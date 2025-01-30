# Adapted from https://github.com/salesforce/CodeT5/blob/e78a61a17f6dc2f3cbb968447d3e2d065b426e7b/CodeT5/_utils.py
import copy
import gzip
import json
import math
import multiprocessing
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from rich import print
from src.commons import project_paths
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm
from transformers import PreTrainedModel, RobertaTokenizer

max_source_length = 256
max_target_length = 128


class InputFeatures(BaseModel):
    """A single training/test features for a example."""

    example_id: int
    source_ids: list[int]
    target_ids: list[int]


class Example(BaseModel):
    """A single training/test example."""

    idx: int
    source: str
    target: str


# Contain 30,000 lines
tiny_python_dataset = (
    project_paths.DATA
    / "Python"
    / "python"
    / "final"
    / "jsonl"
    / "train"
    / "python_train_0.jsonl.gz"
)


def read_summarize_examples(filename: str, data_num: int = -1) -> List[Example]:
    """Read examples from filename. By default, show all data"""
    examples = []
    with gzip.open(filename, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if "idx" not in js:
                js["idx"] = idx
            code = " ".join(js["code_tokens"]).replace("\n", " ")
            code = " ".join(code.strip().split())
            nl = " ".join(js["docstring_tokens"]).replace("\n", "")
            nl = " ".join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def get_token_ids(
    raw_str: str, tokenizer: RobertaTokenizer, max_length: int
) -> List[int]:
    raw_str = raw_str.replace("</s>", "<unk>")
    token_ids = tokenizer.encode(
        raw_str,
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    return token_ids


def convert_examples_to_features(example: Example, tokenizer: RobertaTokenizer):
    source_str = f"Summarize: {example.source}"
    source_ids = get_token_ids(source_str, tokenizer, max_source_length)

    target_str = "<en> " + example.target
    target_ids = get_token_ids(target_str, tokenizer, max_target_length)

    return InputFeatures(
        example_id=example.idx,
        source_ids=source_ids,
        target_ids=target_ids,
    )


def preprocess_training_data(tokenizer: RobertaTokenizer) -> TensorDataset:
    examples = read_summarize_examples(tiny_python_dataset)

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    tuple_examples = [(example, tokenizer) for example in examples]

    features = pool.starmap(
        convert_examples_to_features, tqdm(tuple_examples, total=len(examples))
    )

    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)

    all_source_mask = (all_source_ids != tokenizer.pad_token_id).long()
    all_target_mask = (all_target_ids != tokenizer.pad_token_id).long()

    assert all_source_ids.shape == (len(examples), max_source_length)  # (30,000, 256)
    assert all_target_ids.shape == (len(examples), max_target_length)  # (30,000, 128)

    # Create labels by replacing padding after EOS with -100
    labels = all_target_ids.clone()
    eos_token_id = tokenizer.eos_token_id
    for i in range(labels.size(0)):
        eos_pos = (labels[i] == eos_token_id).nonzero()
        if eos_pos.numel() > 0:
            labels[i, eos_pos[0] + 1 :] = -100  # Mask padding after EOS

    return TensorDataset(
        all_source_ids, all_source_mask, all_target_ids, all_target_mask, labels
    )


# T5 architecture configuration
@dataclass
class T5Config:
    vocab_size = 32128
    d_model = 512  # Size of the encoder layers and the pooler layer
    d_kv = 64  # Size of the key, query, value projections per attention head
    d_ff = 2048  # Size of the intermediate feed forward layer in each `T5Block`
    num_layers = 6  # Number of hidden layers in the Transformer encoder
    num_decoder_layers = 6  # Number of hidden layers in the Transformer decoder
    num_heads = 8  # Number of attention heads for each attention layer in the Transformer encoder
    # The number of buckets to use for each attention layer
    relative_attention_num_buckets = 32
    relative_attention_max_distance = 128
    dropout_rate = 0.1
    layer_norm_epsilon = 1e-6
    initializer_factor = 1.0
    is_encoder_decoder = True
    pad_token_id = 0
    eos_token_id = 1
    is_decoder = False
    decoder_start_token_id = 0  # Usually the same as pad


class T5LayerNorm(nn.Module):
    """T5-specific layer normalization using RMS (Root Mean Square) Normalization

    Unlike traditional Layer Normalization, T5 uses RMS Norm which:
    1. Omits mean subtraction (centering)
    2. Only normalizes by the root mean square
    3. Applies learnable per-feature scaling factors

    Args:
        hidden_size: Dimension of the hidden states (feature dimension)
        eps: Small constant for numerical stability (default: 1e-6)
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        # Learnable parameter initialized to ones; allow the network to learn an optimal scaling for each feature dimension
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # Note that RMS: x/sqrt(variance), which is different from standard layer norm: (x - mean)/sqrt(variance)
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # Each feature is scaled by its own learned factor, allowing the model to amplify important features
        return self.weight * normalized


# Differences between T5 and the original Transformer attention
# 1. T5 uses relative position encodings (`_relative_position_bucket` and `compute_bias`) instead of absolute position encodings
# 2. T5 uses a separate dimension (`d_kv`) for keys and values that can be different from `d_model/num_heads`
# 3. T5 uses pre-norm + residual
class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.n_heads = config.num_heads
        self.is_decoder = config.is_decoder
        self.key_value_proj_dim = (
            config.d_model // config.num_heads
        )  # Dimension size for each attention head
        self.d_model = config.d_model  # Embedding dimension
        self.inner_dim = (
            self.n_heads * self.key_value_proj_dim
        )  # Total dimension across all attention heads
        self.has_relative_attention_bias = has_relative_attention_bias

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.dropout = config.dropout_rate

        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        # Embedding table for relative position buckets
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )

    def relative_position_bucket(
        self, relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        """Translate relative positions into bucket indices"""
        # Handle bidirectional case
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2  # Split buckets into two halves: 0-15 for negative positions and 16-31 for positive positions
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:  # Handle unidirectional case (decoder self-attention)
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )

        max_exact = num_buckets // 2  # First 8 positions get their own exact buckets
        is_small = relative_position < max_exact

        relative_pos_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)  # Normalizing factor
            * (num_buckets - max_exact)  # Remaining buckets
        ).to(torch.long)

        relative_pos_if_large = torch.min(
            relative_pos_if_large,
            torch.full_like(relative_pos_if_large, num_buckets - 1),
        )
        relative_buckets += torch.where(
            is_small, relative_position, relative_pos_if_large
        )

        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute relative position bias between query and key positions"""
        device = self.relative_attention_bias.weight.device

        # (query_length, 1)
        context_position = torch.arange(query_length, device=device).unsqueeze(1)

        # (1, key_length)
        memory_position = torch.arange(key_length, device=device).unsqueeze(0)

        # Calculate relative positions; shape: (query_length, key_length)
        relative_position = memory_position - context_position

        # Convert to bucket indices
        relative_position_bucket = self.relative_position_bucket(
            relative_position,
            bidirectional=not self.is_decoder,  # Bidirectional for encoder, unidirectional for decoder
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )

        # Look up biases from embedding table
        values = self.relative_attention_bias(relative_position_bucket)

        # Shape = (1, num_heads, query_length, key_length)
        values = values.permute([2, 0, 1]).unsqueeze(0)

        return values

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        key_value_states=None,
        position_bias=None,
    ):
        if attention_mask is not None and attention_mask.dim() == 2:
            # e.g. [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask[:, None, None, :]

        # hidden_states: [batch_size, seq_len, d_model]
        batch_size, seq_len = hidden_states.shape[:2]
        is_cross_attention = key_value_states is not None

        q = self.q(hidden_states)

        # key_value_states is from encoder
        current_states = key_value_states if is_cross_attention else hidden_states

        k = self.k(current_states)
        v = self.v(current_states)

        # Project and reshape: (batch_size, seq_len, n_heads, key_value_proj_dim)
        k = k.view(batch_size, -1, self.n_heads, self.key_value_proj_dim)
        q = q.view(batch_size, -1, self.n_heads, self.key_value_proj_dim)
        v = v.view(batch_size, -1, self.n_heads, self.key_value_proj_dim)

        # (batch_size, n_heads, seq_len, key_value_proj_dim)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))
        key_length = k.shape[-2]

        # Handle position bias (either cached or computed)
        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, seq_len, key_length),
                    device=scores.device,
                    dtype=scores.dtype,
                )
            else:
                position_bias = self.compute_bias(seq_len, key_length)

            # Add attention mask if provided
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : k.shape[-2]]
                position_bias = position_bias + causal_mask

        # Add position bias to scores
        scores += position_bias

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project back (batch_size, seq_len, n_heads, key_value_proj_dim)
        # .contiguous() ensures tensor is stored contiguously in memory for efficient reshaping
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.inner_dim)
        attn_output = self.o(attn_output)

        # attn_output shape: (batch_size, seq_len, d_model)
        # position_bias shape:(1, n_heads, seq_len, key_length)
        return attn_output, position_bias


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.attention = T5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, attention_mask=None, position_bias=None):
        # Layer norm first (pre-norm architecture)
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output, position_bias = self.attention(
            normed_hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
        )
        hidden_states = hidden_states + self.dropout(attention_output)

        return hidden_states, position_bias


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.encoder_decoder_attn = T5Attention(
            config, has_relative_attention_bias=False
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        attention_mask=None,
        position_bias=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output, position_bias = self.encoder_decoder_attn(
            normed_hidden_states,
            key_value_states=key_value_states,
            attention_mask=encoder_attention_mask,
            position_bias=position_bias,
        )
        hidden_states = hidden_states + self.dropout(attention_output)

        return hidden_states, position_bias


class T5DenseActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()

        # First projection (expansion)
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)

        # Second projection (contraction)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = nn.ReLU()  # Activation function

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)

        return hidden_states


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = nn.GELU()

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))  # Activated branch
        hidden_linear = self.wi_1(hidden_states)  # Linear gating branch
        hidden_states = hidden_gelu * hidden_linear  # Element-wise multiplication
        hidden_states = self.dropout(hidden_states)  # Regularization
        hidden_states = self.wo(hidden_states)  # Project back to d_model
        return hidden_states


class T5LayerFeedForward(nn.Module):
    """Feed-forward layer with layer norm and residual connection"""

    def __init__(self, config: T5Config):
        super().__init__()
        self.feedforward = T5DenseGatedActDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        normed_hidden_states = self.layer_norm(hidden_states)
        ff_output = self.feedforward(normed_hidden_states)

        return hidden_states + self.dropout(ff_output)


class T5Block(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        """
        Self-attention allows the model to relate different positions within the same sequence
        Cross-attention allows the decoder to attent to encoder outputs
        Feed-forward processes each position independently with a small neural network
        """
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        # Always add self-attention as first layer
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias))

        # For decoder only, add cross-attention as second layer
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        # Always add feed-forward as last layer
        self.layer.append(T5LayerFeedForward(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        # Self attention
        hidden_states, position_bias = self.layer[0](hidden_states, attention_mask)

        # Cross attention (decoder only)
        if self.is_decoder and encoder_hidden_states is not None:
            hidden_states, _ = self.layer[1](
                hidden_states,  # Queries: "What should come after 'The cat is'?"
                key_value_states=encoder_hidden_states,  # Keys/Values: Look up information from "Le chat est noir"
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )

        # Feed-forward
        hidden_states = self.layer[-1](hidden_states)

        return hidden_states, position_bias


class T5Stack(nn.Module):
    """Stack of encoder/decoder blocks"""

    def __init__(self, config: T5Config, embed_tokens):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [
                # Only the first layer (i==0) gets relative attention bias
                T5Block(config, has_relative_attention_bias=bool(i == 0))
                for i in range(config.num_layers)
            ]
        )
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        input_ids,
        encoder_hidden_states=None,
        attention_mask=None,
        encoder_attention_mask=None,
    ):
        hidden_states = self.embed_tokens(input_ids)
        position_bias = None

        for _, layer_module in enumerate(self.block):
            if self.is_decoder:
                # Reuse position_bias across layers
                hidden_states, position_bias = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                hidden_states, position_bias = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_bias=position_bias,
                )

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class T5Model(PreTrainedModel):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = T5Stack(config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        self.decoder = T5Stack(decoder_config, self.shared)

    def forward(
        self,
        input_ids,
        decoder_input_ids,
        attention_mask=None,
        decoder_attention_mask=None,
    ):
        encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask)
        decoder_outputs = self.decoder(
            decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=attention_mask,
        )

        return decoder_outputs


class T5ForConditionalGeneration(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        self.model_dim = config.d_model

        # Shared embedding layer for encoder and decoder
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # Encoder configuration
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        # Decoder configuration
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        self.decoder = T5Stack(decoder_config, self.shared)

        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def _shift_right(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Standard T5 shift-right:
        - Move all tokens to the right by 1.
        - Place decoder_start_token_id at position 0.
        - Replace any -100 with pad_token_id so we don't feed them into the decoder.
        """
        pad_token_id = self.config.pad_token_id
        start_token_id = self.config.decoder_start_token_id

        shifted = labels.new_zeros(labels.shape)
        shifted[..., 1:] = labels[..., :-1].clone()
        shifted[..., 0] = start_token_id
        # Replace -100 with pad_token_id
        shifted.masked_fill_(shifted == -100, pad_token_id)
        return shifted

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
    ):
        # If training (labels given) but decoder_input_ids are None, shift-right
        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels)

        # Encoding
        encoder_outputs = self.encoder(input_ids=input_ids)

        # Decoding
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=attention_mask,
        )

        # Compute logits
        lm_logits = self.lm_head(decoder_outputs)

        # Compute loss (if labels are provided)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        return (loss, lm_logits) if loss is not None else lm_logits

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=128,
        num_beams=4,
        temperature=1.0,
        pad_token_id=None,
        eos_token_id=None,
        early_stopping=True,
    ):
        """Generate text using beam search decoding."""
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Get encoder outputs first
        encoder_outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Initialize beam state
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
            device=device,
            pad_token_id=(
                pad_token_id if pad_token_id is not None else self.config.pad_token_id
            ),
            eos_token_id=(
                eos_token_id if eos_token_id is not None else self.config.eos_token_id
            ),
        )

        # Expand encoder outputs for beam search
        expanded_return_idx = (
            torch.arange(batch_size).view(-1, 1).repeat(1, num_beams).view(-1)
        ).to(device)
        encoder_outputs = encoder_outputs.index_select(0, expanded_return_idx)

        # Expand attention mask for beam search
        if attention_mask is not None:
            expanded_attention_mask = attention_mask.index_select(
                0, expanded_return_idx
            )
        else:
            expanded_attention_mask = None

        # Initialize decoder input ids
        decoder_input_ids = torch.full(
            (batch_size * num_beams, 1),
            self.config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )

        # Initialize scores tensor
        beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=device
        )
        beam_scores[:, 1:] = float(
            "-inf"
        )  # Initialize all beams except first to large negative value
        beam_scores = beam_scores.view(-1)  # shape: (batch_size * num_beams,)

        for step in range(max_length):
            # Get decoder outputs
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs,
                encoder_attention_mask=expanded_attention_mask,
            )

            # Get next token logits
            next_token_logits = self.lm_head(decoder_outputs)[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Calculate log probabilities
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)

            # Add beam scores to token scores
            next_token_scores = next_token_scores + beam_scores[:, None]

            # Reshape scores for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(
                batch_size, num_beams * vocab_size
            )

            # Get next tokens and their scores
            next_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_beam_indices = torch.div(
                next_tokens, vocab_size, rounding_mode="floor"
            )
            next_tokens = next_tokens % vocab_size

            # Reorder batch dimensions
            beam_outputs = beam_scorer.process(
                decoder_input_ids,
                next_scores,
                next_tokens,
                next_beam_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            decoder_input_ids = torch.cat(
                [decoder_input_ids[beam_idx], beam_next_tokens.unsqueeze(-1)], dim=-1
            )

            if beam_scorer.is_done:
                break

        # Finalize beam search
        return beam_scorer.finalize(
            decoder_input_ids,
            beam_scores,
            next_tokens,
            next_beam_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )


class BeamSearchScorer:
    """A refined beam search scorer that handles batch processing properly."""

    def __init__(
        self,
        batch_size,
        max_length,
        num_beams,
        device,
        pad_token_id=None,
        eos_token_id=None,
    ):
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_beams = num_beams
        self.device = device
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        # Track generated sequences and scores for each batch
        self.saved_sequences = [[] for _ in range(batch_size)]
        self.saved_scores = [[] for _ in range(batch_size)]

        self.is_done = False
        self._done_batches = [False] * batch_size

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: int = None,
        eos_token_id: int = None,
    ) -> dict:
        cur_len = input_ids.shape[-1]
        batch_size = len(self._done_batches)

        next_beam_scores = torch.zeros(
            (batch_size, self.num_beams),
            dtype=next_scores.dtype,
            device=next_scores.device,
        )
        next_beam_tokens = torch.zeros(
            (batch_size, self.num_beams),
            dtype=next_tokens.dtype,
            device=next_tokens.device,
        )
        next_beam_indices = torch.zeros(
            (batch_size, self.num_beams),
            dtype=next_indices.dtype,
            device=next_indices.device,
        )

        for batch_idx in range(batch_size):
            if self._done_batches[batch_idx]:
                continue

            beam_idx = batch_idx * self.num_beams
            next_score = next_scores[batch_idx]
            next_token = next_tokens[batch_idx]
            next_index = next_indices[batch_idx]

            # Update next beam content
            next_beam_scores[batch_idx] = next_score[: self.num_beams]
            next_beam_tokens[batch_idx] = next_token[: self.num_beams]
            next_beam_indices[batch_idx] = next_index[: self.num_beams]

            # Check if batch is done
            if eos_token_id is not None:
                if (next_token[: self.num_beams] == eos_token_id).any():
                    self._done_batches[batch_idx] = True

        self.is_done = all(self._done_batches)

        return {
            "next_beam_scores": next_beam_scores.view(-1),
            "next_beam_tokens": next_beam_tokens.view(-1),
            "next_beam_indices": next_beam_indices.view(-1),
        }

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        pad_token_id: int = None,
        eos_token_id: int = None,
    ) -> torch.LongTensor:
        batch_size = len(self._done_batches)

        # Finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx in range(batch_size):
            if self._done_batches[batch_idx]:
                continue

            # All done so add remaining beams
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                self.saved_sequences[batch_idx].append(input_ids[batch_beam_idx])
                self.saved_scores[batch_idx].append(final_beam_scores[batch_beam_idx])

        # Select best hypotheses
        output_sequences = []
        for batch_idx, (sequences, scores) in enumerate(
            zip(self.saved_sequences, self.saved_scores)
        ):
            if not sequences:
                # If no sequences were saved for this batch, use the input_ids corresponding to the first beam
                output_sequences.append(input_ids[batch_idx * self.num_beams])
            else:
                # Select the sequence with the highest score
                best_score_idx = torch.tensor(scores).argmax()
                output_sequences.append(sequences[best_score_idx])

        return torch.stack(output_sequences)


def main():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    config = T5Config()
    config.decoder_start_token_id = (
        config.pad_token_id
    )  # Typically T5 does <pad> as start
    model = T5ForConditionalGeneration(config)
    model = model.to(device)

    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
    train_data = preprocess_training_data(tokenizer)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=8)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.train()  # Set to training mode

    for epoch in range(1):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")

        for step, batch in enumerate(progress_bar):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            decoder_input_ids = batch[2].to(device)
            decoder_attention_mask = batch[3].to(device)
            labels = batch[4].to(device)

            loss, lm_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
            )

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()
            optimizer.zero_grad()

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())

            # Print progress
            if step % 100 == 0:
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item():.4f}")

    # Save final model after training completes
    final_model_path = "codet5_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
        },
        final_model_path,
    )
    print(f"Saved final model to {final_model_path}")


if __name__ == "__main__":
    main()
