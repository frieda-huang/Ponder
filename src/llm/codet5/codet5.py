# Adapted from https://github.com/salesforce/CodeT5/blob/e78a61a17f6dc2f3cbb968447d3e2d065b426e7b/CodeT5/_utils.py
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
    assert token_ids.count(tokenizer.eos_token_id) == 1
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

    assert all_source_ids.shape == (len(examples), max_source_length)  # (30,000, 256)
    assert all_target_ids.shape == (len(examples), max_target_length)  # (30,000, 128)

    return TensorDataset(all_source_ids, all_target_ids)


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
    def __init__(self, config: T5Config, has_relative_attention_bias=True):
        super().__init__()
        self.n_heads = config.num_heads
        self.key_value_proj_dim = config.d_kv  # Dimension size for each attention head
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
        self.relative_attention_bias = nn.Embedding(
            self.relative_attention_num_buckets, self.n_heads
        )

    def relative_position_bucket(
        self, relative_position, num_buckets=32, max_distance=128
    ):
        """Translate relative positions into bucket indices"""
        # Handle bidirectional case
        relative_buckets = 0
        num_buckets //= 2  # Split buckets into two halves: 0-15 for negative positions and 16-31 for positive positions
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)

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

        context_position = torch.arange(query_length).unsqueeze(1)  # (query_length, 1)
        memory_position = torch.arange(key_length).unsqueeze(0)  # (1, key_length)

        # Calculate relative positions; shape: (query_length, key_length)
        relative_position = memory_position - context_position

        # Convert to bucket indices
        relative_position_bucket = self.relative_position_bucket(
            relative_position,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )

        # Look up biases from embedding table
        values = self.relative_attention_bias(relative_position_bucket)

        # Shape = (1, num_heads, query_length, key_length)
        values = values.permute([2, 0, 1]).unsqueeze(0)

        return values

    def forward(self, hidden_states):
        batch_size, seq_len = hidden_states.shape[:2]

        k = self.k(hidden_states)
        q = self.q(hidden_states)
        v = self.v(hidden_states)

        # Project and reshape: (batch_size, seq_len, n_heads, key_value_proj_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.key_value_proj_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.key_value_proj_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.key_value_proj_dim)

        # (batch_size, n_heads, seq_len, key_value_proj_dim)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))

        if self.has_relative_attention_bias:
            position_bias = self.compute_bias(seq_len, seq_len)
            scores = scores + position_bias

        scores = scores / math.sqrt(self.key_value_proj_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project back (batch_size, seq_len, n_heads, key_value_proj_dim)
        # .contiguous() ensures tensor is stored contiguously in memory for efficient reshaping
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.inner_dim)
        attn_output = self.o(attn_output)

        # (batch_size, seq_len, d_model)
        return attn_output


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.attention = T5Attention(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, mask=None):
        # Layer norm first (pre-norm architecture)
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.attention(normed_hidden_states, mask=mask)
        hidden_states = hidden_states + attention_output

        return hidden_states


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.encoder_decoder_attn = T5Attention(
            config, has_relative_attention_bias=False
        )
        self.layer_norm = T5LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self, hidden_states, key_value_states, attention_mask=None, position_bias=None
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.encoder_decoder_attn(
            normed_hidden_states,
            key_value_states=key_value_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        return layer_output


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


class T5LayerFeedForward(nn.Module):
    """Feed-forward layer with layer norm and residual connection"""

    def __init__(self, config: T5Config):
        super().__init__()
        self.feedforward = T5DenseActDense(config)
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
        encoder_hidden_states=None,
        attention_mask=None,
        encoder_attention_mask=None,
    ):
        # Self attention
        hidden_states = self.layer[0](hidden_states, attention_mask)

        # Cross attention (decoder only)
        if self.is_decoder and encoder_hidden_states is not None:
            hidden_states = self.layer[1](
                hidden_states, encoder_hidden_states, encoder_attention_mask
            )

        # Feed-forward
        hidden_states = self.layer[-1](hidden_states)

        return hidden_states


class T5Stack(nn.Module):
    """Stack of encoder/decoder blocks"""

    def __init__(self, config: T5Config, is_decoder=False):
        super().__init__()
        self.config = config
        self.is_decoder = is_decoder
        self.block = nn.ModuleList(
            [
                T5Block(config, is_decoder)
                for _ in range(
                    config.num_decoder_layers if is_decoder else config.num_layers
                )
            ]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        encoder_attention_mask=None,
    ):
        for block in self.block:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
        return self.final_layer_norm(hidden_states)


class T5Model(PreTrainedModel):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = T5Stack(config, is_decoder=False)
        self.decoder = T5Stack(config, is_decoder=True)

    def forward(
        self,
        input_ids,
        decoder_input_ids,
        attention_mask=None,
        decoder_attention_mask=None,
    ):
        # Encode
        encoder_hidden_states = self.shared(input_ids)
        encoder_hidden_states = self.encoder(
            encoder_hidden_states, attention_mask=attention_mask
        )

        # Decode
        decoder_hidden_states = self.shared(decoder_input_ids)
        decoder_hidden_states = self.decoder(
            decoder_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=attention_mask,
        )

        return decoder_hidden_states


class T5ForConditionalGeneration(nn.Module):
    pass


def main():
    from transformers import T5ForConditionalGeneration

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")
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
            labels = batch[1].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

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


if __name__ == "__main__":
    main()
