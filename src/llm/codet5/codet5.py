# Adapted from https://github.com/salesforce/CodeT5/blob/e78a61a17f6dc2f3cbb968447d3e2d065b426e7b/CodeT5/_utils.py
import copy
import gzip
import json
import multiprocessing
from dataclasses import dataclass
from turtle import forward
from typing import List

import torch
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
    dropout_rate = 0.1
    layer_norm_epsilon = 1e-6
    initializer_factor = 1.0
    is_encoder_decoder = True
    pad_token_id = 0
    eos_token_id = 1
    is_decoder = False


class T5LayerNorm(nn.Module):
    pass


# Differences between T5 and the original Transformer attention
# 1. T5 uses relative position encodings (`_relative_position_bucket` and `compute_bias`) instead of absolute position encodings
# 2. T5 uses a separate dimension (`d_kv`) for keys and values that can be different from `d_model/num_heads`
# 3. T5 uses pre-norm + residual
class T5Attention(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.n_heads = config.num_heads
        self.key_value_proj_dim = config.d_kv
        self.d_model = config.d_model
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.dropout = config.dropout_rate

        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        self.relative_attention_bias = nn.Embedding(
            self.relative_attention_num_buckets, self.n_heads
        )


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


class T5Block(nn.Module):
    pass


class T5Stack(nn.Module):
    pass


class T5Model(PreTrainedModel):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False  # Encoder

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True

        decoder_config.num_layers = config.num_decoder_layers

        self.encoder = T5Stack(encoder_config, self.shared)
        self.decoder = T5Stack(decoder_config, self.shared)


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
