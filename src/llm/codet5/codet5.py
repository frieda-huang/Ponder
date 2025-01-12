# Adapted from https://github.com/salesforce/CodeT5/blob/e78a61a17f6dc2f3cbb968447d3e2d065b426e7b/CodeT5/_utils.py
import gzip
import json
import multiprocessing
from typing import List

import torch
from pydantic import BaseModel
from rich import print
from src.commons import project_paths
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm
from transformers import RobertaTokenizer

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
