import torch
from transformers import RobertaTokenizer
from typing import List, Optional
import sys
import os

# Add the parent directory to the system path so we can import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.codet5.codet5 import T5Config, T5ForConditionalGeneration


def generate_code_summary(
    model,
    tokenizer: RobertaTokenizer,
    code: str,
    device: str,
    max_length: int = 128,
    num_beams: int = 4,
    temperature: float = 0.7,
) -> str:
    """Generate a natural language summary for the given code snippet."""
    model.eval()

    # Prepare input
    source_str = f"Summarize: {code}"
    inputs = tokenizer(
        source_str,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
        )

    decoded_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_summary.strip()


def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model & config
    try:
        checkpoint = torch.load("codet5_model.pt")

        # The config in your checkpoint is already a T5Config instance
        saved_config = checkpoint["config"]
        print("Loaded config:", saved_config)

        # Extract model state dict
        model_state = checkpoint["model_state_dict"]

        # Initialize model with the saved config
        model = T5ForConditionalGeneration(saved_config)
        model.load_state_dict(model_state)

        # Optional: Inspect one of the loaded weights for debugging
        for k, v in model_state.items():
            print(f"Loaded weight: {k}, shape = {v.shape}")
            break

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model = model.to(device)

    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")

    # Test examples
    test_examples = [
        """
        def factorial(n):
            if n == 0:
                return 1
            return n * factorial(n-1)
        """,
        """
        def bubble_sort(arr):
            n = len(arr)
            for i in range(n):
                for j in range(0, n-i-1):
                    if arr[j] > arr[j+1]:
                        arr[j], arr[j+1] = arr[j+1], arr[j]
            return arr
        """,
    ]

    print("\nGenerating summaries for test examples:")
    for i, code in enumerate(test_examples, 1):
        try:
            summary = generate_code_summary(model, tokenizer, code, device)
            print(f"\nExample {i}:")
            print("Code:")
            print(code)
            print("Generated Summary:")
            print(summary)
        except Exception as e:
            print(f"Error generating summary for example {i}: {e}")


if __name__ == "__main__":
    main()
