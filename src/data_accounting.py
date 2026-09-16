import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset


class InstructionDataset(Dataset):
    """
    A PyTorch Dataset that formats instructions and masks prompt tokens with -100.
    """
    def __init__(self, raw_data, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        print("Processing and formatting raw samples...")
        for item in raw_data:
            instruction = item.get("instruction", "")
            user_input = item.get("input", "")
            output = item.get("output", "")

            # If there is additional user input context, append it to instruction
            if user_input.strip():
                full_instruction = f"{instruction}\n\nInput Context:\n{user_input}"
            else:
                full_instruction = instruction

            # Standard ChatML format
            prompt_text = f"<|im_start|>user\n{full_instruction}<|im_end|>\n<|im_start|>assistant\n"
            full_text = prompt_text + f"{output}<|im_end|>"

            # Tokenize prompt and full text without padding yet (padding happens in batches)
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = self.tokenizer(full_text, max_length=self.max_length, truncation=True, add_special_tokens=False)["input_ids"]

            prompt_len = len(prompt_ids)
            full_len = len(full_ids)

            # If the prompt alone exceeded max_length, skip
            if prompt_len >= full_len:
                continue

            labels = list(full_ids)
            labels[:len(prompt_ids)] = [-100] * len(prompt_ids)
            

            self.samples.append({
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, pad_token_id: int):
    """
    Combines a list of sample dicts into a single padded batch tensor.
    """
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "labels": padded_labels
    }


def count_tokens(batch):
    """
    Calculates token statistics for the batch:
    - total_elements: B * L (all positions in the tensor)
    - non_pad_tokens: count of real tokens (where attention_mask == 1)
    - trained_tokens: count of tokens with loss calculated (where labels != -100)
    """

    total_elements = batch["input_ids"].numel()
    non_pad_tokens = (batch["attention_mask"] == 1).sum().item()
    trained_tokens = (batch["labels"] != -100).sum().item()

    return {
        "total_elements": total_elements,
        "non_pad_tokens": non_pad_tokens,
        "trained_tokens": trained_tokens
    }


def demo_data_pipeline():
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Loading tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading 20 samples from yahma/alpaca-cleaned dataset...")
    raw_data = load_dataset("yahma/alpaca-cleaned", split="train[:20]")

    dataset = InstructionDataset(raw_data, tokenizer, max_length=256)
    print(f"Dataset successfully created with {len(dataset)} valid samples.")

    # Create a DataLoader with batch_size=4
    # Note: lambda batch: collate_fn(batch, pad_token_id=tokenizer.pad_token_id) passes the pad id into collate
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id)
    )

    # Inspect the first batch
    for batch_idx, batch in enumerate(loader):
        print(f"\n" + "=" * 50)
        print(f"Batch {batch_idx + 1} Inspection:")
        print(f"input_ids shape:      {list(batch['input_ids'].shape)}  [Batch, Length]")
        print(f"attention_mask shape: {list(batch['attention_mask'].shape)}")
        print(f"labels shape:         {list(batch['labels'].shape)}")

        stats = count_tokens(batch)
        print(f"\nToken Accounting Metrics:")
        print(f"  - Total Tensor Slots (B x L): {stats['total_elements']}")
        print(f"  - Real / Non-Padding Tokens:  {stats['non_pad_tokens']} ({100 * stats['non_pad_tokens'] / stats['total_elements']:.1f}% efficiency)")
        print(f"  - Response Tokens (Trained):  {stats['trained_tokens']} (Loss is computed only on these)")

        # Inspect Sample 0 labels in this batch
        sample_labels = batch["labels"][0].tolist()
        num_masked = sample_labels.count(-100)
        num_active = len(sample_labels) - num_masked
        print(f"\nSample 0 in this batch:")
        print(f"  - Total length: {len(sample_labels)}")
        print(f"  - Masked tokens (-100): {num_masked} (Prompt + Padding)")
        print(f"  - Active prediction tokens: {num_active} (Assistant Response)")
        break


if __name__ == "__main__":
    demo_data_pipeline()