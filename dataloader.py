import torch
from transformers import XLMRobertaTokenizerFast, DataCollatorForTokenClassification
from torch.utils.data import DataLoader

def load_data(batch_size=16, data_path="tokenized_abkhaz_dataset.pth"):
    """
    Load and prepare data loaders for training, validation, and testing.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_tags)
    """

    # === Load tokenized dataset ===
    try:
        data = torch.load(data_path)
        print(f"✅ Loaded tokenized dataset from {data_path}")
    except Exception as e:
        raise RuntimeError(f"❌ Error loading tokenized dataset: {e}")

    # === Sanity check ===
    print(f"🔍 Sample entry from train set:\n{data['train'][0]}")

    # === Load tokenizer (must match preprocessor) ===
    tok = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-large")

    # === Define collator to batch and pad sequences ===
    collate = DataCollatorForTokenClassification(tokenizer=tok, return_tensors="pt")

    # === Build DataLoaders for each split ===
    train_loader = DataLoader(data["train"], batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(data["validation"], batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(data["test"], batch_size=batch_size, shuffle=False, collate_fn=collate)

    # === Count unique labels (excluding ignored tokens) ===
    tag_set = {tag for ex in data["train"] for tag in ex["labels"] if tag != -100}
    num_tags = len(tag_set)

    return train_loader, val_loader, test_loader, num_tags


if __name__ == "__main__":
    train_loader, val_loader, test_loader, num_tags = load_data()
    print(f"✅ Number of POS tags: {num_tags}")


