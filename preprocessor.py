from conllu import parse
from transformers import XLMRobertaTokenizerFast
import torch
import random

# === Load CoNLL-U formatted data ===
conllu_file = "ab_abnc-ud-test.conllu"  # Path to annotated dataset

with open(conllu_file, "r", encoding="utf-8") as f:
    sents = parse(f.read())  # Parse file into list of sentences

# === Extract raw tokens and corresponding POS tags ===
data = []
for sent in sents:
    toks = [tok["form"] for tok in sent]       # Surface words
    tags = [tok["upostag"] for tok in sent]    # Universal POS tags
    data.append({"tokens": toks, "upos": tags})

# === Shuffle dataset for randomness ===
random.shuffle(data)

# === Train/Validation/Test split (80/10/10) ===
n = len(data)
n_train = int(0.8 * n)
n_val = int(0.1 * n)
n_test = n - n_train - n_val

train = data[:n_train]
val = data[n_train:n_train + n_val]
test = data[n_train + n_val:]

# Organize into dictionary for ease of access
splits = {"train": train, "validation": val, "test": test}

# === Load tokenizer ===
# This tokenizer splits words into subword tokens and maps them to input IDs
tok = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")

# === Get unique POS tags in training set for ID mapping ===
tags_set = set(tag for ex in splits["train"] for tag in ex["upos"])
print("\n✅ Unique POS tags:", tags_set)

# === Map POS tags <-> integer IDs for model training ===
tag2id = {tag: i for i, tag in enumerate(sorted(tags_set))}
id2tag = {i: tag for tag, i in tag2id.items()}
print("\n✅ POS tag to ID:", tag2id)


# === Function to tokenize text and align labels to token positions ===
def tokenize_and_align_labels(ex):
    # Tokenize input sequence (word-level split, pad to 128 tokens)
    tok_inp = tok(
        ex["tokens"],
        truncation=True,
        is_split_into_words=True,  # Ensures alignment with original word indices
        padding="max_length",
        max_length=128,
    )
    
    word_ids = tok_inp.word_ids()  # Maps each token to its original word index
    prev_wid = None                # Track word boundaries to assign one label per word
    lbl_ids = []                   # Will store label IDs aligned to token positions
    misaligned = 0                # Count mismatches (e.g., if word index goes out of bounds)

    # Loop through tokens and assign POS label where appropriate
    for wid in word_ids:
        if wid is None:
            # Special tokens (CLS, SEP, PAD) → ignore during loss computation
            lbl_ids.append(-100)
        elif wid != prev_wid:
            # Start of a new word
            if wid < len(ex["upos"]):
                tag = ex["upos"][wid]
                lbl_ids.append(tag2id.get(tag, -100))  # Get ID or fallback
            else:
                # Rare: tokenizer mapped to non-existent word
                misaligned += 1
                lbl_ids.append(-100)
        else:
            # Subword token → ignore for loss calculation
            lbl_ids.append(-100)
        prev_wid = wid

    tok_inp["labels"] = lbl_ids  # Add aligned labels to tokenized input

    # Print warning for misaligned tokens
    if misaligned > 0:
        print(f"⚠ {misaligned} token(s) misaligned: {ex['tokens']}")
    
    return tok_inp


# === Apply preprocessing to all dataset splits ===
print("\n🔄 Tokenizing...")
tok_ds = {
    split: [tokenize_and_align_labels(ex) for ex in splits[split]]
    for split in splits
}

# === Diagnostic: check for POS tags missed due to misalignment ===
n_total = sum(len(splits[sp]) for sp in splits)
n_misaligned = sum(1 for sp in splits for ex in splits[sp] if -100 in ex["upos"])  # Conservative estimate
print(f"\n✅ Tokenization done: {n_total - n_misaligned}/{n_total} aligned")

# === Save the tokenized dataset for model training ===
out_file = "tokenized_abkhaz_dataset.pth"
torch.save(tok_ds, out_file)
print(f"✅ Saved tokenized dataset to {out_file}")

