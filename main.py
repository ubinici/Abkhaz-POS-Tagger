import torch
import os
from dataloader import load_data
from trainer import train_model
from evaluator import evaluate_model

def main():
    """
    Full pipeline for XLM-RoBERTa POS tagging:
    1. Ensures the dataset is preprocessed.
    2. Trains the model.
    3. Evaluates on the test set.
    """

    # === Device setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Running on device: {device}")

    # === Preprocessing check ===
    dataset_path = "tokenized_abkhaz_dataset_fixed.pth"
    if not os.path.exists(dataset_path):
        print("\n🔄 Preprocessing dataset...")
        from preprocessor import tokenize_and_align_labels  # Trigger preprocessing
    else:
        print("\n✅ Preprocessed dataset found — skipping preprocessing.")

    # === Load dataset ===
    print("\n📥 Loading dataset...")
    train_loader, val_loader, test_loader, num_tags = load_data()
    print(f"\n📊 Number of POS tags: {num_tags}")

    # === Train model ===
    print("\n🎯 Training model...")
    model_path = train_model(train_loader, val_loader, num_tags)

    # === Evaluate model ===
    print("\n🧪 Evaluating on test set...")
    evaluate_model(test_loader, num_tags, model_path)

    print("\n✅ Done! Model training and evaluation complete.")

if __name__ == "__main__":
    main()
