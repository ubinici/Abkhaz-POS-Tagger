import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from model import POSModel
from dataloader import load_data

def train_model(train_loader, val_loader, num_tags, epochs=10, lr=2e-5, max_grad_norm=1.0):
    """
    Train the POS tagging model.

    Args:
        train_loader: Dataloader for training set
        val_loader: Dataloader for validation set
        num_tags: Number of POS tag classes
        epochs: Number of epochs
        lr: Learning rate
        max_grad_norm: Clip threshold for gradients

    Returns:
        Path to saved model
    """

    # === Setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = POSModel(num_tags).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)

    loss_history = []

    # === Training loop ===
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0

        for batch in tqdm(train_loader, desc=f"🟢 Epoch {epoch}/{epochs} Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits.view(-1, num_tags), labels.view(-1))
            loss.backward()

            # Clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_train_loss)
        print(f"✅ Epoch {epoch} Train Loss: {avg_train_loss:.4f}")

        # === Validation ===
        val_loss, val_acc = validate_model(model, val_loader, loss_fn, device)

        # === Log loss trend and adjust learning rate ===
        print(f"📈 Training Loss History: {loss_history}")
        scheduler.step(val_loss)

    # === Save model ===
    model_path = "pos_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")
    return model_path


def validate_model(model, val_loader, loss_fn, device):
    """
    Validate the model on the validation set.

    Args:
        model: Trained model
        val_loader: Dataloader for validation
        loss_fn: Loss function
        device: CPU or CUDA

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    val_loss = 0
    total_correct = 0
    total_tokens = 0
    tag_errors = {}

    # === Validation loop ===
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="🔵 Validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            val_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            mask = labels != -100

            total_correct += (preds[mask] == labels[mask]).sum().item()
            total_tokens += mask.sum().item()

            # === Count misclassifications per tag ===
            for p, t in zip(preds[mask].tolist(), labels[mask].tolist()):
                if p != t:
                    tag_errors[t] = tag_errors.get(t, 0) + 1

    avg_loss = val_loss / len(val_loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0

    print(f"✅ Validation Loss: {avg_loss:.4f}")
    print(f"✅ Validation Accuracy: {accuracy:.4f}")

    if tag_errors:
        top_errors = sorted(tag_errors.items(), key=lambda x: -x[1])[:5]
        print(f"🔍 Most Misclassified Tags: {top_errors}")

    return avg_loss, accuracy


if __name__ == "__main__":
    train_loader, val_loader, _, num_tags = load_data()
    train_model(train_loader, val_loader, num_tags)

