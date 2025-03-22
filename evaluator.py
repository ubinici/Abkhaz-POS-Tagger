import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
from collections import defaultdict

from model import POSModel
from dataloader import load_data

def evaluate_model(test_loader, num_tags, model_path="pos_model.pth"):
    """
    Evaluate model performance on the test set.

    Args:
        test_loader: DataLoader for test set
        num_tags: Number of POS tag classes
        model_path: Path to trained model
    """

    # === Setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = POSModel(num_tags).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    total_loss = 0
    total_correct = 0
    total_tokens = 0

    true_labels = []
    pred_labels = []
    incorrect = []
    tag_errors = defaultdict(int)

    # === Evaluation loop ===
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="🔵 Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            mask = labels != -100

            # Collect predictions and gold labels
            pred_list = preds[mask].tolist()
            true_list = labels[mask].tolist()
            true_labels.extend(true_list)
            pred_labels.extend(pred_list)

            # Track misclassified tokens
            for p, t, inp in zip(pred_list, true_list, input_ids[mask].tolist()):
                if p != t:
                    incorrect.append((inp, p, t))
                    tag_errors[t] += 1

            total_correct += (preds[mask] == labels[mask]).sum().item()
            total_tokens += mask.sum().item()

    # === Metrics ===
    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    f1 = f1_score(true_labels, pred_labels, average="macro")

    print(f"\n✅ Test Loss: {avg_loss:.4f}")
    print(f"✅ Test Accuracy: {accuracy:.4f}")
    print(f"✅ Test F1 Score: {f1:.4f}")

    # === Error analysis ===
    if tag_errors:
        top_errors = sorted(tag_errors.items(), key=lambda x: -x[1])[:5]
        print(f"\n🔍 Most Misclassified POS Tags: {top_errors}")

    if incorrect:
        print("\n🔍 Sample Misclassified Tokens (Input ID, Predicted, True Label):")
        for ex in incorrect[:10]:
            print(ex)

    # === Save misclassified samples ===
    torch.save(incorrect, "misclassified_samples.pth")
    print(f"\n📁 Misclassified samples saved to misclassified_samples.pth")


if __name__ == "__main__":
    _, _, test_loader, num_tags = load_data()
    evaluate_model(test_loader, num_tags)
