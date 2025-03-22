import torch
from torch import nn
from transformers import XLMRobertaModel

class POSModel(nn.Module):
    """
    POS tagger based on XLM-RoBERTa with a linear classification head.
    """
    def __init__(self, num_tags, freeze_layers=9):
        """
        Args:
            num_tags (int): Number of unique POS tags.
            freeze_layers (int): Number of encoder layers to freeze.
        """
        super(POSModel, self).__init__()

        # === Load pretrained XLM-RoBERTa model ===
        self.roberta = XLMRobertaModel.from_pretrained("xlm-roberta-large")

        # === Freeze embeddings and early transformer layers ===
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
        for param in self.roberta.encoder.layer[:freeze_layers].parameters():
            param.requires_grad = False

        # === Classification head ===
        self.dropout = nn.Dropout(0.3)  # Regularization
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_tags)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.

        Args:
            input_ids (torch.Tensor): Token IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Logits over POS tag classes for each token.
        """
        # === Run transformer and apply classifier ===
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        logits = self.classifier(self.dropout(hidden))
        return logits

