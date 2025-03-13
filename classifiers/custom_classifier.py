import torch.nn as nn
from transformers import AutoModel


class CustomClassifier(nn.Module):
    def __init__(self, model_path, num_labels, dropout_rate=0.1):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_path)
        hidden_size = self.transformer.config.hidden_size
        self.num_labels = num_labels

        # Multi-layer classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_labels),
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        # Use the [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]

        # Pass through the classifier
        logits = self.classifier(pooled_output)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return (
            {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
        )