import torch
import torch.nn as nn
from transformers import AutoModel


class AttentionClassifier(nn.Module):
    def __init__(self, model_path, num_labels, dropout_rate=0.1):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_path, local_files_only=True)
        hidden_size = self.transformer.config.hidden_size
        self.num_labels = num_labels  # Store num_labels for use in forward

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
            nn.Softmax(dim=1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Get transformer outputs - ignore unexpected kwargs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        # Get sequence output
        sequence_output = outputs.last_hidden_state

        # Apply attention
        attention_weights = self.attention(sequence_output)
        context_vector = torch.sum(attention_weights * sequence_output, dim=1)

        # Classification
        logits = self.classifier(context_vector)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return (
            {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
        )