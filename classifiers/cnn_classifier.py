import torch
import torch.nn as nn
from transformers import AutoModel


class CNNTextClassifier(nn.Module):
    def __init__(self, model_path, num_labels, dropout_rate=0.1):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_path)
        hidden_size = self.transformer.config.hidden_size
        self.num_labels = num_labels

        # CNN layers with different kernel sizes to capture different n-gram features
        self.conv1 = nn.Conv1d(hidden_size, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, 128, kernel_size=4, padding=2)
        self.conv3 = nn.Conv1d(hidden_size, 128, kernel_size=5, padding=2)

        # Pooling layers
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),  # 384 = 128*3 (concatenated CNN features)
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        # Get sequence output and transpose for CNN
        # [batch, seq_len, hidden] -> [batch, hidden, seq_len]
        x = outputs.last_hidden_state.transpose(1, 2)

        # Apply CNN layers
        x1 = self.pool(torch.relu(self.conv1(x))).squeeze(2)  # [batch, 128]
        x2 = self.pool(torch.relu(self.conv2(x))).squeeze(2)  # [batch, 128]
        x3 = self.pool(torch.relu(self.conv3(x))).squeeze(2)  # [batch, 128]

        # Concatenate features from different CNN layers
        x_cat = torch.cat((x1, x2, x3), dim=1)  # [batch, 384]

        # Apply dropout
        x_drop = self.dropout(x_cat)

        # Classification
        logits = self.classifier(x_drop)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return (
            {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
        )