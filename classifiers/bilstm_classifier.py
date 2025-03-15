import torch.nn as nn
from transformers import AutoModel


class BiLSTMClassifier(nn.Module):
    def __init__(self, model_path, num_labels, lstm_hidden_size=256, dropout_rate=0.1):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_path, local_files_only=True)
        hidden_size = self.transformer.config.hidden_size
        self.num_labels = num_labels

        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_size, num_labels),
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Get transformer outputs - full sequence
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        # Process sequence with BiLSTM
        sequence_output = outputs.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)

        # Use the last hidden state of the LSTM
        # Combine forward and backward directions
        pooled_output = lstm_output[:, -1, :]

        # Classification
        logits = self.classifier(pooled_output)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return (
            {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
        )