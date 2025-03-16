import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class BiLSTMClassifier(nn.Module):
    def __init__(self, model_path, num_labels, lstm_hidden_size=256, dropout_rate=0.1, use_multilayer=False, num_layers_to_use=3):
        super().__init__()
        
        # If using multilayer output, configure the model to output all hidden states
        if use_multilayer:
            config = AutoConfig.from_pretrained(model_path, output_hidden_states=True, local_files_only=True)
            self.transformer = AutoModel.from_pretrained(model_path, config=config, local_files_only=True)
        else:
            self.transformer = AutoModel.from_pretrained(model_path, local_files_only=True)
            
        hidden_size = self.transformer.config.hidden_size
        self.num_labels = num_labels
        self.use_multilayer = use_multilayer
        
        # ModernBERT uses 'layers' instead of 'encoder.layer'
        if hasattr(self.transformer, 'layers'):
            self.num_layers_to_use = min(num_layers_to_use, len(self.transformer.layers))
        elif hasattr(self.transformer, 'encoder') and hasattr(self.transformer.encoder, 'layer'):
            self.num_layers_to_use = min(num_layers_to_use, len(self.transformer.encoder.layer))
        else:
            self.num_layers_to_use = num_layers_to_use
            
        # Create a separate BiLSTM for each layer if using multilayer
        if use_multilayer:
            self.lstm_modules = nn.ModuleList([
                nn.LSTM(
                    input_size=hidden_size,
                    hidden_size=lstm_hidden_size,
                    batch_first=True,
                    bidirectional=True,
                ) for _ in range(self.num_layers_to_use)
            ])
            # Classifier will take concatenated output from all LSTMs
            classifier_input_size = lstm_hidden_size * 2 * self.num_layers_to_use
        else:
            # BiLSTM layer (original single layer)
            self.lstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=lstm_hidden_size,
                batch_first=True,
                bidirectional=True,
            )
            # Original single classifier
            classifier_input_size = lstm_hidden_size * 2
            
        # Add layer normalization
        self.layer_norm = nn.LayerNorm(classifier_input_size)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
            
        # Classifier
        self.classifier = nn.Linear(classifier_input_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=self.use_multilayer  # Only get hidden states if needed
        )

        if self.use_multilayer:
            # Get the last n hidden states
            all_hidden_states = outputs.hidden_states
            
            # Use the last num_layers_to_use layers
            selected_hidden_states = all_hidden_states[-self.num_layers_to_use:]
            
            # Process each layer with its own LSTM
            lstm_outputs = []
            
            for i, hidden_states in enumerate(selected_hidden_states):
                # Process this layer's sequence with its LSTM
                lstm_output, _ = self.lstm_modules[i](hidden_states)
                
                # Use the last hidden state of the LSTM
                layer_output = lstm_output[:, -1, :]
                lstm_outputs.append(layer_output)
            
            # Concatenate outputs from all layers
            combined_output = torch.cat(lstm_outputs, dim=1)
            
            # Apply normalization and dropout
            normalized_output = self.layer_norm(combined_output)
            pooled_output = self.dropout(normalized_output)
        else:
            # Original single-layer processing
            sequence_output = outputs.last_hidden_state
            lstm_output, _ = self.lstm(sequence_output)
            
            # Use the last hidden state of the LSTM
            pooled_output = lstm_output[:, -1, :]
            
            # Apply dropout
            pooled_output = self.dropout(pooled_output)

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