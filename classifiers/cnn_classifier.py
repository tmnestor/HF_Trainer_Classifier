import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class CNNTextClassifier(nn.Module):
    def __init__(self, model_path, num_labels, dropout_rate=0.3, use_multilayer=False, num_layers_to_use=3):
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

        # Features per CNN filter
        self.cnn_features = 64
        
        # Reduce feature maps to combat overfitting
        self.conv1 = nn.Conv1d(hidden_size, self.cnn_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, self.cnn_features, kernel_size=4, padding=2)
        self.conv3 = nn.Conv1d(hidden_size, self.cnn_features, kernel_size=5, padding=2)

        # Pooling layers
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Calculate output features
        self.cnn_output_size = self.cnn_features * 3  # 3 CNN filters
        
        # Total classifier input size depends on multilayer setting
        classifier_input_size = self.cnn_output_size * self.num_layers_to_use if use_multilayer else self.cnn_output_size
        
        # Add layer normalization for better stability
        self.layer_norm = nn.LayerNorm(classifier_input_size)

        # Increased dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Simple linear classifier after CNN feature extraction
        self.classifier = nn.Linear(classifier_input_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        if self.use_multilayer:
            # Get the last n hidden states
            all_hidden_states = outputs.hidden_states
            
            # Use the last num_layers_to_use layers (including the final layer)
            selected_hidden_states = all_hidden_states[-self.num_layers_to_use:]
            
            # Apply CNN to each layer and concatenate
            cnn_features = []
            
            for hidden_states in selected_hidden_states:
                # Transpose hidden states for CNN
                # [batch, seq_len, hidden] -> [batch, hidden, seq_len]
                x = hidden_states.transpose(1, 2)
                
                # Apply CNN layers
                x1 = self.pool(torch.relu(self.conv1(x))).squeeze(2)
                x2 = self.pool(torch.relu(self.conv2(x))).squeeze(2)
                x3 = self.pool(torch.relu(self.conv3(x))).squeeze(2)
                
                # Concatenate features from different CNN filters
                x_cat = torch.cat((x1, x2, x3), dim=1)
                
                # Add to features list
                cnn_features.append(x_cat)
            
            # Concatenate features from all layers
            all_features = torch.cat(cnn_features, dim=1)
            
            # Apply layer normalization and dropout
            normalized_features = self.layer_norm(all_features)
            x_drop = self.dropout(normalized_features)
        else:
            # Get sequence output and transpose for CNN (single layer)
            x = outputs.last_hidden_state.transpose(1, 2)

            # Apply CNN layers
            x1 = self.pool(torch.relu(self.conv1(x))).squeeze(2)
            x2 = self.pool(torch.relu(self.conv2(x))).squeeze(2)
            x3 = self.pool(torch.relu(self.conv3(x))).squeeze(2)

            # Concatenate features from different CNN layers
            x_cat = torch.cat((x1, x2, x3), dim=1)

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