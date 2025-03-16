import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class AttentionClassifier(nn.Module):
    def __init__(self, model_path, num_labels, dropout_rate=0.1, use_multilayer=False, num_layers_to_use=3):
        super().__init__()
        
        # If using multilayer output, configure the model to output all hidden states
        if use_multilayer:
            config = AutoConfig.from_pretrained(model_path, output_hidden_states=True, local_files_only=True)
            self.transformer = AutoModel.from_pretrained(model_path, config=config, local_files_only=True)
        else:
            self.transformer = AutoModel.from_pretrained(model_path, local_files_only=True)
            
        hidden_size = self.transformer.config.hidden_size
        self.num_labels = num_labels  # Store num_labels for use in forward
        self.use_multilayer = use_multilayer
        # ModernBERT uses 'layers' instead of 'encoder.layer'
        if hasattr(self.transformer, 'layers'):
            self.num_layers_to_use = min(num_layers_to_use, len(self.transformer.layers))
        elif hasattr(self.transformer, 'encoder') and hasattr(self.transformer.encoder, 'layer'):
            self.num_layers_to_use = min(num_layers_to_use, len(self.transformer.encoder.layer))
        else:
            self.num_layers_to_use = num_layers_to_use
        
        # Calculate input size for the classifier based on multilayer setting
        classifier_input_size = hidden_size * self.num_layers_to_use if use_multilayer else hidden_size

        # Attention mechanism - we'll create separate attention modules for each layer if using multilayer
        if use_multilayer:
            self.attention_modules = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, 1, bias=False),
                    nn.Softmax(dim=1),
                ) for _ in range(self.num_layers_to_use)
            ])
        else:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1, bias=False),
                nn.Softmax(dim=1),
            )

        # Add layer norm for better stability
        self.layer_norm = nn.LayerNorm(classifier_input_size)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
            
        # Deeper classifier network with GELU activation and dropout between layers
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, classifier_input_size),
            nn.GELU(),  # GELU is used in modern transformers and performs better than ReLU
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_input_size, num_labels)
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Get transformer outputs - ignore unexpected kwargs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        if self.use_multilayer:
            # Get the last n hidden states
            all_hidden_states = outputs.hidden_states
            
            # Use the last num_layers_to_use layers (including the final layer)
            selected_hidden_states = all_hidden_states[-self.num_layers_to_use:]
            
            # Apply attention to each layer and concatenate results
            context_vectors = []
            
            for i, hidden_states in enumerate(selected_hidden_states):
                # Apply attention for this layer
                attention_weights = self.attention_modules[i](hidden_states)
                context_vector = torch.sum(attention_weights * hidden_states, dim=1)
                context_vectors.append(context_vector)
            
            # Concatenate context vectors from different layers
            combined_context = torch.cat(context_vectors, dim=1)
            
            # Apply layer norm and dropout
            normalized_context = self.layer_norm(combined_context)
            dropped_context = self.dropout(normalized_context)
            
            # Classification
            logits = self.classifier(dropped_context)
        else:
            # Get sequence output (single layer)
            sequence_output = outputs.last_hidden_state

            # Apply attention
            attention_weights = self.attention(sequence_output)
            context_vector = torch.sum(attention_weights * sequence_output, dim=1)
            
            # Classification
            logits = self.classifier(context_vector)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Use label smoothing (0.1) to reduce overconfidence and improve generalization
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return (
            {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
        )