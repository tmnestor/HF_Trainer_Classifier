import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class CustomClassifier(nn.Module):
    def __init__(self, model_path, num_labels, dropout_rate=0.1, pooling_strategy="cls", use_multilayer=False, num_layers_to_use=3):
        """
        Args:
            model_path: Path to the pre-trained model
            num_labels: Number of target labels
            dropout_rate: Dropout rate for regularization
            pooling_strategy: One of "cls", "mean", or "combined" - determines how transformer outputs are pooled
            use_multilayer: Whether to use outputs from the last several transformer layers
            num_layers_to_use: Number of last layers to use when use_multilayer is True
        """
        super().__init__()
        
        # If using multilayer output, we need to configure the model to output all hidden states
        if use_multilayer:
            config = AutoConfig.from_pretrained(model_path, output_hidden_states=True, local_files_only=True)
            self.transformer = AutoModel.from_pretrained(model_path, config=config, local_files_only=True)
        else:
            self.transformer = AutoModel.from_pretrained(model_path, local_files_only=True)
            
        hidden_size = self.transformer.config.hidden_size
        self.num_labels = num_labels
        self.pooling_strategy = pooling_strategy
        self.use_multilayer = use_multilayer
        # ModernBERT uses 'layers' instead of 'encoder.layer'
        if hasattr(self.transformer, 'layers'):
            self.num_layers_to_use = min(num_layers_to_use, len(self.transformer.layers))
        elif hasattr(self.transformer, 'encoder') and hasattr(self.transformer.encoder, 'layer'):
            self.num_layers_to_use = min(num_layers_to_use, len(self.transformer.encoder.layer))
        else:
            self.num_layers_to_use = num_layers_to_use
        
        # Determine classifier input size based on strategies
        if use_multilayer:
            if pooling_strategy == "combined":
                # Combined pooling with multiple layers: (CLS + mean) × num_layers_to_use
                classifier_input_size = hidden_size * 2 * self.num_layers_to_use
            else:
                # Single pooling strategy with multiple layers
                classifier_input_size = hidden_size * self.num_layers_to_use
        else:
            # Original behavior for single layer
            classifier_input_size = hidden_size * 2 if pooling_strategy == "combined" else hidden_size
        
        # Simple linear classifier
        self.classifier = nn.Linear(classifier_input_size, num_labels)
        
        # Add a layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(classifier_input_size)
        
        # Light dropout before classification
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        if self.use_multilayer:
            # Get the last n hidden states
            all_hidden_states = outputs.hidden_states
            
            # Use the last num_layers_to_use layers (including the final layer)
            selected_hidden_states = all_hidden_states[-self.num_layers_to_use:]
            
            # Process each hidden state according to pooling strategy and concatenate
            pooled_outputs = []
            
            for hidden_states in selected_hidden_states:
                if self.pooling_strategy == "cls":
                    # Use the [CLS] token representation
                    pooled_output = hidden_states[:, 0]
                    pooled_outputs.append(pooled_output)
                
                elif self.pooling_strategy == "mean":
                    # Apply mean pooling
                    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, 1)
                    sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)  # Avoid div by zero
                    pooled_output = sum_embeddings / sum_mask
                    pooled_outputs.append(pooled_output)
                    
                elif self.pooling_strategy == "combined":
                    # Get CLS token
                    cls_output = hidden_states[:, 0]
                    
                    # Apply mean pooling
                    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, 1)
                    sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
                    mean_output = sum_embeddings / sum_mask
                    
                    # Concatenate CLS and mean pooled outputs for this layer
                    pooled_output = torch.cat([cls_output, mean_output], dim=1)
                    pooled_outputs.append(pooled_output)
            
            # Concatenate outputs from all selected layers
            pooled_output = torch.cat(pooled_outputs, dim=1)
            
        else:
            # Original single-layer processing
            hidden_states = outputs.last_hidden_state
            
            if self.pooling_strategy == "cls":
                # Use the [CLS] token representation
                pooled_output = hidden_states[:, 0]
            
            elif self.pooling_strategy == "mean":
                # Apply mean pooling
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, 1)
                sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)  # Avoid div by zero
                pooled_output = sum_embeddings / sum_mask
                
            elif self.pooling_strategy == "combined":
                # Get CLS token
                cls_output = hidden_states[:, 0]
                
                # Apply mean pooling
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, 1)
                sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
                mean_output = sum_embeddings / sum_mask
                
                # Concatenate CLS and mean pooled outputs
                pooled_output = torch.cat([cls_output, mean_output], dim=1)
                
            else:
                raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")
        
        # Apply layer normalization and dropout
        normalized_output = self.layer_norm(pooled_output)
        dropped_output = self.dropout(normalized_output)
        
        # Pass through the classifier
        logits = self.classifier(dropped_output)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return (
            {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
        )