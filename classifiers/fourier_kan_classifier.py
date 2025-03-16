import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModel


class FourierLayer(nn.Module):
    """
    Simplified Fourier layer with improved stability for text classification tasks.
    Uses a more standard architecture with residual-like connections.
    """
    def __init__(self, in_features, out_features, num_frequencies=8, learn_freq=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_frequencies = num_frequencies
        
        # More stable implementation with linear layers
        self.linear1 = nn.Linear(in_features, out_features)
        
        # Fourier projection layer with reduced complexity
        self.freq_proj = nn.Linear(in_features, num_frequencies)
        
        # Output projection after Fourier transform
        self.fourier_out = nn.Linear(2 * num_frequencies, out_features)
        
        # Learnable frequency scaling factors
        if learn_freq:
            # Smaller initial frequencies for better training stability
            self.frequency_scale = nn.Parameter(torch.ones(num_frequencies) * 0.5)
        else:
            # Fixed frequencies with reasonable range
            scale = torch.linspace(0.1, 1.0, num_frequencies)
            self.register_buffer('frequency_scale', scale)
        
        # Layer normalization for better training dynamics
        self.layer_norm = nn.LayerNorm(out_features)
        
        # Initialize with smaller weights for improved stability
        self._reset_parameters()
        
    def _reset_parameters(self):
        # Use smaller initialization for better numerical stability
        nn.init.xavier_uniform_(self.linear1.weight, gain=0.5)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.freq_proj.weight, gain=0.5)
        nn.init.zeros_(self.freq_proj.bias)
        nn.init.xavier_uniform_(self.fourier_out.weight, gain=0.5)
        nn.init.zeros_(self.fourier_out.bias)

    def forward(self, x):
        # Direct linear projection (skip connection)
        linear_out = self.linear1(x)
        
        # Fourier pathway
        # Project to frequency space with controlled scale
        freq_projection = self.freq_proj(x)  # [batch_size, num_frequencies]
        
        # Scale the frequencies to prevent extreme values
        scaled_freqs = freq_projection * self.frequency_scale.unsqueeze(0)  # [batch_size, num_frequencies]
        
        # Apply sine and cosine with normalized inputs to prevent training instability
        sin_components = torch.sin(scaled_freqs)  # [batch_size, num_frequencies]
        cos_components = torch.cos(scaled_freqs)  # [batch_size, num_frequencies]
        
        # Concatenate Fourier components
        fourier_features = torch.cat([sin_components, cos_components], dim=-1)  # [batch_size, 2*num_frequencies]
        
        # Project back to output dimension
        fourier_out = self.fourier_out(fourier_features)  # [batch_size, out_features]
        
        # Add the two pathways (residual-like connection)
        combined = linear_out + fourier_out
        
        # Apply layer normalization for training stability
        return self.layer_norm(combined)


class KANUnit(nn.Module):
    """
    Simplified KANUnit with improved stability for text classification.
    Combines Fourier components with feed-forward layers and strong residual connections.
    """
    def __init__(self, in_features, hidden_dim, out_features, num_frequencies=8):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        
        # First layer: input projection with layer normalization
        self.layer_norm1 = nn.LayerNorm(in_features)
        
        # First Fourier layer
        self.fourier1 = FourierLayer(in_features, hidden_dim, num_frequencies)
        
        # Second layer normalization
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Second Fourier layer
        self.fourier2 = FourierLayer(hidden_dim, out_features, num_frequencies)
        
        # Project residual if dimensions don't match
        self.residual_proj = None
        if in_features != out_features:
            self.residual_proj = nn.Linear(in_features, out_features)
            
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Store input for residual connection
        residual = x
        
        # First normalization
        x = self.layer_norm1(x)
        
        # First Fourier layer
        x = self.fourier1(x)
        
        # Activation - GELU works well with transformers
        x = F.gelu(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Second normalization
        x = self.layer_norm2(x)
        
        # Second Fourier layer
        x = self.fourier2(x)
        
        # Residual connection (with projection if needed)
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
            
        # Add residual connection
        x = x + residual
        
        return x


class FourierKANClassifier(nn.Module):
    """
    A simplified and more stable FourierKAN classifier that combines traditional MLP 
    components with Fourier-based KAN units for improved text classification.
    """
    def __init__(self, model_path, num_labels, dropout_rate=0.1, num_frequencies=8, use_multilayer=False, num_layers_to_use=3):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_path, local_files_only=True)
        hidden_size = self.transformer.config.hidden_size
        self.num_labels = num_labels
        self.use_multilayer = use_multilayer
        self.num_layers_to_use = num_layers_to_use
        
        # Smaller intermediate dimensions for better training stability
        kan_hidden_dim = min(hidden_size, 512)
        
        # Layer normalization for transformer outputs
        self.norm_layer = nn.LayerNorm(hidden_size)
        
        # Initialize input dimension based on whether we're using multiple layers
        input_dim = hidden_size
        if use_multilayer:
            input_dim = hidden_size * num_layers_to_use
            # Add projection layer to handle the increased dimensions
            self.layer_proj = nn.Linear(input_dim, hidden_size)
            self.proj_norm = nn.LayerNorm(hidden_size)
        
        # Initial MLP projection for stability
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_size, kan_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate/2)  # Lighter dropout in first layer
        )
        
        # KAN Unit with strong residual connections
        self.kan_unit = KANUnit(
            kan_hidden_dim, 
            kan_hidden_dim, 
            kan_hidden_dim, 
            num_frequencies
        )
        
        # Simple linear classifier after Fourier KAN processing
        # Apply layer norm before the classifier for stability
        self.layer_norm_final = nn.LayerNorm(kan_hidden_dim)
        self.classifier = nn.Linear(kan_hidden_dim, num_labels)
        
        # Initialize the weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize classifier layer with small weights
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Get transformer outputs with output_hidden_states to get all layers
        outputs = self.transformer(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=self.use_multilayer  # Only get hidden states if needed
        )
        
        if self.use_multilayer:
            # Get the specified number of last layers from hidden states
            hidden_states = outputs.hidden_states
            # Use last num_layers_to_use layers
            last_layers = hidden_states[-self.num_layers_to_use:]
            
            # Extract [CLS] token from each layer and concatenate
            cls_outputs = [layer[:, 0] for layer in last_layers]
            pooled_output = torch.cat(cls_outputs, dim=-1)
            
            # Project the concatenated layers down to original hidden size
            x = self.layer_proj(pooled_output)
            x = self.proj_norm(x)
        else:
            # Use the [CLS] token representation from last layer
            pooled_output = outputs.last_hidden_state[:, 0]
            x = self.norm_layer(pooled_output)
        
        # Initial projection
        x = self.input_proj(x)
        
        # Apply KAN unit
        x = self.kan_unit(x)
        
        # Apply final layer norm and classifier
        x = self.layer_norm_final(x)
        logits = self.classifier(x)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            "loss": loss, 
            "logits": logits
        } if loss is not None else {
            "logits": logits
        }