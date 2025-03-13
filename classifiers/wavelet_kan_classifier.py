import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from transformers import AutoModel


class HaarWaveletFunctions:
    """
    Implementation of Haar wavelet basis functions.
    Haar wavelets are the simplest wavelet family with compact support.
    """
    @staticmethod
    def scaling_function(x):
        """Haar scaling function (father wavelet)"""
        return ((x >= 0) & (x < 1)).float()
    
    @staticmethod
    def wavelet_function(x):
        """Haar wavelet function (mother wavelet)"""
        return ((x >= 0) & (x < 0.5)).float() - ((x >= 0.5) & (x < 1)).float()
    
    @staticmethod
    def wavelet_transform(x, scale, translation):
        """Apply wavelet transform with given scale and translation"""
        # Normalize x to the wavelet domain
        scaled_x = (x - translation) / scale
        # Apply wavelet function
        return HaarWaveletFunctions.wavelet_function(scaled_x)
    
    @staticmethod
    def scaling_transform(x, scale, translation):
        """Apply scaling function with given scale and translation"""
        # Normalize x to the wavelet domain
        scaled_x = (x - translation) / scale
        # Apply scaling function
        return HaarWaveletFunctions.scaling_function(scaled_x)


class MexicanHatWaveletFunctions:
    """
    Implementation of Mexican Hat (Ricker) wavelet basis functions.
    These are derived from the second derivative of a Gaussian function.
    """
    @staticmethod
    def wavelet_function(x):
        """Mexican Hat wavelet function (proportional to second derivative of Gaussian)"""
        return (1.0 - x**2) * torch.exp(-0.5 * x**2)
    
    @staticmethod
    def wavelet_transform(x, scale, translation):
        """Apply Mexican Hat wavelet with given scale and translation"""
        # Normalize x to the wavelet domain
        scaled_x = (x - translation) / scale
        # Apply wavelet function
        return MexicanHatWaveletFunctions.wavelet_function(scaled_x)


class MorletWaveletFunctions:
    """
    Implementation of Morlet wavelet basis functions.
    These are well-localized in both time and frequency.
    """
    @staticmethod
    def wavelet_function(x, omega0=5.0):
        """Morlet wavelet function (complex sinusoid modulated by Gaussian)"""
        # The normalized real part of the Morlet wavelet
        return torch.cos(omega0 * x) * torch.exp(-0.5 * x**2)
    
    @staticmethod
    def wavelet_transform(x, scale, translation, omega0=5.0):
        """Apply Morlet wavelet with given scale and translation"""
        # Normalize x to the wavelet domain
        scaled_x = (x - translation) / scale
        # Apply wavelet function
        return MorletWaveletFunctions.wavelet_function(scaled_x, omega0)


class WaveletLayer(nn.Module):
    """
    Wavelet layer that learns to apply wavelet transformations to input features.
    Combines multiple wavelet families for rich feature extraction.
    """
    def __init__(self, in_features, out_features, num_wavelets=16, wavelet_type="mixed"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_wavelets = num_wavelets
        self.wavelet_type = wavelet_type
        
        # Direct linear path for residual-like connection
        self.linear = nn.Linear(in_features, out_features)
        
        # Project inputs to wavelet space
        self.projection = nn.Linear(in_features, num_wavelets)
        
        # Learnable wavelet parameters
        # Scale parameters control the frequency/width of wavelets
        self.scales = nn.Parameter(torch.ones(num_wavelets) * 0.5)
        # Translation parameters control the center position of wavelets
        self.translations = nn.Parameter(torch.linspace(-2, 2, num_wavelets))
        
        # Separate parameters for different wavelet types if using mixed
        if wavelet_type == "mixed":
            # Number of each type of wavelet
            self.num_haar = num_wavelets // 3
            self.num_mexican = num_wavelets // 3
            self.num_morlet = num_wavelets - (self.num_haar + self.num_mexican)
        
        # Weights to combine wavelet outputs
        self.wavelet_combine = nn.Linear(num_wavelets, out_features)
        
        # Layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(out_features)
        
        # Initialize with appropriate values
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with values suitable for wavelets"""
        # Initialize scales to cover multiple resolutions
        if hasattr(self, 'num_haar'):
            # Different initialization for different wavelet types
            haar_scales = torch.logspace(-1, 0, self.num_haar) 
            mexican_scales = torch.logspace(-0.5, 0.5, self.num_mexican)
            morlet_scales = torch.logspace(-0.5, 0.5, self.num_morlet)
            self.scales.data = torch.cat([haar_scales, mexican_scales, morlet_scales])
        else:
            # Log-spaced scales to cover different frequencies
            self.scales.data = torch.logspace(-1, 0.5, self.num_wavelets)
        
        # Initialize translations to cover the input space
        if hasattr(self, 'num_haar'):
            # Different initialization for different wavelet types
            haar_trans = torch.linspace(-2, 2, self.num_haar)
            mexican_trans = torch.linspace(-2, 2, self.num_mexican)
            morlet_trans = torch.linspace(-2, 2, self.num_morlet)
            self.translations.data = torch.cat([haar_trans, mexican_trans, morlet_trans])
        else:
            # Evenly spaced translations to cover the input domain
            self.translations.data = torch.linspace(-2, 2, self.num_wavelets)
        
        # Initialize weights with small values
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.linear.bias)
        nn.init.kaiming_normal_(self.projection.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.projection.bias)
        nn.init.kaiming_normal_(self.wavelet_combine.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.wavelet_combine.bias)

    def forward(self, x):
        """
        Forward pass applying wavelet transformations.
        Args:
            x: Input tensor [batch_size, in_features]
        Returns:
            Transformed tensor [batch_size, out_features]
        """
        batch_size = x.size(0)
        
        # Direct linear path (residual connection)
        direct_path = self.linear(x)
        
        # Project inputs to wavelet space
        wavelet_input = self.projection(x)  # [batch_size, num_wavelets]
        
        # Create empty tensor to store wavelet activations
        wavelet_outputs = torch.zeros(batch_size, self.num_wavelets, device=x.device)
        
        if self.wavelet_type == "haar":
            # Apply Haar wavelets
            for i in range(self.num_wavelets):
                # Apply both scaling and wavelet functions for complete representation
                if i % 2 == 0:
                    wavelet_outputs[:, i] = HaarWaveletFunctions.wavelet_transform(
                        wavelet_input[:, i], 
                        self.scales[i], 
                        self.translations[i]
                    )
                else:
                    wavelet_outputs[:, i] = HaarWaveletFunctions.scaling_transform(
                        wavelet_input[:, i], 
                        self.scales[i], 
                        self.translations[i]
                    )
                    
        elif self.wavelet_type == "mexican":
            # Apply Mexican Hat wavelets
            for i in range(self.num_wavelets):
                wavelet_outputs[:, i] = MexicanHatWaveletFunctions.wavelet_transform(
                    wavelet_input[:, i], 
                    self.scales[i], 
                    self.translations[i]
                )
                
        elif self.wavelet_type == "morlet":
            # Apply Morlet wavelets
            for i in range(self.num_wavelets):
                wavelet_outputs[:, i] = MorletWaveletFunctions.wavelet_transform(
                    wavelet_input[:, i], 
                    self.scales[i], 
                    self.translations[i]
                )
                
        elif self.wavelet_type == "mixed":
            # Apply a mix of wavelet types for richer representation
            
            # Haar wavelets
            for i in range(self.num_haar):
                wavelet_outputs[:, i] = HaarWaveletFunctions.wavelet_transform(
                    wavelet_input[:, i], 
                    self.scales[i], 
                    self.translations[i]
                )
                
            # Mexican Hat wavelets
            for i in range(self.num_haar, self.num_haar + self.num_mexican):
                wavelet_outputs[:, i] = MexicanHatWaveletFunctions.wavelet_transform(
                    wavelet_input[:, i], 
                    self.scales[i], 
                    self.translations[i]
                )
                
            # Morlet wavelets
            for i in range(self.num_haar + self.num_mexican, self.num_wavelets):
                wavelet_outputs[:, i] = MorletWaveletFunctions.wavelet_transform(
                    wavelet_input[:, i], 
                    self.scales[i], 
                    self.translations[i]
                )
        
        # Apply non-linearity to wavelet outputs
        wavelet_outputs = F.gelu(wavelet_outputs)
        
        # Combine wavelet outputs
        wavelet_path = self.wavelet_combine(wavelet_outputs)
        
        # Combine direct path and wavelet path
        combined = direct_path + wavelet_path
        
        # Apply layer normalization
        return self.layer_norm(combined)


class WaveletKANUnit(nn.Module):
    """
    Wavelet-based KAN unit that applies multi-resolution analysis.
    Learns to decompose the input signal into components at different scales
    and recombine them for feature extraction.
    """
    def __init__(self, in_features, hidden_dim, out_features, num_wavelets=16, wavelet_type="mixed"):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        
        # First layer normalization
        self.norm1 = nn.LayerNorm(in_features)
        
        # First wavelet layer
        self.wavelet1 = WaveletLayer(
            in_features=in_features,
            out_features=hidden_dim,
            num_wavelets=num_wavelets,
            wavelet_type=wavelet_type
        )
        
        # Second layer normalization
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Second wavelet layer
        self.wavelet2 = WaveletLayer(
            in_features=hidden_dim,
            out_features=out_features,
            num_wavelets=num_wavelets,
            wavelet_type=wavelet_type
        )
        
        # Projection for residual if dimensions don't match
        self.residual_proj = None
        if in_features != out_features:
            self.residual_proj = nn.Linear(in_features, out_features)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        """Forward pass through the WaveletKAN unit"""
        # Store input for residual
        residual = x
        
        # First normalization
        x = self.norm1(x)
        
        # First wavelet layer
        x = self.wavelet1(x)
        
        # Apply activation
        x = F.gelu(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Second normalization
        x = self.norm2(x)
        
        # Second wavelet layer
        x = self.wavelet2(x)
        
        # Apply residual connection with projection if needed
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        
        # Add residual connection
        x = x + residual
        
        return x


class WaveletKANClassifier(nn.Module):
    """
    Text classifier using Wavelet-based Kolmogorov-Arnold Networks.
    Uses multi-resolution wavelet analysis for feature extraction and classification.
    """
    def __init__(self, model_path, num_labels, dropout_rate=0.1, num_wavelets=16, wavelet_type="mixed"):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_path)
        hidden_size = self.transformer.config.hidden_size
        self.num_labels = num_labels
        
        # Constrain hidden dimensions for stability
        kan_hidden_dim = min(hidden_size, 512)
        
        # Layer normalization for transformer outputs
        self.norm_layer = nn.LayerNorm(hidden_size)
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_size, kan_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate/2)
        )
        
        # WaveletKAN units
        self.wavelet_kan = WaveletKANUnit(
            in_features=kan_hidden_dim,
            hidden_dim=kan_hidden_dim,
            out_features=kan_hidden_dim,
            num_wavelets=num_wavelets,
            wavelet_type=wavelet_type
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(kan_hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(kan_hidden_dim, kan_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(kan_hidden_dim // 2, num_labels)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the classifier layers"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """Forward pass for the classifier"""
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]
        
        # Normalize outputs
        x = self.norm_layer(pooled_output)
        
        # Initial projection
        x = self.input_proj(x)
        
        # Apply WaveletKAN
        x = self.wavelet_kan(x)
        
        # Classification
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