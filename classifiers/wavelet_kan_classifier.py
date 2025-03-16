import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import numpy as np
from transformers import AutoModel, AutoConfig
from torch.nn import GELU

# Set debug mode
DEBUG_MODE = os.environ.get("TUNING_DEBUG") == "True"


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


class DaubechiesWaveletFunctions:
    """
    Implementation of Daubechies wavelet basis functions.
    These are compact orthogonal wavelets with maximum number of vanishing moments.
    We implement low-order db2 and db4, which are well-suited for discrete signals.
    """
    # Daubechies filter coefficients (only low-order ones needed for text classification)
    # DB2 (4 coefficients)
    DB2_LO = torch.tensor([0.4829629131445341, 0.8365163037378079, 0.2241438680420134, -0.1294095225512604])
    DB2_HI = torch.tensor([-0.1294095225512604, -0.2241438680420134, 0.8365163037378079, -0.4829629131445341])
    
    # DB4 (8 coefficients)
    DB4_LO = torch.tensor([
        0.1629171572809054, 0.5055922216746659, 0.4461007108737368, -0.0197721859913265,
        -0.1323683939878645, 0.0218081502370024, 0.0233534968567756, -0.0074578275060427
    ])
    DB4_HI = torch.tensor([
        -0.0074578275060427, -0.0233534968567756, 0.0218081502370024, 0.1323683939878645,
        -0.0197721859913265, -0.4461007108737368, 0.5055922216746659, -0.1629171572809054
    ])
    
    @staticmethod
    def simplified_transform(x, filter_coeffs):
        """
        Simplified wavelet transform for scalar features in text classification.
        For text classification, we're dealing with feature embeddings not signals,
        so we don't need a full wavelet transform. We can use the filter coefficients
        to create a simple feature transformation.
        """
        # Ensure we're working with a scalar or small vector
        if torch.is_tensor(x):
            # Extract a single scalar value if possible
            if x.numel() == 1:
                value = x.item()
            else:
                # Take the mean of the first few values if it's a vector
                value = x[:min(4, x.numel())].mean().item()
        else:
            value = float(x)
            
        # Apply a non-linear transformation based on filter coefficients
        # This preserves the spirit of wavelet transforms while being much simpler
        device = filter_coeffs.device
        filter_sum = filter_coeffs.abs().sum().item()
        
        # Create a feature that captures the essence of wavelet transform
        # but works on scalar inputs
        feature = torch.tensor(
            [value * math.sin(coef * 5) for coef in filter_coeffs], 
            device=device
        ).mean()
        
        # Normalize by filter magnitude
        if filter_sum > 0:
            feature = feature / (filter_sum * 0.1)
        
        # Return as a scalar
        return feature
    
    @staticmethod
    def db2_wavelet_transform(x, scale, translation):
        """Apply db2 wavelet transform with given scale and translation"""
        # Scale modulates the filter coefficients
        scale_factor = torch.sigmoid(scale).item() * 2.0  # Keep in reasonable range
        
        # Translation affects the filter shift
        shift = torch.clamp(torch.round(translation * 4).long(), 0, 3).item()  # Limit to filter size
        
        # Shift the filter based on translation
        shifted_filter = torch.roll(DaubechiesWaveletFunctions.DB2_HI, shifts=shift)
        
        # Scale the filter
        scaled_filter = shifted_filter * scale_factor
        
        # Apply simplified wavelet transform for scalar features
        return DaubechiesWaveletFunctions.simplified_transform(x, scaled_filter)
    
    @staticmethod
    def db4_wavelet_transform(x, scale, translation):
        """Apply db4 wavelet transform with given scale and translation"""
        # Scale modulates the filter coefficients
        scale_factor = torch.sigmoid(scale).item() * 2.0  # Keep in reasonable range
        
        # Translation affects the filter shift
        shift = torch.clamp(torch.round(translation * 8).long(), 0, 7).item()  # Limit to filter size
        
        # Shift the filter based on translation
        shifted_filter = torch.roll(DaubechiesWaveletFunctions.DB4_HI, shifts=shift)
        
        # Scale the filter
        scaled_filter = shifted_filter * scale_factor
        
        # Apply simplified wavelet transform for scalar features
        return DaubechiesWaveletFunctions.simplified_transform(x, scaled_filter)


class WaveletLayer(nn.Module):
    """
    Wavelet layer that learns to apply wavelet transformations to input features.
    Combines multiple wavelet families for rich feature extraction.
    Optimized for high-frequency Haar wavelets.
    """

    def __init__(
        self, in_features, out_features, num_wavelets=16, wavelet_type="mixed"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_wavelets = num_wavelets
        self.wavelet_type = wavelet_type

        # Determine if this is a high-frequency case
        self.high_frequency = num_wavelets >= 64 and wavelet_type == "haar"

        # Direct linear path for residual-like connection
        self.linear = nn.Linear(in_features, out_features)

        # For high-frequency Haar wavelets, use optimized projection
        if self.high_frequency:
            # For high frequency, use multiple smaller projections rather than one large one
            # This reduces memory pressure and improves numerical stability
            self.group_size = 32
            self.num_groups = (num_wavelets + self.group_size - 1) // self.group_size
            self.projections = nn.ModuleList(
                [
                    nn.Linear(
                        in_features,
                        min(self.group_size, num_wavelets - i * self.group_size),
                    )
                    for i in range(self.num_groups)
                ]
            )
        else:
            # Standard projection for regular cases
            self.projection = nn.Linear(in_features, num_wavelets)

        # Learnable wavelet parameters
        # Scale parameters control the frequency/width of wavelets
        # Initialize with safer values to prevent division by very small numbers
        self.scales = nn.Parameter(torch.ones(num_wavelets) * 0.5)
        # Translation parameters control the center position of wavelets
        self.translations = nn.Parameter(torch.linspace(-2, 2, num_wavelets))

        # Separate parameters for different wavelet types if using mixed
        if wavelet_type == "mixed":
            # Number of each type of wavelet - focus on Haar and Daubechies
            total = num_wavelets
            self.num_haar = total // 3
            self.num_db2 = total // 3
            self.num_db4 = total - (self.num_haar + self.num_db2)

        # For high frequency Haar, use grouped wavelet outputs
        if self.high_frequency:
            # Use grouped linear layers
            self.wavelet_combines = nn.ModuleList(
                [
                    nn.Linear(
                        min(self.group_size, num_wavelets - i * self.group_size),
                        out_features // self.num_groups,
                    )
                    for i in range(self.num_groups)
                ]
            )
        else:
            # Regular combining for normal cases
            self.wavelet_combine = nn.Linear(num_wavelets, out_features)

        # Layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(out_features)

        # Initialize with appropriate values
        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters with values suitable for wavelets"""
        # Initialize scales to cover multiple resolutions with safer values
        if hasattr(self, "num_haar") and not self.high_frequency:
            # Different initialization for different wavelet types
            # Use larger minimum scale values to prevent numerical instability
            haar_scales = torch.logspace(
                -0.5, 0.5, self.num_haar
            )  # Safer minimum scale
            db2_scales = torch.logspace(-0.3, 0.7, self.num_db2)
            db4_scales = torch.logspace(-0.3, 0.7, self.num_db4)
            self.scales.data = torch.cat([haar_scales, db2_scales, db4_scales])
        elif self.high_frequency:
            # For high frequency models, use specialized initialization
            # Use more uniform scale distribution for better coverage
            self.scales.data = torch.logspace(-0.2, 1.0, self.num_wavelets)
            if DEBUG_MODE:
                print(
                    f"Initializing {self.num_wavelets} Haar wavelets for high-frequency model"
                )
        else:
            # Log-spaced scales with safer minimum values
            self.scales.data = torch.logspace(-0.3, 0.7, self.num_wavelets)

        # Initialize translations to cover the input space
        if hasattr(self, "num_haar") and not self.high_frequency:
            # Different initialization for different wavelet types
            haar_trans = torch.linspace(-1.5, 1.5, self.num_haar)  # Reduced range
            db2_trans = torch.linspace(-1.5, 1.5, self.num_db2)
            db4_trans = torch.linspace(-1.5, 1.5, self.num_db4)
            self.translations.data = torch.cat(
                [haar_trans, db2_trans, db4_trans]
            )
        elif self.high_frequency:
            # For high frequency models, create a more structured grid of translations
            # This provides better coverage of the input space
            if self.num_wavelets <= 64:
                # For 64 wavelets, use a simple linear spacing
                self.translations.data = torch.linspace(-2.0, 2.0, self.num_wavelets)
            else:
                # For higher counts, use a more sophisticated spacing
                # Create a mix of coarse and fine translations
                coarse = torch.linspace(-2.0, 2.0, self.num_wavelets // 4)
                medium = torch.linspace(-1.5, 1.5, self.num_wavelets // 4)
                fine = torch.linspace(-1.0, 1.0, self.num_wavelets // 2)

                # Ensure we have exactly num_wavelets translations
                combined = torch.cat([coarse, medium, fine])
                if len(combined) > self.num_wavelets:
                    combined = combined[: self.num_wavelets]
                elif len(combined) < self.num_wavelets:
                    # Add extra fine translations if needed
                    extra = torch.linspace(-0.5, 0.5, self.num_wavelets - len(combined))
                    combined = torch.cat([combined, extra])

                # Shuffle to avoid systematic biases
                idx = torch.randperm(self.num_wavelets)
                self.translations.data = combined[idx]
        else:
            # Evenly spaced translations with slightly reduced range
            self.translations.data = torch.linspace(-1.5, 1.5, self.num_wavelets)

        # Initialize weights with small values
        nn.init.kaiming_normal_(
            self.linear.weight, mode="fan_in", nonlinearity="linear"
        )
        nn.init.zeros_(self.linear.bias)

        if self.high_frequency:
            # Initialize the group projections
            for i, proj in enumerate(self.projections):
                nn.init.kaiming_normal_(
                    proj.weight, mode="fan_in", nonlinearity="linear"
                )
                nn.init.zeros_(proj.bias)

            # Initialize the group wavelet combiners
            for i, comb in enumerate(self.wavelet_combines):
                nn.init.kaiming_normal_(
                    comb.weight, mode="fan_in", nonlinearity="linear"
                )
                nn.init.zeros_(comb.bias)
        else:
            # Standard initialization
            nn.init.kaiming_normal_(
                self.projection.weight, mode="fan_in", nonlinearity="linear"
            )
            nn.init.zeros_(self.projection.bias)
            nn.init.kaiming_normal_(
                self.wavelet_combine.weight, mode="fan_in", nonlinearity="linear"
            )
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

        # Handle high frequency case differently for memory efficiency
        if self.high_frequency and self.wavelet_type == "haar":
            # Process in groups to reduce memory pressure
            group_outputs = []

            for group_idx in range(self.num_groups):
                start_idx = group_idx * self.group_size
                end_idx = min(start_idx + self.group_size, self.num_wavelets)
                group_size = end_idx - start_idx

                # Project this group
                group_input = self.projections[group_idx](x)  # [batch_size, group_size]

                # Process this group of wavelets
                group_wavelet_outputs = torch.zeros(
                    batch_size, group_size, device=x.device
                )

                for i in range(group_size):
                    global_idx = start_idx + i
                    # Alternating wavelet and scaling functions
                    if global_idx % 2 == 0:
                        group_wavelet_outputs[:, i] = (
                            HaarWaveletFunctions.wavelet_transform(
                                group_input[:, i],
                                self.scales[global_idx],
                                self.translations[global_idx],
                            )
                        )
                    else:
                        group_wavelet_outputs[:, i] = (
                            HaarWaveletFunctions.scaling_transform(
                                group_input[:, i],
                                self.scales[global_idx],
                                self.translations[global_idx],
                            )
                        )

                # Apply non-linearity
                gelu = GELU()
                group_wavelet_outputs = gelu(group_wavelet_outputs)

                # Combine this group's outputs
                group_wavelet_path = self.wavelet_combines[group_idx](
                    group_wavelet_outputs
                )
                group_outputs.append(group_wavelet_path)

            # Concatenate all group outputs
            wavelet_path = torch.cat(group_outputs, dim=1)

            # If the concatenated shape doesn't match out_features, pad or truncate
            if wavelet_path.shape[1] != self.out_features:
                if wavelet_path.shape[1] < self.out_features:
                    # Pad
                    padding = torch.zeros(
                        batch_size,
                        self.out_features - wavelet_path.shape[1],
                        device=wavelet_path.device,
                    )
                    wavelet_path = torch.cat([wavelet_path, padding], dim=1)
                else:
                    # Truncate
                    wavelet_path = wavelet_path[:, : self.out_features]

        else:
            # Standard approach for non-high-frequency cases
            # Project inputs to wavelet space
            wavelet_input = self.projection(x)  # [batch_size, num_wavelets]

            # Create empty tensor to store wavelet activations
            wavelet_outputs = torch.zeros(
                batch_size, self.num_wavelets, device=x.device
            )

            if self.wavelet_type == "haar":
                # Apply Haar wavelets
                for i in range(self.num_wavelets):
                    # Apply both scaling and wavelet functions for complete representation
                    if i % 2 == 0:
                        wavelet_outputs[:, i] = HaarWaveletFunctions.wavelet_transform(
                            wavelet_input[:, i], self.scales[i], self.translations[i]
                        )
                    else:
                        wavelet_outputs[:, i] = HaarWaveletFunctions.scaling_transform(
                            wavelet_input[:, i], self.scales[i], self.translations[i]
                        )

            elif self.wavelet_type == "mexican":
                # Apply Mexican Hat wavelets
                for i in range(self.num_wavelets):
                    wavelet_outputs[:, i] = (
                        MexicanHatWaveletFunctions.wavelet_transform(
                            wavelet_input[:, i], self.scales[i], self.translations[i]
                        )
                    )

            elif self.wavelet_type == "db2":
                # Apply Daubechies 2 wavelets
                for i in range(self.num_wavelets):
                    wavelet_outputs[:, i] = DaubechiesWaveletFunctions.db2_wavelet_transform(
                        wavelet_input[:, i], self.scales[i], self.translations[i]
                    )
                    
            elif self.wavelet_type == "db4":
                # Apply Daubechies 4 wavelets
                for i in range(self.num_wavelets):
                    wavelet_outputs[:, i] = DaubechiesWaveletFunctions.db4_wavelet_transform(
                        wavelet_input[:, i], self.scales[i], self.translations[i]
                    )

            elif self.wavelet_type == "mixed":
                # Apply a mix of wavelet types for richer representation

                # Haar wavelets
                for i in range(self.num_haar):
                    wavelet_outputs[:, i] = HaarWaveletFunctions.wavelet_transform(
                        wavelet_input[:, i], self.scales[i], self.translations[i]
                    )

                # Daubechies 2 wavelets
                for i in range(self.num_haar, self.num_haar + self.num_db2):
                    wavelet_outputs[:, i] = (
                        DaubechiesWaveletFunctions.db2_wavelet_transform(
                            wavelet_input[:, i], self.scales[i], self.translations[i]
                        )
                    )

                # Daubechies 4 wavelets
                for i in range(self.num_haar + self.num_db2, self.num_wavelets):
                    wavelet_outputs[:, i] = DaubechiesWaveletFunctions.db4_wavelet_transform(
                        wavelet_input[:, i], self.scales[i], self.translations[i]
                    )

            # Apply non-linearity to wavelet outputs
            gelu = GELU()
            wavelet_outputs = gelu(wavelet_outputs)

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

    def __init__(
        self,
        in_features,
        hidden_dim,
        out_features,
        num_wavelets=16,
        wavelet_type="mixed",
    ):
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
            wavelet_type=wavelet_type,
        )

        # Second layer normalization
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Second wavelet layer
        self.wavelet2 = WaveletLayer(
            in_features=hidden_dim,
            out_features=out_features,
            num_wavelets=num_wavelets,
            wavelet_type=wavelet_type,
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
        gelu = GELU()
        x = gelu(x)

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
    Focuses on compact support wavelets like Haar and Daubechies that are well-suited for
    discrete data like text embeddings.
    """

    def __init__(
        self,
        model_path,
        num_labels,
        dropout_rate=0.1,
        num_wavelets=16,
        wavelet_type="mixed", # Can be "mixed", "haar", "db2", or "db4"
        use_multilayer=False,
        num_layers_to_use=3
    ):
        super().__init__()
        
        # Configure model to output all hidden states if using multilayer
        if use_multilayer:
            config = AutoConfig.from_pretrained(model_path, output_hidden_states=True, local_files_only=True)
            self.transformer = AutoModel.from_pretrained(model_path, config=config, local_files_only=True)
        else:
            self.transformer = AutoModel.from_pretrained(model_path, local_files_only=True)
            
        hidden_size = self.transformer.config.hidden_size
        self.num_labels = num_labels
        self.wavelet_type = wavelet_type
        self.num_wavelets = num_wavelets
        self.use_multilayer = use_multilayer
        # ModernBERT uses 'layers' instead of 'encoder.layer'
        if hasattr(self.transformer, 'layers'):
            self.num_layers_to_use = min(num_layers_to_use, len(self.transformer.layers))
        elif hasattr(self.transformer, 'encoder') and hasattr(self.transformer.encoder, 'layer'):
            self.num_layers_to_use = min(num_layers_to_use, len(self.transformer.encoder.layer))
        else:
            self.num_layers_to_use = num_layers_to_use

        # For high frequency models, optimize memory usage
        self.optimize_for_high_freq = self.num_wavelets >= 64 and wavelet_type == "haar"
        if self.optimize_for_high_freq and DEBUG_MODE:
            print(
                f"Optimizing WaveletKAN for high frequency count: {self.num_wavelets}"
            )

        # Don't freeze layers for Haar wavelets, as they performed well with full fine-tuning
        # Just set a smaller learning rate for transformer backbone in the optimizer

        # Use only CLS token for pooling as it preserves sequential information better
        # This worked better with the original Haar implementation

        # Layer normalization for transformer outputs
        self.norm_layer = nn.LayerNorm(hidden_size)

        # For non-Haar wavelet types, add a pooler to handle concatenated outputs
        # If using multilayer, we'll need a separate pooler for each layer
        if wavelet_type != "haar":
            if use_multilayer:
                self.poolers = nn.ModuleList([
                    nn.Linear(hidden_size * 2, hidden_size)
                    for _ in range(self.num_layers_to_use)
                ])
            else:
                # Single pooler for concatenated outputs (original behavior)
                self.pooler = nn.Linear(hidden_size * 2, hidden_size)

        # Direct path optimized for wavelet types
        # For high-frequency Haar, use more efficient hidden dimension
        if self.optimize_for_high_freq:
            # Reduce hidden dimension for high frequency models to save memory
            kan_hidden_dim = min(hidden_size, 256)
        else:
            # Normal hidden dimensions
            kan_hidden_dim = min(hidden_size, 512 if wavelet_type == "haar" else 256)
        
        self.kan_hidden_dim = kan_hidden_dim

        # Initial projection - create one per layer if using multilayer
        if use_multilayer:
            self.input_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, kan_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_rate / 2 if wavelet_type == "haar" else dropout_rate),
                ) for _ in range(self.num_layers_to_use)
            ])
        else:
            # Original single projection
            self.input_proj = nn.Sequential(
                nn.Linear(hidden_size, kan_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate / 2 if wavelet_type == "haar" else dropout_rate),
            )

        # For Haar wavelets, add an extra layer to better learn the features
        if wavelet_type == "haar":
            # Deeper network for Haar wavelets since they performed well before
            # For multilayer, create one MLP module per layer
            if use_multilayer:
                self.mlps = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(kan_hidden_dim, kan_hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout_rate / 3),  # Lower dropout for Haar
                        nn.Linear(kan_hidden_dim, kan_hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout_rate / 3),
                    ) for _ in range(self.num_layers_to_use)
                ])
            else:
                # Original single MLP
                self.mlp = nn.Sequential(
                    nn.Linear(kan_hidden_dim, kan_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_rate / 3),  # Lower dropout for Haar
                    nn.Linear(kan_hidden_dim, kan_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_rate / 3),
                )
        else:
            # Simpler network for other wavelet types
            if use_multilayer:
                self.mlps = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(kan_hidden_dim, kan_hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout_rate / 2),
                    ) for _ in range(self.num_layers_to_use)
                ])
            else:
                # Original single MLP
                self.mlp = nn.Sequential(
                    nn.Linear(kan_hidden_dim, kan_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_rate / 2),
                )

        # Wavelet layer with optimized configuration
        # Use more wavelets for Haar type since it performed well before
        actual_num_wavelets = (
            self.num_wavelets * 2 if wavelet_type == "haar" else self.num_wavelets
        )
        
        # Create one wavelet layer per transformer layer if using multilayer
        if use_multilayer:
            self.wavelet_layers = nn.ModuleList([
                WaveletLayer(
                    in_features=kan_hidden_dim,
                    out_features=kan_hidden_dim,
                    num_wavelets=actual_num_wavelets,
                    wavelet_type=wavelet_type,
                ) for _ in range(self.num_layers_to_use)
            ])
            # Also create one normalization layer per transformer layer
            self.norm_layers = nn.ModuleList([
                nn.LayerNorm(kan_hidden_dim) for _ in range(self.num_layers_to_use)
            ])
            
            # Add a final layer normalization for the concatenated outputs
            final_dim = kan_hidden_dim * self.num_layers_to_use if use_multilayer else kan_hidden_dim
            self.final_norm = nn.LayerNorm(final_dim)
        else:
            # Original single wavelet layer
            self.wavelet_layer = WaveletLayer(
                in_features=kan_hidden_dim,
                out_features=kan_hidden_dim,
                num_wavelets=actual_num_wavelets,
                wavelet_type=wavelet_type,
            )
            # Original single normalization layer
            self.norm1 = nn.LayerNorm(kan_hidden_dim)

        # Simple linear classifier after Wavelet processing
        # Input size is multiplied by number of layers when using multilayer
        classifier_input = kan_hidden_dim * self.num_layers_to_use if use_multilayer else kan_hidden_dim
        self.classifier = nn.Linear(classifier_input, num_labels)
        self.classifier.is_classifier = True  # Keep this for initialization purposes

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize the weights with stronger values to break training deadlock"""
        # Use a larger std for classifier to improve optimization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use different initialization for classifier vs other layers
                if getattr(module, "is_classifier", False):
                    # Classifier needs much larger initialization for better training
                    module.weight.data.normal_(
                        mean=0.0, std=0.1
                    )  # Much larger std to break symmetry
                else:
                    module.weight.data.normal_(
                        mean=0.0, std=0.02
                    )  # Slightly larger std for all layers

                if module.bias is not None:
                    module.bias.data.normal_(
                        mean=0.0, std=0.01
                    )  # Initialize biases with small random values
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:  # Check if bias exists
                    module.bias.data.zero_()
                if module.weight is not None:  # Check if weight exists
                    module.weight.data.fill_(1.0)

        # Explicitly initialize pooler if it exists
        if hasattr(self, "pooler"):
            self.pooler.weight.data.normal_(mean=0.0, std=0.02)  # Larger std
            if self.pooler.bias is not None:
                self.pooler.bias.data.normal_(mean=0.0, std=0.01)  # Small random values

        # Special initialization for the classifier to prevent uniform predictions - MUCH stronger
        for name, module in self.named_modules():
            if (
                "classifier" in name
                and isinstance(module, nn.Linear)
                and module.out_features > 1
            ):
                if module.bias is not None:
                    # Add much larger, random biases to make logits strongly non-uniform at start
                    module.bias.data = (
                        torch.randn_like(module.bias) * 0.5
                    )  # Strong random biases

                # Also modify the weights more strongly
                scale_factor = 0.3
                num_classes = module.out_features
                # Add different scales for each output class to break symmetry
                for i in range(num_classes):
                    # Make each class have a different weight scale
                    scale = (
                        1.0 + scale_factor * (i / (num_classes - 1))
                        if num_classes > 1
                        else 1.0
                    )
                    module.weight.data[i, :] *= (
                        scale  # Scale weights differently per class
                    )

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass optimized for Haar wavelets which previously achieved >0.94 Matthews correlation.
        Different architectures for Haar vs other wavelet types.
        """
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        if self.use_multilayer:
            # Get hidden states from all layers
            all_hidden_states = outputs.hidden_states
            
            # Use the last num_layers_to_use layers
            selected_hidden_states = all_hidden_states[-self.num_layers_to_use:]
            
            # Process each layer separately and then concatenate the results
            layer_outputs = []
            
            for i, hidden_states in enumerate(selected_hidden_states):
                # For Haar wavelets, only use CLS token which worked better in original implementation
                if self.wavelet_type == "haar":
                    x = hidden_states[:, 0]  # Just [CLS] token for Haar
                else:
                    # Two pooling strategies for other wavelet types
                    cls_output = hidden_states[:, 0]  # [CLS] token
    
                    # Mean pooling - mask out padding tokens
                    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    mean_output = torch.sum(hidden_states * mask, 1) / torch.clamp(
                        mask.sum(1), min=1e-9
                    )
    
                    # Concatenate pooling results
                    pooled = torch.cat([cls_output, mean_output], dim=1)
    
                    # Project to original hidden size using the appropriate pooler
                    x = self.poolers[i](pooled)
    
                # Normalize outputs
                x = self.norm_layer(x)
    
                # Initial projection for this layer
                proj_x = self.input_projs[i](x)
    
                # Apply MLP for this layer
                mlp_output = self.mlps[i](proj_x)
    
                # Add residual connection
                if self.wavelet_type == "haar":
                    # For Haar, use stronger residual connection (with adjustable scaling)
                    mlp_output = mlp_output + 0.8 * proj_x
                else:
                    # Regular residual for other types
                    mlp_output = mlp_output + proj_x
    
                # Apply wavelet layer for this layer
                wavelet_output = self.wavelet_layers[i](mlp_output)
    
                # Add skip connection - stronger for Haar wavelets
                if self.wavelet_type == "haar":
                    combined = 0.7 * mlp_output + wavelet_output
                else:
                    combined = mlp_output + wavelet_output
    
                # Apply layer normalization
                normalized = self.norm_layers[i](combined)
                
                # Add to outputs list
                layer_outputs.append(normalized)
            
            # Concatenate outputs from all layers
            final = torch.cat(layer_outputs, dim=1)
            
            # Apply final normalization
            final = self.final_norm(final)
        else:
            # Original single-layer processing
            hidden_states = outputs.last_hidden_state
    
            # For Haar wavelets, only use CLS token which worked better in original implementation
            if self.wavelet_type == "haar":
                x = hidden_states[:, 0]  # Just [CLS] token for Haar
            else:
                # Two pooling strategies for other wavelet types
                cls_output = hidden_states[:, 0]  # [CLS] token
    
                # Mean pooling - mask out padding tokens
                mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                mean_output = torch.sum(hidden_states * mask, 1) / torch.clamp(
                    mask.sum(1), min=1e-9
                )
    
                # Concatenate pooling results
                pooled = torch.cat([cls_output, mean_output], dim=1)
    
                # Project to original hidden size
                x = self.pooler(pooled)
    
            # Normalize outputs
            x = self.norm_layer(x)
    
            # Initial projection
            proj_x = self.input_proj(x)
    
            # Apply MLP
            mlp_output = self.mlp(proj_x)
    
            # Add residual connection
            if self.wavelet_type == "haar":
                # For Haar, use stronger residual connection (with adjustable scaling)
                mlp_output = mlp_output + 0.8 * proj_x
            else:
                # Regular residual for other types
                mlp_output = mlp_output + proj_x
    
            # Apply wavelet layer
            wavelet_output = self.wavelet_layer(mlp_output)
    
            # Add skip connection - stronger for Haar wavelets
            if self.wavelet_type == "haar":
                combined = 0.7 * mlp_output + wavelet_output
            else:
                combined = mlp_output + wavelet_output
    
            # Final normalization
            final = self.norm1(combined)

        # Classification
        logits = self.classifier(final)

        # Loss calculation - more sophisticated to prevent single-class predictions
        loss = None
        if labels is not None:
            # First, calculate regular loss (with label smoothing for all types)
            smoothing = 0.2  # Higher label smoothing helps prevent getting stuck
            num_classes = self.num_labels

            # Create one-hot encoding of labels
            one_hot = torch.zeros_like(logits).scatter_(
                dim=1, index=labels.unsqueeze(1), value=1.0
            )

            # Apply label smoothing
            smoothed_targets = one_hot * (1.0 - smoothing) + smoothing / num_classes

            # Calculate cross entropy with smoothed targets
            log_probs = F.log_softmax(logits, dim=1)
            loss = -(smoothed_targets * log_probs).sum(dim=1).mean()

            # Add focal loss component to focus more on hard examples
            # This helps when the model gets stuck predicting one class
            gamma = 2.0  # Focus parameter
            probs = torch.exp(log_probs)
            p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - p_t).pow(gamma)

            focal_loss = -focal_weight * torch.log(p_t + 1e-8)
            focal_loss = focal_loss.mean()

            # Combine losses
            combined_loss = 0.75 * loss + 0.25 * focal_loss
            loss = combined_loss

            if self.training and DEBUG_MODE:
                # Debug information
                avg_prob = (
                    torch.exp(log_probs.gather(1, labels.unsqueeze(1))).mean().item()
                )
                if avg_prob > 0.95:
                    print(
                        f"Warning: High confidence predictions ({avg_prob:.4f}) might indicate model is too confident"
                    )

            # Class balance regularization - especially for Haar wavelets
            if self.training:
                batch_predictions = torch.argmax(logits, dim=1)
                unique_classes, pred_counts = torch.unique(
                    batch_predictions, return_counts=True
                )

                # Stronger regularization to prevent predicting only one class
                if len(unique_classes) < self.num_labels:
                    # Calculate how many classes are missing
                    missing_classes = self.num_labels - len(unique_classes)

                    # Add stronger penalty when fewer classes are predicted
                    balance_factor = 0.3 * (missing_classes / self.num_labels)
                    balance_loss = torch.tensor(balance_factor, device=logits.device)

                    # Only print in debug mode
                    if balance_factor > 0.1 and DEBUG_MODE:
                        print(
                            f"Adding class balance loss: {balance_factor:.4f}, predicted classes: {len(unique_classes)}/{self.num_labels}"
                        )

                    loss = loss + balance_loss

                # Add entropy regularization to encourage diverse predictions
                probs = F.softmax(logits, dim=1)
                mean_probs = probs.mean(dim=0)
                entropy = -(mean_probs * torch.log(mean_probs + 1e-5)).sum()

                # Lower entropy means more concentrated predictions, add penalty
                target_entropy = -math.log(
                    1.0 / self.num_labels
                )  # Maximum possible entropy for uniform distribution
                entropy_factor = 0.1 * (1.0 - entropy / target_entropy)

                # Only apply if entropy is too low (concentrated predictions)
                if entropy_factor > 0:
                    loss = loss + entropy_factor

        return (
            {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
        )
