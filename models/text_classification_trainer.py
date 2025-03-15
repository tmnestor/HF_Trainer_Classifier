import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import logging

# Set logging level for transformers
logging.getLogger("transformers").setLevel(logging.WARNING)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    get_scheduler,
)
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score

from classifiers import (
    CustomClassifier,
    BiLSTMClassifier,
    AttentionClassifier,
    CNNTextClassifier,
    FourierKANClassifier,
    WaveletKANClassifier,
)
from callbacks import LearningCurveCallback


class TextClassificationTrainer:
    def __init__(
        self,
        model_path,
        num_labels=None,
        max_length=64,
        output_dir="./results",
        learning_rate=2e-5,
        batch_size=8,
        num_epochs=3,
        weight_decay=0.01,
    ):
        self.model_path = model_path
        self.num_labels = num_labels
        self.max_length = max_length
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay

        # Initialize tokenizer from local path only (no download)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True  # Prevent attempting to download from Hugging Face
        )

        # Model will be initialized when prepare_model is called
        self.model = None
        self.trainer = None
        self.label_to_id = None
        self.id_to_label = None
        self.learning_curve_callback = None

    def prepare_data(self, dataset_path, split=True, **split_kwargs):
        """
        Load dataset from CSV, optionally split it, and prepare for training
        """
        if split:
            dataset_dict = self.stratified_split_csv(dataset_path, **split_kwargs)
        else:
            # Load pre-split dataset
            df = pd.read_csv(dataset_path)
            dataset_dict = DatasetDict.from_pandas(df)

        # Auto-detect number of labels if not specified
        if self.num_labels is None and self.label_to_id is not None:
            self.num_labels = len(self.label_to_id)

        # Tokenize datasets
        tokenized_datasets = self.tokenize_datasets(dataset_dict)
        return tokenized_datasets

    def stratified_split_csv(
        self,
        csv_path,
        text_column="text",
        label_column="label",
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        random_state=42,
    ):
        """
        Load a CSV file and create a stratified train-val-test split as a HuggingFace Dataset.
        Converts text labels to numerical indices.
        """
        # Verify split sizes add up to 1
        assert abs(train_size + val_size + test_size - 1.0) < 1e-10, (
            "Split sizes must add up to 1"
        )

        # Load the data
        df = pd.read_csv(csv_path)

        # Convert text labels to numerical indices
        if df[label_column].dtype == "object":  # If labels are strings
            # Get unique labels and create a mapping
            unique_labels = df[label_column].unique()
            self.label_to_id = {label: i for i, label in enumerate(unique_labels)}

            # Inverse mapping for later
            self.id_to_label = {i: label for label, i in self.label_to_id.items()}

            # Store the mapping for reference (only in non-tuning context)
            if not os.environ.get('TUNING_MODE'):
                print(f"Label mapping: {self.label_to_id}")

            # Convert labels to indices
            df[label_column] = df[label_column].map(self.label_to_id)

            # Set number of labels if not already specified
            self.num_labels = len(self.label_to_id)

        # First split: train and temp (validation + test)
        train_df, temp_df = train_test_split(
            df,
            train_size=train_size,
            stratify=df[label_column],
            random_state=random_state,
        )

        # Calculate relative sizes for validation and test from the temp set
        relative_val_size = val_size / (val_size + test_size)

        # Second split: validation and test from temp
        val_df, test_df = train_test_split(
            temp_df,
            train_size=relative_val_size,
            stratify=temp_df[label_column],
            random_state=random_state,
        )

        # Create HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)

        # Create and return DatasetDict
        dataset_dict = DatasetDict(
            {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
        )

        return dataset_dict

    def tokenize_datasets(self, dataset_dict):
        """Tokenize all datasets in the dictionary"""

        def preprocess_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,
            )

        # Use map to apply tokenization
        tokenized_datasets = dataset_dict.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on dataset",
            remove_columns=["text"],  # Remove text column after tokenization
        )

        # Format dataset to return PyTorch tensors
        tokenized_datasets = tokenized_datasets.with_format("torch")
        return tokenized_datasets

    def prepare_model(
        self,
        model_type="standard",
        dropout_rate=0.3,
        num_frequencies=8,
        wavelet_type="mixed",
    ):
        """Initialize the model with the correct number of labels"""
        # Try to determine num_labels from label_to_id if it exists
        if self.num_labels is None and self.label_to_id is not None:
            self.num_labels = len(self.label_to_id)
            
        if self.num_labels is None:
            raise ValueError(
                "Number of labels not specified and could not be determined from data"
            )
        

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        # Only output in non-tuning context or if tuning debug is enabled
        debug_mode = os.environ.get('TUNING_DEBUG') == 'True'
        if not os.environ.get('TUNING_MODE') or debug_mode:
            print(f"\nPreparing model on device: {device}")

        if model_type == "standard":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=self.num_labels,
                ignore_mismatched_sizes=True,
                local_files_only=True  # Prevent attempting to download from Hugging Face
            )
        elif model_type == "custom":
            self.model = CustomClassifier(
                model_path=self.model_path,
                num_labels=self.num_labels,
                dropout_rate=dropout_rate,
            )
        elif model_type == "bilstm":
            self.model = BiLSTMClassifier(
                model_path=self.model_path,
                num_labels=self.num_labels,
                dropout_rate=dropout_rate,
            )
        elif model_type == "attention":
            self.model = AttentionClassifier(
                model_path=self.model_path,
                num_labels=self.num_labels,
                dropout_rate=dropout_rate,
            )
        elif model_type == "cnn":
            self.model = CNNTextClassifier(
                model_path=self.model_path,
                num_labels=self.num_labels,
                dropout_rate=dropout_rate,
            )
        elif model_type == "fourier_kan":
            self.model = FourierKANClassifier(
                model_path=self.model_path,
                num_labels=self.num_labels,
                dropout_rate=dropout_rate,
                num_frequencies=num_frequencies,
            )
        elif model_type == "wavelet_kan":
            self.model = WaveletKANClassifier(
                model_path=self.model_path,
                num_labels=self.num_labels,
                dropout_rate=dropout_rate,
                num_wavelets=num_frequencies,
                wavelet_type=wavelet_type,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Move model to device and print device info
        self.model.to(device)
        debug_mode = os.environ.get('TUNING_DEBUG') == 'True'
        if not os.environ.get('TUNING_MODE') or debug_mode:
            print(f"Model moved to: {next(self.model.parameters()).device}")

        return self.model

    def get_training_args(self, metric_for_best_model="matthews_correlation"):
        """Define training arguments"""
        # Determine if we're in tuning mode to disable checkpoint saving
        is_tuning_mode = bool(os.environ.get('TUNING_MODE'))
        
        # Completely disable saving checkpoints during training
        save_strategy = "no"
        
        # Since we're not saving checkpoints, we can't load the best model at the end
        load_best_model_at_end = False
        
        # Create output_dir and logging_dir as Path objects and convert to strings
        output_dir_path = Path(self.output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        logging_dir_path = output_dir_path / "logs"
        logging_dir_path.mkdir(parents=True, exist_ok=True)
        
        return TrainingArguments(
            output_dir=str(output_dir_path),  # Convert to string for compatibility
            eval_strategy="epoch",  # Evaluate at the end of each epoch
            # evaluation_strategy="epoch",  # Alias for eval_strategy (newer HF versions)
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=self.weight_decay
            * 2,  # Increased weight decay for regularization
            push_to_hub=False,
            logging_dir=str(logging_dir_path),  # Convert to string for compatibility
            logging_strategy="epoch",  # Only log at end of each epoch
            disable_tqdm=False,  # Keep progress bars but reduce other output
            save_strategy=save_strategy,  # Match save strategy with evaluation strategy
            load_best_model_at_end=load_best_model_at_end,
            # early_stopping_patience is not a valid argument in this version of transformers
            save_total_limit=1,  # Only keep the best model checkpoint
            remove_unused_columns=True,
            report_to="none",  # Disable reporting to wandb, tensorboard, etc.
            warmup_ratio=0.1,  # Default warmup ratio
            gradient_accumulation_steps=1,  # Default - no gradient accumulation
            eval_accumulation_steps=None,  # Default - no accumulation during eval
            fp16=False,  # Mixed precision training (faster on modern GPUs)
            # Enable GPU acceleration
            no_cuda=False,  # Set to True to disable CUDA even when available
        )

    @staticmethod
    def compute_metrics(eval_pred):
        """Compute metrics for evaluation"""
        from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        # Calculate multiple metrics with zero_division=0 to handle missing classes
        accuracy = accuracy_score(labels, predictions)
        
        # Use zero_division=0 to avoid UndefinedMetricWarning
        precision = precision_score(labels, predictions, average="macro", zero_division=0)
        recall = recall_score(labels, predictions, average="macro", zero_division=0)
        f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
        
        # For matthews correlation, we use the standard implementation
        # (it handles class imbalance well)
        matthews = matthews_corrcoef(labels, predictions)

        # For early epochs when model might be random, ensure we still have valid metrics
        if np.isnan(matthews):
            matthews = 0.0
        
        # Add some debug info about class distribution
        unique_preds, pred_counts = np.unique(predictions, return_counts=True)
        pred_distribution = {int(label): int(count) for label, count in zip(unique_preds, pred_counts)}

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_macro": f1_macro,
            "matthews_correlation": matthews,
            "pred_distribution": str(pred_distribution),  # Debug info
        }

    def create_optimizer_and_scheduler(self, model_type="standard"):
        """
        Creates an optimizer with different learning rates for encoder and classifier,
        along with a learning rate scheduler.
        """
        if self.model is None:
            raise ValueError("Model must be initialized before creating optimizer")

        # Group parameters based on which component they belong to
        encoder_params = []
        classifier_params = []

        # Different grouping based on model type
        if model_type == "standard":
            # For standard HF model
            encoder_name = (
                "transformer" if hasattr(self.model, "transformer") else "bert"
            )

            for name, param in self.model.named_parameters():
                if name.startswith(encoder_name):
                    encoder_params.append(param)
                else:
                    classifier_params.append(param)
        else:
            # For custom models with explicit transformer and classifier components
            for name, param in self.model.named_parameters():
                if "transformer" in name:
                    encoder_params.append(param)
                else:
                    classifier_params.append(param)

        # Define different learning rates - adjusted for wavelet_kan to avoid getting stuck
        if model_type == "wavelet_kan":
            # For WaveletKAN, use higher learning rates to prevent one-class predictions
            encoder_lr = self.learning_rate * 0.2  # Higher LR for encoder in WaveletKAN
            classifier_lr = self.learning_rate * 2.0  # Much higher LR for classifier to avoid sticking
            print(f"Using adjusted learning rates for WaveletKAN - encoder: {encoder_lr:.2e}, classifier: {classifier_lr:.2e}")
        else:
            # Standard learning rates for other models
            encoder_lr = self.learning_rate * 0.1  # Lower learning rate for pretrained encoder
            classifier_lr = self.learning_rate  # Higher learning rate for classifier

        # Create optimizer with parameter groups
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": encoder_lr},
                {"params": classifier_params, "lr": classifier_lr},
            ],
            weight_decay=self.weight_decay,
        )

        return optimizer

    def setup_trainer(
        self,
        tokenized_datasets,
        model_type="standard",
        metric_for_best_model="matthews_correlation",
        model_params=None,
        callback_params=None,
        training_args=None,
    ):
        """Set up the trainer with the model and datasets with custom optimizer"""
        # Default parameters
        if model_params is None:
            model_params = {}
        if callback_params is None:
            callback_params = {"model_type": model_type, "patience": 2}
        if training_args is None:
            training_args = {}

        # Prepare model with additional parameters
        if self.model is None:
            if model_type == "fourier_kan":
                # Pass both dropout_rate and num_frequencies to FourierKAN
                dropout_rate = model_params.get("dropout_rate", 0.3)
                num_frequencies = model_params.get("num_frequencies", 8)
                self.prepare_model(
                    model_type=model_type,
                    dropout_rate=dropout_rate,
                    num_frequencies=num_frequencies,
                )
            elif model_type == "wavelet_kan":
                # Pass parameters for WaveletKAN
                dropout_rate = model_params.get("dropout_rate", 0.3)
                num_frequencies = model_params.get("num_frequencies", 8)
                wavelet_type = model_params.get("wavelet_type", "mixed")
                self.prepare_model(
                    model_type=model_type,
                    dropout_rate=dropout_rate,
                    num_frequencies=num_frequencies,
                    wavelet_type=wavelet_type,
                )
            elif "dropout_rate" in model_params and model_type in [
                "cnn",
                "bilstm",
                "attention",
                "custom",
            ]:
                self.prepare_model(
                    model_type=model_type, dropout_rate=model_params["dropout_rate"]
                )
            else:
                self.prepare_model(model_type=model_type)

        # Custom optimizer with different learning rates
        optimizer = self.create_optimizer_and_scheduler(model_type)

        # Get training arguments with proper metric and update with provided args
        base_training_args = self.get_training_args(
            metric_for_best_model=metric_for_best_model
        )
        
        # Update training arguments with provided overrides
        for key, value in training_args.items():
            setattr(base_training_args, key, value)
        
        # Use the updated training arguments
        training_args = base_training_args

        # Create a function to return the custom optimizer
        def model_init():
            return self.model

        # Calculate number of training steps
        num_training_steps = (
            len(tokenized_datasets["train"])
            // training_args.per_device_train_batch_size
            * training_args.num_train_epochs
        )

        # Create the learning rate scheduler
        warmup_ratio = callback_params.get("warmup_ratio", 0.1)
        # Override with training_args if specified
        if hasattr(training_args, "warmup_ratio"):
            warmup_ratio = training_args.warmup_ratio
        
        lr_scheduler = get_scheduler(
            name="linear",  # You can use "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
            optimizer=optimizer,
            num_warmup_steps=int(
                warmup_ratio * num_training_steps
            ),  # Use configured warmup ratio
            num_training_steps=num_training_steps,
        )

        # Initialize the learning curve callback with parameters
        patience = callback_params.get("patience", 2)
        callback_model_type = callback_params.get("model_type", model_type)

        self.learning_curve_callback = LearningCurveCallback(
            eval_dataset=tokenized_datasets["validation"],
            model_type=callback_model_type,
            patience=patience,
        )

        # Setup Trainer with custom optimizer and scheduler
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            compute_metrics=self.compute_metrics,
            optimizers=(
                optimizer,
                lr_scheduler,
            ),  # Custom optimizer and scheduler instance
            callbacks=[self.learning_curve_callback],  # Add learning curve callback
        )

        return self.trainer

    def train(self, tokenized_datasets):
        """Train the model"""
        if self.trainer is None:
            self.setup_trainer(tokenized_datasets)

        debug_mode = os.environ.get('TUNING_DEBUG') == 'True'
        if not os.environ.get('TUNING_MODE') or debug_mode:
            print("\nStarting model training...")
        return self.trainer.train()

    def evaluate(self, dataset):
        """Evaluate the model on a dataset"""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call setup_trainer() first")

        debug_mode = os.environ.get('TUNING_DEBUG') == 'True'
        if not os.environ.get('TUNING_MODE') or debug_mode:
            print(f"\nEvaluating model...")
        return self.trainer.evaluate(dataset)

    def save_model(self, path=None):
        """Save the model"""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call setup_trainer() first")

        # Use pathlib for path management
        save_path = path or Path(self.output_dir) / "model"
        # Ensure save_path is a Path object
        if not isinstance(save_path, Path):
            save_path = Path(save_path)
            
        # Make sure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        self.trainer.save_model(str(save_path))  # Convert to string for compatibility
        print(f"Model saved to {save_path}")

        # Also save the label mapping if available
        if self.label_to_id is not None:
            label_map_path = save_path / "label_mapping.json"
            with open(label_map_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"label_to_id": self.label_to_id, "id_to_label": self.id_to_label},
                    f,
                    indent=2,
                )

        return save_path
