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
    Trainer as HFTrainer,
    get_scheduler,
)
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from classifiers import (
    CustomClassifier,
    BiLSTMClassifier,
    AttentionClassifier,
    CNNTextClassifier,
    FourierKANClassifier,
    WaveletKANClassifier,
)
from callbacks import TrainingMonitorCallback


class TextClassificationTrainer:
    def __init__(
        self,
        model_path,
        num_labels=None,
        max_length=64,
        output_dir="./results",
        learning_rate=2e-5,
        batch_size=16,
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
            local_files_only=True,  # Prevent attempting to download from Hugging Face
        )

        # Model will be initialized when prepare_model is called
        self.model = None
        self.trainer = None
        self.label_to_id = None
        self.id_to_label = None
        self.training_monitor_callback = None

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
            if not os.environ.get("TUNING_MODE"):
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
        use_multilayer=False,
        num_layers_to_use=3,
        pooling_strategy=None,
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
        debug_mode = os.environ.get("TUNING_DEBUG") == "True"
        if not os.environ.get("TUNING_MODE") or debug_mode:
            print(f"\nPreparing model on device: {device}")

        if model_type == "standard":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=self.num_labels,
                ignore_mismatched_sizes=True,
                local_files_only=True,  # Prevent attempting to download from Hugging Face
            )
        elif model_type == "custom":
            # Ensure pooling_strategy has a default if not specified
            if pooling_strategy is None:
                pooling_strategy = "cls"

            self.model = CustomClassifier(
                model_path=self.model_path,
                num_labels=self.num_labels,
                dropout_rate=dropout_rate,
                pooling_strategy=pooling_strategy,
                use_multilayer=use_multilayer,
                num_layers_to_use=num_layers_to_use,
            )
        elif model_type == "bilstm":
            self.model = BiLSTMClassifier(
                model_path=self.model_path,
                num_labels=self.num_labels,
                dropout_rate=dropout_rate,
                # Add these new parameters (if BiLSTM was also updated)
                use_multilayer=use_multilayer,
                num_layers_to_use=num_layers_to_use,
            )
        elif model_type == "attention":
            self.model = AttentionClassifier(
                model_path=self.model_path,
                num_labels=self.num_labels,
                dropout_rate=dropout_rate,
                use_multilayer=use_multilayer,
                num_layers_to_use=num_layers_to_use,
            )
        elif model_type == "cnn":
            self.model = CNNTextClassifier(
                model_path=self.model_path,
                num_labels=self.num_labels,
                dropout_rate=dropout_rate,
                use_multilayer=use_multilayer,
                num_layers_to_use=num_layers_to_use,
            )
        elif model_type == "fourier_kan":
            self.model = FourierKANClassifier(
                model_path=self.model_path,
                num_labels=self.num_labels,
                dropout_rate=dropout_rate,
                num_frequencies=num_frequencies,
                use_multilayer=use_multilayer,
                num_layers_to_use=num_layers_to_use,
            )
        elif model_type == "wavelet_kan":
            self.model = WaveletKANClassifier(
                model_path=self.model_path,
                num_labels=self.num_labels,
                dropout_rate=dropout_rate,
                num_wavelets=num_frequencies,
                wavelet_type=wavelet_type,
                use_multilayer=use_multilayer,
                num_layers_to_use=num_layers_to_use,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Move model to device and print device info
        self.model.to(device)
        debug_mode = os.environ.get("TUNING_DEBUG") == "True"
        if not os.environ.get("TUNING_MODE") or debug_mode:
            print(f"Model moved to: {next(self.model.parameters()).device}")

        return self.model

    def get_training_args(self, metric_for_best_model="matthews_correlation"):
        """Define training arguments"""
        # Determine if we're in tuning mode to disable checkpoint saving
        is_tuning_mode = bool(os.environ.get("TUNING_MODE"))

        # Completely disable saving checkpoints during training
        save_strategy = "no"

        # Since we're not saving checkpoints, we can't load the best model at the end
        load_best_model_at_end = False

        # Determine if we should disable progress bars during tuning
        disable_tqdm = bool(os.environ.get("TUNING_MODE"))

        # Create output_dir and logging_dir as Path objects and convert to strings
        output_dir_path = Path(self.output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        logging_dir_path = output_dir_path / "logs"
        logging_dir_path.mkdir(parents=True, exist_ok=True)

        # Ensure we always have a valid metric for evaluation, defaulting to matthews_correlation
        if metric_for_best_model is None:
            metric_for_best_model = "matthews_correlation"

        # Make sure the metric is prefixed with "eval_" as required by the Trainer
        if not metric_for_best_model.startswith("eval_"):
            metric_key = f"eval_{metric_for_best_model}"
        else:
            metric_key = metric_for_best_model

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
            disable_tqdm=disable_tqdm,  # Disable progress bars during tuning
            save_strategy=save_strategy,  # Match save strategy with evaluation strategy
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_key,  # Explicitly set the metric to use
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
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
        )

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        # Calculate multiple metrics with zero_division=0 to handle missing classes
        accuracy = accuracy_score(labels, predictions)

        # Use zero_division=0 to avoid UndefinedMetricWarning
        precision = precision_score(
            labels, predictions, average="macro", zero_division=0
        )
        recall = recall_score(labels, predictions, average="macro", zero_division=0)
        f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)

        # For matthews correlation, we use the standard implementation
        # (it handles class imbalance well)
        matthews = matthews_corrcoef(labels, predictions)

        # For early epochs when model might be random, ensure we still have valid metrics
        if np.isnan(matthews):
            matthews = 0.0

        # Add pred_distribution for internal use by pruning callback
        unique_preds, pred_counts = np.unique(predictions, return_counts=True)
        pred_distribution = {
            int(label): int(count) for label, count in zip(unique_preds, pred_counts)
        }

        # Use a special prefix for internal metrics to make them easy to identify and remove
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_macro": f1_macro,
            "matthews_correlation": matthews,
            # Prefix with _internal_ to ensure it's filtered out by _clean_metrics
            "_internal_pred_distribution": str(pred_distribution),
        }

    def create_optimizer_and_scheduler(
        self, model_type="standard", optimizer_type="adamw"
    ):
        """
        Creates an optimizer with different learning rates for encoder and classifier,
        along with a learning rate scheduler.

        Args:
            model_type: Type of the model (standard, custom, bilstm, etc.)
            optimizer_type: Type of optimizer to use (adamw, sgd, rmsprop)
        """
        if self.model is None:
            raise ValueError("Model must be initialized before creating optimizer")

        # Group parameters based on which component they belong to
        encoder_params = []
        classifier_params = []

        # Different grouping based on model type
        if model_type == "standard":
            # For standard HF model - ensure encoder_name is never None
            if hasattr(self.model, "transformer"):
                encoder_name = "transformer"
            elif hasattr(self.model, "bert"):
                encoder_name = "bert"
            elif hasattr(self.model, "roberta"):
                encoder_name = "roberta"
            elif hasattr(self.model, "model"):
                encoder_name = "model"
            else:
                # Fallback to classify all as classifier params if we can't determine
                encoder_name = "THIS_STRING_WONT_MATCH_ANYTHING"

            for name, param in self.model.named_parameters():
                if (
                    encoder_name != "THIS_STRING_WONT_MATCH_ANYTHING"
                    and name.startswith(encoder_name)
                ):
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

        # Scientific comparison mode: freeze transformer for all models
        # Set high initial learning rate for classifiers with reduction on plateau

        # Freeze the transformer (encoder) - set requires_grad=False
        for param in encoder_params:
            param.requires_grad = False

        # Set classifier learning rate to 1e-3 (high) for all models
        # This will be reduced later with ReduceLROnPlateau
        encoder_lr = 0  # Doesn't matter since encoder is frozen
        classifier_lr = 1e-3  # High initial learning rate for classifier

        # Only show this in non-tuning mode
        if not os.environ.get("TUNING_MODE"):
            print("SCIENTIFIC COMPARISON MODE:")
            print("- Transformer is frozen (no update)")
            print(f"- Optimizer: {optimizer_type}")
            print(f"- Classifier learning rate: {classifier_lr:.2e}")
            print("- Learning rate will reduce on plateau")

        # Prepare parameter groups
        param_groups = [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": classifier_params, "lr": classifier_lr},
        ]

        # Create optimizer based on type
        if optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=self.weight_decay,
            )
        elif optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(
                param_groups,
                momentum=0.9,  # Use momentum with SGD
                weight_decay=self.weight_decay,
            )
        elif optimizer_type.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop(
                param_groups,
                alpha=0.99,  # Default RMSprop smoothing constant
                weight_decay=self.weight_decay,
            )
        else:
            # Default to AdamW if optimizer type is not recognized
            print(
                f"Warning: Unrecognized optimizer type {optimizer_type}. Using AdamW instead."
            )
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=self.weight_decay,
            )

        return optimizer

    # Create a custom Trainer class that filters and formats metrics
    class CleanMetricsTrainer(HFTrainer):
        """Custom Trainer class that filters and formats metrics"""

        def log(self, logs, start_time=None):
            """Override log to clean metrics before they're logged"""
            # Clean the metrics
            cleaned_logs = {}

            for key, value in logs.items():
                # Skip unwanted metrics
                if any(
                    unwanted in key
                    for unwanted in [
                        "runtime",
                        "samples_per_second",
                        "steps_per_second",
                        "_internal_",
                    ]
                ):
                    continue

                # Format floats to 4 decimal places
                if isinstance(value, float):
                    cleaned_logs[key] = round(value, 4)
                elif key == "epoch" and isinstance(value, float):
                    # Make epoch a 1-based integer
                    cleaned_logs[key] = int(value) + 1
                else:
                    cleaned_logs[key] = value

            # Call parent log with cleaned metrics - match parent signature
            super().log(cleaned_logs, start_time)

        def evaluate(self, *args, **kwargs):
            """Override evaluate to clean metrics"""
            results = super().evaluate(*args, **kwargs)

            # Clean the results
            cleaned_results = {}
            for key, value in results.items():
                # Skip unwanted metrics
                if any(
                    unwanted in key
                    for unwanted in [
                        "runtime",
                        "samples_per_second",
                        "steps_per_second",
                        "_internal_",
                    ]
                ):
                    continue

                # Format floats to 4 decimal places
                if isinstance(value, float):
                    cleaned_results[key] = round(value, 4)
                elif key == "epoch" and isinstance(value, float):
                    # Make epoch a 1-based integer
                    cleaned_results[key] = int(value) + 1
                else:
                    cleaned_results[key] = value

            return cleaned_results

        def train(self, *args, **kwargs):
            """Override train to clean metrics in results"""
            results = super().train(*args, **kwargs)

            # Clean metrics in the results
            if isinstance(results, dict) and "metrics" in results:
                cleaned_metrics = {}

                for key, value in results["metrics"].items():
                    # Skip unwanted metrics
                    if any(
                        unwanted in key
                        for unwanted in [
                            "runtime",
                            "samples_per_second",
                            "steps_per_second",
                            "_internal_",
                        ]
                    ):
                        continue

                    # Format floats to 4 decimal places
                    if isinstance(value, float):
                        cleaned_metrics[key] = round(value, 4)
                    elif key == "epoch" and isinstance(value, float):
                        # Make epoch a 1-based integer
                        cleaned_metrics[key] = int(value) + 1
                    else:
                        cleaned_metrics[key] = value

                # Replace with cleaned metrics
                results["metrics"] = cleaned_metrics

            return results

    def setup_trainer(
        self,
        tokenized_datasets,
        model_type="standard",
        metric_for_best_model="matthews_correlation",
        model_params=None,
        callback_params=None,
        training_args=None,
        disable_progress_bar=False,
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
                use_multilayer = model_params.get("use_multilayer", False)
                num_layers_to_use = model_params.get("num_layers_to_use", 3)
                self.prepare_model(
                    model_type=model_type,
                    dropout_rate=dropout_rate,
                    num_frequencies=num_frequencies,
                    use_multilayer=use_multilayer,
                    num_layers_to_use=num_layers_to_use,
                )
            elif model_type == "wavelet_kan":
                # Pass parameters for WaveletKAN
                dropout_rate = model_params.get("dropout_rate", 0.3)
                num_frequencies = model_params.get("num_frequencies", 8)
                wavelet_type = model_params.get("wavelet_type", "mixed")
                use_multilayer = model_params.get("use_multilayer", False)
                num_layers_to_use = model_params.get("num_layers_to_use", 3)
                self.prepare_model(
                    model_type=model_type,
                    dropout_rate=dropout_rate,
                    num_frequencies=num_frequencies,
                    wavelet_type=wavelet_type,
                    use_multilayer=use_multilayer,
                    num_layers_to_use=num_layers_to_use,
                )
            elif model_type in ["cnn", "bilstm", "attention", "custom"]:
                dropout_rate = model_params.get("dropout_rate", 0.3)
                use_multilayer = model_params.get("use_multilayer", False)
                num_layers_to_use = model_params.get("num_layers_to_use", 3)
                pooling_strategy = (
                    model_params.get("pooling_strategy", "cls")
                    if model_type == "custom"
                    else None
                )

                # Prepare params dict to pass to prepare_model
                prepare_params = {
                    "model_type": model_type,
                    "dropout_rate": dropout_rate,
                    "use_multilayer": use_multilayer,
                    "num_layers_to_use": num_layers_to_use,
                }

                # Only add pooling_strategy for custom classifier
                if model_type == "custom" and pooling_strategy:
                    prepare_params["pooling_strategy"] = pooling_strategy

                self.prepare_model(**prepare_params)
            else:
                self.prepare_model(model_type=model_type)

        # Get optimizer type from model_params
        optimizer_type = model_params.get("optimizer_type", "adamw")

        # Custom optimizer with different learning rates
        optimizer = self.create_optimizer_and_scheduler(model_type, optimizer_type)

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

        # Use ReduceLROnPlateau for scientific comparison
        # This will reduce the learning rate when performance plateaus
        # Use the same metric for the scheduler as for model selection
        metric_for_lr_scheduler = metric_for_best_model
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",  # Since we're maximizing metrics like accuracy or F1
            factor=0.5,  # Reduce LR by half when plateau is detected
            patience=1,  # Wait for 1 epoch of no improvement before reducing
            verbose=True,  # Print message when LR is reduced
            min_lr=1e-6,  # Don't reduce LR below this threshold
        )

        # Store the metric name for use in callbacks
        lr_scheduler.metric_name = metric_for_lr_scheduler

        if not os.environ.get("TUNING_MODE"):
            print(
                f"Using ReduceLROnPlateau scheduler to adjust learning rate based on {metric_for_lr_scheduler}"
            )

        # Initialize the training monitor callback with parameters
        patience = callback_params.get("patience", 2)
        callback_model_type = callback_params.get("model_type", model_type)
        unfreeze_threshold = callback_params.get("unfreeze_threshold", 1e-5)

        # For tuning, set quiet mode to reduce output
        quiet_mode = bool(os.environ.get("TUNING_MODE"))

        self.training_monitor_callback = TrainingMonitorCallback(
            eval_dataset=tokenized_datasets["validation"],
            model_type=callback_model_type,
            patience=patience,
            unfreeze_threshold=unfreeze_threshold,
            quiet_mode=quiet_mode,
        )

        # Setup our custom Trainer with filtering and formatting
        self.trainer = self.CleanMetricsTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            compute_metrics=self.compute_metrics,
            optimizers=(
                optimizer,
                lr_scheduler,
            ),  # Custom optimizer and scheduler instance
            callbacks=[self.training_monitor_callback],  # Add training monitor callback
        )

        # For ReduceLROnPlateau, pass the optimizer, scheduler, and model to the callback
        # This allows the callback to update the learning rate based on metrics
        # and unfreeze the transformer when learning rate gets small
        self.training_monitor_callback.on_train_begin(
            args=training_args,
            state=None,
            control=None,
            model=self.model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

        return self.trainer

    def train(self, tokenized_datasets):
        """Train the model"""
        if self.trainer is None:
            self.setup_trainer(tokenized_datasets)

        debug_mode = os.environ.get("TUNING_DEBUG") == "True"
        # Only show starting message in non-tuning mode
        if not os.environ.get("TUNING_MODE"):
            print("\nStarting model training...")

        # Train the model - our CleanMetricsTrainer will handle metrics cleaning
        results = self.trainer.train()
        return results

    def evaluate(self, dataset):
        """Evaluate the model on a dataset"""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call setup_trainer() first")

        debug_mode = os.environ.get("TUNING_DEBUG") == "True"
        if not os.environ.get("TUNING_MODE") or debug_mode:
            print(f"\nEvaluating model...")

        # Get the evaluation results - our CleanMetricsTrainer will handle the cleaning
        results = self.trainer.evaluate(dataset)
        return results

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
