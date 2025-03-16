import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import TrainerCallback, EarlyStoppingCallback


class TrainingMonitorCallback(TrainerCallback):
    """Callback to monitor training progress, track metrics, adjust learning rate, and implement early stopping"""

    def __init__(self, eval_dataset=None, model_type="standard", patience=2, pruning_callback=None, 
                 unfreeze_threshold=1e-5, quiet_mode=False):
        self.eval_dataset = eval_dataset
        self.model_type = model_type
        self.patience = patience
        self.pruning_callback = pruning_callback
        self.unfreeze_threshold = unfreeze_threshold
        self.transformer_unfrozen = False
        self.quiet_mode = quiet_mode  # Set to True to reduce output during tuning
        # Initialize empty DataFrame for metrics
        self.metrics_df = pd.DataFrame()
        
        # Track best metrics for early stopping implementation
        self.best_metric = None
        self.no_improvement_count = 0
        
        # Set in on_train_begin if we have a ReduceLROnPlateau scheduler
        self.lr_scheduler = None
        self.metric_name = None
        
    def on_train_begin(self, args, state, control, model=None, train_dataloader=None, **kwargs):
        """Extract and store LR scheduler from trainer if it's ReduceLROnPlateau"""
        # Store the model for later use in unfreezing
        if model is not None:
            self.model = model
            
        if "optimizer" in kwargs and "lr_scheduler" in kwargs:
            lr_scheduler = kwargs.get("lr_scheduler")
            # Check if it's a ReduceLROnPlateau scheduler
            if hasattr(lr_scheduler, "__class__") and lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
                self.lr_scheduler = lr_scheduler
                # Get the metric name to monitor, falling back to matthews_correlation if not specified
                self.metric_name = getattr(lr_scheduler, "metric_name", "matthews_correlation")
                if not os.environ.get('TUNING_MODE'):
                    print(f"TrainingMonitorCallback will manage ReduceLROnPlateau scheduler based on {self.metric_name}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when the Trainer logs metrics"""
        if logs is None:
            return

        # Add step/epoch info to the logs
        logs = logs.copy()
        if "epoch" not in logs and state.epoch is not None:
            # Format epoch as 1-based integer
            logs["epoch"] = int(state.epoch) + 1
        if "step" not in logs and state.global_step is not None:
            logs["step"] = state.global_step

        # Rename internal pred_distribution for our logs if needed
        if "_internal_pred_distribution" in logs:
            logs["pred_distribution"] = logs["_internal_pred_distribution"]
            
        # Filter logs to keep only essential metrics
        filtered_logs = logs.copy()
        keys_to_remove = []
        for key in filtered_logs:
            # Remove runtime metrics and any internal metrics
            if any(unwanted in key for unwanted in ['runtime', 'samples_per_second', 'steps_per_second', '_internal_', 'pred_distribution']):
                keys_to_remove.append(key)
        
        # Remove unwanted keys
        for key in keys_to_remove:
            filtered_logs.pop(key, None)
            
        # Convert filtered logs to DataFrame
        log_df = pd.DataFrame([filtered_logs])

        # Append to the metrics DataFrame
        self.metrics_df = pd.concat([self.metrics_df, log_df], ignore_index=True)
        
        # Implement early stopping and LR scheduling based on specified metric
        # Use the metric name from the LR scheduler if available, otherwise use matthews_correlation
        metric_to_monitor = self.metric_name or "matthews_correlation"
        metric_key = f"eval_{metric_to_monitor}"
        
        if metric_key in logs:
            current_metric = logs[metric_key]
            
            # First evaluation or new best
            if self.best_metric is None or current_metric > self.best_metric:
                # Only print first evaluation and improvements when not in quiet mode
                if self.best_metric is not None and not self.quiet_mode:
                    print(f"Improved: {current_metric:.4f} (prev: {self.best_metric:.4f})")
                
                self.best_metric = current_metric
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
                
            # Update the ReduceLROnPlateau scheduler if available
            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, "step"):
                # For ReduceLROnPlateau, we need to pass the metric
                if self.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    self.lr_scheduler.step(current_metric)
                # For other schedulers that don't need the metric, we would just call step()
                # But we skip this since the Trainer handles other types of schedulers
                
                # Get the current learning rate and log it if optimizer is available (when not in quiet mode)
                if hasattr(self.lr_scheduler, "optimizer") and len(self.lr_scheduler.optimizer.param_groups) > 1:
                    current_lr = self.lr_scheduler.optimizer.param_groups[1]['lr']  # Index 1 for classifier
                    # Use scientific notation for small values - only print in non-quiet mode
                    if not self.quiet_mode:
                        print(f"Current classifier learning rate: {current_lr:.2e}")
                    
                    # Check if we should unfreeze the transformer based on learning rate threshold
                    if not self.transformer_unfrozen and current_lr <= self.unfreeze_threshold:
                        self._unfreeze_transformer()
            
            # Check if we should stop training - increased patience for ReduceLROnPlateau
            effective_patience = self.patience * 2 if self.lr_scheduler is not None else self.patience
            if self.no_improvement_count >= effective_patience:
                # Always show early stopping message even in quiet mode
                if not self.quiet_mode:
                    print(f"\nEarly stopping triggered after {self.no_improvement_count} evaluations without improvement")
                control.should_training_stop = True
            
            # If we have a pruning callback, call it with the evaluation metrics
            if self.pruning_callback is not None:
                try:
                    import optuna
                    
                    # Check for pruning without verbose output
                    pruning_result = self.pruning_callback.on_evaluate(logs)
                    
                except optuna.exceptions.TrialPruned:
                    print("Trial pruned")
                    control.should_training_stop = True
                    # Re-raise to ensure pruning is properly reported to Optuna
                    raise

    def plot_learning_curves(self, save_path=None):
        """Plot learning curves from collected metrics"""
        # Use classifier-specific path if not provided
        if save_path is None:
            save_path = Path("evaluation") / self.model_type / "learning_curves.png"

        # Create the directory if it doesn't exist
        save_path = Path(save_path)  # Ensure it's a Path object
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Filter for training and evaluation metrics
        train_metrics = self.metrics_df.filter(regex="^(loss|train|epoch|step)").dropna(
            subset=["loss"]
        )
        eval_metrics = self.metrics_df.filter(regex="^(eval|epoch|step)").dropna(
            subset=["eval_loss"]
        )

        # Set up the figure with 2 rows and 2 columns
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Add classifier type in the figure title
        fig.suptitle(
            f"Learning Curves for {self.model_type.capitalize()} Classifier",
            fontsize=16,
        )

        # Plot Loss curves
        if not train_metrics.empty and not eval_metrics.empty:
            ax = axes[0, 0]
            sns.lineplot(
                x="epoch", y="loss", data=train_metrics, label="Training Loss", ax=ax
            )
            sns.lineplot(
                x="epoch",
                y="eval_loss",
                data=eval_metrics,
                label="Validation Loss",
                ax=ax,
            )
            ax.set_title("Loss vs. Epoch")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()

        # Plot Accuracy curves
        if "eval_accuracy" in eval_metrics.columns:
            ax = axes[0, 1]
            sns.lineplot(
                x="epoch",
                y="eval_accuracy",
                data=eval_metrics,
                label="Validation Accuracy",
                ax=ax,
            )
            ax.set_title("Accuracy vs. Epoch")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.legend()
            
        # Plot F1 Macro
        if "eval_f1_macro" in eval_metrics.columns:
            ax = axes[1, 0]
            sns.lineplot(
                x="epoch",
                y="eval_f1_macro",
                data=eval_metrics,
                label="F1 Macro",
                ax=ax,
            )
            ax.set_title("F1 Macro vs. Epoch")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("F1 Score")
            ax.legend()
        
        # Plot Matthews Correlation
        if "eval_matthews_correlation" in eval_metrics.columns:
            ax = axes[1, 1]
            sns.lineplot(
                x="epoch",
                y="eval_matthews_correlation",
                data=eval_metrics,
                label="Matthews Correlation",
                ax=ax,
            )
            ax.set_title("Matthews Correlation vs. Epoch")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MCC")
            ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        plt.savefig(save_path)
        plt.close()

        return str(save_path)

    def save_metrics(self, save_path=None):
        """Save metrics to CSV"""
        # Use classifier-specific path if not provided
        if save_path is None:
            save_path = Path("evaluation") / self.model_type / "metrics.csv"
            
        # Ensure save_path is a Path object
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Format floating point numbers to 4 decimal places before saving
        formatted_df = self.metrics_df.copy()
        
        # Format floating point columns to 4 decimal places
        for col in formatted_df.columns:
            if formatted_df[col].dtype == 'float64' or formatted_df[col].dtype == 'float32':
                formatted_df[col] = formatted_df[col].map(lambda x: round(x, 4))
                
        # Make sure epoch is an integer
        if 'epoch' in formatted_df.columns:
            formatted_df['epoch'] = formatted_df['epoch'].astype(int)
            
        # Save formatted metrics to CSV
        formatted_df.to_csv(save_path, index=False)
        return str(save_path)
        
    def _unfreeze_transformer(self):
        """Unfreeze transformer layers and adjust learning rate"""
        if self.model is None or not hasattr(self.lr_scheduler, "optimizer"):
            print("Cannot unfreeze transformer: model or optimizer not available")
            return
            
        # Detect transformer parameters group (usually index 0)
        transformer_param_group = self.lr_scheduler.optimizer.param_groups[0]
        
        # Unfreeze all parameters in the transformer group
        unfrozen_count = 0
        for param in transformer_param_group['params']:
            if not param.requires_grad:
                param.requires_grad = True
                unfrozen_count += 1
                
        # Set a small learning rate for the transformer (10% of the current classifier lr)
        current_classifier_lr = self.lr_scheduler.optimizer.param_groups[1]['lr']  # Index 1 for classifier
        transformer_lr = current_classifier_lr * 0.1  # Use a smaller lr for transformer
        transformer_param_group['lr'] = transformer_lr
        
        # Mark as unfrozen so we don't do this again
        self.transformer_unfrozen = True
        
        # Only print when not in quiet mode
        if not self.quiet_mode:
            print(f"\n🔓 Unfroze {unfrozen_count} transformer parameters with lr={transformer_lr:.2e}")
            print(f"   Training will continue with both transformer and classifier layers active")