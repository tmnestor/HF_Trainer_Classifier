import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import TrainerCallback, EarlyStoppingCallback


class LearningCurveCallback(TrainerCallback):
    """Callback to track training and evaluation metrics for learning curves"""

    def __init__(self, eval_dataset=None, model_type="standard", patience=2):
        self.eval_dataset = eval_dataset
        self.model_type = model_type
        self.patience = patience
        # Initialize empty DataFrame for metrics
        self.metrics_df = pd.DataFrame()
        
        # Track best metrics for early stopping implementation
        self.best_metric = None
        self.no_improvement_count = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when the Trainer logs metrics"""
        if logs is None:
            return

        # Add step/epoch info to the logs
        logs = logs.copy()
        if "epoch" not in logs and state.epoch is not None:
            logs["epoch"] = round(state.epoch, 2)
        if "step" not in logs and state.global_step is not None:
            logs["step"] = state.global_step

        # Convert logs to DataFrame and add evaluation prefix if needed
        log_df = pd.DataFrame([logs])

        # Append to the metrics DataFrame
        self.metrics_df = pd.concat([self.metrics_df, log_df], ignore_index=True)
        
        # Implement early stopping manually based on Matthews Correlation
        if "eval_matthews_correlation" in logs:
            current_metric = logs["eval_matthews_correlation"]
            
            # First evaluation or new best
            if self.best_metric is None or current_metric > self.best_metric:
                self.best_metric = current_metric
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
                
            # Check if we should stop training
            if self.no_improvement_count >= self.patience:
                print(f"\nEarly stopping triggered after {self.no_improvement_count} evaluations without improvement")
                control.should_training_stop = True

    def plot_learning_curves(self, save_path=None):
        """Plot learning curves from collected metrics"""
        # Use classifier-specific path if not provided
        if save_path is None:
            save_path = f"./evaluation/{self.model_type}/learning_curves.png"

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

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
            
        # Plot F1 Score curves
        if "eval_f1_macro" in eval_metrics.columns and "eval_f1_weighted" in eval_metrics.columns:
            ax = axes[1, 0]
            sns.lineplot(
                x="epoch",
                y="eval_f1_macro",
                data=eval_metrics,
                label="F1 Macro",
                ax=ax,
            )
            sns.lineplot(
                x="epoch",
                y="eval_f1_weighted",
                data=eval_metrics,
                label="F1 Weighted",
                ax=ax,
            )
            ax.set_title("F1 Scores vs. Epoch")
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

        return save_path

    def save_metrics(self, save_path=None):
        """Save metrics to CSV"""
        # Use classifier-specific path if not provided
        if save_path is None:
            save_path = f"./evaluation/{self.model_type}/metrics.csv"

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.metrics_df.to_csv(save_path, index=False)
        return save_path