import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models import TextClassificationTrainer


def train_classifier(classifier_type, model_path, data_path, num_epochs=5):
    """Train a single classifier and save its results"""
    # Create output directory based on classifier
    output_dir = f"./results/{classifier_type}"
    model_save_path = f"./models/{classifier_type}"

    # Create trainer instance
    trainer = TextClassificationTrainer(
        model_path=model_path,
        output_dir=output_dir,
        num_epochs=num_epochs,
    )

    # Prepare data
    tokenized_datasets = trainer.prepare_data(
        data_path,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
    )

    print(f"Training {classifier_type.upper()} classifier")

    # Setup trainer with the specified model type
    trainer.setup_trainer(tokenized_datasets, model_type=classifier_type)

    # Train
    trainer.train(tokenized_datasets)

    # Generate and save learning curves
    if trainer.learning_curve_callback:
        # Save metrics to CSV
        metrics_path = trainer.learning_curve_callback.save_metrics()
        print(f"Metrics saved to {metrics_path}")

        # Plot and save learning curves
        curves_path = trainer.learning_curve_callback.plot_learning_curves()
        print(f"Learning curves saved to {curves_path}")

    # Evaluate
    test_results = trainer.evaluate(tokenized_datasets["test"])
    print(f"Test results for {classifier_type}: {test_results}")

    # Save model
    trainer.save_model(model_save_path)
    print(f"Model saved to {model_save_path}")

    return test_results


def train_all_classifiers(model_path, data_path, num_epochs=5, classifier_types=None):
    """Train multiple classifiers and create a comparison"""
    if classifier_types is None:
        classifier_types = ["standard", "custom", "bilstm", "attention", "cnn"]

    results = {}

    for classifier_type in classifier_types:
        print(f"\n{'=' * 50}")
        print(f"Training {classifier_type.upper()} classifier")
        print(f"{'=' * 50}\n")

        try:
            test_results = train_classifier(
                classifier_type=classifier_type,
                model_path=model_path,
                data_path=data_path,
                num_epochs=num_epochs,
            )
            results[classifier_type] = test_results
        except Exception as e:
            print(f"Error training {classifier_type} classifier: {e}")

    # Create comparison plot
    compare_classifiers(classifier_types)

    # Create results summary dataframe
    summary = pd.DataFrame.from_dict(results, orient="index")
    summary.index.name = "classifier"
    summary.reset_index(inplace=True)

    # Save summary
    summary_path = "./evaluation/classifier_results_summary.csv"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"\nResults summary saved to {summary_path}")

    return summary


def compare_classifiers(classifiers=None):
    """
    Create a comparison plot of learning curves from multiple classifier results

    Args:
        classifiers: List of classifier names to compare. If None, uses all available.
    """
    if classifiers is None:
        classifiers = ["standard", "custom", "bilstm", "attention", "cnn"]

    # Set up the figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Classifier Comparison", fontsize=18)

    # For consistent colors across plots
    colors = sns.color_palette("husl", len(classifiers))

    # Plot Loss and Accuracy for each classifier
    for i, classifier in enumerate(classifiers):
        metrics_path = f"./evaluation/{classifier}/metrics.csv"

        try:
            # Check if metrics file exists
            if not os.path.exists(metrics_path):
                print(f"No metrics found for {classifier} classifier")
                continue

            # Load metrics
            metrics_df = pd.read_csv(metrics_path)

            # Filter evaluation metrics
            eval_metrics = metrics_df.filter(regex="^(eval|epoch)").dropna(
                subset=["eval_loss"]
            )

            if not eval_metrics.empty:
                # Plot Loss
                sns.lineplot(
                    x="epoch",
                    y="eval_loss",
                    data=eval_metrics,
                    label=f"{classifier.capitalize()}",
                    ax=axes[0],
                    color=colors[i],
                )

                # Plot Accuracy
                if "eval_accuracy" in eval_metrics.columns:
                    sns.lineplot(
                        x="epoch",
                        y="eval_accuracy",
                        data=eval_metrics,
                        label=f"{classifier.capitalize()}",
                        ax=axes[1],
                        color=colors[i],
                    )
        except Exception as e:
            print(f"Error processing {classifier} metrics: {e}")

    # Set titles and labels
    axes[0].set_title("Validation Loss Comparison", fontsize=14)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].legend(fontsize=10)

    axes[1].set_title("Validation Accuracy Comparison", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].legend(fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle

    # Save the comparison plot
    save_path = "./evaluation/classifier_comparison.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Classifier comparison saved to {save_path}")
    return save_path