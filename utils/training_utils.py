import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models import TextClassificationTrainer


def train_classifier(
    classifier_type,
    model_path,
    data_path,
    num_epochs=5,
    batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    early_stopping_patience=2,
    dropout_rate=0.3,
    output_dir=None,
    save_dir=None,
    max_length=64,
    train_size=0.6,
    val_size=0.2,
    test_size=0.2,
    metric_for_best_model="matthews_correlation",
    num_frequencies=8,
    num_wavelets=16,
    wavelet_type="mixed",
    use_multilayer=False,
    num_layers_to_use=3,
    optimizer_type="adamw",
):
    """Train a single classifier and save its results"""
    # Set default output directories if not provided
    if output_dir is None:
        output_dir = f"./results/{classifier_type}"
    if save_dir is None:
        save_dir = f"./models/{classifier_type}"

    # Create a small test model to verify GPU usage
    import torch

    # Explicitly print detailed GPU info
    if torch.cuda.is_available():
        print(f"\nCUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA current device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, "mps"):
        print(f"\nMPS available: {torch.backends.mps.is_available()}")
        print(f"MPS built: {torch.backends.mps.is_built()}")

    # Create test tensor to verify device placement
    print("\nTesting GPU with tensor operation:")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Create test tensor
    x = torch.ones(1, 3).to(device)
    print(f"Test tensor device: {x.device}")

    # Create trainer instance with all parameters
    trainer = TextClassificationTrainer(
        model_path=model_path,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_length=max_length,
    )

    # Prepare data
    tokenized_datasets = trainer.prepare_data(
        data_path,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )

    print(f"Training {classifier_type.upper()} classifier")

    # Pass the model-specific parameters
    model_params = {
        "dropout_rate": dropout_rate,
        "use_multilayer": use_multilayer,
        "num_layers_to_use": num_layers_to_use,
        "optimizer_type": optimizer_type
    }
    
    if classifier_type == "fourier_kan":
        model_params["num_frequencies"] = num_frequencies
    elif classifier_type == "wavelet_kan":
        model_params["num_wavelets"] = num_wavelets
        model_params["wavelet_type"] = wavelet_type

    # Setup early stopping in callback
    callback_params = {
        "patience": early_stopping_patience,
        "model_type": classifier_type,
        "metric_for_best_model": metric_for_best_model  # Use the same metric for callback
    }

    # Handle special pooling configurations that use the CustomClassifier
    actual_model_type = classifier_type
    if classifier_type == "mean_pooling":
        actual_model_type = "custom"
        model_params["pooling_strategy"] = "mean"
        model_params["dropout_rate"] = dropout_rate
        model_params["use_multilayer"] = use_multilayer
        model_params["num_layers_to_use"] = num_layers_to_use
    elif classifier_type == "combined_pooling":
        actual_model_type = "custom"
        model_params["pooling_strategy"] = "combined"
        model_params["dropout_rate"] = dropout_rate
        model_params["use_multilayer"] = use_multilayer
        model_params["num_layers_to_use"] = num_layers_to_use
            
    # Setup trainer with the specified model type and parameters
    trainer.setup_trainer(
        tokenized_datasets,
        model_type=actual_model_type,
        metric_for_best_model=metric_for_best_model,
        model_params=model_params,
        callback_params=callback_params,
    )

    # Train
    trainer.train(tokenized_datasets)

    # Generate and save learning curves
    if trainer.training_monitor_callback:
        # Save metrics to CSV
        metrics_path = trainer.training_monitor_callback.save_metrics()
        print(f"Metrics saved to {metrics_path}")

        # Plot and save learning curves
        curves_path = trainer.training_monitor_callback.plot_learning_curves()
        print(f"Learning curves saved to {curves_path}")

    # Evaluate
    test_results = trainer.evaluate(tokenized_datasets["test"])
    print(f"Test results for {classifier_type}: {test_results}")

    # Save model
    trainer.save_model(save_dir)
    print(f"Model saved to {save_dir}")

    return test_results


def train_all_classifiers(
    model_path,
    data_path,
    num_epochs=5,
    batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    early_stopping_patience=2,
    dropout_rate=0.3,
    output_dir="./results",
    save_dir="./models",
    max_length=64,
    train_size=0.6,
    val_size=0.2,
    test_size=0.2,
    metric_for_best_model="matthews_correlation",
    num_frequencies=8,
    num_wavelets=16,
    wavelet_type="mixed",
    use_multilayer=False,
    num_layers_to_use=3,
    optimizer_type="adamw",
    classifier_types=None,
):
    """Train multiple classifiers and create a comparison"""
    if classifier_types is None:
        classifier_types = [
            "attention",
            "bilstm", 
            "cnn",
            "combined_pooling",
            "fourier_kan",
            "mean_pooling",
            "standard",
            "wavelet_kan",
        ]

    results = {}
    
    # Sort classifiers alphabetically for consistent ordering
    sorted_classifier_types = sorted(classifier_types)
    
    for classifier_type in sorted_classifier_types:
        print(f"\n{'=' * 50}")
        print(f"Training {classifier_type.upper()} classifier")
        print(f"{'=' * 50}\n")

        # Setup classifier-specific directories
        classifier_output_dir = f"{output_dir}/{classifier_type}"
        classifier_save_dir = f"{save_dir}/{classifier_type}"

        try:
            test_results = train_classifier(
                classifier_type=classifier_type,
                model_path=model_path,
                data_path=data_path,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                early_stopping_patience=early_stopping_patience,
                dropout_rate=dropout_rate,
                output_dir=classifier_output_dir,
                save_dir=classifier_save_dir,
                max_length=max_length,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                metric_for_best_model=metric_for_best_model,
                num_frequencies=num_frequencies,
                num_wavelets=num_wavelets,
                wavelet_type=wavelet_type,
                use_multilayer=use_multilayer,
                num_layers_to_use=num_layers_to_use,
                optimizer_type=optimizer_type,
            )
            results[classifier_type] = test_results
        except Exception as e:
            print(f"Error training {classifier_type} classifier: {e}")

    # Create comparison plot
    compare_classifiers(sorted_classifier_types)

    # Create results summary dataframe
    summary = pd.DataFrame.from_dict(results, orient="index")
    summary.index.name = "classifier"
    summary.reset_index(inplace=True)

    # Save summary
    output_dir_path = Path(output_dir)
    summary_path = output_dir_path / "classifier_results_summary.csv"
    output_dir_path.mkdir(parents=True, exist_ok=True)
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
        classifiers = ["attention", "bilstm", "cnn", "combined_pooling", "fourier_kan", "mean_pooling", "standard", "wavelet_kan"]

    # Set up the figure with 3x2 grid (to include precision and recall)
    fig, axes = plt.subplots(3, 2, figsize=(18, 21))
    fig.suptitle("Classifier Comparison", fontsize=18)

    # For consistent colors across plots
    colors = sns.color_palette("husl", len(classifiers))

    # Plot Loss, Accuracy, Precision, Recall, F1, and Matthews Correlation
    for i, classifier in enumerate(classifiers):
        metrics_path = Path("evaluation") / classifier / "metrics.csv"

        try:
            # Check if metrics file exists
            if not metrics_path.exists():
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
                    ax=axes[0, 0],
                    color=colors[i],
                )

                # Plot Accuracy
                if "eval_accuracy" in eval_metrics.columns:
                    sns.lineplot(
                        x="epoch",
                        y="eval_accuracy",
                        data=eval_metrics,
                        label=f"{classifier.capitalize()}",
                        ax=axes[0, 1],
                        color=colors[i],
                    )

                # Plot Precision
                if "eval_precision" in eval_metrics.columns:
                    sns.lineplot(
                        x="epoch",
                        y="eval_precision",
                        data=eval_metrics,
                        label=f"{classifier.capitalize()}",
                        ax=axes[1, 0],
                        color=colors[i],
                    )

                # Plot Recall
                if "eval_recall" in eval_metrics.columns:
                    sns.lineplot(
                        x="epoch",
                        y="eval_recall",
                        data=eval_metrics,
                        label=f"{classifier.capitalize()}",
                        ax=axes[1, 1],
                        color=colors[i],
                    )

                # Plot F1 Score (macro)
                if "eval_f1_macro" in eval_metrics.columns:
                    sns.lineplot(
                        x="epoch",
                        y="eval_f1_macro",
                        data=eval_metrics,
                        label=f"{classifier.capitalize()}",
                        ax=axes[2, 0],
                        color=colors[i],
                    )

                # Plot Matthews Correlation
                if "eval_matthews_correlation" in eval_metrics.columns:
                    sns.lineplot(
                        x="epoch",
                        y="eval_matthews_correlation",
                        data=eval_metrics,
                        label=f"{classifier.capitalize()}",
                        ax=axes[2, 1],
                        color=colors[i],
                    )
        except Exception as e:
            print(f"Error processing {classifier} metrics: {e}")

    # Set titles and labels
    axes[0, 0].set_title("Validation Loss Comparison", fontsize=14)
    axes[0, 0].set_xlabel("Epoch", fontsize=12)
    axes[0, 0].set_ylabel("Loss", fontsize=12)
    axes[0, 0].legend(fontsize=10)

    axes[0, 1].set_title("Validation Accuracy Comparison", fontsize=14)
    axes[0, 1].set_xlabel("Epoch", fontsize=12)
    axes[0, 1].set_ylabel("Accuracy", fontsize=12)
    axes[0, 1].legend(fontsize=10)

    axes[1, 0].set_title("Precision Comparison", fontsize=14)
    axes[1, 0].set_xlabel("Epoch", fontsize=12)
    axes[1, 0].set_ylabel("Precision", fontsize=12)
    axes[1, 0].legend(fontsize=10)

    axes[1, 1].set_title("Recall Comparison", fontsize=14)
    axes[1, 1].set_xlabel("Epoch", fontsize=12)
    axes[1, 1].set_ylabel("Recall", fontsize=12)
    axes[1, 1].legend(fontsize=10)

    axes[2, 0].set_title("F1 Score (Macro) Comparison", fontsize=14)
    axes[2, 0].set_xlabel("Epoch", fontsize=12)
    axes[2, 0].set_ylabel("F1 Score", fontsize=12)
    axes[2, 0].legend(fontsize=10)

    axes[2, 1].set_title("Matthews Correlation Comparison", fontsize=14)
    axes[2, 1].set_xlabel("Epoch", fontsize=12)
    axes[2, 1].set_ylabel("MCC", fontsize=12)
    axes[2, 1].legend(fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle

    # Save the comparison plot
    save_path = Path("evaluation") / "classifier_comparison.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Classifier comparison saved to {save_path}")
    return str(save_path)
