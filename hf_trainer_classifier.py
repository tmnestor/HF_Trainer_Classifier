import os
import torch
import argparse
from pathlib import Path
from utils import (
    train_classifier, 
    train_all_classifiers, 
    compare_classifiers,
    tune_classifier,
    tune_all_classifiers
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Hugging Face Text Classification Trainer CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Create argument groups
    model_group = parser.add_argument_group("Model Configuration")
    training_group = parser.add_argument_group("Training Parameters")
    dataset_group = parser.add_argument_group("Dataset Configuration")
    optimization_group = parser.add_argument_group("Optimization Parameters")
    output_group = parser.add_argument_group("Output Configuration")

    # Action arguments (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--train", action="store_true", help="Train a single classifier"
    )
    action_group.add_argument(
        "--train-all", action="store_true", help="Train all classifier types"
    )
    action_group.add_argument(
        "--compare", action="store_true", help="Compare trained models"
    )
    action_group.add_argument(
        "--tune", action="store_true", help="Tune hyperparameters for a single classifier"
    )
    action_group.add_argument(
        "--tune-all", action="store_true", help="Tune hyperparameters for all classifier types"
    )

    # Model arguments
    model_group.add_argument(
        "--model-name",
        type=str,
        default="ModernBERT-base",
        help="Name of the Hugging Face model to use",
    )
    model_group.add_argument(
        "--model-path",
        type=str,
        help="Custom path to pretrained model (overrides --model-name)",
    )
    model_group.add_argument(
        "--classifier",
        type=str,
        default="cnn",
        choices=["standard", "custom", "bilstm", "attention", "cnn", "fourier_kan", "wavelet_kan"],
        help="Type of classifier to use",
    )
    model_group.add_argument(
        "--classifiers",
        type=str,
        nargs="+",
        default=["standard", "custom", "bilstm", "attention", "cnn", "fourier_kan", "wavelet_kan"],
        help="List of classifiers to train when using --train-all",
    )

    # Dataset arguments
    dataset_group.add_argument(
        "--data-path",
        type=str,
        help="Path to dataset CSV file (defaults to $DATADIR/bbc-text/bbc-text.csv)",
    )
    dataset_group.add_argument(
        "--max-length",
        type=int,
        default=64,
        help="Maximum sequence length for tokenization",
    )
    dataset_group.add_argument(
        "--train-size",
        type=float,
        default=0.6,
        help="Proportion of data to use for training",
    )
    dataset_group.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for validation",
    )
    dataset_group.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing",
    )

    # Training parameters
    training_group.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    training_group.add_argument(
        "--batch-size", type=int, default=8, help="Training batch size"
    )
    training_group.add_argument(
        "--early-stopping",
        type=int,
        default=2,
        help="Early stopping patience (set to 0 to disable)",
    )
    training_group.add_argument(
        "--metric",
        type=str,
        default="matthews_correlation",
        choices=["accuracy", "precision", "recall", "f1_macro", "matthews_correlation"],
        help="Metric to use for model selection and early stopping",
    )
    
    # Tuning parameters
    tuning_group = parser.add_argument_group("Hyperparameter Tuning")
    tuning_group.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of hyperparameter optimization trials",
    )
    tuning_group.add_argument(
        "--tuning-epochs",
        type=int,
        default=5,
        help="Number of epochs for each tuning trial",
    )
    tuning_group.add_argument(
        "--hyperparams-dir",
        type=str,
        default="./hyperparams",
        help="Directory to save hyperparameter configurations",
    )

    # Optimization parameters
    optimization_group.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate for training"
    )
    optimization_group.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for regularization",
    )
    optimization_group.add_argument(
        "--dropout", type=float, default=0.3, help="Dropout rate for classifier"
    )
    optimization_group.add_argument(
        "--num-frequencies", type=int, default=16, 
        help="Number of frequency components for FourierKAN classifier"
    )
    optimization_group.add_argument(
        "--num-wavelets", type=int, default=16, 
        help="Number of wavelet components for WaveletKAN classifier"
    )
    optimization_group.add_argument(
        "--wavelet-type", type=str, default="mixed", 
        choices=["mixed", "haar", "mexican", "morlet"],
        help="Type of wavelet to use for WaveletKAN classifier"
    )
    optimization_group.add_argument(
        "--no-cuda", action="store_true", help="Disable CUDA even when available"
    )

    # Output arguments
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save training results",
    )
    output_group.add_argument(
        "--save-dir",
        type=str,
        default="./models",
        help="Directory to save trained models",
    )

    args = parser.parse_args()

    # Default to training a single classifier if no action specified
    if not (args.train or args.train_all or args.compare or args.tune or args.tune_all):
        args.train = True

    # Set default model path if not provided
    if args.model_path is None:
        args.model_path = str(
            Path(os.environ.get("LLM_MODELS_PATH", ".")) / args.model_name
        )

    # Set default data path if not provided
    if args.data_path is None:
        args.data_path = (
            Path(os.environ.get("DATADIR", ".")) / "bbc-text" / "bbc-text.csv"
        )

    return args


def check_device(no_cuda=False):
    """Check and print available devices"""
    if no_cuda:
        device = torch.device("cpu")
        print("CUDA disabled by user, using CPU")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("No GPU found, using CPU")

    return device


def main():
    args = parse_arguments()
    device = check_device(args.no_cuda)

    # Determine classifier_type
    classifier_type = args.classifier if args.train else None

    # Create output directories with classifier type if single training
    if args.train:
        # output_dir = f"{args.output_dir}/{args.classifier}"
        output_dir = Path(args.output_dir) / args.classifier
        save_dir = Path(args.save_dir) / args.classifier
        # save_dir = f"{args.save_dir}/{args.classifier}"
    else:
        output_dir = Path(args.output_dir)
        save_dir = Path(args.save_dir)

    # Additional parameters to pass to training functions
    train_params = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "early_stopping_patience": args.early_stopping,
        "dropout_rate": args.dropout,
        "output_dir": output_dir,
        "save_dir": save_dir,
        "max_length": args.max_length,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "metric_for_best_model": args.metric,
    }
    
    # Add KAN-specific parameters if needed
    # For FourierKAN
    if args.classifier == 'fourier_kan' or (hasattr(args, 'classifiers') and 'fourier_kan' in args.classifiers):
        train_params['num_frequencies'] = args.num_frequencies
        
    # For WaveletKAN
    if args.classifier == 'wavelet_kan' or (hasattr(args, 'classifiers') and 'wavelet_kan' in args.classifiers):
        train_params['num_wavelets'] = args.num_wavelets
        train_params['wavelet_type'] = args.wavelet_type

    if args.train:
        print(f"\n{'=' * 50}")
        print(f"Training {args.classifier.upper()} classifier")
        print(f"{'=' * 50}\n")

        train_classifier(classifier_type=args.classifier, **train_params)

    elif args.train_all:
        train_all_classifiers(classifier_types=args.classifiers, **train_params)

    elif args.compare:
        print("\nComparing trained models...")
        compare_classifiers(classifiers=args.classifiers)
        
    elif args.tune:
        print(f"\n{'=' * 50}")
        print(f"Tuning {args.classifier.upper()} classifier hyperparameters")
        print(f"{'=' * 50}\n")
        
        tune_params = {
            "classifier_type": args.classifier,
            "model_path": args.model_path,
            "data_path": args.data_path,
            "n_trials": args.n_trials,
            "num_epochs": args.tuning_epochs,
            "metric_for_best_model": args.metric,
            "output_dir": output_dir,
            "save_dir": args.hyperparams_dir,
            "max_length": args.max_length,
            "train_size": args.train_size,
            "val_size": args.val_size,
            "test_size": args.test_size,
        }
        
        tune_classifier(**tune_params)
        
    elif args.tune_all:
        print(f"\n{'=' * 50}")
        print(f"Tuning ALL classifier hyperparameters")
        print(f"{'=' * 50}\n")
        
        tune_params = {
            "model_path": args.model_path,
            "data_path": args.data_path,
            "classifier_types": args.classifiers,
            "n_trials": args.n_trials,
            "num_epochs": args.tuning_epochs,
            "metric_for_best_model": args.metric,
            "output_dir": output_dir,
            "save_dir": args.hyperparams_dir,
            "max_length": args.max_length,
            "train_size": args.train_size,
            "val_size": args.val_size,
            "test_size": args.test_size,
        }
        
        tune_all_classifiers(**tune_params)


if __name__ == "__main__":
    main()
