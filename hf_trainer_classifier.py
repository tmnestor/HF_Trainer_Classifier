import os
import sys
import torch
import argparse
import pandas as pd
from pathlib import Path
from utils import (
    train_classifier, 
    train_all_classifiers, 
    compare_classifiers,
    tune_classifier,
    tune_all_classifiers
)

# Note: Since we're using in-memory Optuna studies, there's no need to clean up databases
# This function is kept for reference but no longer necessary
def clean_study_files(classifier_type=None):
    """This function is no longer needed as we're using in-memory studies"""
    pass


def load_best_configs(classifier_type=None):
    """
    Load best configurations from YAML files
    
    Args:
        classifier_type: If provided, load only this classifier's config
        
    Returns:
        Dictionary mapping classifier types to their best configs
    """
    import yaml
    from pathlib import Path
    
    hyperparams_dir = Path("./hyperparams")
    configs = {}
    
    # Function to load a single config
    def load_config(clf_type):
        config_path = hyperparams_dir / f"{clf_type}_best_config.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                try:
                    config = yaml.safe_load(f)
                    # Most importantly, get the best_params
                    if config and "best_params" in config:
                        return config["best_params"]
                except Exception as e:
                    print(f"Error loading config for {clf_type}: {e}")
        return None
    
    if classifier_type:
        # Load just one config
        config = load_config(classifier_type)
        if config:
            configs[classifier_type] = config
    else:
        # Load all available configs
        for config_file in hyperparams_dir.glob("*_best_config.yaml"):
            clf_type = config_file.stem.replace("_best_config", "")
            config = load_config(clf_type)
            if config:
                configs[clf_type] = config
    
    return configs

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
        default="standard",
        choices=["standard", "bilstm", "attention", "cnn", "fourier_kan", "wavelet_kan", "mean_pooling", "combined_pooling"],
        help="Type of classifier to use",
    )
    model_group.add_argument(
        "--classifiers",
        type=str,
        nargs="+",
        default=["attention", "bilstm", "cnn", "combined_pooling", "fourier_kan", "mean_pooling", "standard", "wavelet_kan"],
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
        "--batch-size", type=int, default=16, help="Training batch size"
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
        default="f1_macro",
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
        "--optimizer", type=str, default="adamw", choices=["adamw", "sgd", "rmsprop"],
        help="Optimizer to use for training"
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
        choices=["mixed", "haar", "db2", "db4"],
        help="Type of wavelet to use for WaveletKAN classifier"
    )
    optimization_group.add_argument(
        "--no-cuda", action="store_true", help="Disable CUDA even when available"
    )
    optimization_group.add_argument(
        "--use-multilayer", action="store_true", 
        help="Use outputs from multiple transformer layers for classification"
    )
    optimization_group.add_argument(
        "--num-layers-to-use", type=int, default=3,
        help="Number of transformer layers to use when --use-multilayer is enabled"
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
    
    # Track which arguments were explicitly specified on the command line
    # This helps us determine which values to override with best configs
    args.batch_size_specified = 'batch_size' in sys.argv
    args.learning_rate_specified = 'learning_rate' in sys.argv or '--learning-rate' in sys.argv
    args.weight_decay_specified = 'weight_decay' in sys.argv or '--weight-decay' in sys.argv
    args.dropout_specified = 'dropout' in sys.argv or '--dropout' in sys.argv
    args.num_frequencies_specified = 'num_frequencies' in sys.argv or '--num-frequencies' in sys.argv
    args.num_wavelets_specified = 'num_wavelets' in sys.argv or '--num-wavelets' in sys.argv
    args.wavelet_type_specified = 'wavelet_type' in sys.argv or '--wavelet-type' in sys.argv
    args.use_multilayer_specified = 'use_multilayer' in sys.argv or '--use-multilayer' in sys.argv
    args.num_layers_to_use_specified = 'num_layers_to_use' in sys.argv or '--num-layers-to-use' in sys.argv
    args.optimizer_specified = 'optimizer' in sys.argv or '--optimizer' in sys.argv

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
    
    # Load best configurations from YAML files
    best_configs = {}
    if args.train:
        # Load config for single classifier
        best_configs = load_best_configs(classifier_type)
        
        # Apply best config to CLI args if corresponding CLI options weren't specified
        if best_configs and classifier_type in best_configs:
            best_cfg = best_configs[classifier_type]
            print(f"Using best configuration for {classifier_type} from hyperparameter tuning")
            
            # Only override if not specified in CLI
            if not args.batch_size_specified and "batch_size" in best_cfg:
                args.batch_size = best_cfg["batch_size"]
                print(f"  batch_size: {args.batch_size}")
                
            if not args.learning_rate_specified and "learning_rate" in best_cfg:
                args.learning_rate = best_cfg["learning_rate"]
                print(f"  learning_rate: {args.learning_rate}")
                
            if not args.weight_decay_specified and "weight_decay" in best_cfg:
                args.weight_decay = best_cfg["weight_decay"]
                print(f"  weight_decay: {args.weight_decay}")
                
            if not args.dropout_specified and "dropout_rate" in best_cfg:
                args.dropout = best_cfg["dropout_rate"]
                print(f"  dropout_rate: {args.dropout}")
                
            if not args.optimizer_specified and "optimizer_type" in best_cfg:
                args.optimizer = best_cfg["optimizer_type"]
                print(f"  optimizer: {args.optimizer}")
            
            # Force use_multilayer and num_layers_to_use from best config
            if not args.use_multilayer_specified and "use_multilayer" in best_cfg:
                args.use_multilayer = best_cfg["use_multilayer"]
                print(f"  use_multilayer: {args.use_multilayer}")
                
            if not args.num_layers_to_use_specified and "num_layers_to_use" in best_cfg:
                args.num_layers_to_use = best_cfg["num_layers_to_use"]
                print(f"  num_layers_to_use: {args.num_layers_to_use}")
            
            # Apply model-specific parameters
            if classifier_type == "fourier_kan" and not args.num_frequencies_specified and "num_frequencies" in best_cfg:
                args.num_frequencies = best_cfg["num_frequencies"]
                print(f"  num_frequencies: {args.num_frequencies}")
            
            if classifier_type == "wavelet_kan":
                if not args.num_wavelets_specified and "num_wavelets" in best_cfg:
                    args.num_wavelets = best_cfg["num_wavelets"]
                    print(f"  num_wavelets: {args.num_wavelets}")
                
                if not args.wavelet_type_specified and "wavelet_type" in best_cfg:
                    args.wavelet_type = best_cfg["wavelet_type"]
                    print(f"  wavelet_type: {args.wavelet_type}")
            
            # Print additional metadata if available
            if "training_metadata" in best_configs[classifier_type]:
                meta = best_configs[classifier_type]["training_metadata"]
                print("\nTraining metadata from best configuration:")
                if "optimizer" in meta:
                    print(f"  Optimizer: {meta['optimizer']}")
                if "architecture" in meta:
                    arch = meta["architecture"]
                    print(f"  Architecture: {arch.get('classifier_type', classifier_type)}")
                    print(f"  Multilayer: {arch.get('use_multilayer', False)}")
                    print(f"  Num layers: {arch.get('num_layers_to_use', 1)}")
    elif args.train_all:
        # Load configs for all classifiers
        best_configs = load_best_configs()
        print(f"Loaded best configs for {len(best_configs)} classifiers")
    
    # Apply best config parameters to respective classifiers for train_all
    # Otherwise, CLI arguments will override these values

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
    
    # Add multilayer parameters for all model types
    train_params["use_multilayer"] = args.use_multilayer
    train_params["num_layers_to_use"] = args.num_layers_to_use
    train_params["optimizer_type"] = args.optimizer
    
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
        # For train_all, we'll create specialized parameter sets for each classifier
        # based on their best configurations when available
        
        # Sort classifiers alphabetically
        sorted_classifiers = sorted(args.classifiers)
        
        all_results = {}
        for clf_type in sorted_classifiers:
            print(f"\n{'=' * 50}")
            print(f"Training {clf_type.upper()} classifier")
            print(f"{'=' * 50}\n")
            
            # Start with base parameters
            clf_params = train_params.copy()
            
            # Setup classifier-specific directories
            clf_params["output_dir"] = Path(args.output_dir) / clf_type
            clf_params["save_dir"] = Path(args.save_dir) / clf_type
            
            # Apply best config parameters if available
            if clf_type in best_configs:
                best_cfg = best_configs[clf_type]
                print(f"Using best configuration for {clf_type} from hyperparameter tuning")
                
                # Override with best hyperparameters if not explicitly set in CLI
                # This ensures CLI args take precedence
                if not args.batch_size_specified:
                    clf_params["batch_size"] = best_cfg.get("batch_size", clf_params["batch_size"])
                    print(f"  batch_size: {clf_params['batch_size']}")
                    
                if not args.learning_rate_specified:
                    clf_params["learning_rate"] = best_cfg.get("learning_rate", clf_params["learning_rate"])
                    print(f"  learning_rate: {clf_params['learning_rate']}")
                    
                if not args.weight_decay_specified:
                    clf_params["weight_decay"] = best_cfg.get("weight_decay", clf_params["weight_decay"])
                    print(f"  weight_decay: {clf_params['weight_decay']}")
                    
                if not args.dropout_specified:
                    clf_params["dropout_rate"] = best_cfg.get("dropout_rate", clf_params["dropout_rate"])
                    print(f"  dropout_rate: {clf_params['dropout_rate']}")
                
                # Model-specific parameters
                if clf_type == "fourier_kan" and not args.num_frequencies_specified:
                    clf_params["num_frequencies"] = best_cfg.get("num_frequencies", clf_params.get("num_frequencies", 8))
                    print(f"  num_frequencies: {clf_params['num_frequencies']}")
                
                if clf_type == "wavelet_kan":
                    if not args.num_wavelets_specified:
                        clf_params["num_wavelets"] = best_cfg.get("num_wavelets", clf_params.get("num_wavelets", 16))
                        print(f"  num_wavelets: {clf_params['num_wavelets']}")
                    
                    if not args.wavelet_type_specified:
                        clf_params["wavelet_type"] = best_cfg.get("wavelet_type", clf_params.get("wavelet_type", "mixed"))
                        print(f"  wavelet_type: {clf_params['wavelet_type']}")
            
            # Train this classifier with its specialized parameters
            try:
                test_results = train_classifier(classifier_type=clf_type, **clf_params)
                all_results[clf_type] = test_results
            except Exception as e:
                print(f"Error training {clf_type} classifier: {e}")
        
        # Create comparison after all classifiers are trained
        compare_classifiers(classifiers=sorted_classifiers)
        
        # Create results summary dataframe
        summary = pd.DataFrame.from_dict(all_results, orient="index")
        summary.index.name = "classifier"
        summary.reset_index(inplace=True)
        
        # Save summary
        output_dir_path = Path(args.output_dir)
        summary_path = output_dir_path / "classifier_results_summary.csv"
        output_dir_path.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"\nResults summary saved to {summary_path}")

    elif args.compare:
        print("\nComparing trained models...")
        compare_classifiers(classifiers=args.classifiers)
        
    elif args.tune:
        print(f"\n{'=' * 50}")
        print(f"Tuning {args.classifier.upper()} classifier hyperparameters")
        print(f"Running {args.n_trials} trials")
        print(f"{'=' * 50}\n")
        
        # Clean any existing study files for this classifier before tuning
        clean_study_files(args.classifier)
        
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
        
        # Clean all study files before tuning
        clean_study_files()
        
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
