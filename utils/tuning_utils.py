import os
import yaml
import optuna
import numpy as np
import logging
import time
from pathlib import Path
from models import TextClassificationTrainer

# Configure logging
logging.basicConfig(level=logging.WARNING)

# Set debug mode
DEBUG_MODE = os.environ.get('TUNING_DEBUG') == 'True'

def save_best_config(config, classifier_type, output_dir="./hyperparams"):
    """
    Save the best hyperparameters for a classifier to a YAML file
    
    Args:
        config: Dictionary of hyperparameters
        classifier_type: Type of classifier
        output_dir: Directory to save the YAML file
    """
    os.makedirs(output_dir, exist_ok=True)
    config_path = Path(output_dir) / f"{classifier_type}_best_config.yaml"
    
    # Add classifier type to config
    config["classifier_type"] = classifier_type
    
    # Save to YAML
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Best config saved to {config_path}")
    return config_path

def load_best_config(classifier_type, output_dir="./hyperparams"):
    """
    Load the best hyperparameters for a classifier from a YAML file
    
    Args:
        classifier_type: Type of classifier
        output_dir: Directory where the YAML file is stored
    
    Returns:
        Dictionary of hyperparameters or None if file not found
    """
    config_path = Path(output_dir) / f"{classifier_type}_best_config.yaml"
    
    if not config_path.exists():
        return None
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config

def create_all_configs_summary(output_dir="./hyperparams", save_dir="./hyperparams"):
    """
    Create a summary of all best configurations
    
    Args:
        output_dir: Directory where the YAML files are stored
        save_dir: Directory to save the summary YAML file
    """
    os.makedirs(save_dir, exist_ok=True)
    summary_path = Path(save_dir) / "all_best_configs.yaml"
    
    all_configs = {}
    
    # Load all configs
    for config_file in Path(output_dir).glob("*_best_config.yaml"):
        classifier_type = config_file.stem.replace("_best_config", "")
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        all_configs[classifier_type] = config
    
    # Save summary
    with open(summary_path, "w") as f:
        yaml.dump(all_configs, f, default_flow_style=False)
    
    print(f"All best configs saved to {summary_path}")
    return summary_path

def objective(trial, classifier_type, model_path, data_path, tokenized_datasets, 
              metric_for_best_model='matthews_correlation', num_epochs=3, output_dir=None):
    """
    Objective function for Optuna optimization
    
    Args:
        trial: Optuna trial object
        classifier_type: Type of classifier to optimize
        model_path: Path to the pretrained model
        data_path: Path to the dataset
        tokenized_datasets: Pre-tokenized datasets
        metric_for_best_model: Metric to optimize
        num_epochs: Number of epochs to train
        output_dir: Output directory for results
        
    Returns:
        Value of the metric to optimize
    """
    # Set up pruning handler with much more aggressive pruning
    class PruningCallback:
        def __init__(self, trial, metric, interval_steps=1):
            self.trial = trial
            self.metric = metric
            self.interval_steps = interval_steps
            self.step = 0
            self.best_value = float('-inf')
            self.no_improvement_steps = 0
            self.start_time = time.time()
            
        def on_evaluate(self, eval_metrics):
            self.step += 1
            current_metric = eval_metrics.get(f"eval_{self.metric}", 0)
            current_time = time.time()
            
            # Get class distribution for better pruning decisions
            pred_distribution = eval_metrics.get("eval_pred_distribution", "{}")
            num_classes_predicted = pred_distribution.count(':')
            
            # Check for improvement
            previous_best = self.best_value  # Store the previous best value
            if current_metric > self.best_value:
                # Use a proper default for the first step
                if previous_best == float('-inf'):
                    # Don't print anything for first evaluation
                    pass
                else:
                    print(f"New best: {current_metric:.4f}")
                self.best_value = current_metric
                self.no_improvement_steps = 0
            else:
                self.no_improvement_steps += 1
                
            # Report intermediate value for pruning
            self.trial.report(current_metric, self.step)
            
            # AGGRESSIVE PRUNING STRATEGY:
            should_prune = False
            pruning_reason = None
            
            # 1. Standard Optuna pruning
            if self.trial.should_prune():
                should_prune = True
                pruning_reason = "Standard Optuna pruning"
                
            # 2. Zero/Near-zero Matthews correlation after step 1
            elif self.metric == 'matthews_correlation' and current_metric <= 0.01 and self.step >= 1:
                should_prune = True
                pruning_reason = "Low metric value"
            
            # 3. Single class prediction after step 2
            elif self.step >= 2 and num_classes_predicted == 1:
                should_prune = True
                pruning_reason = "Single class prediction"
                
            # 4. No improvement for multiple steps and low metric value
            elif self.no_improvement_steps >= 2 and current_metric < 0.1:
                should_prune = True 
                pruning_reason = "No improvement with low metric"
                
            # 5. Time-based pruning for slow-converging trials
            elif current_time - self.start_time > 120 and current_metric < 0.2:
                should_prune = True
                pruning_reason = "Slow convergence"
            
            # Execute pruning if any condition met
            if should_prune:
                print(f"PRUNING Trial {self.trial.number}: {pruning_reason}")
                
                # Add reason to trial user attributes
                self.trial.set_user_attr("pruning_reason", pruning_reason)
                self.trial.set_user_attr("pruned_at_step", self.step)
                
                raise optuna.exceptions.TrialPruned()
            
            # Add step info to trial metadata
            self.trial.set_user_attr(f"step_{self.step}_{self.metric}", current_metric)
            
            return current_metric
    # During tuning, use a temporary directory that will be cleaned up
    import tempfile
    
    # Create a temporary directory for this trial that will be auto-deleted
    if os.environ.get('TUNING_MODE'):
        temp_dir = tempfile.mkdtemp(prefix=f"{classifier_type}_trial_{trial.number}_")
        output_dir = temp_dir
    else:
        # Only used for final model, not trials
        if output_dir is None:
            output_dir = f"./results/{classifier_type}_trial_{trial.number}"
        else:
            output_dir = Path(output_dir) / f"{classifier_type}_trial_{trial.number}"
    
    # Common hyperparameters with custom ranges for specific classifiers
    if classifier_type == "wavelet_kan":
        # Directly suggest wavelet type rather than using an index
        wavelet_type = trial.suggest_categorical("wavelet_type", ["haar", "mixed", "morlet"])
        
        # Use uniform set of wavelet counts across all types to avoid distribution errors
        # Standard choices that work for all wavelet types
        num_wavelets = trial.suggest_categorical("num_wavelets", [8, 16, 32, 64, 128, 256])
        
        # Different hyperparameter search for each wavelet type
        if wavelet_type == "haar":
            # For Haar wavelets, use larger counts where appropriate
            batch_size = trial.suggest_categorical("batch_size", [16, 32])
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
            # Lower weight decay for complex models
            weight_decay = trial.suggest_float("weight_decay", 0.001, 0.02, log=True)
            # Lower dropout for Haar wavelets with high frequencies
            dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.3)
        else:
            # For other wavelet types, use standard search space
            batch_size = trial.suggest_categorical("batch_size", [16, 32])
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True)
            weight_decay = trial.suggest_float("weight_decay", 0.005, 0.05, log=True)
            dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.4)
            
        # Print info about hyperparameter ranges only in debug mode
        if DEBUG_MODE:
            print(f"Hyperparameter ranges for {wavelet_type} wavelet:")
            print(f"- Learning rate: {learning_rate:.2e}")
            print(f"- Wavelets: {num_wavelets}")
            print(f"- Weight decay: {weight_decay:.4f}")
            print(f"- Dropout: {dropout_rate:.2f}")
    else:
        # Default ranges for other classifiers
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        learning_rate = trial.suggest_float("learning_rate", 5e-6, 5e-5, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.001, 0.1, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    
    # Model-specific hyperparameters
    model_params = {"dropout_rate": dropout_rate}
    
    # Handle model-specific parameters
    if classifier_type in ["fourier_kan", "wavelet_kan"]:
        if classifier_type == "wavelet_kan":
            # num_wavelets already selected above, no need to suggest again
            # Just add to model params
            model_params["num_wavelets"] = num_wavelets
            
            # Get wavelet type from earlier selection
            model_params["wavelet_type"] = wavelet_type
            
            # Learning rate warmup is important for transformer stability
            model_params["warmup_ratio"] = trial.suggest_float("warmup_ratio", 0.05, 0.2)
            
            # Special optimizations for trial 0
            if trial.number == 0 and DEBUG_MODE:
                print("For WaveletKAN, optimizing with specialized initialization...")
                # Increase weight decay for better generalization
                weight_decay = max(weight_decay, 0.01)
                # Use more conservative learning rate for stability
                learning_rate = min(learning_rate, 3e-5)
        else:
            # Wider range for FourierKAN which can handle more frequencies
            num_frequencies = trial.suggest_int("num_frequencies", 4, 64)
            model_params["num_frequencies"] = num_frequencies
    
    # Create trainer with num_labels from tokenized datasets
    # Extract num_labels from label_to_id mapping
    num_labels = len(tokenized_datasets["train"].unique("label"))
    
    # Create pruning callback
    pruning_cb = PruningCallback(trial, metric_for_best_model)
    
    trainer = TextClassificationTrainer(
        model_path=model_path,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_labels=num_labels,
    )
    
    # Setup early stopping with enhanced parameters for WaveletKAN
    callback_params = {
        "patience": 2 if classifier_type == "wavelet_kan" else 2,  # Less patience for wavelet_kan for faster training
        "model_type": classifier_type,
        "warmup_ratio": 0.1 if classifier_type == "wavelet_kan" else 0.1,  # Lower warmup for faster progress
        "pruning_callback": pruning_cb  # Add our pruning callback
    }
    
    # Setup trainer with improved training parameters
    # Use a warmup ratio from model_params if available
    warmup_ratio = model_params.get("warmup_ratio", callback_params.get("warmup_ratio", 0.1))
    
    # Adjust training arguments based on classifier type
    training_args = {
        "warmup_ratio": warmup_ratio
    }
    
    # For WaveletKAN, use additional training settings for stability
    if classifier_type == "wavelet_kan":
        training_args["gradient_accumulation_steps"] = 2
        training_args["fp16"] = True  # Mixed precision training
        training_args["eval_accumulation_steps"] = 4
    
    # Setup trainer with custom training args
    trainer.setup_trainer(
        tokenized_datasets,
        model_type=classifier_type,
        metric_for_best_model=metric_for_best_model,
        model_params=model_params,
        callback_params=callback_params,
        training_args=training_args
    )
    
    # Train (suppress stdout to reduce noise)
    import sys
    from io import StringIO
    
    # Conditionally capture output based on debug mode
    original_stdout = sys.stdout
    debug_mode = os.environ.get('TUNING_DEBUG') == 'True'
    
    if not debug_mode:
        # Capture output only if not in debug mode
        sys.stdout = StringIO()
    else:
        print(f"\n=== TRAINING TRIAL {trial.number} WITH PARAMS: {trial.params} ===\n")
    
    try:
        # Train (with or without captured output)
        trainer.train(tokenized_datasets)
        
        # Evaluate (with or without captured output)
        eval_results = trainer.evaluate(tokenized_datasets["validation"])
        
        # Call pruning callback with final results but DON'T raise pruning exception
        # This ensures that good trial results always get captured even with pruning
        if pruning_cb:
            try:
                metric_value = pruning_cb.on_evaluate(eval_results)
                # Store the best trial parameters and value
                trial.set_user_attr("final_metric_value", metric_value)
            except optuna.exceptions.TrialPruned:
                # Handle pruning gracefully but don't actually raise the exception
                # This is to avoid skipping the recording of good results
                sys.stdout = original_stdout
                print(f"Trial {trial.number} would be pruned at final evaluation, but keeping results")
                # Store the metric value anyway
                metric_value = eval_results.get(f"eval_{metric_for_best_model}", 0)
                trial.set_user_attr("final_metric_value", metric_value)
                # DON'T raise the exception - this allows us to save good pruned results
    finally:
        # Restore stdout
        sys.stdout = original_stdout
        
        # Clean up any remaining temporary files in the output directory
        if os.environ.get('TUNING_MODE') and os.path.exists(output_dir):
            import shutil
            try:
                # Try to remove the entire directory tree
                shutil.rmtree(output_dir, ignore_errors=True)
            except Exception:
                pass
    
    # Get metric value (higher is better, but Optuna minimizes by default)
    metric_value = eval_results.get(f"eval_{metric_for_best_model}", 0)
    
    # Print more detailed information about this trial
    pred_distribution = eval_results.get("eval_pred_distribution", "Unknown")
    accuracy = eval_results.get("eval_accuracy", 0)
    
    print(f"\nTrial {trial.number} completed:")
    print(f"  {metric_for_best_model}: {metric_value:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Class distribution: {pred_distribution}")
    
    # Record additional metrics in trial user attributes for easier analysis
    trial.set_user_attr("accuracy", accuracy)
    trial.set_user_attr("pred_distribution", str(pred_distribution))
    trial.set_user_attr("params_str", str(trial.params))
    
    # For metrics where higher is better, return negative
    return -metric_value  # Negate since Optuna minimizes by default

def tune_classifier(
    classifier_type,
    model_path,
    data_path,
    n_trials=20,
    num_epochs=3,
    metric_for_best_model="matthews_correlation",
    output_dir=None,
    save_dir=None,
    max_length=64,
    train_size=0.6,
    val_size=0.2,
    test_size=0.2,
):
    """
    Tune hyperparameters for a single classifier using Optuna
    
    Args:
        classifier_type: Type of classifier to tune
        model_path: Path to the pretrained model
        data_path: Path to the dataset
        n_trials: Number of Optuna trials
        num_epochs: Number of epochs for each trial
        metric_for_best_model: Metric to optimize
        output_dir: Output directory for results
        save_dir: Directory to save best hyperparameters
        max_length: Maximum sequence length
        train_size: Proportion of data for training
        val_size: Proportion of data for validation
        test_size: Proportion of data for testing
        
    Returns:
        Dictionary with best parameters and best score
    """
    # Set default directories
    if output_dir is None:
        output_dir = f"./results/{classifier_type}_tuning"
    if save_dir is None:
        save_dir = "./hyperparams"
    
    # Set environment variables for tuning mode
    os.environ['TUNING_MODE'] = 'True'
    os.environ['TUNING_DEBUG'] = 'True'  # Enable debug output during tuning
    
    # Create a temporary trainer to prepare data
    # Determine number of labels from data if possible
    temp_trainer = TextClassificationTrainer(
        model_path=model_path,
        max_length=max_length,
    )
    
    # Prepare data once for all trials
    tokenized_datasets = temp_trainer.prepare_data(
        data_path,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )
    
    # Define the study name
    study_name = f"{classifier_type}_optimization"
    
    # Create an Optuna study with pruning and storage for persistence
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=2,  # Fewer startup trials, since we're being aggressive with pruning
        n_warmup_steps=0,    # Number of steps in a trial before pruning
        interval_steps=1     # Interval between pruning checks
    )
    
    # Create storage directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Use SQLite storage to persist the study between runs
    storage_path = f"sqlite:///{save_dir}/{classifier_type}_study.db"
    
    # Load or create study
    try:
        # Try to load existing study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            direction="maximize",  # We're maximizing the metric
            pruner=pruner,         # Enable pruning
            load_if_exists=True    # Load existing study if it exists
        )
        print(f"Loaded existing study with {len(study.trials)} previous trials")
        
        # Check if we have any completed trials
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed_trials:
            print(f"Study has {len(completed_trials)} completed trials. Best value so far: {-study.best_value:.4f}")
        
    except Exception as e:
        print(f"Error loading study: {e}. Creating new study.")
        # Create new study if loading fails
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",  # We're maximizing the metric
            pruner=pruner          # Enable pruning
        )
    
    # Configure optuna logging to be less verbose
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Set environment variable for tuning mode to reduce output
    os.environ['TUNING_MODE'] = 'True'
    
    # Look for existing best configuration to use as a starting point
    config_path = Path(save_dir) / f"{classifier_type}_best_config.yaml"
    previous_best_value = None
    
    if config_path.exists():
        print(f"Found existing configuration at {config_path}. Using as starting point for tuning.")
        with open(config_path, "r") as f:
            previous_config = yaml.safe_load(f)
        
        if "best_params" in previous_config:
            # Use the previous best parameters as a starting point
            best_params = previous_config["best_params"]
            previous_best_value = previous_config.get("best_value")
            print(f"Starting with previous best parameters: {best_params}")
            print(f"Previous best {metric_for_best_model}: {previous_best_value:.4f}")
            
            # Make sure hyperparameters are within valid ranges before enqueueing
            valid_params = best_params.copy()
            
            # Fix wavelet_kan specific parameter ranges
            if classifier_type == "wavelet_kan":
                # Adjust weight_decay to be within valid range
                if "weight_decay" in valid_params and (valid_params["weight_decay"] < 0.01 or valid_params["weight_decay"] > 0.2):
                    print(f"Adjusting weight_decay from {valid_params['weight_decay']} to be within range [0.01, 0.2]")
                    valid_params["weight_decay"] = max(min(valid_params["weight_decay"], 0.2), 0.01)
                
                # Ensure batch_size is one of the valid choices
                if "batch_size" in valid_params and valid_params["batch_size"] not in [16, 32]:
                    print(f"Adjusting batch_size from {valid_params['batch_size']} to nearest valid value")
                    valid_params["batch_size"] = 16 if valid_params["batch_size"] < 24 else 32
                
                # Add warmup_ratio if not present
                if "warmup_ratio" not in valid_params:
                    if DEBUG_MODE:
                        print("Adding warmup_ratio parameter")
                    valid_params["warmup_ratio"] = 0.1
            
            if DEBUG_MODE:
                print("Adding previous best parameters as a trial")
            study.enqueue_trial(valid_params)
            
            # For WaveletKAN, specifically add trials with larger wavelets
            if classifier_type == "wavelet_kan":
                # If current best params don't have very high wavelets and is haar type, add some
                if (valid_params.get("wavelet_type") == "haar" and 
                    valid_params.get("num_wavelets", 0) < 128):
                    
                    # Add trials with larger wavelets from our canonical wavelets list
                    for wavelets in [64, 128, 256]:
                        # Skip if already at this wavelets count
                        if valid_params.get("num_wavelets") == wavelets:
                            continue
                        
                        # Ensure we only use valid wavelets from our suggest_categorical call
                        if wavelets not in [8, 16, 32, 64, 128, 256]:
                            continue
                            
                        high_freq_params = valid_params.copy()
                        high_freq_params["num_wavelets"] = wavelets
                        # Use appropriate learning rate for high frequencies
                        high_freq_params["learning_rate"] = 5e-4
                        # Lower dropout for high frequencies
                        high_freq_params["dropout_rate"] = 0.15
                        
                        # Only print in debug mode
                        if DEBUG_MODE:
                            print(f"  Adding Haar trial with {wavelets} wavelets")
                        study.enqueue_trial(high_freq_params)
                
                # Always add a mixed wavelet type trial for comparison
                if valid_params.get("wavelet_type") == "haar":
                    mixed_params = valid_params.copy()
                    mixed_params["wavelet_type"] = "mixed"
                    # Use a value from our canonical categorical choice list
                    mixed_params["num_wavelets"] = 32  # Good default for mixed
                    mixed_params["learning_rate"] = 8e-4  # Higher LR for mixed
                    
                    # Only print in debug mode
                    if DEBUG_MODE:
                        print(f"  Adding mixed wavelet trial with {mixed_params['num_wavelets']} wavelets")
                    study.enqueue_trial(mixed_params)
            
            # For other classifiers, use smaller variations
            elif valid_params.get("learning_rate") is not None:
                # Enqueue a trial with slightly higher learning rate
                higher_lr_params = valid_params.copy()
                higher_lr_params["learning_rate"] = min(valid_params["learning_rate"] * 2.0, 5e-4)
                study.enqueue_trial(higher_lr_params)
                
                # Enqueue a trial with slightly lower learning rate
                lower_lr_params = valid_params.copy()
                lower_lr_params["learning_rate"] = max(valid_params["learning_rate"] * 0.5, 1e-5)
                study.enqueue_trial(lower_lr_params)
    
    # Run the optimization
    try:
        study.optimize(
            lambda trial: objective(
                trial, 
                classifier_type, 
                model_path, 
                data_path, 
                tokenized_datasets,
                metric_for_best_model, 
                num_epochs, 
                output_dir
            ),
            n_trials=n_trials,
        )
    finally:
        # Clean up environment variable
        if 'TUNING_MODE' in os.environ:
            del os.environ['TUNING_MODE']
    
    # Get best parameters
    best_params = study.best_params
    best_value = -study.best_value  # Negate to get the actual value (since we negated in objective)
    
    # Print best parameters
    print(f"\nBest {metric_for_best_model} value: {best_value:.4f}")
    print("Best hyperparameters:")
    for param, value in best_params.items():
        print(f"    {param}: {value}")
        
    # Display a more detailed study summary
    print("\n" + "="*50)
    print(f"STUDY SUMMARY FOR {classifier_type.upper()}")
    print("="*50)
    
    # Show improvement from previous run if available
    if previous_best_value is not None:
        improvement = best_value - previous_best_value
        improvement_percent = (improvement / previous_best_value) * 100 if previous_best_value > 0 else 0
        
        if improvement > 0:
            print(f"✨ Improved {metric_for_best_model} by {improvement:.4f} ({improvement_percent:.2f}%)")
        else:
            print(f"No improvement from previous best {metric_for_best_model} value.")
    
    # Display top 3 trials
    print("\nTop 3 trials:")
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))[:3]
    
    for i, trial in enumerate(top_trials):
        if trial.value is None:
            continue
            
        print(f"\n{i+1}. Trial {trial.number}: {metric_for_best_model} = {-trial.value:.4f}")
        print(f"   Parameters: {trial.params}")
        
        # Show class distribution if available
        if hasattr(trial, "user_attrs") and "pred_distribution" in trial.user_attrs:
            print(f"   Class distribution: {trial.user_attrs['pred_distribution']}")
        
    # Print the total number of trials and how many were pruned
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    
    print(f"\nTotal trials: {len(study.trials)}")
    print(f"Completed: {len(completed_trials)}")
    print(f"Pruned: {len(pruned_trials)}")
    
    # Get top trials and check if they have better metrics than our best trial
    # This helps avoid getting stuck with a model that predicts only one class
    
    # Consider both completed and pruned trials for finding the best configuration
    valid_trials = [t for t in study.trials if 
                    (t.state == optuna.trial.TrialState.COMPLETE or t.state == optuna.trial.TrialState.PRUNED) 
                    and t.value is not None]
    
    # If we have pruned trials, print a message about considering them
    pruned_with_values = [t for t in pruned_trials if t.value is not None]
    if pruned_with_values:
        print(f"Including {len(pruned_with_values)} pruned trials with valid metrics in best configuration selection.")
    
    if len(valid_trials) > 0:
        # Sort trials by main metric value
        sorted_trials = sorted(valid_trials, key=lambda t: t.value)
        
        best_trial = sorted_trials[0]
        best_params = best_trial.params
        best_value = -best_trial.value  # Negate back to original value
        
        # Check if we found a better trial than what was reported above
        print(f"\nSelected best trial {best_trial.number} with {metric_for_best_model} = {best_value:.4f}")
        
        # Check class distribution if available
        if hasattr(best_trial, "user_attrs") and "pred_distribution" in best_trial.user_attrs:
            print(f"Class distribution: {best_trial.user_attrs['pred_distribution']}")
            
            # Verify if best trial predicts only one class
            dist_str = best_trial.user_attrs['pred_distribution']
            
            # If it looks like all predictions are one class, see if we have a better diverse trial
            if dist_str.count(':') == 1:  # Only one class in distribution
                print("Warning: Best trial may be predicting only one class. Looking for more diverse predictions...")
                
                # Try to find a trial with multiple classes that has decent performance
                for trial in sorted_trials[1:5]:  # Check next 4 best trials
                    if hasattr(trial, "user_attrs") and "pred_distribution" in trial.user_attrs:
                        trial_dist = trial.user_attrs['pred_distribution']
                        if trial_dist.count(':') > 1:  # More than one class
                            print(f"Found better trial {trial.number} with multiple classes predicted:")
                            print(f"  {metric_for_best_model}: {-trial.value:.4f}")
                            print(f"  Class distribution: {trial_dist}")
                            
                            # Use this trial instead
                            best_trial = trial
                            best_params = trial.params
                            best_value = -trial.value
                            break
    
    # Save only the best parameters to YAML (minimal storage)
    best_config = {
        "best_params": best_params,
        "best_value": best_value,
        "metric": metric_for_best_model,
    }
    save_best_config(best_config, classifier_type, save_dir)
    
    # Train a final model with best parameters
    final_output_dir = Path(output_dir) / "best_model"
    final_save_dir = Path(save_dir) / classifier_type
    
    # Extract best params
    batch_size = best_params.get("batch_size", 16)
    learning_rate = best_params.get("learning_rate", 2e-5)
    weight_decay = best_params.get("weight_decay", 0.01)
    dropout_rate = best_params.get("dropout_rate", 0.3)
    
    # Model-specific params
    model_params = {"dropout_rate": dropout_rate}
    
    if classifier_type in ["fourier_kan", "wavelet_kan"]:
        if classifier_type == "wavelet_kan":
            # WaveletKANClassifier expects num_wavelets parameter
            num_wavelets = best_params.get("num_wavelets", 8)
            model_params["num_wavelets"] = num_wavelets
            
            # Get wavelet type
            wavelet_type = best_params.get("wavelet_type", "mixed")
            
            model_params["wavelet_type"] = wavelet_type
            if DEBUG_MODE:
                print(f"Using wavelet type: {wavelet_type} for final model with {num_wavelets} wavelets")
        else:
            # FourierKAN uses num_frequencies parameter
            num_frequencies = best_params.get("num_frequencies", 8)
            model_params["num_frequencies"] = num_frequencies
    
    # Create trainer with best parameters
    # Get num_labels from the tokenized datasets
    num_labels = len(tokenized_datasets["train"].unique("label"))
    
    # For WaveletKAN, ensure we use a sufficiently high learning rate to avoid getting stuck
    if classifier_type == "wavelet_kan":
        # Force a minimum learning rate for WaveletKAN to prevent stuck predictions
        orig_learning_rate = learning_rate
        learning_rate = max(learning_rate, 1e-4)  # Enforce minimum learning rate
        if learning_rate != orig_learning_rate:
            print(f"Increasing learning rate from {orig_learning_rate:.2e} to {learning_rate:.2e} to prevent training issues")
    
    final_trainer = TextClassificationTrainer(
        model_path=model_path,
        output_dir=final_output_dir,
        num_epochs=num_epochs * 2,  # Double epochs for final model
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_length=max_length,
        num_labels=num_labels,
    )
    
    # Setup trainer with best parameters
    callback_params = {"patience": 3, "model_type": classifier_type}  # More patience for final model
    
    final_trainer.setup_trainer(
        tokenized_datasets,
        model_type=classifier_type,
        metric_for_best_model=metric_for_best_model,
        model_params=model_params,
        callback_params=callback_params
    )
    
    # Train final model
    print(f"\nTraining final {classifier_type} model with best hyperparameters...")
    
    # Turn off tuning mode for final model
    if 'TUNING_MODE' in os.environ:
        del os.environ['TUNING_MODE']
    final_trainer.train(tokenized_datasets)
    
    # Generate and save learning curves for final model
    if final_trainer.learning_curve_callback:
        metrics_path = final_trainer.learning_curve_callback.save_metrics()
        print(f"Metrics saved to {metrics_path}")
        curves_path = final_trainer.learning_curve_callback.plot_learning_curves()
        print(f"Learning curves saved to {curves_path}")
    
    # Evaluate final model
    test_results = final_trainer.evaluate(tokenized_datasets["test"])
    print(f"Test results for best {classifier_type}: {test_results}")
    
    # Save final model
    final_trainer.save_model(final_save_dir)
    print(f"Best model saved to {final_save_dir}")
    
    # Update best config with test results only (no history or timestamps)
    best_config["test_results"] = test_results
    save_best_config(best_config, classifier_type, save_dir)
    
    return best_config

def tune_all_classifiers(
    model_path,
    data_path,
    classifier_types=None,
    n_trials=15,
    num_epochs=3,
    metric_for_best_model="matthews_correlation",
    output_dir="./results/tuning",
    save_dir="./hyperparams",
    max_length=64,
    train_size=0.6,
    val_size=0.2,
    test_size=0.2,
):
    """
    Tune hyperparameters for multiple classifiers
    
    Args:
        model_path: Path to the pretrained model
        data_path: Path to the dataset
        classifier_types: List of classifier types to tune
        n_trials: Number of Optuna trials
        num_epochs: Number of epochs for each trial
        metric_for_best_model: Metric to optimize
        output_dir: Output directory for results
        save_dir: Directory to save best hyperparameters
        max_length: Maximum sequence length
        train_size: Proportion of data for training
        val_size: Proportion of data for validation
        test_size: Proportion of data for testing
        
    Returns:
        Dictionary with best configurations for each classifier
    """
    if classifier_types is None:
        classifier_types = ["standard", "custom", "bilstm", "attention", "cnn", "fourier_kan", "wavelet_kan"]
    
    best_configs = {}
    
    for classifier_type in classifier_types:
        print(f"\n{'=' * 50}")
        print(f"Tuning {classifier_type.upper()} classifier")
        print(f"{'=' * 50}\n")
        
        # Setup classifier-specific directories
        classifier_output_dir = Path(output_dir) / classifier_type
        
        # Tune classifier
        try:
            best_config = tune_classifier(
                classifier_type=classifier_type,
                model_path=model_path,
                data_path=data_path,
                n_trials=n_trials,
                num_epochs=num_epochs,
                metric_for_best_model=metric_for_best_model,
                output_dir=classifier_output_dir,
                save_dir=save_dir,
                max_length=max_length,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
            )
            best_configs[classifier_type] = best_config
        except Exception as e:
            print(f"Error tuning {classifier_type} classifier: {e}")
    
    # Create a summary of all best configurations
    create_all_configs_summary(save_dir)
    
    # Find the best overall classifier
    best_classifier = None
    best_score = -float("inf")
    
    for classifier_type, config in best_configs.items():
        score = config.get("best_value", -float("inf"))
        
        if score > best_score:
            best_score = score
            best_classifier = classifier_type
    
    if best_classifier:
        print(f"\n{'=' * 50}")
        print(f"Best overall classifier: {best_classifier.upper()}")
        print(f"Best {metric_for_best_model}: {best_score:.4f}")
        print(f"{'=' * 50}\n")
        
        # Save best overall classifier
        best_overall = {
            "best_classifier": best_classifier,
            "best_score": best_score,
            "metric": metric_for_best_model,
        }
        
        best_overall_path = Path(save_dir) / "best_overall_classifier.yaml"
        with open(best_overall_path, "w") as f:
            yaml.dump(best_overall, f, default_flow_style=False)
        
        print(f"Best overall classifier saved to {best_overall_path}")
    
    return best_configs