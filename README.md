# HF Trainer Classifier

A modular framework for fine-tuning and benchmarking various text classification architectures using Hugging Face's transformer models and training utilities.

## Overview

This project provides a flexible and extensible solution for text classification tasks using Hugging Face's ecosystem. It allows you to:

1. Fine-tune various classifier architectures on top of pre-trained transformer models
2. Compare performance across different architectures
3. Visualize learning curves and metrics during training
4. Apply custom optimizers with different learning rates for transformer and classifier components

## Key Features

- **Multiple Classifier Architectures**: Standard linear head, custom MLP, BiLSTM, self-attention, and CNN-based classifiers
- **Differential Learning Rates**: Apply lower learning rates to pre-trained components and higher rates to new classifier layers
- **Integrated Evaluation**: Built-in metrics tracking and visualization
- **Modular Design**: Easily extend with your own architectures or components

## Classifier Architectures

### Standard Classifier
Uses Hugging Face's `AutoModelForSequenceClassification` with a simple linear classification head.

```python
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
```

### Custom MLP Classifier
A multi-layer perceptron with multiple hidden layers, taking the [CLS] token representation as input.

```python
# Architecture:
# Input -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear -> Output
```

### BiLSTM Classifier
Processes transformer outputs through a bidirectional LSTM to capture sequential dependencies.

```python
# Architecture:
# Transformer outputs -> BiLSTM -> Linear -> ReLU -> Dropout -> Linear -> Output
```

### Attention Classifier
Applies a learned attention mechanism to weight the importance of different token positions in the sequence.

```python
# Architecture:
# Transformer outputs -> Self-Attention -> Weighted Sum -> Linear -> ReLU -> Dropout -> Linear -> Output
```

### CNN Classifier
Uses multiple parallel convolutional layers with different kernel sizes to capture n-gram patterns of different lengths.

```python
# Architecture:
# Transformer outputs -> Parallel CNNs -> MaxPooling -> Concatenate -> Linear -> ReLU -> Dropout -> Linear -> Output
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hf-trainer-classifier.git
cd hf-trainer-classifier

# Install dependencies
pip install torch transformers datasets pandas seaborn matplotlib scikit-learn
```

## Usage Examples

### Training a Single Classifier

```python
import os
from pathlib import Path
from utils import train_classifier

# Configuration
MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"  # or path to local model
DATA_PATH = "path/to/your/data.csv"
CLASSIFIER_TYPE = "custom"  # options: standard, custom, bilstm, attention, cnn

# Train the classifier
results = train_classifier(
    classifier_type=CLASSIFIER_TYPE,
    model_path=MODEL_PATH,
    data_path=DATA_PATH,
    num_epochs=5
)

print(f"Test results: {results}")
```

### Training and Comparing Multiple Classifiers

```python
from utils import train_all_classifiers

# Train multiple classifiers and create comparison visualizations
summary = train_all_classifiers(
    model_path="sentence-transformers/all-MiniLM-L6-v2",
    data_path="path/to/your/data.csv",
    num_epochs=5,
    classifier_types=["standard", "custom", "bilstm", "attention", "cnn"]
)

print(summary)
```

### Using a Pre-trained Classifier for Inference

```python
import torch
from transformers import AutoTokenizer
from models import TextClassificationTrainer
from classifiers import CustomClassifier

# Load the model and tokenizer
model_path = "path/to/saved/model"
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Load label mapping
import json
with open(f"{model_path}/label_mapping.json", "r") as f:
    mapping = json.load(f)
    id_to_label = mapping["id_to_label"]

# Load the model
model = CustomClassifier.from_pretrained(model_path)
model.eval()

# Prepare input text
text = "This is a sample text for classification"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs["logits"]
    predicted_class_id = logits.argmax().item()
    predicted_label = id_to_label[str(predicted_class_id)]

print(f"Predicted class: {predicted_label}")
```

### Custom Dataset Preparation

```python
from models import TextClassificationTrainer

# Initialize trainer
trainer = TextClassificationTrainer(
    model_path="sentence-transformers/all-MiniLM-L6-v2",
    max_length=128  # Increase max sequence length for longer texts
)

# Prepare dataset with custom split and column names
tokenized_datasets = trainer.prepare_data(
    dataset_path="path/to/your/data.csv",
    text_column="content",  # Custom text column name
    label_column="category",  # Custom label column name
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    random_state=42
)

# Continue with model setup and training
trainer.setup_trainer(tokenized_datasets, model_type="bilstm")
trainer.train(tokenized_datasets)
```

### Fine-tuning with Custom Training Parameters

```python
from models import TextClassificationTrainer

# Initialize trainer with custom hyperparameters
trainer = TextClassificationTrainer(
    model_path="sentence-transformers/all-MiniLM-L6-v2",
    num_labels=5,
    max_length=64,
    output_dir="./custom_results",
    learning_rate=3e-5,
    batch_size=16,
    num_epochs=8,
    weight_decay=0.05
)

# Prepare data and train
tokenized_datasets = trainer.prepare_data("path/to/your/data.csv")
trainer.setup_trainer(tokenized_datasets, model_type="attention")
trainer.train(tokenized_datasets)

# Evaluate and save
test_results = trainer.evaluate(tokenized_datasets["test"])
trainer.save_model("./saved_models/attention_model")
```

## Project Structure

```
HF_Trainer_Classifier/
├── callbacks/                  # Training callbacks
│   ├── __init__.py
│   └── learning_curve_callback.py
├── classifiers/                # Model architectures
│   ├── __init__.py
│   ├── attention_classifier.py
│   ├── bilstm_classifier.py
│   ├── cnn_classifier.py
│   └── custom_classifier.py
├── models/                     # Core training components
│   ├── __init__.py
│   └── text_classification_trainer.py
├── utils/                      # Utility functions
│   ├── __init__.py
│   └── training_utils.py
├── hf_trainer_classifier.py    # Main script
└── README.md
```

## Hugging Face Integration

This project leverages several key components from the Hugging Face ecosystem:

1. **Transformers Library**: Uses `AutoModel`, `AutoTokenizer`, and other components for loading pre-trained models

2. **Trainer API**: Utilizes Hugging Face's `Trainer` class for streamlined training with features like:
   - Automatic gradient accumulation
   - Mixed precision training
   - Learning rate scheduling
   - Checkpointing
   - Evaluation during training

3. **Datasets Library**: Employs the `Dataset` and `DatasetDict` classes for efficient data handling with features like:
   - Memory-mapped storage
   - Efficient data transformations
   - Batched processing
   - Easy integration with the Trainer API

4. **Model Hub Integration**: Support for loading models directly from Hugging Face's model hub

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for their excellent transformers and datasets libraries
- PyTorch for the underlying deep learning framework