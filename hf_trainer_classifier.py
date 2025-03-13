import os
from pathlib import Path
from utils import train_classifier, train_all_classifiers, compare_classifiers


def main():
    # Configuration
    MODEL_PATH = str(Path(os.environ["LLM_MODELS_PATH"]) / "all-MiniLM-L6-v2")
    DATA_PATH = Path(os.environ["DATADIR"]) / "bbc-text" / "bbc-text.csv"
    NUM_EPOCHS = 10
    CLASSIFIER = "custom"  # choices: standard, custom, bilstm, attention, cnn

    # Train a single classifier
    # train_classifier(
    #     classifier_type=CLASSIFIER,
    #     model_path=MODEL_PATH,
    #     data_path=DATA_PATH,
    #     num_epochs=NUM_EPOCHS
    # )

    # Uncomment to train all classifiers
    train_all_classifiers(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        num_epochs=NUM_EPOCHS,
        classifier_types=["standard", "custom", "bilstm", "attention", "cnn"],
    )


if __name__ == "__main__":
    main()

    # Uncomment to generate comparison after training all models
    compare_classifiers()