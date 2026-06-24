"""
=============================================================================================================
Python script to demonstrates training a deep neural network to predict whether a student will pass or fail.
=============================================================================================================
This program demonstrates student performance prediction using a deep neural network (DNN). It illustrates
the complete deep learning workflow applied to a real-world educational classification problem: predicting
whether a student will pass or fail a course based on academic and engagement indicators.

Deep Learning Workflow:
    1. Load student dataset
    2. Validate dataset
    3. Explore dataset structure
    4. Select features
    5. Scale data
    6. Build neural network
    7. Apply ReLU activation
    8. Apply Sigmoid activation
    9. Train model using Backpropagation
    10. Optimise using Gradient Descent
    11. Apply Regularization
    12. Evaluate model
    13. Generate predictions
    14. Visualise results
    15. Save output figures

Dataset:
    files/student_scores.csv

Outputs:
    files/results/student_performance_class_distribution.png
    files/results/student_performance_correlation_heatmap.png
    files/results/student_performance_training_history.png
    files/results/student_performance_confusion_matrix.png
    files/results/student_performance_prediction_examples.png


Requirements:
    !pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras

Author: Nyanjui
Date: 23 June 2026
"""
from tabnanny import verbose

# --------------------------------------------------------------------------------
# 0. Import required modules
# --------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

import warnings

# Suppress warnings for cleaner output demo
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------
# 1. Configuration Constants
# --------------------------------------------------------------------------------
DATA_FILE: Path = Path("../files/student_scores.csv") #Path(__file__).parent.joinpath('files/student_scores.csv')
RESULTS_DIR: Path = Path("../files/results/")
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
EPOCHS: int = 50
BATCH_SIZE: int = 32
LEARNING_RATE: float = 0.001
REGULARIZATION_STRENGTH: float = 0.001

# Columns in the student_scores.csv file
REQUIRED_COLUMNS: list[str] = [
    "student_id",
    "attendance_pct",
    "cat1_score",
    "cat2_score",
    "assignment_avg",
    "practical_avg",
    "lms_activity_pct",
    "study_hours_week",
    "final_exam_score",
    "overall_mark",
    "pass_fail",
]

FEATURE_COLUMNS: list[str] = [
    "attendance_pct",
    "cat1_score",
    "cat2_score",
    "assignment_avg",
    "practical_avg",
    "lms_activity_pct",
    "study_hours_week",
]

TARGET_COLUMN: str = "pass_fail"

# --------------------------------------------------------------------------------
# 2. Utility Functions
# --------------------------------------------------------------------------------
def create_results_directory() -> None:

    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OSError(f"Unable to create results directory: {error}") from error

# --------------------------------------------------------------------------------
# 3. Dataset Function
# --------------------------------------------------------------------------------
def load_dataset(file_path: Path) -> pd.DataFrame:

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    try:
        dataset = pd.read_csv(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}") from e

    return dataset
# --------------------------------------------------------------------------------
# 4. Validation Functions
# --------------------------------------------------------------------------------
def validate_dataset(dataset: pd.DataFrame) -> None:

    if dataset.empty:
        raise ValueError("Dataset is empty")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if dataset["student_id"].duplicated().any():
        raise ValueError(f"Dataset contains duplicated student IDs.")

    if dataset[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError(f"Dataset contains missing values.")

    if not dataset["attendance_pct"].between(0,100).all():
        raise ValueError(f"Attendance percentage must be between 0 and 100.")

    for cat_column in ["cat1_score", "cat2_score"]:
        if not dataset[cat_column].between(0,30).all():
            raise ValueError(f"{cat_column} values must be between 0 and 30.")

    if not dataset[TARGET_COLUMN].isin([0,1]).all():
        raise ValueError(f"'pass_fail' column must contain only binary values (0 or 1).")

# --------------------------------------------------------------------------------
# 5. Analysis Functions
# --------------------------------------------------------------------------------
def display_dataset_information(dataset: pd.DataFrame) -> None:

    print("\n---- STUDENT DATASET INFORMATION ----")
    print(f"Shape: {dataset.shape[0]} rows, {dataset.shape[1]} columns.")
    print(f"Column Data Types:\n{dataset.dtypes}")
    print(f"First 5 records:\n{dataset.head(5)}")

def display_dataset_statistics(dataset: pd.DataFrame) -> None:

    print("\n---- STUDENT DATASET STATISTICS ----")
    print(dataset.describe())

# --------------------------------------------------------------------------------
# 6. Visualisation Functions
# --------------------------------------------------------------------------------
def plot_class_distribution(dataset: pd.DataFrame) -> None:

    try:
        plt.figure(figsize=(8,7))
        sns.countplot(x=TARGET_COLUMN, data=dataset, hue=TARGET_COLUMN,palette="viridis", legend=False)
        plt.title("Class Distribution: Pass (1) vs. Fail (0)")
        plt.xlabel("Outcome")
        plt.ylabel("Number of Students")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "student_performance_class_distribution.png")
        plt.close()
    except Exception as e:
        raise RuntimeError(f"Failed to save class distribution plot: {e}") from e

def plot_correlation_heatmap(dataset: pd.DataFrame) -> None:

    try:
        numeric_data = dataset[FEATURE_COLUMNS + [TARGET_COLUMN]]
        plt.figure(figsize=(9,7))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "student_performance_correlation_heatmap.png")
        plt.close()
    except Exception as e:
        raise RuntimeError(f"Failed to save correlation heatmap: {e}") from e

def plot_training_history(history: tf.keras.callbacks.History) -> None:

    try:
        plt.figure(figsize=(8,7))
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "student_performance_training_history.png")
        plt.close()
    except Exception as e:
        raise RuntimeError(f"Failed to save training history plot: {e}") from e

def plot_confusion_matrix(y_true:np.ndarray, y_pred:np.ndarray) -> None:

    try:
        matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8,7))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Outcome")
        plt.ylabel("Actual Outcome")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "student_performance_confusion_matrix.png")
        plt.close()
    except Exception as e:
        raise RuntimeError(f"Failed to save confusion matrix: {e}") from e

def display_prediction_examples(
        student_ids:pd.Series,
        y_true:np.ndarray,
        y_pred:np.ndarray,
        y_proba:np.ndarray,
) -> None:

    sample_size = min(10,len(y_true))
    examples = pd.DataFrame({
        "student_id": student_ids.values[:sample_size],
        "actual": y_true[:sample_size],
        "predicted": y_pred[:sample_size],
        "probability": y_proba[:sample_size].flatten()
    })

    print("\n---- PREDICTION EXAMPLES (First 10 students)----")
    print(examples.to_string(index=False))

    try:
        plt.figure(figsize=(10,7))
        x_positions = np.arange(sample_size)
        plt.bar(x_positions - 0.2, examples["actual"], width=0.4, label="Actual")
        plt.bar(x_positions + 0.2, examples["predicted"], width=0.4, label="Predicted")
        plt.xticks(x_positions,examples["student_id"], rotation=90)
        plt.title("Prediction Examples: Actual vs Predicted Outcomes")
        plt.xlabel("Student ID")
        plt.ylabel("Outcome (0 = Fail, 1 = Pass)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "student_performance_prediction_examples.png")
        plt.close()
    except Exception as e:
        raise RuntimeError(f"Failed to save prediction examples plot: {e}") from e

# --------------------------------------------------------------------------------
# 7. Preprocessing Functions
# --------------------------------------------------------------------------------
def select_features_and_target(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:

    features = dataset[FEATURE_COLUMNS].copy()
    target = dataset[TARGET_COLUMN].copy()
    return features, target

def scale_features(features: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler

def split_dataset(
        features: np.ndarray,
        target: pd.Series,
        student_ids: pd.Series,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, pd.Series, pd.Series]:

    (x_train, x_test, y_train, y_test,id_train,id_test) = train_test_split(
        features,
        target,
        student_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target
    )

    return x_train, x_test, y_train, y_test, id_train, id_test
# --------------------------------------------------------------------------------
# 8. Validation Functions
# --------------------------------------------------------------------------------
def build_neural_network(input_dimension: int) -> Sequential:

    model = Sequential([
        Dense(
            64,
            activation="relu",
            kernel_regularizer=l2(REGULARIZATION_STRENGTH),
            input_shape=(input_dimension,)
        ),
        Dense(
            32,
            activation="relu",
            kernel_regularizer=l2(REGULARIZATION_STRENGTH)
        ),
        Dense(
            16,
            activation="relu",
            kernel_regularizer=l2(REGULARIZATION_STRENGTH)
        ),
        # Sigmoid squashes the final linear combination into a probability,
        # which is required for binary cross-entropy based classification.
        Dense(1, activation="sigmoid")
    ])

    # Adam (Adaptive Moment Estimation) performs Gradient Descent based
    # optimisation, adjusting each weight's learning rate individually using
    # estimates of first and second moments of gradients.
    # This accelerates convergence

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"])

    return model

def train_model(
        model: Sequential,
        x_train: np.ndarray,
        y_train: pd.Series,
)-> tf.keras.callbacks.History:

    try:
        history = model.fit(
            x_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            verbose=1
        )
    except Exception as e:
        raise RuntimeError(f"Model training failed: {e}") from e

    return history

# --------------------------------------------------------------------------------
# 9. Validation Functions
# --------------------------------------------------------------------------------
def evaluate_model(
        model: Sequential,
        x_test: np.ndarray,
        y_test: pd.Series,
) -> tuple[np.ndarray, np.ndarray, float]:

    probabilities = model.predict(x_test, verbose=0)
    predictions = (probabilities > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, predictions)

    print("\n---- MODEL EVALUATION ----")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return predictions, probabilities, accuracy

# --------------------------------------------------------------------------------
# 10. Prediction Functions
# --------------------------------------------------------------------------------
def predict_student_outcomes(model:Sequential, x_test: np.ndarray) -> np.ndarray:

    return model.predict(x_test,verbose=0)

# --------------------------------------------------------------------------------
# 11. Utility/Summary Functions
# --------------------------------------------------------------------------------
def display_summary(
    train_count: int,
    test_count: int,
    accuracy: float
) -> None:
    """
    Purpose:
        Display a structured closing summary of the demonstration,
        recapping dataset sizes, model accuracy, and the Deep Learning
        concepts illustrated by this program.

    Parameters:
        train_count (int): Number of training records.
        test_count (int): Number of testing records.
        accuracy (float): Final test accuracy achieved by the model.

    Returns:
        None.
    """
    print("\n" + "-" * 60)
    print("STUDENT PERFORMANCE PREDICTION RESULTS")
    print("-" * 60)
    print(f"Training Records: {train_count}")
    print(f"Testing Records: {test_count}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nDeep Learning Concepts Demonstrated:")
    print("  * Forward Propagation")
    print("  * Backpropagation")
    print("  * Gradient Descent")
    print("  * Regularization")
    print("  * ReLU Activation")
    print("  * Sigmoid Activation")
    print(f"\nOutput files saved to: {RESULTS_DIR}/")
    print("-" * 60)
    print("END OF DEMONSTRATION")
    print("-" * 60)


# --------------------------------------------------------------------------------
# 12. Main Execution Function
# --------------------------------------------------------------------------------
def main() -> None:

    try:
        create_results_directory()

        dataset = load_dataset(DATA_FILE)
        validate_dataset(dataset)

        display_dataset_information(dataset)
        display_dataset_statistics(dataset)

        plot_class_distribution(dataset)
        plot_correlation_heatmap(dataset)

        features, target = select_features_and_target(dataset)
        scaled_features, _scaler = scale_features(features)

        x_train, x_test, y_train, y_test, id_train, id_test = split_dataset(
            scaled_features, target, dataset["student_id"]
        )

        model = build_neural_network(input_dimension=x_train.shape[1])
        history = train_model(model, x_train, y_train)
        plot_training_history(history)

        predictions, probabilities, accuracy = evaluate_model(model, x_test, y_test)
        plot_confusion_matrix(y_test.to_numpy(),predictions)

        predict_student_outcomes(model, x_test)
        display_prediction_examples(id_test, y_test.to_numpy(), predictions, probabilities)

        display_summary(len(x_train), len(x_test), accuracy)
    except (FileNotFoundError, RuntimeError, ValueError,OSError) as e:
        print(f"\n[ERROR] Program terminated: {e}")



# --------------------------------------------------------------------------------
# 13. Run the script by invoking it's main() function
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    main()