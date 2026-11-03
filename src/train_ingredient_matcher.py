
from pathlib import Path
from difflib import SequenceMatcher
import json
import shutil

import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


# ==================================================
# 1. PATHS AND SETTINGS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models" / "ingredient_matching"

CSV_FILE = PROCESSED_DIR / "ingredient_pairs.csv"
EMBEDDINGS_FILE = PROCESSED_DIR / "ingredient_embeddings.npz"

MODEL_FILE = MODEL_DIR / "ingredient_equivalence_model.joblib"
METRICS_FILE = MODEL_DIR / "evaluation_metrics.json"

BACKUP_FILE = PROCESSED_DIR / "ingredient_pairs_before_classifier.bak"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CONFIDENCE_THRESHOLD = 0.75
TEST_SIZE = 0.20
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "cosine_similarity",
    "embedding_distance",
    "word_overlap",
    "token_containment",
    "character_similarity",
    "length_difference",
    "normalized_length_difference",
]


# ==================================================
# 2. NORMALIZATION AND FEATURE FUNCTIONS
# ==================================================

def normalize_name(value):
    if pd.isna(value):
        return ""

    return " ".join(str(value).lower().strip().split())


def normalize_label(value):
    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    if value in {"1", "1.0"}:
        return 1

    if value in {"0", "0.0"}:
        return 0

    return None


def calculate_text_features(name_a, name_b):

    tokens_a = set(name_a.split())
    tokens_b = set(name_b.split())

    union = tokens_a | tokens_b
    intersection = tokens_a & tokens_b

    word_overlap = (
        len(intersection) / len(union)
        if union else 0
    )

    token_containment = (
        len(intersection) / min(len(tokens_a), len(tokens_b))
        if tokens_a and tokens_b else 0
    )

    length_difference = abs(len(name_a) - len(name_b))

    maximum_length = max(len(name_a), len(name_b), 1)

    normalized_length_difference = (
        length_difference / maximum_length
    )

    character_similarity = SequenceMatcher(
        None, name_a, name_b
    ).ratio()

    return {
        "word_overlap": word_overlap,
        "token_containment": token_containment,
        "character_similarity": character_similarity,
        "length_difference": length_difference,
        "normalized_length_difference": normalized_length_difference,
    }


# ==================================================
# 3. LOAD DATA AND PRESERVE ORIGINAL LABELS
# ==================================================

def load_data():

    if not CSV_FILE.exists():
        raise FileNotFoundError(f"Missing dataset: {CSV_FILE}")

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Missing embeddings: {EMBEDDINGS_FILE}"
        )

    df = pd.read_csv(CSV_FILE)

    required_columns = {"ingredient_a", "ingredient_b"}

    if not required_columns.issubset(df.columns):
        raise ValueError("The CSV is missing ingredient name columns.")

    # Preserve the existing CSV before modifying it.
    if not BACKUP_FILE.exists():
        shutil.copy2(CSV_FILE, BACKUP_FILE)
        print(f"Backup created: {BACKUP_FILE}")

    df["ingredient_a"] = df["ingredient_a"].apply(normalize_name)
    df["ingredient_b"] = df["ingredient_b"].apply(normalize_name)

    # Preserve the existing manually assigned labels.
    if "manual_label" not in df.columns:

        df["manual_label"] = df["label"].apply(normalize_label)

    else:

        df["manual_label"] = df["manual_label"].apply(normalize_label)

    # Do not use automatic labels as training targets.
    df["label"] = df["manual_label"]

    return df


# ==================================================
# 4. LOAD EMBEDDINGS AND GENERATE MISSING ONES
# ==================================================

def load_embeddings(df):

    saved = np.load(EMBEDDINGS_FILE, allow_pickle=True)

    vocabulary = [
        normalize_name(name)
        for name in saved["ingredients"]
    ]

    vectors = saved["embeddings"]

    embedding_lookup = {
        name: vector
        for name, vector in zip(vocabulary, vectors)
    }

    required_names = set(df["ingredient_a"]) | set(df["ingredient_b"])

    missing_names = sorted(
        name for name in required_names
        if name not in embedding_lookup
    )

    if missing_names:

        print(
            f"Generating embeddings for {len(missing_names)} "
            "previously unseen ingredient names..."
        )

        model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        new_vectors = model.encode(
            missing_names,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=64,
        )

        for name, vector in zip(missing_names, new_vectors):
            embedding_lookup[name] = vector

    return embedding_lookup


# ==================================================
# 5. CALCULATE FEATURES FOR EVERY PAIR
# ==================================================

def calculate_all_features(df, embedding_lookup):

    feature_rows = []

    for _, row in df.iterrows():

        name_a = row["ingredient_a"]
        name_b = row["ingredient_b"]

        vector_a = embedding_lookup[name_a]
        vector_b = embedding_lookup[name_b]

        cosine_similarity = float(
            np.dot(vector_a, vector_b)
        )

        features = calculate_text_features(name_a, name_b)

        features["cosine_similarity"] = cosine_similarity
        features["embedding_distance"] = 1 - cosine_similarity

        feature_rows.append(features)

    feature_df = pd.DataFrame(feature_rows, index=df.index)

    for column in FEATURE_COLUMNS:
        df[column] = feature_df[column]

    return df


# ==================================================
# 6. PREPARE TRAINING DATA
# ==================================================

def prepare_training_data(df):

    labelled = df[df["manual_label"].notna()].copy()

    if len(labelled) < 20:
        raise ValueError(
            "At least 20 manually labelled pairs are required."
        )

    X = labelled[FEATURE_COLUMNS].astype(float)

    y = labelled["manual_label"].astype(int)

    class_counts = y.value_counts()

    print("\nTRAINING LABEL DISTRIBUTION")
    print(class_counts.to_string())

    if len(class_counts) != 2 or class_counts.min() < 2:
        raise ValueError(
            "Both classes need at least two labelled examples."
        )

    return X, y


# ==================================================
# 7. CREATE AND TRAIN THE CLASSIFIER
# ==================================================

def create_classifier():

    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


# ==================================================
# 8. EVALUATE ON HELD-OUT DATA
# ==================================================

def evaluate_classifier(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    evaluation_model = create_classifier()

    evaluation_model.fit(X_train, y_train)

    predictions = evaluation_model.predict(X_test)

    probabilities = evaluation_model.predict_proba(X_test)[:, 1]

    metrics = {
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            zero_division=0,
        ),
    }

    if len(np.unique(y_test)) == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(y_test, probabilities)
        )

    print("\nHELD-OUT EVALUATION")
    print("=" * 45)

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    ]:
        if metric in metrics:
            print(f"{metric}: {metrics[metric]:.4f}")

    print("\nConfusion matrix:")
    print(metrics["confusion_matrix"])

    print("\nClassification report:")
    print(metrics["classification_report"])

    return metrics


# ==================================================
# 9. TRAIN FINAL MODEL ON ALL MANUAL LABELS
# ==================================================

def train_final_model(X, y):

    model = create_classifier()

    model.fit(X, y)

    return model


# ==================================================
# 10. PREDICT ALL INGREDIENT PAIRS
# ==================================================

def label_all_pairs(df, model):

    X_all = df[FEATURE_COLUMNS].astype(float)

    probabilities = model.predict_proba(X_all)

    predicted_classes = model.predict(X_all)

    confidence_values = probabilities.max(axis=1)

    raw_predictions = []

    for prediction, confidence in zip(
        predicted_classes,
        confidence_values,
    ):

        if confidence < CONFIDENCE_THRESHOLD:
            raw_predictions.append("REVIEW")

        else:
            raw_predictions.append(int(prediction))

    df["auto_label"] = raw_predictions

    df["confidence"] = confidence_values

    # Preserve the original manual labels.
    # For unlabelled pairs, convert REVIEW to 0 as requested.
    final_labels = []
    statuses = []

    for _, row in df.iterrows():

        manual_label = row["manual_label"]
        automatic_label = row["auto_label"]

        if pd.notna(manual_label):

            final_labels.append(int(manual_label))
            statuses.append("MANUAL_SEED")

        elif automatic_label == "REVIEW":

            final_labels.append(0)
            statuses.append("REVIEW_CONVERTED_TO_0")

        else:

            final_labels.append(int(automatic_label))
            statuses.append("AUTO_ASSIGNED")

    df["label"] = final_labels
    df["status"] = statuses

    return df


# ==================================================
# 11. SAVE MODEL, METRICS AND CENTRAL CSV
# ==================================================

def save_results(df, model, metrics):

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    df.to_csv(CSV_FILE, index=False)

    print("\nSAVED RESULTS")
    print("=" * 45)

    print(f"Updated dataset: {CSV_FILE}")
    print(f"Trained model: {MODEL_FILE}")
    print(f"Evaluation metrics: {METRICS_FILE}")
    print(f"Original CSV backup: {BACKUP_FILE}")

    print("\nFINAL LABEL COUNTS")
    print(df["label"].value_counts().to_string())

    print("\nAUTOMATIC LABELLING COUNTS")
    print(df["status"].value_counts().to_string())


# ==================================================
# 12. MAIN
# ==================================================

def main():

    print("\nINGREDIENT EQUIVALENCE CLASSIFIER")
    print("=" * 45)

    df = load_data()

    print(f"\nLoaded ingredient pairs: {len(df)}")

    embedding_lookup = load_embeddings(df)

    df = calculate_all_features(df, embedding_lookup)

    X, y = prepare_training_data(df)

    metrics = evaluate_classifier(X, y)

    print("\nTraining final classifier using all manual labels...")

    final_model = train_final_model(X, y)

    df = label_all_pairs(df, final_model)

    save_results(df, final_model, metrics)

    print("\nCLASSIFIER TRAINING COMPLETED.")


if __name__ == "__main__":
    main()
