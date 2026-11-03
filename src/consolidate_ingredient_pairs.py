
from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

OUTPUT_FILE = PROCESSED_DIR / "ingredient_pairs.csv"

# Files that may contain ingredient-pair information.
INPUT_FILES = {
    "manual": PROCESSED_DIR / "ingredient_matching_tests.csv",
    "review": PROCESSED_DIR / "ingredient_pair_review.csv",
    "candidates": PROCESSED_DIR / "ingredient_pair_candidates_200.csv",
    "features": PROCESSED_DIR / "ingredient_pair_features.csv",
    "automatic": PROCESSED_DIR / "ingredient_pair_auto_labels.csv",
}

# Features we want to preserve.
FEATURE_COLUMNS = [
    "cosine_similarity",
    "embedding_distance",
    "word_overlap",
    "token_containment",
    "character_similarity",
    "length_difference",
    "normalized_length_difference",
]


# --------------------------------------------------
# 2. Helper functions
# --------------------------------------------------

def normalize_name(value):
    """Standardize ingredient names."""
    if pd.isna(value):
        return ""

    return " ".join(str(value).lower().strip().split())


def normalize_label(value):
    """Accept only verified binary labels."""
    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    if value in {"1", "1.0"}:
        return 1

    if value in {"0", "0.0"}:
        return 0

    return None


def make_pair_key(name_a, name_b):
    """Create an order-independent pair identifier."""
    return tuple(sorted((name_a, name_b)))


def safe_float(value):
    """Convert a feature to a number where possible."""
    if pd.isna(value):
        return None

    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# --------------------------------------------------
# 3. Read existing CSV files
# --------------------------------------------------

def load_input_files():

    loaded_files = {}

    for source, file_path in INPUT_FILES.items():

        if not file_path.exists():
            print(f"Skipping missing file: {file_path.name}")
            continue

        df = pd.read_csv(file_path)

        required = {"ingredient_a", "ingredient_b"}

        if not required.issubset(df.columns):
            print(f"Skipping invalid file: {file_path.name}")
            continue

        loaded_files[source] = df

        print(
            f"Loaded {file_path.name}: {len(df)} rows"
        )

    if not loaded_files:
        raise FileNotFoundError(
            "No valid ingredient-pair CSV files were found."
        )

    return loaded_files


# --------------------------------------------------
# 4. Determine which labels are trusted
# --------------------------------------------------

def get_trusted_label(row, source):
    """
    Only use human-provided labels.

    Never use auto_label or final_label as verified labels.
    """

    if source == "automatic":

        # The automatic file has a separate manual_label column.
        if "manual_label" in row.index:
            return normalize_label(row["manual_label"])

        return None

    # Labels in the manual, review, candidate and feature
    # files are accepted as supplied human labels.
    if "label" in row.index:
        return normalize_label(row["label"])

    return None


# --------------------------------------------------
# 5. Consolidate all ingredient pairs
# --------------------------------------------------

def consolidate_pairs(loaded_files):

    consolidated = {}

    # Priority determines which source supplies the first
    # available feature value and preferred auto prediction.
    priority = [
        "manual",
        "review",
        "candidates",
        "features",
        "automatic",
    ]

    for source in priority:

        if source not in loaded_files:
            continue

        df = loaded_files[source]

        for _, row in df.iterrows():

            name_a = normalize_name(row["ingredient_a"])
            name_b = normalize_name(row["ingredient_b"])

            if not name_a or not name_b:
                continue

            if name_a == name_b:
                continue

            pair_key = make_pair_key(name_a, name_b)

            # Keep a consistent ingredient order.
            name_a, name_b = pair_key

            if pair_key not in consolidated:

                consolidated[pair_key] = {
                    "ingredient_a": name_a,
                    "ingredient_b": name_b,
                    "label": None,
                    "auto_label": "REVIEW",
                    "status": "PENDING",
                    "source": set(),
                    "reason": "",
                    "conflict": False,
                }

                for feature in FEATURE_COLUMNS:
                    consolidated[pair_key][feature] = None

            record = consolidated[pair_key]

            record["source"].add(source)

            # Preserve the first available feature value.
            for feature in FEATURE_COLUMNS:

                if feature in row.index:

                    value = safe_float(row[feature])

                    if (
                        value is not None
                        and record[feature] is None
                    ):
                        record[feature] = value

            # Preserve automatic predictions separately.
            if source == "automatic" and "auto_label" in row.index:

                prediction = str(row["auto_label"]).strip().upper()

                if prediction in {"0", "0.0"}:
                    record["auto_label"] = 0

                elif prediction in {"1", "1.0"}:
                    record["auto_label"] = 1

                elif prediction == "REVIEW":
                    record["auto_label"] = "REVIEW"

            # Preserve a useful explanation.
            if "reason" in row.index and pd.notna(row["reason"]):

                reason = str(row["reason"]).strip()

                if reason and not record["reason"]:
                    record["reason"] = reason

            # Only trusted labels are considered here.
            trusted_label = get_trusted_label(row, source)

            if trusted_label is not None:

                if record["label"] is None:

                    record["label"] = trusted_label

                elif record["label"] != trusted_label:

                    record["conflict"] = True

    return consolidated


# --------------------------------------------------
# 6. Build the final DataFrame
# --------------------------------------------------

def build_output(consolidated):

    results = []

    for record in consolidated.values():

        if record["conflict"]:

            status = "CONFLICT"

            # Do not silently choose between conflicting labels.
            label = None

        elif record["label"] is not None:

            status = "REVIEWED"
            label = record["label"]

        else:

            status = "PENDING"
            label = None

        results.append({
            "ingredient_a": record["ingredient_a"],
            "ingredient_b": record["ingredient_b"],
            **{
                feature: record[feature]
                for feature in FEATURE_COLUMNS
            },
            "label": label if label is not None else "",
            "auto_label": record["auto_label"],
            "status": status,
            "source": ";".join(sorted(record["source"])),
            "reason": record["reason"],
        })

    output_df = pd.DataFrame(results)

    return output_df.sort_values(
        by=["status", "ingredient_a", "ingredient_b"]
    ).reset_index(drop=True)


# --------------------------------------------------
# 7. Main program
# --------------------------------------------------

def main():

    print("\nINGREDIENT PAIR MIGRATION")
    print("=" * 45)

    loaded_files = load_input_files()

    consolidated = consolidate_pairs(loaded_files)

    output_df = build_output(consolidated)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Prevent accidental overwriting of an existing central dataset.
    if OUTPUT_FILE.exists():

        raise FileExistsError(
            f"\n{OUTPUT_FILE.name} already exists.\n"
            "Back it up or rename it before running this migration."
        )

    output_df.to_csv(OUTPUT_FILE, index=False)

    print("\nMIGRATION COMPLETE")
    print("=" * 45)

    print(f"Total unique pairs: {len(output_df)}")

    print(
        "Reviewed pairs:",
        (output_df["status"] == "REVIEWED").sum()
    )

    print(
        "Pending pairs:",
        (output_df["status"] == "PENDING").sum()
    )

    print(
        "Conflicting pairs:",
        (output_df["status"] == "CONFLICT").sum()
    )

    print(f"\nOutput saved to:\n{OUTPUT_FILE}")

    print("\nFirst 15 consolidated pairs:")

    print(
        output_df[
            [
                "ingredient_a",
                "ingredient_b",
                "label",
                "auto_label",
                "status",
                "source",
            ]
        ].head(15).to_string(index=False)
    )


if __name__ == "__main__":
    main()
