"""Compare the trained recipe model with an embedding/semantic-search model.

The trained scikit-learn model is kept as the baseline.  The embedding model
provides an independent semantic-search view of the same recipes so that the
two approaches can be compared side by side.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from recommend import (
    TOP_N,
    calculate_features,
    load_recipes,
    normalize_ingredient,
    normalize_ingredient_list,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "recipe_match_model.joblib"
EMBEDDINGS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "ingredient_embeddings.npz"
)


def load_trained_model():
    """Load the existing trained model without retraining it."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    try:
        import joblib
    except ImportError as error:
        raise RuntimeError(
            "Comparing with the trained model requires joblib and scikit-learn."
        ) from error

    return joblib.load(MODEL_PATH)


def _feature_row(user_ingredients, recipe_ingredients):
    features = calculate_features(user_ingredients, recipe_ingredients)
    return {
        "jaccard_similarity": features["jaccard_similarity"],
        "user_coverage": features["user_coverage"],
        "recipe_coverage": features["recipe_coverage"],
        "matched_ingredients": features["matched_count"],
        "missing_ingredients": max(
            len(normalize_ingredient_list(recipe_ingredients))
            - features["matched_count"],
            0,
        ),
        "user_ingredient_count": len(
            normalize_ingredient_list(user_ingredients)
        ),
        "recipe_ingredient_count": len(
            normalize_ingredient_list(recipe_ingredients)
        ),
    }


def current_model_recommendations(user_ingredients, recipes, model=None, top_n=TOP_N):
    """Rank recipes using the trained model's predicted probability."""
    if model is None:
        model = load_trained_model()

    feature_columns = [
        "jaccard_similarity",
        "user_coverage",
        "recipe_coverage",
        "matched_ingredients",
        "missing_ingredients",
        "user_ingredient_count",
        "recipe_ingredient_count",
    ]

    rows = []

    for recipe in recipes:
        row = _feature_row(user_ingredients, recipe["ingredients"])
        row["title"] = recipe["title"]
        row["ingredients"] = recipe["ingredients"]
        rows.append(row)

    if not rows:
        return []

    frame = pd.DataFrame(rows)
    probabilities = model.predict_proba(frame[feature_columns])[:, 1]
    frame["model_probability"] = probabilities

    frame = frame.sort_values(
        "model_probability", ascending=False
    ).head(top_n)

    return frame.to_dict("records")


def _load_embedding_lookup():
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Ingredient embeddings not found: {EMBEDDINGS_PATH}"
        )

    data = np.load(EMBEDDINGS_PATH)
    names = [str(name).strip().lower() for name in data["ingredients"]]
    vectors = data["embeddings"]

    return {
        name: vector
        for name, vector in zip(names, vectors)
    }


def _embedding_for(name, lookup):
    normalized = normalize_ingredient(name).lower().strip()

    if normalized in lookup:
        return lookup[normalized]

    return None


def semantic_search_recommendations(user_ingredients, recipes, top_n=TOP_N):
    """Rank recipes by embedding similarity between ingredient sets."""
    lookup = _load_embedding_lookup()
    user_vectors = []

    for ingredient in normalize_ingredient_list(user_ingredients):
        vector = _embedding_for(ingredient, lookup)

        if vector is not None:
            user_vectors.append(vector)

    if not user_vectors:
        return []

    user_matrix = np.array(user_vectors)
    recommendations = []

    for recipe in recipes:
        recipe_vectors = []

        for ingredient in normalize_ingredient_list(recipe["ingredients"]):
            vector = _embedding_for(ingredient, lookup)

            if vector is not None:
                recipe_vectors.append(vector)

        if not recipe_vectors:
            continue

        recipe_matrix = np.array(recipe_vectors)

        similarity_matrix = recipe_matrix @ user_matrix.T
        best_matches = similarity_matrix.max(axis=0)
        score = float(best_matches.mean())

        recommendations.append(
            {
                "title": recipe["title"],
                "ingredients": recipe["ingredients"],
                "semantic_score": score,
            }
        )

    recommendations.sort(
        key=lambda item: item["semantic_score"], reverse=True
    )

    return recommendations[:top_n]


def compare(user_ingredients, top_n=TOP_N):
    """Print a side-by-side comparison of the two approaches."""
    recipes = load_recipes()

    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print("User ingredients:", ", ".join(user_ingredients))

    print("\n--- Current trained model ---")
    try:
        current = current_model_recommendations(
            user_ingredients, recipes, top_n=top_n
        )

        for index, recipe in enumerate(current, start=1):
            print(
                f"{index}. {recipe['title']} "
                f"(probability: {recipe['model_probability']:.3f})"
            )

    except Exception as error:
        print(f"Could not run trained model: {error}")

    print("\n--- Embedding / semantic search model ---")
    try:
        semantic = semantic_search_recommendations(
            user_ingredients, recipes, top_n=top_n
        )

        for index, recipe in enumerate(semantic, start=1):
            print(
                f"{index}. {recipe['title']} "
                f"(semantic score: {recipe['semantic_score']:.3f})"
            )

    except Exception as error:
        print(f"Could not run semantic search: {error}")


if __name__ == "__main__":
    sample = ["milk", "sugar", "flour"]
    compare(sample)
