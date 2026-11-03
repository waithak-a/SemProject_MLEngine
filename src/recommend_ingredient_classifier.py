import ast
import re
import sys
from pathlib import Path
from difflib import SequenceMatcher

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "Recipes_cleaned.csv"

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "ingredient_matching"
    / "ingredient_equivalence_model.joblib"
)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CONFIDENCE_THRESHOLD = 0.75
TOP_RESULTS = 10


# --------------------------------------------------
# 2. LOAD MODEL AND RECIPE DATA
# --------------------------------------------------

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Ingredient classifier not found: {MODEL_PATH}"
        )

    print("Loading ingredient equivalence classifier...")
    return joblib.load(MODEL_PATH)


def load_recipes():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Recipe dataset not found: {DATA_PATH}"
        )

    print("Loading recipes...")
    recipes = pd.read_csv(DATA_PATH)

    if "ingredients_normalized" not in recipes.columns:
        raise ValueError(
            "The dataset does not contain 'ingredients_normalized'."
        )

    name_column = next(
        (
            column for column in
            ["name", "recipe_name", "title", "recipe_title"]
            if column in recipes.columns
        ),
        None
    )

    if name_column is None:
        raise ValueError(
            "Could not identify the recipe name column. "
            "Check the columns in Recipes_cleaned.csv."
        )

    return recipes, name_column


# --------------------------------------------------
# 3. LOAD EMBEDDING MODEL
# --------------------------------------------------

def load_embedding_model():
    print("Loading ingredient embedding model...")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# --------------------------------------------------
# 4. PROCESS INGREDIENTS
# --------------------------------------------------

def normalize_ingredient(ingredient):
    ingredient = str(ingredient).strip().lower()
    ingredient = re.sub(r"\s+", " ", ingredient)
    return ingredient


def parse_ingredients(value):
    """
    Convert the ingredients stored in the CSV into a list.
    Supports Python-style lists and comma-separated strings.
    """

    if pd.isna(value):
        return []

    if isinstance(value, list):
        ingredients = value

    else:
        value = str(value).strip()

        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, list):
                ingredients = parsed
            else:
                ingredients = value.split(",")

        except (ValueError, SyntaxError):
            ingredients = value.split(",")

    cleaned = []

    for ingredient in ingredients:
        ingredient = normalize_ingredient(ingredient)

        if ingredient and ingredient not in cleaned:
            cleaned.append(ingredient)

    return cleaned


def get_user_ingredients():
    print("\nEnter ingredients separated by commas.")

    user_input = input("Your ingredients: ").strip()

    ingredients = [
        normalize_ingredient(item)
        for item in user_input.split(",")
    ]

    ingredients = list(dict.fromkeys(
        ingredient for ingredient in ingredients if ingredient
    ))

    if not ingredients:
        raise ValueError("You must enter at least one ingredient.")

    return ingredients


# --------------------------------------------------
# 5. GENERATE CLASSIFIER FEATURES
# --------------------------------------------------

def calculate_features(ingredient_a, ingredient_b, embedding_cache):

    ingredient_a = normalize_ingredient(ingredient_a)
    ingredient_b = normalize_ingredient(ingredient_b)

    embedding_a = embedding_cache[ingredient_a]
    embedding_b = embedding_cache[ingredient_b]

    cosine_similarity = float(
        np.dot(embedding_a, embedding_b)
    )

    embedding_distance = 1.0 - cosine_similarity

    words_a = set(ingredient_a.split())
    words_b = set(ingredient_b.split())

    intersection = words_a & words_b
    union = words_a | words_b

    word_overlap = (
        len(intersection) / len(union)
        if union else 0.0
    )

    token_containment = (
        len(intersection) / min(len(words_a), len(words_b))
        if words_a and words_b else 0.0
    )

    character_similarity = SequenceMatcher(
        None,
        ingredient_a,
        ingredient_b
    ).ratio()

    length_difference = abs(
        len(ingredient_a) - len(ingredient_b)
    )

    maximum_length = max(
        len(ingredient_a),
        len(ingredient_b),
        1
    )

    normalized_length_difference = (
        length_difference / maximum_length
    )

    return [
        cosine_similarity,
        embedding_distance,
        word_overlap,
        token_containment,
        character_similarity,
        length_difference,
        normalized_length_difference
    ]


# --------------------------------------------------
# 6. CREATE INGREDIENT EMBEDDINGS
# --------------------------------------------------

def create_embedding_cache(ingredients, embedding_model):

    unique_ingredients = list(dict.fromkeys(ingredients))

    embeddings = embedding_model.encode(
        unique_ingredients,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return {
        ingredient: embedding
        for ingredient, embedding in zip(
            unique_ingredients,
            embeddings
        )
    }


# --------------------------------------------------
# 7. PREDICT INGREDIENT EQUIVALENCE
# --------------------------------------------------

def predict_equivalence(
    ingredient_a,
    ingredient_b,
    classifier,
    embedding_cache
):

    feature_names = [
        "cosine_similarity",
        "embedding_distance",
        "word_overlap",
        "token_containment",
        "character_similarity",
        "length_difference",
        "normalized_length_difference"
    ]

    features = calculate_features(
        ingredient_a,
        ingredient_b,
        embedding_cache
    )

    features_df = pd.DataFrame(
        [features],
        columns=feature_names
    )

    probabilities = classifier.predict_proba(features_df)[0]

    positive_class_index = list(
        classifier.classes_
    ).index(1)

    return float(probabilities[positive_class_index])

# --------------------------------------------------
# 8. MATCH USER INGREDIENTS TO A RECIPE
# --------------------------------------------------

def match_recipe(
    user_ingredients,
    recipe_ingredients,
    classifier,
    embedding_cache
):

    matched = []
    missing = []

    for user_ingredient in user_ingredients:

        best_match = None
        best_score = 0.0

        for recipe_ingredient in recipe_ingredients:

            score = predict_equivalence(
                user_ingredient,
                recipe_ingredient,
                classifier,
                embedding_cache
            )

            if score > best_score:
                best_score = score
                best_match = recipe_ingredient

        if best_match is not None and best_score >= CONFIDENCE_THRESHOLD:

            matched.append({
                "user_ingredient": user_ingredient,
                "recipe_ingredient": best_match,
                "confidence": best_score
            })

        else:
            missing.append(user_ingredient)

    return matched, missing


# --------------------------------------------------
# 9. RANK RECIPES
# --------------------------------------------------

def recommend_recipes(
    user_ingredients,
    recipes,
    name_column,
    classifier,
    embedding_model
):

    print("\nPreparing ingredient comparisons...")

    parsed_recipes = []

    all_ingredients = list(user_ingredients)

    for _, row in recipes.iterrows():

        recipe_ingredients = parse_ingredients(
            row["ingredients_normalized"]
        )

        if not recipe_ingredients:
            continue

        parsed_recipes.append({
            "name": row[name_column],
            "ingredients": recipe_ingredients
        })

        all_ingredients.extend(recipe_ingredients)

    embedding_cache = create_embedding_cache(
        all_ingredients,
        embedding_model
    )

    print("Comparing ingredients with recipes...")

    results = []

    for recipe in parsed_recipes:

        matched, missing = match_recipe(
            user_ingredients,
            recipe["ingredients"],
            classifier,
            embedding_cache
        )

        matched_count = len(matched)

        score = (
            matched_count / len(user_ingredients)
        ) * 100

        results.append({
            "name": recipe["name"],
            "ingredients": recipe["ingredients"],
            "matched": matched,
            "missing": missing,
            "score": score,
            "matched_count": matched_count
        })

    results.sort(
        key=lambda item: (
            item["score"],
            item["matched_count"]
        ),
        reverse=True
    )

    return results[:TOP_RESULTS]




def display_recommendations(results, user_ingredients):

    print("\n" + "=" * 65)
    print("           INGREDIENT-BASED RECIPE RECOMMENDATIONS")
    print("=" * 65)

    print("\nYour ingredients:")
    print(", ".join(user_ingredients))

    if not results:
        print("\nNo suitable recipes were found.")
        return

    for index, recipe in enumerate(results, start=1):

        print("\n" + "-" * 65)
        print(f"{index}. {recipe['name']}")

        print(
            f"Match score: {recipe['score']:.2f}% "
            f"({recipe['matched_count']}/{len(user_ingredients)} "
            f"ingredients matched)"
        )

        print("\nMatched ingredients:")

        if recipe["matched"]:
            for item in recipe["matched"]:
                print(
                    f"  {item['user_ingredient']} "
                    f"-> {item['recipe_ingredient']} "
                    f"({item['confidence']:.2%})"
                )
        else:
            print("  No ingredients matched.")

        print("\nIngredients you do not have a match for:")

        if recipe["missing"]:
            for ingredient in recipe["missing"]:
                print(f"  - {ingredient}")
        else:
            print("  None")

        print("\nRecipe ingredients:")
        print(", ".join(recipe["ingredients"]))


def main():

    try:

        classifier = load_model()

        recipes, name_column = load_recipes()

        embedding_model = load_embedding_model()

        user_ingredients = get_user_ingredients()

        results = recommend_recipes(
            user_ingredients,
            recipes,
            name_column,
            classifier,
            embedding_model
        )

        display_recommendations(
            results,
            user_ingredients
        )

    except Exception as error:

        print(f"\nError: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()