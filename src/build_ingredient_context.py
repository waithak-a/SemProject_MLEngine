
from pathlib import Path
import ast
import json

import numpy as np
import pandas as pd


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "Recipes_cleaned.csv"

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ingredient_context.npz"
)


# ==================================================
# 2. INGREDIENT PARSING
# ==================================================

def parse_ingredients(value):

    if pd.isna(value):
        return []

    if isinstance(value, list):
        ingredients = value

    else:
        value = str(value).strip()

        if not value:
            return []

        try:
            ingredients = ast.literal_eval(value)

        except (ValueError, SyntaxError):
            try:
                ingredients = json.loads(value)

            except (ValueError, TypeError):
                ingredients = value.split("|")

    if not isinstance(ingredients, (list, tuple, set)):
        return []

    cleaned = []

    for ingredient in ingredients:

        if not isinstance(ingredient, str):
            continue

        ingredient = " ".join(
            ingredient.lower().strip().split()
        )

        if ingredient and ingredient not in cleaned:
            cleaned.append(ingredient)

    return cleaned


# ==================================================
# 3. LOAD RECIPE DATASET
# ==================================================

def load_recipes():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    ingredient_column = "ingredients_normalized"

    if ingredient_column not in df.columns:
        raise ValueError(
            f"Missing column: {ingredient_column}"
        )

    recipes = df[ingredient_column].apply(parse_ingredients)

    recipes = [
        ingredients
        for ingredients in recipes
        if len(ingredients) >= 2
    ]

    if not recipes:
        raise ValueError(
            "No valid recipes with at least two ingredients were found."
        )

    return recipes


# ==================================================
# 4. BUILD INGREDIENT VOCABULARY
# ==================================================

def build_vocabulary(recipes):

    vocabulary = sorted({
        ingredient
        for recipe in recipes
        for ingredient in recipe
    })

    ingredient_to_index = {
        ingredient: index
        for index, ingredient in enumerate(vocabulary)
    }

    return vocabulary, ingredient_to_index


# ==================================================
# 5. BUILD CO-OCCURRENCE MATRIX
# ==================================================

def build_cooccurrence_matrix(
    recipes,
    vocabulary,
    ingredient_to_index
):

    number_of_ingredients = len(vocabulary)

    cooccurrence = np.zeros(
        (
            number_of_ingredients,
            number_of_ingredients,
        ),
        dtype=np.float32,
    )

    frequencies = np.zeros(
        number_of_ingredients,
        dtype=np.float32,
    )

    for recipe in recipes:

        indices = sorted({
            ingredient_to_index[ingredient]
            for ingredient in recipe
        })

        for index in indices:
            frequencies[index] += 1

        for i in range(len(indices)):

            for j in range(i + 1, len(indices)):

                index_a = indices[i]
                index_b = indices[j]

                cooccurrence[index_a, index_b] += 1
                cooccurrence[index_b, index_a] += 1

    return cooccurrence, frequencies


# ==================================================
# 6. CALCULATE PPMI CONTEXT PROFILES
# ==================================================

def calculate_context_profiles(
    cooccurrence,
    frequencies,
    number_of_recipes
):

    context = np.zeros_like(cooccurrence)

    denominator = (
        frequencies[:, None] * frequencies[None, :]
    )

    valid = cooccurrence > 0

    with np.errstate(divide="ignore", invalid="ignore"):

        pmi = np.log(
            (cooccurrence * number_of_recipes)
            / np.maximum(denominator, 1)
        )

    context[valid] = np.maximum(pmi[valid], 0)

    # An ingredient should not be its own context.
    np.fill_diagonal(context, 0)

    # Normalize each context vector.
    norms = np.linalg.norm(context, axis=1, keepdims=True)

    context = np.divide(
        context,
        norms,
        out=np.zeros_like(context),
        where=norms != 0,
    )

    return context


# ==================================================
# 7. SAVE CONTEXT PROFILES
# ==================================================

def save_context_profiles(
    vocabulary,
    context,
    frequencies,
    number_of_recipes
):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_FILE,
        ingredients=np.array(vocabulary),
        context_vectors=context,
        recipe_frequencies=frequencies,
        number_of_recipes=number_of_recipes,
    )

    print(f"\nContext profiles saved to:\n{OUTPUT_FILE}")


# ==================================================
# 8. MAIN
# ==================================================

def main():

    print("\nBUILDING INGREDIENT CONTEXT PROFILES")
    print("=" * 50)

    recipes = load_recipes()

    print(f"Recipes processed: {len(recipes)}")

    vocabulary, ingredient_to_index = build_vocabulary(recipes)

    print(f"Unique ingredients: {len(vocabulary)}")

    cooccurrence, frequencies = build_cooccurrence_matrix(
        recipes,
        vocabulary,
        ingredient_to_index,
    )

    context = calculate_context_profiles(
        cooccurrence,
        frequencies,
        len(recipes),
    )

    save_context_profiles(
        vocabulary,
        context,
        frequencies,
        len(recipes),
    )

    print("\nCONTEXT PROFILE SUMMARY")
    print("=" * 50)

    print(f"Number of recipes: {len(recipes)}")
    print(f"Unique ingredients: {len(vocabulary)}")
    print(f"Context matrix shape: {context.shape}")
    print(f"Non-zero context values: {np.count_nonzero(context)}")

    print("\nBUILD COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
