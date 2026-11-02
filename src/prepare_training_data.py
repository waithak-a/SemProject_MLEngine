
import ast
from importlib.resources import path
import json
import random
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = Path("data/processed/Recipes_cleaned.csv")
OUTPUT_FILE = Path("data/processed/Training_data")

RANDOM_SEED = 42
NEGATIVE_EXAMPLES_PER_RECIPE = 3


def parse_ingredients(value):
    """Convert a stored ingredient list into a normalized set."""

    if isinstance(value, list):
        ingredients = value

    elif pd.isna(value):
        return set()

    elif isinstance(value, str):
        value = value.strip()

        if not value:
            return set()

        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, list):
                ingredients = parsed
            else:
                ingredients = value.split("|")

        except (ValueError, SyntaxError):
            ingredients = value.split("|")

    else:
        return set()

    return {
        str(ingredient).strip().lower()
        for ingredient in ingredients
        if str(ingredient).strip()
    }


def calculate_features(user_ingredients, recipe_ingredients):
    """Calculate ingredient similarity features for a user–recipe pair."""

    matched = user_ingredients.intersection(recipe_ingredients)
    union = user_ingredients.union(recipe_ingredients)

    user_count = len(user_ingredients)
    recipe_count = len(recipe_ingredients)
    matched_count = len(matched)

    jaccard_similarity = (
        len(matched) / len(union)
        if union
        else 0.0
    )

    user_coverage = (
        matched_count / user_count
        if user_count
        else 0.0
    )

    recipe_coverage = (
        matched_count / recipe_count
        if recipe_count
        else 0.0
    )

    missing_ingredients = recipe_count - matched_count

    return {
        "jaccard_similarity": jaccard_similarity,
        "user_coverage": user_coverage,
        "recipe_coverage": recipe_coverage,
        "matched_ingredients": matched_count,
        "missing_ingredients": missing_ingredients,
        "user_ingredient_count": user_count,
        "recipe_ingredient_count": recipe_count,
    }


def create_user_ingredients(recipe_ingredients, rng):
    """Simulate ingredients available to a user."""

    ingredients = sorted(recipe_ingredients)

    if not ingredients:
        return set()

    if len(ingredients) == 1:
        return set(ingredients)

    minimum = max(1, int(len(ingredients) * 0.4))
    maximum = max(minimum, int(len(ingredients) * 0.8))
    maximum = min(maximum, len(ingredients) - 1)

    selected_count = rng.randint(minimum, maximum)

    return set(rng.sample(ingredients, selected_count))


def generate_examples(recipes, split_name, rng):
    """Generate positive and negative training examples."""

    examples = []

    recipes = recipes.reset_index(drop=True)

    ingredient_sets = recipes["ingredient_set"].tolist()
    titles = recipes["title_clean"].astype(str).tolist()

    for anchor_index, anchor_ingredients in enumerate(ingredient_sets):

        if not anchor_ingredients:
            continue

        user_ingredients = create_user_ingredients(
            anchor_ingredients,
            rng,
        )

        candidate_indices = [
            index
            for index in range(len(recipes))
            if index != anchor_index
        ]

        if not candidate_indices:
            continue

        negative_count = min(
            NEGATIVE_EXAMPLES_PER_RECIPE,
            len(candidate_indices),
        )

        negative_indices = rng.sample(
            candidate_indices,
            negative_count,
        )

        candidates = [(anchor_index, 1)]

        candidates.extend(
            (index, 0)
            for index in negative_indices
        )

        for candidate_index, label in candidates:

            candidate_ingredients = ingredient_sets[candidate_index]

            features = calculate_features(
                user_ingredients,
                candidate_ingredients,
            )

            example = {
                "split": split_name,
                "user_ingredients": json.dumps(
                    sorted(user_ingredients)
                ),
                "candidate_title": titles[candidate_index],
                "label": label,
                **features,
            }

            examples.append(example)

    return pd.DataFrame(examples)


def split_recipes(df):
    """Split recipes by title group to reduce data leakage."""

    groups = (
        df["title_clean"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=RANDOM_SEED,
    )

    train_indices, remaining_indices = next(
        first_split.split(df, groups=groups)
    )

    train_recipes = df.iloc[train_indices].copy()
    remaining_recipes = df.iloc[remaining_indices].copy()

    remaining_groups = groups.iloc[remaining_indices]

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=RANDOM_SEED,
    )

    validation_indices, test_indices = next(
        second_split.split(
            remaining_recipes,
            groups=remaining_groups,
        )
    )

    validation_recipes = remaining_recipes.iloc[
        validation_indices
    ].copy()

    test_recipes = remaining_recipes.iloc[
        test_indices
    ].copy()

    return train_recipes, validation_recipes, test_recipes


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_FILE}\n"
            "Run src/preprocessing.py first."
        )

    OUTPUT_FILE.mkdir(parents=True, exist_ok=True)

    rng = random.Random(RANDOM_SEED)

    df = pd.read_csv(INPUT_FILE, dtype=str)

    if "title_clean" not in df.columns:
        raise ValueError(
            "The cleaned dataset must contain 'title_clean'."
        )

    ingredient_column = (
        "ingredients_normalized"
        if "ingredients_normalized" in df.columns
        else "ingredients_list"
    )

    if ingredient_column not in df.columns:
        raise ValueError(
            "No ingredient column was found. Expected "
            "'ingredients_normalized' or 'ingredients_list'. "
            f"Available columns: {df.columns.tolist()}"
        )

    df["ingredient_set"] = df[ingredient_column].apply(
        parse_ingredients
    )

    original_count = len(df)

    df = df[
        df["ingredient_set"].map(len) > 0
    ].copy()

    print("========== DATA PREPARATION ==========")
    print(f"Recipes loaded: {original_count}")
    print(f"Recipes with usable ingredients: {len(df)}")
    print(f"Ingredient column used: {ingredient_column}")

    train_recipes, validation_recipes, test_recipes = (
        split_recipes(df)
    )

    print("\n========== RECIPE SPLITS ==========")
    print(f"Training recipes: {len(train_recipes)}")
    print(f"Validation recipes: {len(validation_recipes)}")
    print(f"Test recipes: {len(test_recipes)}")

    train_examples = generate_examples(
        train_recipes,
        "train",
        rng,
    )

    validation_examples = generate_examples(
        validation_recipes,
        "validation",
        rng,
    )

    test_examples = generate_examples(
        test_recipes,
        "test",
        rng,
    )

    datasets = {
        "training_examples.csv": train_examples,
        "validation_examples.csv": validation_examples,
        "test_examples.csv": test_examples,
    }

    for filename, dataset in datasets.items():
        output_path = OUTPUT_FILE / filename
        dataset.to_csv(output_path, index=False)

        print(f"\nSaved: {output_path}")
        print(f"Examples: {len(dataset)}")

        if not dataset.empty:
            print(
                "Label distribution:",
                dataset["label"].value_counts().sort_index().to_dict(),
            )

    print("\n========== FEATURE COLUMNS ==========")

    if not train_examples.empty:
        print(
            train_examples[
                [
                    "jaccard_similarity",
                    "user_coverage",
                    "recipe_coverage",
                    "matched_ingredients",
                    "missing_ingredients",
                    "user_ingredient_count",
                    "recipe_ingredient_count",
                    "label",
                ]
            ].head()
        )

    print("\nTraining-data generation completed.")


if __name__ == "__main__":
    main()
