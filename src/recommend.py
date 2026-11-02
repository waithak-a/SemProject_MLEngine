
import ast
import re
from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RECIPE_FILE = Path("data/processed/Recipes_cleaned.csv")

TOP_N = 10

# Semantic matching is enabled, but should be evaluated carefully.
ENABLE_SEMANTIC_MATCHING = True

# Conservative starting threshold, not a guarantee of correctness.
SEMANTIC_THRESHOLD = 0.62

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# INGREDIENT NORMALISATION


PREPARATION_WORDS = {
    "fresh",
    "ripe",
    "raw",
    "cooked",
    "boiled",
    "chopped",
    "diced",
    "sliced",
    "peeled",
    "grated",
    "crushed",
    "frozen",
}

# Ingredient synonyms/aliases. These are foods that should count as the
# same ingredient even though their names differ.
INGREDIENT_ALIASES = {
    "aubergine": "eggplant",
    "eggplants": "eggplant",
    "courgette": "zucchini",
    "courgettes": "zucchini",
    "capsicum": "bell pepper",
    "capsicums": "bell pepper",
    "cilantro": "coriander",
    "coriander leaves": "coriander",
    "scallions": "spring onion",
    "green onions": "spring onion",
    "spring onions": "spring onion",
    "garbanzo beans": "chickpea",
    "garbanzo bean": "chickpea",
    "chickpeas": "chickpea",
    "black eyed peas": "cowpea",
    "black eyed pea": "cowpea",
    "black-eyed peas": "cowpea",
    "black-eyed pea": "cowpea",
    "black eyed beans": "cowpea",
    "black eyed bean": "cowpea",
    "black-eyed beans": "cowpea",
    "black-eyed bean": "cowpea",
    "cow peas": "cowpea",
    "cow pea": "cowpea",
    "cowpeas": "cowpea",
    "lobia": "cowpea",
    "gajjar": "carrot",
    "minced beef": "ground beef",
    "minced meat": "ground meat",
    "icing sugar": "powdered sugar",
    "confectioners sugar": "powdered sugar",
    "all purpose flour": "all-purpose flour",
    "plain flour": "all-purpose flour",
    "baking soda": "bicarbonate of soda",
    "cow milk": "milk",
}

# Foods that are wording-similar but are NOT the same food.
CONFLICTING_INGREDIENT_PAIRS = {
    ("milk", "coconut milk"),
    ("milk", "almond milk"),
    ("milk", "oat milk"),
    ("milk", "soy milk"),
    ("milk", "soya milk"),
    ("milk", "rice milk"),
    ("potato", "sweet potato"),
    ("green peas", "black eyed peas"),
    ("green peas", "cowpea"),
    ("white rice", "brown rice"),
    ("beef", "chicken"),
}

NON_PLURAL_S_WORDS = {
    "asparagus",
    "couscous",
    "hummus",
    "molasses",
}
EMBEDDING_CACHE = {}

SEMANTIC_MODEL = None


def parse_ingredients(value):
    """Convert the dataset ingredient field into a list."""
    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, (list, tuple, set)):
            return [
                str(item).strip()
                for item in parsed
                if str(item).strip()
            ]

    except (ValueError, SyntaxError):
        pass

    if "|" in text:
        return [
            item.strip()
            for item in text.split("|")
            if item.strip()
        ]

    return [text]


def singularize_word(word):
    """Handle common plural ingredient names."""
    if word in NON_PLURAL_S_WORDS:
        return word

    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"

    if len(word) > 4 and word.endswith("oes"):
        return word[:-2]

    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]

    return word


def clean_ingredient_text(ingredient):
    """
    Remove parenthetical descriptions and preparation words
    while retaining the meaningful ingredient name.
    """
    ingredient = str(ingredient).lower().strip()

    ingredient = re.sub(r"\([^)]*\)", "", ingredient)

    ingredient = ingredient.replace("_", " ")
    ingredient = ingredient.replace("-", " ")

    ingredient = re.sub(r"[^a-z0-9\s]", " ", ingredient)
    ingredient = re.sub(r"\s+", " ", ingredient).strip()

    words = ingredient.split()

    while words and words[0] in PREPARATION_WORDS:
        words.pop(0)

    return " ".join(words)


def normalize_ingredient(ingredient):
    """Create a canonical name for exact ingredient matching."""
    cleaned = clean_ingredient_text(ingredient)

    if not cleaned:
        return ""

    # Apply synonym aliases before singularisation.
    if cleaned in INGREDIENT_ALIASES:
        return INGREDIENT_ALIASES[cleaned]

    words = cleaned.split()
    words[-1] = singularize_word(words[-1])

    normalized = " ".join(words)

    if normalized in INGREDIENT_ALIASES:
        return INGREDIENT_ALIASES[normalized]

    return normalized


def normalize_ingredient_list(ingredients):
    """Return unique normalised ingredient names."""
    return {
        normalize_ingredient(item)
        for item in ingredients
        if normalize_ingredient(item)
    }



# SEMANTIC MATCHING


def load_semantic_model():
    """Load the pretrained embedding model when required."""
    global SEMANTIC_MODEL

    if SEMANTIC_MODEL is not None:
        return SEMANTIC_MODEL

    try:
        from sentence_transformers import SentenceTransformer

        SEMANTIC_MODEL = SentenceTransformer(MODEL_NAME)

        return SEMANTIC_MODEL

    except ImportError as error:
        raise RuntimeError(
            "Semantic matching requires sentence-transformers. "
        ) from error


def get_embeddings(ingredient_names):
    """
    Encode ingredient names and cache their embeddings.
    Embeddings are normalised so their dot product gives
    cosine similarity.
    """
    unique_names = list(dict.fromkeys(
        name for name in ingredient_names if name
    ))

    missing_names = [
        name for name in unique_names
        if name not in EMBEDDING_CACHE
    ]

    if missing_names:
        model = load_semantic_model()

        embeddings = model.encode(
            missing_names,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        for name, embedding in zip(missing_names, embeddings):
            EMBEDDING_CACHE[name] = embedding

    return {
        name: EMBEDDING_CACHE[name]
        for name in unique_names
    }


def is_conflicting_milk_match(first, second):
    
    plant_milks = {
        "coconut milk",
        "almond milk",
        "oat milk",
        "soy milk",
        "soya milk",
        "rice milk",
    }

    first = normalize_ingredient(first)
    second = normalize_ingredient(second)

    if first == second:
        return False

    return (
        (first == "milk" and second in plant_milks)
        or (second == "milk" and first in plant_milks)
    )


def is_conflicting_match(first, second):
    """Return True when two ingredients must never be treated as the same food."""
    first_norm = normalize_ingredient(first)
    second_norm = normalize_ingredient(second)

    if first_norm == second_norm:
        return False

    if (first_norm, second_norm) in CONFLICTING_INGREDIENT_PAIRS:
        return True

    if (second_norm, first_norm) in CONFLICTING_INGREDIENT_PAIRS:
        return True

    # Milk families are never interchangeable.
    if is_conflicting_milk_match(first, second):
        return True

    return False


def are_known_synonyms(first, second):
    """Return True only for explicitly accepted synonym pairs."""
    first_norm = normalize_ingredient(first)
    second_norm = normalize_ingredient(second)

    if not first_norm or not second_norm:
        return False

    if first_norm == second_norm:
        return True

    return False
def semantic_similarity(first, second, embeddings):
    """Calculate cosine similarity between two ingredient names."""
    first_text = clean_ingredient_text(first)
    second_text = clean_ingredient_text(second)

    if not first_text or not second_text:
        return 0.0

    first_embedding = embeddings.get(first_text)
    second_embedding = embeddings.get(second_text)

    if first_embedding is None or second_embedding is None:
        return 0.0

    return float(np.dot(first_embedding, second_embedding))



def match_ingredients(user_ingredients, recipe_ingredients):
    """
    Match ingredients in two stages.

    Stage 1: exact normalised matches.
    Stage 2: semantic matches for unmatched ingredients.

    A user ingredient and a recipe ingredient can each be
    matched only once.
    """
    user_items = {}

    for item in user_ingredients:
        normalized = normalize_ingredient(item)

        if normalized:
            user_items.setdefault(normalized, item)

    recipe_items = {}

    for item in recipe_ingredients:
        normalized = normalize_ingredient(item)

        if normalized:
            recipe_items.setdefault(normalized, item)

    unmatched_users = set(user_items)
    unmatched_recipes = set(recipe_items)

    matched_pairs = []

    # Stage 1: exact matches after normalisation.
    exact_matches = sorted(
        unmatched_users & unmatched_recipes
    )

    for name in exact_matches:
        matched_pairs.append({
            "user_ingredient": user_items[name],
            "recipe_ingredient": recipe_items[name],
            "similarity": 1.0,
            "match_type": "Exact",
        })

        unmatched_users.remove(name)
        unmatched_recipes.remove(name)

    # Stage 2: semantic matching for remaining ingredients.
    # Synonym pairs are already exact matches after canonicalisation, so
    # there is normally nothing to do here.  Keep the guard so the
    # embedding model is never loaded unless an accepted synonym pair
    # actually remains unmatched.
    synonym_pairs_exist = any(
        are_known_synonyms(user_items[u], recipe_items[r])
        and not is_conflicting_match(user_items[u], recipe_items[r])
        for u in unmatched_users
        for r in unmatched_recipes
    )

    if (
        ENABLE_SEMANTIC_MATCHING
        and unmatched_users
        and unmatched_recipes
        and synonym_pairs_exist
    ):
        user_candidates = [
            user_items[name]
            for name in unmatched_users
        ]

        recipe_candidates = [
            recipe_items[name]
            for name in unmatched_recipes
        ]

        candidate_texts = [
            clean_ingredient_text(item)
            for item in user_candidates + recipe_candidates
        ]

        embeddings = get_embeddings(candidate_texts)

        possible_matches = []

        for user_name in unmatched_users:
            user_original = user_items[user_name]

            for recipe_name in unmatched_recipes:
                recipe_original = recipe_items[recipe_name]

                # Protect against known ingredient conflicts.
                if is_conflicting_match(
                    user_original,
                    recipe_original,
                ):
                    continue

                # Semantic matching must not merge foods that are merely
                # wording-similar. Only explicitly accepted synonyms may be
                # matched here; all other pairs stay distinct.
                if not are_known_synonyms(
                    user_original,
                    recipe_original,
                ):
                    continue

                score = semantic_similarity(
                    user_original,
                    recipe_original,
                    embeddings,
                )

                if score >= SEMANTIC_THRESHOLD:
                    possible_matches.append(
                        (
                            score,
                            user_name,
                            recipe_name,
                        )
                    )

        # Consider the strongest matches first.
        possible_matches.sort(reverse=True)

        for score, user_name, recipe_name in possible_matches:
            if (
                user_name not in unmatched_users
                or recipe_name not in unmatched_recipes
            ):
                continue

            matched_pairs.append({
                "user_ingredient": user_items[user_name],
                "recipe_ingredient": recipe_items[recipe_name],
                "similarity": score,
                "match_type": "Semantic",
            })

            unmatched_users.remove(user_name)
            unmatched_recipes.remove(recipe_name)

    return matched_pairs


# ============================================================
# RECIPE SCORING AND RANKING
# ============================================================

def calculate_features(user_ingredients, recipe_ingredients):
    """Calculate ingredient overlap and recommendation scores."""
    matched_pairs = match_ingredients(
        user_ingredients,
        recipe_ingredients,
    )

    user_set = normalize_ingredient_list(user_ingredients)
    recipe_set = normalize_ingredient_list(recipe_ingredients)

    matched_count = len(matched_pairs)

    user_coverage = (
        matched_count / len(user_set)
        if user_set else 0.0
    )

    recipe_coverage = (
        matched_count / len(recipe_set)
        if recipe_set else 0.0
    )

    # Use the number of matched pairs and the normalised sets.
    matched_user_names = {
        normalize_ingredient(pair["user_ingredient"])
        for pair in matched_pairs
    }

    matched_recipe_names = {
        normalize_ingredient(pair["recipe_ingredient"])
        for pair in matched_pairs
    }

    union_count = (
        len(user_set)
        + len(recipe_set)
        - len(matched_user_names & matched_recipe_names)
    )

    # For semantic matches, the two normalised names may differ.
    # Compute a practical overlap score from the matched count.
    union_count = max(
        len(user_set) + len(recipe_set) - matched_count,
        0,
    )

    jaccard_similarity = (
        matched_count / union_count
        if union_count else 0.0
    )

    exact_count = sum(
        pair["match_type"] == "Exact"
        for pair in matched_pairs
    )

    semantic_count = sum(
        pair["match_type"] == "Semantic"
        for pair in matched_pairs
    )

    semantic_scores = [
        pair["similarity"]
        for pair in matched_pairs
        if pair["match_type"] == "Semantic"
    ]

    average_semantic_similarity = (
        sum(semantic_scores) / len(semantic_scores)
        if semantic_scores else 1.0
    )

    return {
        "matched_pairs": matched_pairs,
        "matched_count": matched_count,
        "exact_count": exact_count,
        "semantic_count": semantic_count,
        "average_semantic_similarity": average_semantic_similarity,
        "user_coverage": user_coverage,
        "recipe_coverage": recipe_coverage,
        "jaccard_similarity": jaccard_similarity,
    }


def load_recipes():
    """Load recipes from the cleaned CSV dataset."""
    if not RECIPE_FILE.exists():
        raise FileNotFoundError(
            f"Recipe dataset not found: {RECIPE_FILE}"
        )

    df = pd.read_csv(RECIPE_FILE, dtype=str)

    if "ingredients_normalized" not in df.columns:
        raise ValueError(
            "Missing 'ingredients_normalized' column in dataset."
        )

    if "title_clean" in df.columns:
        title_column = "title_clean"
    elif "title" in df.columns:
        title_column = "title"
    else:
        raise ValueError(
            "The dataset does not contain a recipe title column."
        )

    recipes = []

    for _, row in df.iterrows():
        ingredients = parse_ingredients(
            row["ingredients_normalized"]
        )

        if not normalize_ingredient_list(ingredients):
            continue

        title = str(row[title_column]).strip()

        if not title or title.lower() == "nan":
            title = "Untitled Recipe"

        recipes.append({
            "title": title,
            "ingredients": ingredients,
        })

    return recipes


def recommend_recipes(user_ingredients, recipes, top_n=TOP_N):
    """Recommend recipes with the strongest ingredient overlap."""
    if not normalize_ingredient_list(user_ingredients):
        return []

    recommendations = []

    for recipe in recipes:
        features = calculate_features(
            user_ingredients,
            recipe["ingredients"],
        )

        if features["matched_count"] == 0:
            continue

        recommendations.append({
            **recipe,
            **features,
        })

    # More matched ingredients always take priority.
    # Exact matches take priority over semantic matches on ties.
    # Similarity and coverage break remaining ties.
    recommendations.sort(
        key=lambda recipe: (
            recipe["matched_count"],
            recipe["exact_count"],
            recipe["average_semantic_similarity"],
            recipe["user_coverage"],
            recipe["jaccard_similarity"],
            recipe["recipe_coverage"],
        ),
        reverse=True,
    )

    return recommendations[:top_n]



def main():
    print("=" * 50)
    print("          RECIPE RECOMMENDATION SYSTEM")
    print("=" * 50)

    user_input = input(
        "\nEnter ingredients separated by commas: "
    ).strip()

    if not user_input:
        print("Please enter at least one ingredient.")
        return

    user_ingredients = [
        item.strip()
        for item in user_input.split(",")
        if item.strip()
    ]

    normalized_user = normalize_ingredient_list(
        user_ingredients
    )

    print(f"\nIngredients entered: {len(normalized_user)}")
    print("Ingredients:", ", ".join(sorted(normalized_user)))

    try:
        recipes = load_recipes()

        recommendations = recommend_recipes(
            user_ingredients,
            recipes,
        )

    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"\nError: {error}")
        return

    print("\n========== TOP RECOMMENDED RECIPES ==========")

    if not recommendations:
        print("No matching recipes were found.")
        print("Try entering additional ingredients.")
        return

    for index, recipe in enumerate(recommendations, start=1):
        print(f"\n{index}. {recipe['title']}")

        print(
            f"   Matched ingredients: "
            f"{recipe['matched_count']} of {len(normalized_user)}"
        )

        print(
            f"   User ingredient coverage: "
            f"{recipe['user_coverage']:.1%}"
        )

        print(
            f"   Recipe ingredient coverage: "
            f"{recipe['recipe_coverage']:.1%}"
        )

        print(
            f"   Jaccard similarity: "
            f"{recipe['jaccard_similarity']:.1%}"
        )

        print("   Matched ingredient details:")

        for pair in recipe["matched_pairs"]:
            user_item = pair["user_ingredient"]
            recipe_item = pair["recipe_ingredient"]
            match_type = pair["match_type"]

            if match_type == "Semantic":
                print(
                    f"      - {user_item} -> {recipe_item} "
                    f"(semantic similarity: {pair['similarity']:.3f})"
                )
            else:
                print(
                    f"      - {user_item} "
                    f"(dataset: {recipe_item}; exact match)"
                )


if __name__ == "__main__":
    main()
