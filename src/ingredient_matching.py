import re
from functools import lru_cache


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
    "black-eyed peas": "cowpea",
    "black eyed beans": "cowpea",
    "black-eyed beans": "cowpea",
    "cow peas": "cowpea",
    "cow pea": "cowpea",
    "minced beef": "ground beef",
    "minced meat": "ground meat",
    "icing sugar": "powdered sugar",
    "confectioners sugar": "powdered sugar",
    "all purpose flour": "all-purpose flour",
    "plain flour": "all-purpose flour",
    "baking soda": "bicarbonate of soda",
    "cow milk": "milk",
    "gajjar": "carrot",
    "cow peas": "cowpea",
    "cow pea": "cowpea",
    "cowpeas": "cowpea",
    "black eyed peas": "cowpea",
    "black eyed pea": "cowpea",
    "black-eyed peas": "cowpea",
    "black-eyed pea": "cowpea",
    "black eyed beans": "cowpea",
    "black eyed bean": "cowpea",
    "black-eyed beans": "cowpea",
    "black-eyed bean": "cowpea",
    "lobia": "cowpea",
}


MILK_TYPES = {
    "milk",
    "whole milk",
    "full cream milk",
    "full fat milk",
    "skimmed milk",
    "skim milk",
    "low fat milk",
    "semi skimmed milk",
    "evaporated milk",
    "condensed milk",
    "powdered milk",
    "milk powder",
    "coconut milk",
    "almond milk",
    "oat milk",
    "soy milk",
    "soya milk",
    "rice milk",
}


PLANT_BASED_MILKS = {
    "coconut milk",
    "almond milk",
    "oat milk",
    "soy milk",
    "soya milk",
    "rice milk",
}


DAIRY_MILKS = {
    "milk",
    "whole milk",
    "full cream milk",
    "full fat milk",
    "skimmed milk",
    "skim milk",
    "low fat milk",
    "semi skimmed milk",
    "evaporated milk",
    "condensed milk",
    "powdered milk",
    "milk powder",
}


def clean_ingredient_name(ingredient):
    """Clean an ingredient name while preserving useful parenthetical aliases."""
    if ingredient is None:
        return ""

    text = str(ingredient).lower().strip()

    text = text.replace("&", " and ")
    text = text.replace("_", " ")
    text = text.replace("-", " ")

    text = re.sub(r"[^\w\s()]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def singularize_ingredient(ingredient):
    """Apply basic plural normalization to common ingredient names."""
    words = ingredient.split()

    if not words:
        return ""

    last_word = words[-1]

    if last_word.endswith("ies") and len(last_word) > 4:
        words[-1] = last_word[:-3] + "y"
    elif (
        last_word.endswith("s")
        and not last_word.endswith("ss")
        and len(last_word) > 3
    ):
        words[-1] = last_word[:-1]

    return " ".join(words)


def canonicalize_ingredient(ingredient):
    """
    Convert common ingredient variations to a standard name.

    Parenthetical names are checked separately so that names such as
    'black eyed peas (lobia)' can still be recognized.
    """
    cleaned = clean_ingredient_name(ingredient)

    if not cleaned:
        return ""

    parenthetical_names = re.findall(r"\((.*?)\)", cleaned)
    main_name = re.sub(r"\(.*?\)", " ", cleaned)
    main_name = re.sub(r"\s+", " ", main_name).strip()

    candidates = [main_name] + parenthetical_names

    for candidate in candidates:
        candidate = clean_ingredient_name(candidate)
        candidate = re.sub(r"[()]", "", candidate).strip()

        if candidate in INGREDIENT_ALIASES:
            return INGREDIENT_ALIASES[candidate]

        singular_candidate = singularize_ingredient(candidate)

        if singular_candidate in INGREDIENT_ALIASES:
            return INGREDIENT_ALIASES[singular_candidate]

    main_name = re.sub(r"[()]", "", main_name).strip()

    if main_name in INGREDIENT_ALIASES:
        return INGREDIENT_ALIASES[main_name]

    main_name = singularize_ingredient(main_name)

    return INGREDIENT_ALIASES.get(main_name, main_name)


def are_ingredients_equivalent(ingredient_a, ingredient_b):
    """Check whether two ingredient names refer to the same ingredient."""
    if not ingredient_a or not ingredient_b:
        return False

    name_a = canonicalize_ingredient(ingredient_a)
    name_b = canonicalize_ingredient(ingredient_b)

    if not name_a or not name_b:
        return False

    if name_a == name_b:
        return True

    return False


def get_milk_type(ingredient):
    """
    Classify recognized milk names.

    Returns:
        'plant_based', 'dairy', or None.
    """
    name = clean_ingredient_name(ingredient)
    name = re.sub(r"[()]", "", name).strip()
    name = singularize_ingredient(name)

    if name in PLANT_BASED_MILKS:
        return "plant_based"

    if name in DAIRY_MILKS:
        return "dairy"

    return None


def can_match_ingredients(user_ingredient, recipe_ingredient):
    """
    Decide whether two ingredient names may be treated as a match.

    Exact canonical matches are preferred. Milk types are kept distinct
    unless their canonical names are equivalent.
    """
    user_name = canonicalize_ingredient(user_ingredient)
    recipe_name = canonicalize_ingredient(recipe_ingredient)

    if not user_name or not recipe_name:
        return False

    if user_name == recipe_name:
        return True

    user_milk_type = get_milk_type(user_ingredient)
    recipe_milk_type = get_milk_type(recipe_ingredient)

    if user_milk_type or recipe_milk_type:
        return False

    return False


@lru_cache(maxsize=1)
def get_semantic_model():
    """
    Load the sentence-transformer model only when semantic matching is
    needed. Install sentence-transformers to enable this feature.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def semantic_similarity(ingredient_a, ingredient_b):
    """
    Calculate semantic similarity for investigation or candidate matching.

    Similarity alone does not establish ingredient equivalence.
    Use can_match_ingredients() for accepted matches.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    name_a = clean_ingredient_name(ingredient_a)
    name_b = clean_ingredient_name(ingredient_b)

    if not name_a or not name_b:
        return 0.0

    if can_match_ingredients(name_a, name_b):
        return 1.0

    model = get_semantic_model()
    embeddings = model.encode([name_a, name_b])

    return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])