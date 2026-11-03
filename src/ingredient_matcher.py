
from pathlib import Path
from difflib import SequenceMatcher
import re

import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# FILE PATHS AND MODEL SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

EMBEDDING_FILE = PROCESSED_DIR / "ingredient_embeddings.npz"
CONTEXT_FILE = PROCESSED_DIR / "ingredient_context.npz"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_CANDIDATES = 5

DISTINGUISHING_MODIFIERS = {
    "coconut", "sweet", "green", "black", "white", "brown",
    "red", "yellow", "purple", "orange", "soy", "almond",
    "oat", "rice", "cashew", "peanut", "sesame", "whole",
    "skimmed", "low-fat", "fat-free", "dried", "fresh",
    "smoked", "salted", "unsalted", "ground", "powdered"
}


# --------------------------------------------------
# TEXT PROCESSING
# --------------------------------------------------

def normalize_ingredient(ingredient):
    ingredient = str(ingredient).lower().strip()
    ingredient = re.sub(r"[_-]+", " ", ingredient)
    ingredient = re.sub(r"\s+", " ", ingredient)
    return ingredient


def get_tokens(ingredient):
    return set(re.findall(r"[a-z0-9]+", normalize_ingredient(ingredient)))


def get_distinguishing_modifiers(ingredient):
    tokens = get_tokens(ingredient)
    return tokens.intersection(DISTINGUISHING_MODIFIERS)


def has_modifier_conflict(query, candidate):
    query_modifiers = get_distinguishing_modifiers(query)
    candidate_modifiers = get_distinguishing_modifiers(candidate)

    return query_modifiers != candidate_modifiers


# --------------------------------------------------
# SIMILARITY CALCULATIONS
# --------------------------------------------------

def calculate_lexical_similarity(query, candidate):
    query = normalize_ingredient(query)
    candidate = normalize_ingredient(candidate)

    sequence_score = SequenceMatcher(
        None, query, candidate
    ).ratio()

    query_tokens = get_tokens(query)
    candidate_tokens = get_tokens(candidate)

    union = query_tokens.union(candidate_tokens)

    if not union:
        token_score = 0.0
    else:
        token_score = (
            len(query_tokens.intersection(candidate_tokens))
            / len(union)
        )

    return (sequence_score + token_score) / 2


def calculate_context_similarity(
    candidate_index,
    known_user_indices,
    context_vectors
):
    if not known_user_indices:
        return 0.0

    candidate_vector = context_vectors[candidate_index]

    if np.linalg.norm(candidate_vector) == 0:
        return 0.0

    similarities = []

    for user_index in known_user_indices:
        if user_index == candidate_index:
            continue

        user_vector = context_vectors[user_index]

        if np.linalg.norm(user_vector) == 0:
            continue

        similarity = float(np.dot(candidate_vector, user_vector))
        similarities.append(max(0.0, similarity))

    if not similarities:
        return 0.0

    return float(np.mean(similarities))


def calculate_context_reliability(frequency):
    return min(float(frequency) / 20.0, 1.0)


# --------------------------------------------------
# INGREDIENT MATCHER
# --------------------------------------------------

class IngredientMatcher:

    def __init__(self):
        print("Loading ingredient data...")

        if not EMBEDDING_FILE.exists():
            raise FileNotFoundError(EMBEDDING_FILE)

        if not CONTEXT_FILE.exists():
            raise FileNotFoundError(CONTEXT_FILE)

        embedding_data = np.load(
            EMBEDDING_FILE,
            allow_pickle=True
        )

        context_data = np.load(
            CONTEXT_FILE,
            allow_pickle=True
        )

        self.ingredients = embedding_data["ingredients"].tolist()
        self.embeddings = embedding_data["embeddings"].astype(
            np.float32
        )

        context_ingredients = context_data["ingredients"].tolist()

        self.context_vectors = context_data[
            "context_vectors"
        ].astype(np.float32)

        self.frequencies = context_data[
            "recipe_frequencies"
        ].astype(np.float32)

        self.ingredient_to_index = {
            normalize_ingredient(ingredient): index
            for index, ingredient in enumerate(self.ingredients)
        }

        self.context_to_index = {
            normalize_ingredient(ingredient): index
            for index, ingredient in enumerate(context_ingredients)
        }

        self.embedding_to_context = {}

        for index, ingredient in enumerate(self.ingredients):
            normalized = normalize_ingredient(ingredient)

            if normalized in self.context_to_index:
                self.embedding_to_context[index] = (
                    self.context_to_index[normalized]
                )

        print("Loading sentence-transformer model...")

        self.model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        norms = np.linalg.norm(
            self.embeddings,
            axis=1,
            keepdims=True
        )

        self.embeddings = np.divide(
            self.embeddings,
            norms,
            out=np.zeros_like(self.embeddings),
            where=norms != 0
        )

        print("Ingredient matcher ready.")
        print(f"Vocabulary size: {len(self.ingredients)}")

    # --------------------------------------------------
    # EMBEDDING GENERATION
    # --------------------------------------------------

    def encode_unknown_ingredient(self, ingredient):
        vector = self.model.encode(
            [ingredient],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        return vector.astype(np.float32)

    # --------------------------------------------------
    # EXACT MATCHING
    # --------------------------------------------------

    def find_exact_matches(self, user_ingredients):
        exact_matches = []
        unknown_ingredients = []

        for ingredient in user_ingredients:
            normalized = normalize_ingredient(ingredient)

            if normalized in self.ingredient_to_index:
                exact_matches.append(
                    self.ingredients[
                        self.ingredient_to_index[normalized]
                    ]
                )
            else:
                unknown_ingredients.append(ingredient)

        return exact_matches, unknown_ingredients

    # --------------------------------------------------
    # CANDIDATE RETRIEVAL
    # --------------------------------------------------

    def find_candidates(
        self,
        query,
        known_user_ingredients,
        top_k=TOP_CANDIDATES
    ):
        query = normalize_ingredient(query)

        if query in self.ingredient_to_index:
            exact = self.ingredients[
                self.ingredient_to_index[query]
            ]

            return [{
                "ingredient": exact,
                "semantic": 1.0,
                "lexical": 1.0,
                "context": 0.0,
                "score": 1.0
            }]

        query_embedding = self.encode_unknown_ingredient(query)

        semantic_scores = self.embeddings @ query_embedding

        known_user_indices = []

        for ingredient in known_user_ingredients:
            normalized = normalize_ingredient(ingredient)

            if normalized in self.context_to_index:
                known_user_indices.append(
                    self.context_to_index[normalized]
                )

        candidates = []

        for index, candidate in enumerate(self.ingredients):
            semantic_score = float(semantic_scores[index])

            lexical_score = calculate_lexical_similarity(
                query,
                candidate
            )

            modifier_conflict = has_modifier_conflict(
                query,
                candidate
            )

            context_score = 0.0
            reliability = 0.0

            if index in self.embedding_to_context:
                context_index = self.embedding_to_context[index]

                context_score = calculate_context_similarity(
                    context_index,
                    known_user_indices,
                    self.context_vectors
                )

                reliability = calculate_context_reliability(
                    self.frequencies[context_index]
                )

            score = (
                0.75 * max(semantic_score, 0.0)
                + 0.15 * lexical_score
                + 0.10 * context_score * reliability
            )

            if modifier_conflict:
                score *= 0.70

            candidates.append({
                "ingredient": candidate,
                "semantic": semantic_score,
                "lexical": lexical_score,
                "context": context_score,
                "score": score
            })

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return candidates[:top_k]

    # --------------------------------------------------
    # DISPLAY MATCHING RESULTS
    # --------------------------------------------------

    def recommend_matches(self, user_ingredients):
        normalized_inputs = [
            normalize_ingredient(item)
            for item in user_ingredients
            if normalize_ingredient(item)
        ]

        exact_matches, unknown_ingredients = (
            self.find_exact_matches(normalized_inputs)
        )

        print("\nEXACT INGREDIENT MATCHES")
        print("=" * 60)

        if exact_matches:
            for ingredient in exact_matches:
                print(f"  {ingredient}")
        else:
            print("  No exact matches found.")

        print("\nPOTENTIAL INGREDIENT MATCHES")
        print("=" * 60)

        for query in unknown_ingredients:
            print(f"\nInput: {query}")

            candidates = self.find_candidates(
                query,
                exact_matches
            )

            for candidate in candidates:
                print(
                    f"  {candidate['ingredient']}\n"
                    f"    Overall:  {candidate['score']:.4f}\n"
                    f"    Semantic: {candidate['semantic']:.4f}\n"
                    f"    Lexical:  {candidate['lexical']:.4f}\n"
                    f"    Context:  {candidate['context']:.4f}"
                )

        print("\nMATCHING COMPLETED.")


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

def main():
    matcher = IngredientMatcher()

    print("\nINGREDIENT MATCHING TEST")
    print("=" * 60)

    user_input = input(
        "Enter ingredients separated by commas: "
    )

    user_ingredients = [
        ingredient.strip()
        for ingredient in user_input.split(",")
        if ingredient.strip()
    ]

    if not user_ingredients:
        print("No ingredients entered.")
        return

    matcher.recommend_matches(user_ingredients)


if __name__ == "__main__":
    main()
