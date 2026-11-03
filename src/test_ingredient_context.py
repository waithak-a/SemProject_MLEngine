
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
CONTEXT_FILE = BASE_DIR / "data" / "processed" / "ingredient_context.npz"


def main():
    print("\nTESTING INGREDIENT CONTEXT PROFILES")
    print("=" * 50)

    data = np.load(CONTEXT_FILE, allow_pickle=True)

    ingredients = data["ingredients"]
    context_vectors = data["context_vectors"]
    frequencies = data["recipe_frequencies"]

    print(f"Ingredients: {len(ingredients)}")
    print(f"Context vectors: {context_vectors.shape}")
    print(f"Recipe frequencies: {frequencies.shape}")

    ingredient_to_index = {
        ingredient: index
        for index, ingredient in enumerate(ingredients)
    }

    test_ingredients = [
        "carrot",
        "milk",
        "coconut milk",
        "potato",
        "sweet potato",
        "black eyed peas (lobia)",
    ]

    print("\nINGREDIENT CONTEXT SIMILARITIES")
    print("=" * 50)

    for ingredient in test_ingredients:
        if ingredient not in ingredient_to_index:
            print(f"\n{ingredient}: Not found")
            continue

        index = ingredient_to_index[ingredient]
        vector = context_vectors[index]

        similarities = context_vectors @ vector
        similarities[index] = -1

        top_indices = np.argsort(similarities)[::-1][:5]

        print(f"\n{ingredient}")
        print(f"Appears in {int(frequencies[index])} recipes")

        for candidate_index in top_indices:
            candidate = ingredients[candidate_index]
            score = similarities[candidate_index]

            print(f"  {candidate}: {score:.4f}")

    print("\nCONTEXT PROFILE TEST COMPLETED.")


if __name__ == "__main__":
    main()
