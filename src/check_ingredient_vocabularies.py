
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

EMBEDDING_FILE = PROCESSED_DIR / "ingredient_embeddings.npz"
CONTEXT_FILE = PROCESSED_DIR / "ingredient_context.npz"


def main():
    embeddings = np.load(EMBEDDING_FILE, allow_pickle=True)
    contexts = np.load(CONTEXT_FILE, allow_pickle=True)

    embedding_ingredients = set(embeddings["ingredients"].tolist())
    context_ingredients = set(contexts["ingredients"].tolist())

    print("\nVOCABULARY COMPARISON")
    print("=" * 50)

    print(f"Embedding vocabulary: {len(embedding_ingredients)}")
    print(f"Context vocabulary: {len(context_ingredients)}")

    only_embeddings = sorted(embedding_ingredients - context_ingredients)
    only_context = sorted(context_ingredients - embedding_ingredients)

    print(f"\nOnly in embeddings: {len(only_embeddings)}")
    for ingredient in only_embeddings[:20]:
        print(f"  {ingredient}")

    print(f"\nOnly in context: {len(only_context)}")
    for ingredient in only_context[:20]:
        print(f"  {ingredient}")

    print("\nSEARCHING FOR BLACK-EYED PEAS")
    print("=" * 50)

    for ingredient in sorted(embedding_ingredients | context_ingredients):
        if "black" in ingredient.lower() or "lobia" in ingredient.lower():
            print(ingredient)

    print("\nVOCABULARY CHECK COMPLETED.")


if __name__ == "__main__":
    main()
