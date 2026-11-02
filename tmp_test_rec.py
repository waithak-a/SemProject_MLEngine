import sys
sys.path.insert(0, "src")
import pandas as pd
import recommend

df = pd.read_csv("data/processed/Recipes_cleaned.csv").head(50)
recipes = []
for _, r in df.iterrows():
    ings = recommend.parse_ingredients(r["ingredients_normalized"])
    recipes.append({"title": str(r["title_clean"]), "ingredients": ings})

out = recommend.recommend_recipes(["milk", "sugar"], recipes, top_n=3)
for rec in out:
    print(rec["title"], rec["matched_count"], rec["matched_pairs"])