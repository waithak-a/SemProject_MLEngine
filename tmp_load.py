import sys
sys.path.insert(0, "src")
import recommend
r = recommend.load_recipes()
print("COUNT", len(r))
print("FIRST", r[0]["title"])
print("INGS", r[0]["ingredients"][:5])