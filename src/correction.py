"""
correction.py - the "verification layer" that turns a raw model prediction
into something a pharmacist can safely use.

Three jobs:

1. CONFIDENCE CHECK (human-in-the-loop)
   The model always outputs SOME answer, even for garbage. So we look at its
   probability and decide:
     - ACCEPT            brand probability >= threshold
     - INGREDIENT ONLY   brand is unsure, BUT the top guesses are all brands
                         of the SAME active ingredient (e.g. Napa / Ace /
                         Aceta are all Paracetamol). The patient-safety
                         question "which drug?" is still answered.
     - VERIFY            unsure -> a pharmacist must check.

2. INDIAN EQUIVALENTS
   The handwriting data is from Bangladesh; most brands are not sold in
   India. So we map brand -> active ingredient -> Indian brands with the SAME
   single ingredient (and the same form: cream, shampoo, eye drop...), using
   the A-Z Medicine Dataset of India, with prices. This is how pharmacists
   think about substitutes, and it enables a price comparison.

3. FUZZY TEXT MATCHING
   If a user types what they THINK the prescription says ("Montiar"), we
   find the closest real medicine names ("Montair") using string similarity
   (difflib's SequenceMatcher, a relative of edit distance).
"""
import difflib
import json
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from src import config

MEDICINE_INFO_PATH = config.MODELS_DIR / "brand_to_generic.json"

# Salt / chemical-form words that don't change the active drug.
# "Cetirizine Hydrochloride" and "Cetirizine" are the same medicine for us.
SALT_WORDS = {"hydrochloride", "dihydrate", "sodium", "potassium", "hcl"}

# Form written in brackets in the dataset -> words to look for in Indian names
FORM_KEYWORDS = {
    None: ["tablet", "capsule"],      # default: oral tablets/capsules
    "tablet": ["tablet"],
    "cream": ["cream"],
    "shampoo": ["shampoo"],
    "ophthalmic": ["eye"],
}


# ---------------------------------------------------------------------------
# Brand -> generic mapping (saved as a small JSON so the app doesn't need
# the whole image dataset to run)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def brand_to_generic():
    if MEDICINE_INFO_PATH.exists():
        return json.loads(MEDICINE_INFO_PATH.read_text())
    from src.data_loader import get_brand_to_generic, load_split
    mapping = {str(k): str(v) for k, v in get_brand_to_generic(load_split("train")).items()}
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MEDICINE_INFO_PATH.write_text(json.dumps(mapping, indent=2, sort_keys=True))
    return mapping


# ---------------------------------------------------------------------------
# 1. Confidence check
# ---------------------------------------------------------------------------
def generic_probabilities(probs, classes):
    """Add up brand probabilities per active ingredient.
    e.g. P(Paracetamol) = P(Napa) + P(Ace) + P(Aceta) + ..."""
    mapping = brand_to_generic()
    totals = {}
    for p, brand in zip(probs, classes):
        g = mapping[brand]
        totals[g] = totals.get(g, 0.0) + float(p)
    return dict(sorted(totals.items(), key=lambda kv: -kv[1]))


def analyse_prediction(probs, classes, threshold=config.CONFIDENCE_THRESHOLD, k=config.TOP_K):
    """probs: 1-D array of 78 probabilities. Returns a dict for the app."""
    mapping = brand_to_generic()
    order = np.argsort(probs)[::-1][:k]
    top = [{"brand": classes[i], "generic": mapping[classes[i]], "prob": float(probs[i])}
           for i in order]
    gen_probs = generic_probabilities(probs, classes)
    best_generic, best_generic_prob = next(iter(gen_probs.items()))

    if top[0]["prob"] >= threshold:
        status = "ACCEPT"
        message = f"Confident: {top[0]['brand']} ({top[0]['generic']})."
    elif best_generic_prob >= threshold:
        status = "INGREDIENT_ONLY"
        message = (f"Brand is uncertain, but the active ingredient is very likely "
                   f"{best_generic} ({best_generic_prob:.0%}). Pharmacist should confirm the brand.")
    else:
        status = "VERIFY"
        message = "Uncertain - a pharmacist must verify this word."
    return {"top": top, "status": status, "message": message,
            "generic": best_generic, "generic_prob": best_generic_prob}


# ---------------------------------------------------------------------------
# 2. Indian equivalents
# ---------------------------------------------------------------------------
def parse_generic(generic):
    """'Ketoconazole (Shampoo)' -> (['ketoconazole'], 'shampoo')
       'Cetirizine Hydrochloride' -> (['cetirizine'], None)
       'Vitamin B Complex + Zinc' -> (['vitamin b complex', 'zinc'], None)"""
    form = None
    match = re.search(r"\(([^)]*)\)", generic)
    if match:
        form = match.group(1).strip().lower()
        generic = generic[:match.start()]
    ingredients = []
    for part in generic.split("+"):
        words = [w for w in part.lower().split() if w not in SALT_WORDS]
        if words:
            ingredients.append(" ".join(words))
    return ingredients, form


@lru_cache(maxsize=1)
def load_medicine_db():
    """Load the 254k-row Indian medicine database once and cache it."""
    db = pd.read_csv(config.MEDICINE_DB_PATH)
    db = db[~db["Is_discontinued"]].copy()
    db["composition"] = (db["short_composition1"].fillna("") + " + "
                         + db["short_composition2"].fillna("")).str.strip(" +")
    db["n_ingredients"] = 1 + db["short_composition2"].notna().astype(int)
    db["comp_lower"] = db["composition"].str.lower()
    db["name_lower"] = db["name"].str.lower()
    # "strip of 10 tablets" -> 10, so we can compare price PER TABLET/CAPSULE.
    # (For creams/syrups "tube of 15 gm" is not comparable, so we leave it empty.)
    is_countable = db["pack_size_label"].str.contains(r"tablet|capsule", case=False, na=False)
    units = db["pack_size_label"].str.extract(r"of (\d+)")[0].astype(float)
    db["price_per_unit"] = (db["price(₹)"] / units).where(is_countable).round(2)
    return db


def find_indian_equivalents(generic, max_rows=10):
    """Indian medicines with the SAME active ingredient(s) and form.

    Returns (DataFrame of cheapest options, summary dict)."""
    db = load_medicine_db()
    ingredients, form = parse_generic(generic)
    mask = db["n_ingredients"] == len(ingredients)   # no extra ingredients
    for ing in ingredients:
        mask &= db["comp_lower"].str.contains(ing, regex=False)
    keywords = FORM_KEYWORDS.get(form, [form] if form else FORM_KEYWORDS[None])
    mask &= db["name_lower"].str.contains("|".join(map(re.escape, keywords)))
    found = db[mask]

    summary = {"generic": generic, "ingredients": ingredients, "form": form or "tablet/capsule",
               "n_products": int(len(found)),
               "n_manufacturers": int(found["manufacturer_name"].nunique())}
    if len(found):
        per_unit = found["price_per_unit"].dropna()
        if len(per_unit) >= len(found) / 2:     # tablets/capsules: compare per unit
            prices, basis = per_unit, "per tablet/capsule"
        else:                                   # creams, shampoos, drops: per pack
            prices, basis = found["price(₹)"], "per pack"
        summary.update({"price_basis": basis,
                        "min_price": float(prices.min()),
                        "median_price": float(prices.median()),
                        "max_price": float(prices.max())})
    cols = {"name": "Indian brand", "manufacturer_name": "Manufacturer", "composition": "Composition",
            "pack_size_label": "Pack", "price(₹)": "Price (₹)", "price_per_unit": "₹ per unit"}
    table = (found.sort_values(["price_per_unit", "price(₹)"])
             .head(max_rows)[list(cols)].rename(columns=cols).reset_index(drop=True))
    return table, summary


# ---------------------------------------------------------------------------
# 3. Fuzzy text matching
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _indian_brand_index():
    """first word of each Indian brand name -> one example row index"""
    db = load_medicine_db()
    first = db["name_lower"].str.split().str[0]
    return dict(zip(first[::-1], db.index[::-1]))  # keep first occurrence


def similarity(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def fuzzy_search(text, n=5, cutoff=0.6):
    """Closest medicine names to a (possibly misspelt) word.

    Returns two lists: matches among the 78 dataset brands, and matches among
    ~135k Indian brand names."""
    text = text.strip().lower()
    if not text:
        return [], []
    mapping = brand_to_generic()
    known = {b.lower(): b for b in mapping}
    dataset_matches = [
        {"match": known[m], "active ingredient": mapping[known[m]],
         "similarity": round(similarity(text, m), 2)}
        for m in difflib.get_close_matches(text, list(known), n=n, cutoff=cutoff)
    ]
    index = _indian_brand_index()
    db = load_medicine_db()
    indian_matches = []
    for m in difflib.get_close_matches(text, list(index), n=n, cutoff=cutoff):
        row = db.loc[index[m]]
        indian_matches.append({"match": row["name"], "active ingredient": row["composition"],
                               "similarity": round(similarity(text, m), 2)})
    return dataset_matches, indian_matches
