from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 80)


DATA_DIR = Path.cwd()
MOVIES_FILE = DATA_DIR / "tmdb_5000_movies.csv"
CREDITS_FILE = DATA_DIR / "tmdb_5000_credits.csv"

missing_files = [path.name for path in (MOVIES_FILE, CREDITS_FILE) if not path.exists()]
if missing_files:
    raise FileNotFoundError(f"Missing required file(s): {', '.join(missing_files)}")

movies_raw = pd.read_csv(MOVIES_FILE)
credits_raw = pd.read_csv(CREDITS_FILE)

print(f"Movies:  {movies_raw.shape[0]:,} rows x {movies_raw.shape[1]} columns")
print(f"Credits: {credits_raw.shape[0]:,} rows x {credits_raw.shape[1]} columns")

 
movies_raw.head(3)
credits_raw.head(3)

def parse_json_list(value: object) -> list[dict]:
    """Return a parsed list for TMDB JSON-like cells; invalid or empty cells become []."""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def names_from_json(value: object, limit: int | None = None) -> list[str]:
    names = [item.get("name", "").strip() for item in parse_json_list(value) if item.get("name")]
    return names[:limit] if limit else names


def director_from_crew(value: object) -> str:
    for person in parse_json_list(value):
        if person.get("job") == "Director":
            return person.get("name", "").strip()
    return ""


def normalize_token(text: str) -> str:
    """Normalize a phrase into a stable token without losing the human-readable name."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def weighted_tokens(row: pd.Series) -> list[str]:
    tokens: list[str] = []

    tokens.extend(f"genre:{normalize_token(name)}" for name in row["genres"])
    tokens.extend(f"keyword:{normalize_token(name)}" for name in row["keywords"])

    # Lead cast and director are stronger taste signals than loose keywords.
    for name in row["cast"]:
        tokens.extend([f"cast:{normalize_token(name)}"] * 2)

    if row["director"]:
        tokens.extend([f"director:{normalize_token(row['director'])}"] * 3)

    return [token for token in tokens if not token.endswith(":")]

credits = credits_raw.copy()
credits["cast"] = credits["cast"].apply(lambda value: names_from_json(value, limit=4))
credits["director"] = credits["crew"].apply(director_from_crew)

movies = movies_raw.merge(
    credits[["movie_id", "cast", "director"]],
    left_on="id",
    right_on="movie_id",
    how="left",
)

movies = movies.assign(
    genres=lambda df: df["genres"].apply(names_from_json),
    keywords=lambda df: df["keywords"].apply(names_from_json),
    cast=lambda df: df["cast"].apply(lambda value: value if isinstance(value, list) else []),
    director=lambda df: df["director"].fillna(""),
)

movies = movies.loc[
    (movies["vote_average"] > 0) & (movies["director"].str.len() > 0),
    ["id", "original_title", "genres", "keywords", "cast", "director", "vote_average", "vote_count", "popularity"],
].reset_index(drop=True)

movies["feature_tokens"] = movies.apply(weighted_tokens, axis=1)
movies["token_counter"] = movies["feature_tokens"].apply(Counter)
movies["token_norm"] = movies["token_counter"].apply(lambda counts: float(np.sqrt(sum(value * value for value in counts.values()))))
movies["new_id"] = np.arange(len(movies))

print(f"Clean movie table: {movies.shape[0]:,} movies")
movies.head(5)

top_genres = (
    movies.explode("genres")
    .query("genres != ''")
    ["genres"]
    .value_counts()
    .head(10)
    .rename_axis("genre")
    .reset_index(name="movies")
)

top_genres

top_directors = (
    movies.loc[movies["director"].ne(""), "director"]
    .value_counts()
    .head(10)
    .rename_axis("director")
    .reset_index(name="movies")
)

top_directors


def cosine_similarity(counter_a: Counter, norm_a: float, counter_b: Counter, norm_b: float) -> float:
    if norm_a == 0 or norm_b == 0:
        return 0.0

    if len(counter_a) > len(counter_b):
        counter_a, counter_b = counter_b, counter_a

    dot_product = sum(weight * counter_b.get(token, 0) for token, weight in counter_a.items())
    return float(dot_product / (norm_a * norm_b))


def find_movie(title: str) -> pd.Series:
    if not title or not title.strip():
        raise ValueError("Please provide a movie title.")

    query = title.strip().casefold()
    titles = movies["original_title"].str.casefold()

    exact_match = titles.eq(query)
    if exact_match.any():
        return movies.loc[exact_match].iloc[0]

    partial = movies.loc[titles.str.contains(query, regex=False, na=False)].copy()
    if not partial.empty:
        partial["match_start"] = partial["original_title"].str.casefold().str.find(query)
        partial["title_length"] = partial["original_title"].str.len()
        return partial.sort_values(["match_start", "title_length", "vote_count"], ascending=[True, True, False]).iloc[0]

    suggestions = movies.loc[
        titles.str.startswith(query[:1], na=False),
        "original_title",
    ].head(5).tolist()
    hint = f" Try one of these: {suggestions}" if suggestions else ""
    raise LookupError(f"No movie found for '{title}'.{hint}")


def recommend_movies(title: str, k: int = 10) -> pd.DataFrame:
    base_movie = find_movie(title)
    base_counter = base_movie["token_counter"]
    base_norm = base_movie["token_norm"]

    scores = []
    for _, candidate in movies.iterrows():
        if candidate["new_id"] == base_movie["new_id"]:
            continue
        similarity = cosine_similarity(base_counter, base_norm, candidate["token_counter"], candidate["token_norm"])
        scores.append((candidate["new_id"], similarity))

    nearest_ids = [movie_id for movie_id, _ in sorted(scores, key=lambda item: item[1], reverse=True)[:k]]
    nearest = movies.set_index("new_id").loc[nearest_ids].reset_index()
    nearest["similarity"] = [score for _, score in sorted(scores, key=lambda item: item[1], reverse=True)[:k]]

    columns = ["original_title", "genres", "director", "vote_average", "similarity"]
    return nearest[columns]


def predict_score(title: str, k: int = 10) -> tuple[pd.DataFrame, float]:
    selected = find_movie(title)
    recommendations = recommend_movies(selected["original_title"], k=k)
    predicted_rating = float(recommendations["vote_average"].mean())

    print(f"Selected movie: {selected['original_title']}")
    print(f"Actual rating:   {selected['vote_average']:.2f}")
    print(f"Predicted rating from {k} neighbors: {predicted_rating:.2f}")
    return recommendations, predicted_rating


recommend_movies("The Godfather", k=10)

recommendations, predicted = predict_score("Donnie Darko", k=10)
recommendations


for title in ["Godfather", "Notting Hill", "Despicable Me"]:
    print("-" * 72)
    predict_score(title, k=10)
    print()

assert not movies.empty
assert movies["original_title"].isna().sum() == 0
assert movies["token_norm"].gt(0).any()
assert len(recommend_movies("Avatar", k=5)) == 5

print("All checks passed.")
