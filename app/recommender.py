from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT
MOVIES_FILE_CANDIDATES = (DATA_DIR / "movies_5000.csv", DATA_DIR / "tmdb_5000_movies.csv")
CREDITS_FILE_CANDIDATES = (DATA_DIR / "movie_credits_5000.csv", DATA_DIR / "tmdb_5000_credits.csv")


def first_existing_path(candidates: tuple[Path, ...]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


class MovieNotFoundError(LookupError):
    """Raised when a requested movie title cannot be matched."""


def parse_json_list(value: object) -> list[dict[str, Any]]:
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
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def weighted_tokens(row: pd.Series) -> list[str]:
    tokens: list[str] = []
    tokens.extend(f"genre:{normalize_token(name)}" for name in row["genres"])
    tokens.extend(f"keyword:{normalize_token(name)}" for name in row["keywords"])

    for name in row["cast"]:
        tokens.extend([f"cast:{normalize_token(name)}"] * 2)

    if row["director"]:
        tokens.extend([f"director:{normalize_token(row['director'])}"] * 3)

    return [token for token in tokens if not token.endswith(":")]


def cosine_similarity(counter_a: Counter, norm_a: float, counter_b: Counter, norm_b: float) -> float:
    if norm_a == 0 or norm_b == 0:
        return 0.0

    if len(counter_a) > len(counter_b):
        counter_a, counter_b = counter_b, counter_a

    dot_product = sum(weight * counter_b.get(token, 0) for token, weight in counter_a.items())
    return float(dot_product / (norm_a * norm_b))


class MovieRecommender:
    def __init__(
        self,
        movies_file: Path | None = None,
        credits_file: Path | None = None,
    ) -> None:
        movies_file = movies_file or first_existing_path(MOVIES_FILE_CANDIDATES)
        credits_file = credits_file or first_existing_path(CREDITS_FILE_CANDIDATES)
        self.movies_file = movies_file
        self.credits_file = credits_file
        self.movies = self._load_movies()
        self.movies_by_id = self.movies.set_index("new_id", drop=False)

    def _load_movies(self) -> pd.DataFrame:
        missing_files = [path.name for path in (self.movies_file, self.credits_file) if not path.exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing required file(s): {', '.join(missing_files)}")

        movies_raw = pd.read_csv(self.movies_file)
        credits_raw = pd.read_csv(self.credits_file)

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
            [
                "id",
                "original_title",
                "overview",
                "release_date",
                "genres",
                "keywords",
                "cast",
                "director",
                "vote_average",
                "vote_count",
                "popularity",
            ],
        ].reset_index(drop=True)

        movies["feature_tokens"] = movies.apply(weighted_tokens, axis=1)
        movies["token_counter"] = movies["feature_tokens"].apply(Counter)
        movies["token_norm"] = movies["token_counter"].apply(
            lambda counts: float(np.sqrt(sum(value * value for value in counts.values())))
        )
        movies["new_id"] = np.arange(len(movies))
        return movies

    def stats(self) -> dict[str, Any]:
        top_genres = (
            self.movies.explode("genres")
            .query("genres != ''")
            ["genres"]
            .value_counts()
            .head(8)
            .to_dict()
        )
        return {
            "movies": int(len(self.movies)),
            "genres": len({genre for genres in self.movies["genres"] for genre in genres}),
            "directors": int(self.movies["director"].nunique()),
            "top_genres": top_genres,
        }

    def search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []

        titles = self.movies["original_title"].str.casefold()
        matches = self.movies.loc[titles.str.contains(query.casefold(), regex=False, na=False)].copy()
        if matches.empty:
            return []

        matches["match_start"] = matches["original_title"].str.casefold().str.find(query.casefold())
        matches["title_length"] = matches["original_title"].str.len()
        matches = matches.sort_values(["match_start", "title_length", "vote_count"], ascending=[True, True, False])
        return [self._movie_summary(row) for _, row in matches.head(limit).iterrows()]

    def find_movie(self, title: str) -> pd.Series:
        if not title or not title.strip():
            raise ValueError("Please provide a movie title.")

        query = title.strip().casefold()
        titles = self.movies["original_title"].str.casefold()

        exact_match = titles.eq(query)
        if exact_match.any():
            return self.movies.loc[exact_match].iloc[0]

        partial = self.movies.loc[titles.str.contains(query, regex=False, na=False)].copy()
        if not partial.empty:
            partial["match_start"] = partial["original_title"].str.casefold().str.find(query)
            partial["title_length"] = partial["original_title"].str.len()
            return partial.sort_values(
                ["match_start", "title_length", "vote_count"], ascending=[True, True, False]
            ).iloc[0]

        suggestions = self.search(title[:1], limit=5)
        suggestion_titles = [item["title"] for item in suggestions]
        raise MovieNotFoundError(f"No movie found for '{title}'.", suggestion_titles)

    def recommend(self, title: str, limit: int = 10) -> dict[str, Any]:
        limit = min(max(limit, 1), 25)
        selected = self.find_movie(title)
        scores = []

        for _, candidate in self.movies.iterrows():
            if candidate["new_id"] == selected["new_id"]:
                continue
            similarity = cosine_similarity(
                selected["token_counter"],
                selected["token_norm"],
                candidate["token_counter"],
                candidate["token_norm"],
            )
            scores.append((int(candidate["new_id"]), similarity))

        nearest = sorted(scores, key=lambda item: item[1], reverse=True)[:limit]
        recommendations = []
        for movie_id, similarity in nearest:
            row = self.movies_by_id.loc[movie_id]
            item = self._movie_summary(row)
            item["similarity"] = round(float(similarity), 4)
            recommendations.append(item)

        predicted_rating = float(np.mean([movie["rating"] for movie in recommendations])) if recommendations else None
        return {
            "selected": self._movie_summary(selected),
            "predicted_rating": round(predicted_rating, 2) if predicted_rating is not None else None,
            "recommendations": recommendations,
        }

    def _movie_summary(self, row: pd.Series) -> dict[str, Any]:
        release_date = row.get("release_date")
        year = None
        if isinstance(release_date, str) and len(release_date) >= 4:
            year = release_date[:4]

        return {
            "id": int(row["id"]),
            "title": row["original_title"],
            "year": year,
            "genres": row["genres"],
            "cast": row["cast"],
            "director": row["director"],
            "rating": float(row["vote_average"]),
            "vote_count": int(row["vote_count"]),
            "popularity": float(row["popularity"]),
            "overview": row["overview"] if isinstance(row.get("overview"), str) else "",
        }


@lru_cache(maxsize=1)
def get_recommender() -> MovieRecommender:
    return MovieRecommender()
