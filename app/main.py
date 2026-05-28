from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .recommender import MovieNotFoundError, MovieRecommender, get_recommender


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

app = FastAPI(
    title="TMDB Movie Recommender",
    description="A deploy-ready FastAPI app for content-based movie recommendations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health(recommender: Annotated[MovieRecommender, Depends(get_recommender)]) -> dict[str, object]:
    return {"status": "ok", **recommender.stats()}


@app.get("/api/search")
def search_movies(
    q: Annotated[str, Query(min_length=1, max_length=80)],
    recommender: Annotated[MovieRecommender, Depends(get_recommender)],
    limit: Annotated[int, Query(ge=1, le=20)] = 8,
) -> dict[str, object]:
    return {"query": q, "results": recommender.search(q, limit=limit)}


@app.get("/api/recommend")
def recommend_movies(
    title: Annotated[str, Query(min_length=1, max_length=120)],
    recommender: Annotated[MovieRecommender, Depends(get_recommender)],
    limit: Annotated[int, Query(ge=1, le=25)] = 10,
) -> dict[str, object]:
    try:
        return recommender.recommend(title, limit=limit)
    except MovieNotFoundError as error:
        suggestions = error.args[1] if len(error.args) > 1 else []
        raise HTTPException(status_code=404, detail={"message": error.args[0], "suggestions": suggestions}) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
