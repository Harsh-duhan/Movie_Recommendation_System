# TMDB Movie Recommender

A deploy-ready FastAPI app that serves a content-based movie recommendation API and a polished web UI.

## Features

- FastAPI backend with health, search, and recommendation endpoints.
- Static responsive UI served by the same app.
- Local TMDB CSV loading with cached recommender startup.
- Content-based similarity from genres, keywords, lead cast, and director.
- Docker and Render deployment files included.

## Project Structure

```text
app/
  main.py              FastAPI app and routes
  recommender.py       Movie data loading and recommendation engine
  static/
    index.html         Web UI
    styles.css         UI styling
    app.js             Browser interactions
tmdb_5000_movies.csv
tmdb_5000_credits.csv
requirements.txt
Dockerfile
render.yaml
Procfile
```

## Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## API

```text
GET /health
GET /api/search?q=godfather&limit=8
GET /api/recommend?title=The%20Godfather&limit=10
```

## Deploy With Docker

```bash
docker build -t tmdb-movie-recommender .
docker run -p 8000:8000 tmdb-movie-recommender
```

## Deploy On Render

1. Push this folder to a GitHub repository.
2. Create a new Render Blueprint from the repository.
3. Render will use `render.yaml` and the `Dockerfile`.
4. After deploy, visit the service URL. The health check is `/health`.

## Notes

The CSV files are included in the Docker image because the app loads them at startup. For larger datasets, move data to object storage or a database and update `app/recommender.py`.
