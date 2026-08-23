# Binge Recommender — AI-Powered Movie Recommendation System

A full-stack movie recommendation application that combines **content-based machine learning recommendations** with a **LangChain-powered movie knowledge agent**.

Search for a movie, discover similar titles, and ask the AI agent natural-language questions about movies, web series, actors, directors, genres, plots, and recommendations.

## Live Demo

**Web App:** https://movie-recommendation-system-theta-henna.vercel.app/

**GitHub:** https://github.com/Harsh-duhan/Movie_Recommendation_System

## Features

### Movie Recommendation Engine

The application provides content-based movie recommendations using movie metadata such as:

* Genres
* Keywords
* Lead cast
* Director
* Movie metadata and similarity signals

Users can search for a movie and receive a list of similar movies through the recommendation API.

### AI Movie Knowledge Agent

The project now includes a conversational AI agent built with **LangChain** and **Groq**.

Users can open the **Ask Agent** interface and ask questions such as:

* Who directed Inception?
* Tell me about the cast of The Dark Knight.
* What genre is Parasite?
* Tell me about Interstellar.
* Recommend movies similar to a particular movie.
* Who is the director of a movie?

The agent is specifically scoped to movie and web-series related questions.

### Interactive Web UI

The application includes:

* Movie title search
* Quick movie selections
* Configurable recommendation count
* Movie recommendation cards
* Movie metadata
* Interactive AI movie assistant
* Responsive frontend
* FastAPI-powered backend

## Architecture

```text
                    ┌──────────────────────┐
                    │      User Browser    │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        Movie Recommendation           Ask Movie Agent
                 │                           │
                 ▼                           ▼
             FastAPI                    FastAPI
                 │                           │
                 ▼                           ▼
       Content-Based Engine              LangChain
                 │                           │
                 ▼                           ▼
          Movie CSV Data                 Groq LLM
```

## API Endpoints

### Health Check

```http
GET /health
```

### Search Movies

```http
GET /api/search?q=godfather&limit=8
```

### Get Recommendations

```http
GET /api/recommend?title=The%20Godfather&limit=10
```

### Ask the AI Movie Agent

```http
POST /api/agent/chat
Content-Type: application/json
```

Request:

```json
{
  "message": "Who directed The Godfather?"
}
```

Response:

```json
{
  "answer": "..."
}
```

FastAPI interactive API documentation is available at:

```text
/docs
```

when the application is running.

## Tech Stack

| Layer            | Technology                   |
| ---------------- | ---------------------------- |
| Backend          | FastAPI                      |
| Machine Learning | Python, Pandas, NumPy        |
| Recommendation   | Content-Based Similarity     |
| AI Agent         | LangChain                    |
| LLM Provider     | Groq                         |
| Frontend         | HTML, CSS, JavaScript        |
| Data             | Movie & Credits CSV datasets |
| Server           | Uvicorn                      |
| Containerization | Docker                       |
| Deployment       | Render / Vercel              |

The current dependency configuration includes FastAPI, Uvicorn, Pandas, NumPy, python-dotenv, LangChain, and LangChain-Groq.

## Project Structure

```text
Movie_Recommendation_System/
│
├── app/
│   ├── main.py
│   ├── recommender.py
│   │
│   └── static/
│       ├── index.html
│       ├── styles.css
│       └── app.js
│
├── core_code.py
├── movie_recommendation_modern.py
├── movie-ratings-and-recommendation-using-knn.ipynb
│
├── movies_5000.csv
├── movie_credits_5000.csv
│
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── render.yaml
└── Procfile
```

## Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Harsh-duhan/Movie_Recommendation_System.git
cd Movie_Recommendation_System
```

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the AI Agent

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

`GROQ_API_KEY` is required for the AI movie agent.

Never commit your API key to GitHub.

### 5. Start the Application

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Docker

Build the image:

```bash
docker build -t binge-recommender .
```

Run the container:

```bash
docker run --env-file .env -p 8000:8000 binge-recommender
```

Open:

```text
http://127.0.0.1:8000
```

The Docker configuration installs the application dependencies, copies the FastAPI application and movie datasets, exposes port `8000`, and starts Uvicorn.

## Environment Variables

| Variable       | Required         | Description                  |
| -------------- | ---------------- | ---------------------------- |
| `GROQ_API_KEY` | Yes for AI Agent | API key for Groq             |
| `GROQ_MODEL`   | Optional         | Groq model used by the agent |

Default model:

```text
openai/gpt-oss-120b
```

## How the AI Agent Works

The AI assistant is implemented in `core_code.py`.

The application:

1. Loads environment variables using `python-dotenv`.
2. Initializes a LangChain chat model through Groq.
3. Applies a movie-focused system prompt.
4. Maintains conversational messages during the running application.
5. Sends the user's question to the configured LLM.
6. Returns the generated answer through the FastAPI `/api/agent/chat` endpoint.

The FastAPI application connects the frontend agent widget to this endpoint.

## Deployment

The repository contains Docker and Render configuration for deployment.

For deployments using the AI agent, configure:

```text
GROQ_API_KEY
GROQ_MODEL
```

as environment variables on the deployment platform.

## Limitations

* Movie data is currently loaded from local CSV files.
* The AI agent requires a Groq API key.
* Agent conversation history is held in application memory and is not persistent.
* The current recommendation system is primarily content-based.
* Production deployments should add authentication, rate limiting, persistent storage, and monitoring.

## Future Improvements

* Hybrid recommendation system
* Collaborative filtering
* Personalized user profiles
* Retrieval-Augmented Generation (RAG)
* Movie posters and trailers
* Ratings and streaming availability
* Persistent AI conversations
* Recommendation explanations
* Agent tool calling
* Automated recommendation evaluation
* Production observability

## Author

**Harsh Duhan**

GitHub: https://github.com/Harsh-duhan

Portfolio: https://harsh-duhan-portfolio.vercel.app/
