from __future__ import annotations

import os
import re
from pathlib import Path
from threading import Lock
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  
    load_dotenv = None

try:
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
except ImportError: 
    init_chat_model = None
    AIMessage = HumanMessage = SystemMessage = None


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = """
You are a movie and web-series assistant for Binge Recommender.
Answer only questions about movies, web series, actors, directors, genres,
plots, recommendations, or viewing knowledge. If the user asks about anything
else, reply exactly: I can not help with that.
Use original wording and concise, helpful explanations.
""".strip()


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MD_BOLD_ITALIC_RE = re.compile(r"\*{1,3}([^*]+?)\*{1,3}")
_MD_HEADING_RE = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_MD_HR_RE = re.compile(r"^[-*_]{3,}\s*$", re.MULTILINE)
_EXTRA_BLANK_RE = re.compile(r"\n{3,}")


def clean_response(text: str) -> str:
    """Return *text* as clean, human-readable plain text.

    Removes:
    * Markdown bold / italic markers  (**word**, *word*, ***word***)
    * Markdown ATX headings           (## Title  ->  Title)
    * Markdown horizontal rules       (---, ***)
    * HTML tags                       (<b>, <br>, etc.)

    Then collapses excessive blank lines and strips leading/trailing whitespace.
    """
    text = _HTML_TAG_RE.sub("", text)           # strip HTML tags
    text = _MD_BOLD_ITALIC_RE.sub(r"\1", text)  # **bold** / *italic* -> plain
    text = _MD_HEADING_RE.sub("", text)          # ## Heading -> Heading
    text = _MD_HR_RE.sub("", text)               # --- / *** dividers
    text = _EXTRA_BLANK_RE.sub("\n\n", text)     # max one blank line
    return text.strip()


def load_agent_environment() -> None:
    if load_dotenv is not None:
        load_dotenv(ENV_FILE, override=False)


def agent_config_status() -> dict[str, object]:
    load_agent_environment()
    return {
        "env_file": str(ENV_FILE),
        "env_file_exists": ENV_FILE.exists(),
        "groq_api_key_set": bool(os.environ.get("GROQ_API_KEY")),
        "model": os.environ.get("GROQ_MODEL", GROQ_MODEL),
    }


class MovieKnowledgeAgent:
    def __init__(self) -> None:
        load_agent_environment()
        if init_chat_model is None or SystemMessage is None:
            raise RuntimeError(
                "Movie agent dependencies are missing. Install langchain, langchain-groq, and python-dotenv."
            )
        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError("The movie agent cannot answer yet. Try Again Later")

        self.llm = init_chat_model(os.environ.get("GROQ_MODEL", GROQ_MODEL), model_provider="groq")
        self.messages: list[Any] = [SystemMessage(content=SYSTEM_PROMPT)]
        self.lock = Lock()

    def ask(self, prompt: str) -> str:
        question = prompt.strip()
        if not question:
            raise ValueError("Please ask a movie or web-series question.")

        with self.lock:
            self.messages.append(HumanMessage(content=question))
            response = self.llm.invoke(self.messages)
            answer = clean_response(str(getattr(response, "content", response)))
            self.messages.append(AIMessage(content=answer))
            return answer


_agent: MovieKnowledgeAgent | None = None
_agent_lock = Lock()


def get_movie_agent() -> MovieKnowledgeAgent:
    global _agent
    with _agent_lock:
        if _agent is None:
            _agent = MovieKnowledgeAgent()
        return _agent


def ask_movie_agent(prompt: str) -> str:
    return get_movie_agent().ask(prompt)


if __name__ == "__main__":
    print("Movie agent ready. Press Ctrl+C to exit.")
    while True:
        user_prompt = input("Enter movie question: ")
        print(ask_movie_agent(user_prompt))
