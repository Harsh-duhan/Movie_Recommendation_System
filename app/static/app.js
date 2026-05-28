const form = document.querySelector("#recommend-form");
const titleInput = document.querySelector("#movie-title");
const limitSelect = document.querySelector("#limit-select");
const message = document.querySelector("#message");
const selectedCard = document.querySelector("#selected-card");
const recommendations = document.querySelector("#recommendations");
const subtitle = document.querySelector("#recommendation-subtitle");

const posterColors = [
  ["#d3422f", "#e9b44c"],
  ["#147a72", "#243c5a"],
  ["#6f4ca5", "#d3422f"],
  ["#1f2937", "#e9b44c"],
  ["#8a3ffc", "#147a72"],
];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function compactNumber(value) {
  return new Intl.NumberFormat("en", { notation: "compact" }).format(value);
}

function chips(values, limit = 4) {
  return (values || [])
    .slice(0, limit)
    .map((item) => `<span class="chip">${escapeHtml(item)}</span>`)
    .join("");
}

function poster(title, index = 0) {
  const colors = posterColors[index % posterColors.length];
  return `
    <div class="poster-art" style="--poster-a:${colors[0]};--poster-b:${colors[1]}">
      <strong>${escapeHtml(title)}</strong>
    </div>
  `;
}

function setMessage(text, isHidden = false) {
  message.hidden = isHidden;
  message.textContent = text;
}

function renderSelected(movie, predictedRating) {
  selectedCard.innerHTML = `
    ${poster(movie.title)}
    <h2>${escapeHtml(movie.title)}</h2>
    <p class="overview">${escapeHtml(movie.overview || "No overview available.")}</p>
    <div class="meta-line">
      <span class="chip">${escapeHtml(movie.year || "Unknown year")}</span>
      <span class="chip">${escapeHtml(movie.director)}</span>
    </div>
    <div class="genre-list">${chips(movie.genres, 5)}</div>
    <div class="score-strip">
      <div>
        <strong>${movie.rating.toFixed(1)}</strong>
        <span>Actual rating</span>
      </div>
      <div>
        <strong>${predictedRating.toFixed(1)}</strong>
        <span>Neighbor score</span>
      </div>
    </div>
  `;
}

function renderRecommendations(items) {
  recommendations.innerHTML = items
    .map(
      (movie, index) => `
        <article class="movie-card">
          <header>
            <h3>${escapeHtml(movie.title)}</h3>
            <span class="rating">${movie.rating.toFixed(1)}</span>
          </header>
          <p>${escapeHtml(movie.year || "Unknown year")} • ${escapeHtml(movie.director)}</p>
          <div class="genre-list">${chips(movie.genres, 3)}</div>
          <p>${escapeHtml((movie.overview || "").slice(0, 150))}${movie.overview && movie.overview.length > 150 ? "..." : ""}</p>
          <p><strong>${Math.round(movie.similarity * 100)}%</strong> taste overlap</p>
        </article>
      `
    )
    .join("");
}

async function loadStats() {
  const response = await fetch("/health");
  const stats = await response.json();
  document.querySelector("#stat-movies").textContent = compactNumber(stats.movies);
  document.querySelector("#stat-genres").textContent = compactNumber(stats.genres);
  document.querySelector("#stat-directors").textContent = compactNumber(stats.directors);
}

async function recommend(title = titleInput.value) {
  const query = title.trim();
  if (!query) {
    setMessage("Type a movie title first.");
    titleInput.focus();
    return;
  }

  document.body.classList.add("loading");
  setMessage("Finding movies with a similar signal...");

  try {
    const params = new URLSearchParams({ title: query, limit: limitSelect.value });
    const response = await fetch(`/api/recommend?${params}`);
    const payload = await response.json();

    if (!response.ok) {
      const detail = payload.detail || {};
      const suggestions = detail.suggestions?.length ? ` Try: ${detail.suggestions.join(", ")}.` : "";
      throw new Error(`${detail.message || "Movie not found."}${suggestions}`);
    }

    renderSelected(payload.selected, payload.predicted_rating);
    renderRecommendations(payload.recommendations);
    subtitle.textContent = `Because you searched for ${payload.selected.title}.`;
    setMessage("", true);
  } catch (error) {
    setMessage(error.message || "Something went wrong.");
  } finally {
    document.body.classList.remove("loading");
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  recommend();
});

limitSelect.addEventListener("change", () => recommend());

document.querySelectorAll("[data-title]").forEach((button) => {
  button.addEventListener("click", () => {
    titleInput.value = button.dataset.title;
    recommend(button.dataset.title);
  });
});

loadStats().catch(() => {
  document.querySelector("#stat-movies").textContent = "--";
});
recommend();
