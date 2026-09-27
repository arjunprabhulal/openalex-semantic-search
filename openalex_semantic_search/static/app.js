const form = document.querySelector("#search-form");
const resultsList = document.querySelector("#results");
const template = document.querySelector("#result-template");
const previewTemplate = document.querySelector("#preview-template");
const notice = document.querySelector("#notice");
const emptyState = document.querySelector("#empty-state");
const summary = document.querySelector("#search-summary");
const skeletons = document.querySelector("#skeletons");
const noResults = document.querySelector("#no-results");
const pager = document.querySelector("#pager");
const pagePrevious = document.querySelector("#page-previous");
const pageNext = document.querySelector("#page-next");
const pageCurrent = document.querySelector("#page-current");
const filtersToggle = document.querySelector("#filters-toggle");
const filtersPanel = document.querySelector("#filters");
const previewRail = document.querySelector("#preview-rail");
const previewSheet = document.querySelector("#preview-sheet");
const queryInput = document.querySelector("#query");
const suggestions = document.querySelector("#search-suggestions");
const format = new Intl.NumberFormat("en-US");

const PAGE_SIZE = 10;
const SUGGESTION_LIMIT = 8;
const SUGGESTION_MIN_CHARS = 3;
const SUGGESTION_DEBOUNCE_MS = 300;
let currentPayload = null;
let currentOffset = 0;
let totalMatches = 0;
let activeCard = null;
let suggestionTimer = null;
let suggestionController = null;
let suggestionResults = [];
let activeSuggestion = -1;

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

const compactCount = (value) =>
  new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value);

function renderCorpusProgress(progress) {
  const panel = document.querySelector("#corpus-progress");
  if (!progress.available) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  document.querySelector("#verified-records").textContent = format.format(
    progress.verified_embedding_records,
  );
  document.querySelector("#indexed-records").textContent = compactCount(progress.indexed_records);
  document.querySelector("#committed-shards").textContent = format.format(
    progress.committed_shards,
  );

  const state = document.querySelector("#corpus-progress-state");
  state.textContent = progress.state === "complete" ? "Complete" : "Live";
  state.classList.toggle("complete", progress.state === "complete");

  const track = document.querySelector("#source-progress");
  const fill = document.querySelector("#source-progress-fill");
  const sourceFiles = document.querySelector("#source-files");
  if (progress.source_progress_percent == null) {
    track.hidden = true;
    sourceFiles.textContent = "Source scan initializing";
  } else {
    const percent = Math.max(0, Math.min(100, progress.source_progress_percent));
    track.hidden = false;
    track.setAttribute("aria-valuenow", String(percent));
    track.setAttribute("aria-valuetext", `${percent}% of snapshot files reached`);
    fill.style.width = `${percent}%`;
    sourceFiles.textContent =
      `Source ${format.format(progress.source_files_reached)} of ${format.format(progress.source_files_total)} files`;
  }

  const checkpoint = document.querySelector("#last-checkpoint");
  if (progress.last_checkpoint_at) {
    const date = new Date(progress.last_checkpoint_at);
    checkpoint.dateTime = progress.last_checkpoint_at;
    checkpoint.textContent = `Checkpoint ${date.toLocaleString([], {
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    })}`;
  } else {
    checkpoint.removeAttribute("datetime");
    checkpoint.textContent = "Waiting for first checkpoint";
  }
}

async function refreshCorpusProgress() {
  try {
    let key = "";
    try {
      key = sessionStorage.getItem("openalex-semantic-search-key") || "";
    } catch {
      /* No storage: the keyed progress summary is simply not shown. */
    }
    if (!key) return;
    const response = await fetch("/index-progress", {
      cache: "no-store",
      headers: { "X-API-Key": key },
    });
    if (!response.ok) return;
    renderCorpusProgress(await response.json());
  } catch {
    /* Search remains usable when optional build telemetry is unavailable. */
  }
}

refreshCorpusProgress();
setInterval(refreshCorpusProgress, 30_000);

/* R-F7 parity: "cited 4,100" — locale grouping, never a raw number. */
const fmtCited = (n) => (n > 0 ? `cited ${format.format(n)}` : null);

const workTypeLabel = (type) => {
  switch (type) {
    case "journal-article":
      return "Journal article";
    case "proceedings-article":
      return "Conference paper";
    case "book-chapter":
      return "Book chapter";
    case "":
      return "Paper";
    default:
      return (type.charAt(0).toUpperCase() + type.slice(1)).replace(/-/g, " ");
  }
};

/* R-F6 parity: 2–3 line abstract snippet for the card. */
const cardSnippet = (abstract, len = 220) => {
  if (!abstract) return "";
  const s = abstract.replace(/\s+/g, " ").trim();
  return s.length > len ? s.slice(0, len).trimEnd() + "…" : s;
};

const storedKey = () => {
  try {
    return sessionStorage.getItem("openalex-semantic-search-key") || "";
  } catch {
    return "";
  }
};

function hideSuggestions() {
  if (suggestionTimer) clearTimeout(suggestionTimer);
  suggestionTimer = null;
  if (suggestionController) suggestionController.abort();
  suggestionController = null;
  suggestionResults = [];
  activeSuggestion = -1;
  suggestions.hidden = true;
  suggestions.replaceChildren();
  queryInput.setAttribute("aria-expanded", "false");
  queryInput.removeAttribute("aria-activedescendant");
}

function paintActiveSuggestion() {
  const options = [...suggestions.querySelectorAll(".search-suggestion")];
  options.forEach((option, index) => {
    option.setAttribute("aria-selected", String(index === activeSuggestion));
  });
  const active = options[activeSuggestion];
  if (active) {
    queryInput.setAttribute("aria-activedescendant", active.id);
    active.scrollIntoView({ block: "nearest" });
  } else {
    queryInput.removeAttribute("aria-activedescendant");
  }
}

function chooseSuggestion(paper) {
  queryInput.value = paper.title;
  hideSuggestions();
  currentPayload = buildPayload();
  runSearch(0);
}

function renderSuggestions(papers) {
  suggestionResults = papers;
  activeSuggestion = -1;
  suggestions.replaceChildren();
  if (!papers.length) {
    hideSuggestions();
    return;
  }
  papers.forEach((paper, index) => {
    const option = document.createElement("button");
    option.type = "button";
    option.id = `search-suggestion-${index}`;
    option.className = "search-suggestion";
    option.setAttribute("role", "option");
    option.setAttribute("aria-selected", "false");

    const title = document.createElement("span");
    title.className = "search-suggestion-title";
    title.textContent = paper.title;
    const meta = document.createElement("span");
    meta.className = "search-suggestion-meta";
    meta.textContent = [paper.authors.slice(0, 2).join(", "), paper.publication_year || null]
      .filter(Boolean)
      .join(" · ");
    const cited = document.createElement("span");
    cited.className = "search-suggestion-cited";
    cited.textContent = paper.cited_by_count
      ? `${format.format(paper.cited_by_count)} citations`
      : "Not yet cited";
    option.append(title, meta, cited);
    option.addEventListener("click", () => chooseSuggestion(paper));
    suggestions.appendChild(option);
  });
  suggestions.hidden = false;
  queryInput.setAttribute("aria-expanded", "true");
}

async function fetchSuggestions(query) {
  const keyField = document.querySelector("#api-key");
  const key = keyField.value.trim() || storedKey();
  if (!key) {
    hideSuggestions();
    return;
  }
  if (keyField.value.trim()) {
    try {
      sessionStorage.setItem("openalex-semantic-search-key", key);
    } catch {
      /* best-effort */
    }
  }
  if (suggestionController) suggestionController.abort();
  const controller = new AbortController();
  suggestionController = controller;
  suggestions.replaceChildren();
  const status = document.createElement("p");
  status.className = "search-suggestion-status";
  status.textContent = "Finding matching papers…";
  suggestions.appendChild(status);
  suggestions.hidden = false;
  queryInput.setAttribute("aria-expanded", "true");
  try {
    const response = await fetch("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": key },
      body: JSON.stringify({ query, limit: SUGGESTION_LIMIT, sort: "relevance" }),
      signal: controller.signal,
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || `Suggestions failed (${response.status})`);
    if (queryInput.value.trim() === query) renderSuggestions(body.results);
  } catch (error) {
    if (error.name !== "AbortError") hideSuggestions();
  } finally {
    if (suggestionController === controller) suggestionController = null;
  }
}

function scheduleSuggestions() {
  if (suggestionTimer) clearTimeout(suggestionTimer);
  if (suggestionController) suggestionController.abort();
  const query = queryInput.value.trim();
  if (query.length < SUGGESTION_MIN_CHARS) {
    hideSuggestions();
    return;
  }
  suggestionTimer = setTimeout(() => fetchSuggestions(query), SUGGESTION_DEBOUNCE_MS);
}

queryInput.addEventListener("input", scheduleSuggestions);
queryInput.addEventListener("keydown", (event) => {
  if (suggestions.hidden || !suggestionResults.length) return;
  if (event.key === "ArrowDown") {
    event.preventDefault();
    activeSuggestion = (activeSuggestion + 1) % suggestionResults.length;
    paintActiveSuggestion();
  } else if (event.key === "ArrowUp") {
    event.preventDefault();
    activeSuggestion =
      (activeSuggestion - 1 + suggestionResults.length) % suggestionResults.length;
    paintActiveSuggestion();
  } else if (event.key === "Enter" && activeSuggestion >= 0) {
    event.preventDefault();
    chooseSuggestion(suggestionResults[activeSuggestion]);
  } else if (event.key === "Escape") {
    event.preventDefault();
    hideSuggestions();
  }
});
document.addEventListener("pointerdown", (event) => {
  if (!event.target.closest(".find-search-field")) hideSuggestions();
});

const savedPapers = () => {
  try {
    return JSON.parse(localStorage.getItem("openalex-saved") || "{}");
  } catch {
    return {};
  }
};

const persistSaved = (papers) => {
  try {
    localStorage.setItem("openalex-saved", JSON.stringify(papers));
  } catch {
    /* private windows may refuse storage; saving is best-effort */
  }
};

const toggleSaved = (paper) => {
  const papers = savedPapers();
  if (papers[paper.openalex_id]) {
    delete papers[paper.openalex_id];
  } else {
    papers[paper.openalex_id] = {
      id: paper.openalex_id,
      title: paper.title,
      year: paper.publication_year,
      url: paper.view_url,
    };
  }
  persistSaved(papers);
  return Boolean(papers[paper.openalex_id]);
};

const isSaved = (paper) => Boolean(savedPapers()[paper.openalex_id]);

const apaCitation = (paper) => {
  const authors = paper.authors.length ? paper.authors.join(", ") : null;
  const year = paper.publication_year ? `(${paper.publication_year})` : "(n.d.)";
  const venue = paper.venue ? ` ${paper.venue}.` : "";
  const link = paper.doi || paper.view_url;
  return [authors, year + ".", `${paper.title}.`, venue.trim(), link]
    .filter(Boolean)
    .join(" ");
};

const bibtexCitation = (paper) => {
  const key =
    (paper.authors[0] || "openalex").split(" ").pop().toLowerCase().replace(/[^a-z]/g, "") +
    (paper.publication_year || "");
  const entryType = paper.work_type === "article" ? "article" : "misc";
  const lines = [`@${entryType}{${key || "paper"},`, `  title = {${paper.title}},`];
  if (paper.authors.length) lines.push(`  author = {${paper.authors.join(" and ")}},`);
  if (paper.publication_year) lines.push(`  year = {${paper.publication_year}},`);
  if (paper.venue) lines.push(`  journal = {${paper.venue}},`);
  if (paper.doi) lines.push(`  doi = {${paper.doi.replace("https://doi.org/", "")}},`);
  else lines.push(`  url = {${paper.view_url}},`);
  lines.push("}");
  return lines.join("\n");
};

const copyText = async (text, button) => {
  try {
    await navigator.clipboard.writeText(text);
    const original = button.textContent;
    button.textContent = "Copied";
    setTimeout(() => (button.textContent = original), 1500);
  } catch {
    notice.textContent = "Clipboard unavailable in this browser context.";
    notice.hidden = false;
  }
};

/* Preview pane — right rail on wide screens, bottom sheet below (R-K4 parity). */
function closePreview() {
  previewRail.hidden = true;
  previewRail.replaceChildren();
  previewSheet.hidden = true;
  previewSheet.querySelector(".sheet-panel").replaceChildren();
  if (activeCard) activeCard.classList.remove("active");
  activeCard = null;
}

function buildPreview(paper) {
  const node = previewTemplate.content.firstElementChild.cloneNode(true);
  node.querySelector(".preview-title").textContent = paper.title;
  node.querySelector(".preview-meta").textContent = [
    paper.authors.join(", ") || null,
    paper.publication_year || null,
    paper.venue || null,
  ]
    .filter(Boolean)
    .join(" · ");

  const cited = fmtCited(paper.cited_by_count);
  const citedEl = node.querySelector(".preview-cited");
  if (cited) {
    citedEl.textContent = cited;
    citedEl.hidden = false;
  }
  const doiLink = node.querySelector(".preview-doi");
  if (paper.doi) {
    doiLink.href = paper.doi;
    doiLink.hidden = false;
  }
  const sourceLink = node.querySelector(".preview-source");
  if (paper.view_url) {
    sourceLink.href = paper.view_url;
    sourceLink.hidden = false;
  }

  if (paper.snippet) {
    node.querySelector(".preview-abstract-wrap").hidden = false;
    node.querySelector(".preview-abstract").textContent = paper.snippet;
  } else {
    /* Q2 parity: never leave a blank pane. */
    node.querySelector(".preview-no-abstract").hidden = false;
  }

  const save = node.querySelector(".preview-save");
  const paintSave = () => {
    const saved = isSaved(paper);
    save.textContent = saved ? "Saved ✓" : "Save";
    save.classList.toggle("saved", saved);
  };
  paintSave();
  save.addEventListener("click", () => {
    toggleSaved(paper);
    paintSave();
    paintCardSave(paper);
  });

  node.querySelector(".preview-view").href = paper.view_url;
  node.querySelector(".preview-close").addEventListener("click", closePreview);
  return node;
}

function showPreview(paper, card) {
  if (activeCard) activeCard.classList.remove("active");
  activeCard = card;
  card.classList.add("active");
  previewRail.replaceChildren(buildPreview(paper));
  previewRail.hidden = false;
  previewSheet.querySelector(".sheet-panel").replaceChildren(buildPreview(paper));
  previewSheet.hidden = false;
}

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closePreview();
});
previewSheet.querySelector(".sheet-backdrop").addEventListener("click", closePreview);

const cardSaveButtons = new Map();
const paintCardSave = (paper) => {
  const button = cardSaveButtons.get(paper.openalex_id);
  if (!button) return;
  const saved = isSaved(paper);
  button.textContent = saved ? "Saved ✓" : "Save";
  button.classList.toggle("saved", saved);
};

function renderPaper(paper) {
  const item = template.content.firstElementChild.cloneNode(true);
  const card = item.querySelector(".result");
  item.querySelector(".work-type").textContent = workTypeLabel(paper.work_type);

  const sourceBadge = item.querySelector(".source-badge");
  if (paper.doi) {
    sourceBadge.href = paper.doi;
    sourceBadge.title = "Indexed by OpenAlex (CC0). Opens the publisher record via DOI.";
    sourceBadge.hidden = false;
    sourceBadge.addEventListener("click", (e) => e.stopPropagation());
  }
  if (paper.is_oa) item.querySelector(".trust").hidden = false;
  const cited = fmtCited(paper.cited_by_count);
  if (cited) {
    const citedEl = item.querySelector(".cited");
    citedEl.textContent = cited;
    citedEl.title =
      "Citation count reported by OpenAlex metadata. Counts vary by index and update on different schedules.";
    citedEl.hidden = false;
  }

  const title = item.querySelector(".title");
  title.textContent = paper.title;
  title.setAttribute("aria-label", `Preview ${paper.title}`);

  item.querySelector(".authors").textContent = paper.authors.join(", ") || "Unknown authors";
  const publication = item.querySelector(".publication");
  const publicationText = [paper.publication_year || null, paper.venue || null]
    .filter(Boolean)
    .join(" · ");
  if (publicationText) {
    publication.textContent = publicationText;
    publication.hidden = false;
  }

  const snip = cardSnippet(paper.snippet);
  if (snip) {
    const abstract = item.querySelector(".abstract");
    abstract.textContent = snip;
    abstract.hidden = false;
  }

  const open = () => showPreview(paper, card);
  card.addEventListener("click", open);
  card.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.target === card) open();
  });
  title.addEventListener("click", (e) => {
    e.stopPropagation();
    open();
  });

  const viewAction = item.querySelector(".view-action");
  viewAction.href = paper.view_url;
  viewAction.title = "Open the paper in a new tab";
  viewAction.addEventListener("click", (e) => e.stopPropagation());

  const save = item.querySelector(".save-action");
  cardSaveButtons.set(paper.openalex_id, save);
  paintCardSave(paper);
  save.addEventListener("click", (e) => {
    e.stopPropagation();
    toggleSaved(paper);
    paintCardSave(paper);
  });

  const citePanel = item.querySelector(".cite-panel");
  const cite = item.querySelector(".cite-action");
  cite.setAttribute("aria-expanded", "false");
  cite.addEventListener("click", (e) => {
    e.stopPropagation();
    const opening = citePanel.hidden;
    citePanel.hidden = !opening;
    cite.setAttribute("aria-expanded", String(opening));
    if (opening) {
      citePanel.querySelector(".cite-apa").textContent = apaCitation(paper);
      citePanel.querySelector(".cite-bibtex").textContent = bibtexCitation(paper);
    }
  });
  citePanel.addEventListener("click", (e) => e.stopPropagation());
  citePanel
    .querySelector(".copy-apa")
    .addEventListener("click", (e) => copyText(apaCitation(paper), e.target));
  citePanel
    .querySelector(".copy-bibtex")
    .addEventListener("click", (e) => copyText(bibtexCitation(paper), e.target));

  return item;
}

async function fetchPage(offset) {
  const field = document.querySelector("#api-key");
  const key = field.value.trim() || storedKey();
  if (field.value.trim()) {
    try {
      sessionStorage.setItem("openalex-semantic-search-key", key);
    } catch {
      /* best-effort */
    }
  }
  const response = await fetch("/search", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-API-Key": key },
    body: JSON.stringify({ ...currentPayload, limit: PAGE_SIZE, offset }),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || `Search failed (${response.status})`);
  return body;
}

function applyResponse(body) {
  emptyState.hidden = true;
  summary.hidden = false;
  resultsList.replaceChildren();
  cardSaveButtons.clear();
  closePreview();
  body.results.forEach((paper) => resultsList.appendChild(renderPaper(paper)));
  currentOffset = body.offset;
  totalMatches = body.total_matches;
  const firstShown = body.results.length ? currentOffset + 1 : 0;
  const lastShown = currentOffset + body.results.length;
  const currentPage = Math.floor(currentOffset / PAGE_SIZE) + 1;
  const pageCount = Math.max(1, Math.ceil(totalMatches / PAGE_SIZE));

  const count = document.querySelector("#results-heading");
  count.textContent = `Top ${format.format(totalMatches)} ${totalMatches === 1 ? "match" : "matches"}`;
  document.querySelector("#search-summary-shown").textContent =
    `Showing ${format.format(firstShown)}–${format.format(lastShown)} of ${format.format(totalMatches)} · ${PAGE_SIZE} per page`;
  document.querySelector("#search-details-state").textContent =
    `${body.elapsed_ms.toFixed(0)} ms`;
  document.querySelector("#metric-corpus").textContent = format.format(body.corpus_records);
  document.querySelector("#metric-filtered").textContent = format.format(body.filtered_records);
  document.querySelector("#metric-candidates").textContent = format.format(body.candidates);
  document.querySelector("#metric-time").textContent = `${body.elapsed_ms.toFixed(1)} ms`;

  pager.hidden = totalMatches <= PAGE_SIZE;
  document.querySelector("#pager-note").textContent =
    `${format.format(firstShown)}–${format.format(lastShown)} of ${format.format(totalMatches)} ranked results`;
  pageCurrent.textContent = `Page ${format.format(currentPage)} of ${format.format(pageCount)}`;
  pagePrevious.disabled = currentOffset === 0;
  pageNext.disabled = !body.has_more;

  noResults.hidden = body.results.length !== 0;
  if (body.results.length > 0 && !reducedMotion.matches) {
    count.focus({ preventScroll: true });
    summary.scrollIntoView({ behavior: "smooth", block: "start" });
  } else if (body.results.length > 0) {
    count.focus();
  }
}

async function runSearch(offset) {
  notice.hidden = true;
  resultsList.setAttribute("aria-busy", "true");
  const button = document.querySelector("#search-button");
  button.disabled = true;
  pagePrevious.disabled = true;
  pageNext.disabled = true;
  skeletons.hidden = false;
  noResults.hidden = true;
  resultsList.replaceChildren();
  pager.hidden = true;
  try {
    const body = await fetchPage(offset);
    applyResponse(body);
  } catch (error) {
    notice.textContent = error.message;
    notice.hidden = false;
  } finally {
    skeletons.hidden = true;
    resultsList.setAttribute("aria-busy", "false");
    button.disabled = false;
  }
}

function buildPayload() {
  const payload = {
    query: document.querySelector("#query").value,
    sort: document.querySelector("#sort").value,
  };
  const yearMin = document.querySelector("#year-min").value;
  const yearMax = document.querySelector("#year-max").value;
  const citations = document.querySelector("#min-citations").value;
  if (yearMin) payload.year_min = Number(yearMin);
  if (yearMax) payload.year_max = Number(yearMax);
  if (citations) payload.min_citations = Number(citations);
  if (document.querySelector("#oa-only").checked) payload.open_access_only = true;
  return payload;
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  hideSuggestions();
  currentPayload = buildPayload();
  runSearch(0);
});

pagePrevious.addEventListener("click", () => {
  if (currentPayload && currentOffset > 0) {
    runSearch(Math.max(0, currentOffset - PAGE_SIZE));
  }
});

pageNext.addEventListener("click", () => {
  if (currentPayload) runSearch(currentOffset + PAGE_SIZE);
});

document.querySelector("#sort").addEventListener("change", () => {
  if (currentPayload) {
    currentPayload = buildPayload();
    runSearch(0);
  }
});

filtersToggle.addEventListener("click", () => {
  const open = filtersPanel.hidden;
  filtersPanel.hidden = !open;
  filtersToggle.setAttribute("aria-expanded", String(open));
});

document.querySelector("#clear-filters").addEventListener("click", () => {
  ["#year-min", "#year-max", "#min-citations"].forEach((selector) => {
    document.querySelector(selector).value = "";
  });
  document.querySelector("#oa-only").checked = false;
  if (currentPayload) {
    currentPayload = buildPayload();
    runSearch(0);
  }
});

document.querySelectorAll(".find-example").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelector("#query").value = button.textContent;
    currentPayload = buildPayload();
    runSearch(0);
  });
});

try {
  const stored = storedKey();
  if (stored) document.querySelector("#api-key").value = stored;
} catch {
  /* best-effort */
}
