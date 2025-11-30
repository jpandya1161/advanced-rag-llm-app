const state = {
  busy: false,
};

const healthBadge = document.querySelector("#healthBadge");
const stats = document.querySelector("#stats");
const providerBadge = document.querySelector("#providerBadge");
const documentCount = document.querySelector("#documentCount");
const documentsList = document.querySelector("#documentsList");
const uploadButton = document.querySelector("#uploadButton");
const sampleButton = document.querySelector("#sampleButton");
const clearButton = document.querySelector("#clearButton");
const evalButton = document.querySelector("#evalButton");
const evalOutput = document.querySelector("#evalOutput");
const documentInput = document.querySelector("#documentInput");
const askForm = document.querySelector("#askForm");
const questionInput = document.querySelector("#questionInput");
const chatLog = document.querySelector("#chatLog");
const sourcesList = document.querySelector("#sourcesList");
const latencyBadge = document.querySelector("#latencyBadge");

function refreshIcons() {
  if (window.lucide) {
    window.lucide.createIcons({ attrs: { "stroke-width": 1.8 } });
  }
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = typeof body === "object" && body.detail ? body.detail : body;
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return body;
}

function setBusy(value) {
  state.busy = value;
  uploadButton.disabled = value;
  sampleButton.disabled = value;
  clearButton.disabled = value;
  evalButton.disabled = value;
  askForm.querySelector("button").disabled = value;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderAnswer(text) {
  return escapeHtml(text).replace(/\[(\d+)\]/g, '<span class="citation-chip">[$1]</span>');
}

function addMessage(role, text) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  article.innerHTML = `
    <div class="message-meta">${role === "user" ? "You" : '<span class="assistant-avatar"><i data-lucide="sparkles"></i></span><span>Knowledge Assistant</span>'}</div>
    <p>${role === "assistant" ? renderAnswer(text) : escapeHtml(text)}</p>
  `;
  chatLog.appendChild(article);
  chatLog.scrollTop = chatLog.scrollHeight;
  refreshIcons();
}

function renderDocuments(documents) {
  documentCount.textContent = documents.length;
  if (!documents.length) {
    documentsList.innerHTML = '<p class="empty-documents">No documents indexed yet.</p>';
    return;
  }
  documentsList.innerHTML = documents
    .map(
      (doc) => `
        <div class="doc-item">
          <span class="doc-icon"><i data-lucide="file-text"></i></span>
          <div>
            <strong>${escapeHtml(doc.title)}</strong>
            <span>${escapeHtml(doc.chunk_count)} chunks · ${escapeHtml(doc.source)}</span>
          </div>
        </div>
      `
    )
    .join("");
  refreshIcons();
}

function renderSources(sources = [], diagnostics = {}) {
  latencyBadge.textContent = diagnostics.latency_ms ? `${diagnostics.latency_ms} ms` : "idle";
  if (!sources.length) {
    sourcesList.innerHTML = `
      <div class="empty-state">
        <span class="empty-icon"><i data-lucide="scan-search"></i></span>
        <p>${diagnostics.refused ? "The evidence gate declined this question." : "No sources returned."}</p>
      </div>
    `;
    refreshIcons();
    return;
  }
  sourcesList.innerHTML = sources
    .map(
      (source) => `
        <div class="source-item">
          <div class="source-topline">
            <span class="source-index">${escapeHtml(source.number)}</span>
            <strong>${escapeHtml(source.title)}</strong>
          </div>
          <span class="source-meta">${escapeHtml(source.source)} · chunk ${escapeHtml(source.position)} · <span class="source-score">score ${escapeHtml(source.score)}</span></span>
          <p>${escapeHtml(source.excerpt)}</p>
        </div>
      `
    )
    .join("");
}

async function refreshHealth() {
  try {
    const health = await api("/api/health");
    healthBadge.textContent = health.status;
    healthBadge.classList.remove("is-offline");
    healthBadge.classList.add("is-online");
    stats.textContent = `${health.documents} documents · ${health.chunks} chunks`;
    providerBadge.innerHTML = `<i data-lucide="bot"></i>${escapeHtml(health.provider || "offline")}`;
    const docs = await api("/api/documents");
    renderDocuments(docs.documents || []);
    refreshIcons();
  } catch (error) {
    healthBadge.textContent = "offline";
    healthBadge.classList.remove("is-online");
    healthBadge.classList.add("is-offline");
    documentsList.innerHTML = `<p class="empty-documents">${escapeHtml(error.message)}</p>`;
    documentCount.textContent = "0";
  }
}

uploadButton.addEventListener("click", async () => {
  const file = documentInput.files[0];
  if (!file) {
    addMessage("assistant", "Choose a document before uploading.");
    return;
  }
  const form = new FormData();
  form.append("file", file);
  setBusy(true);
  try {
    const result = await api("/api/documents", { method: "POST", body: form });
    addMessage("assistant", `Indexed ${result.title} into ${result.chunk_count} chunks. [1]`);
    await refreshHealth();
  } catch (error) {
    addMessage("assistant", error.message);
  } finally {
    setBusy(false);
  }
});

sampleButton.addEventListener("click", async () => {
  setBusy(true);
  try {
    const result = await api("/api/ingest-path", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: "sample_docs", reset: false }),
    });
    addMessage("assistant", `Loaded ${result.count} sample documents into the local index.`);
    await refreshHealth();
  } catch (error) {
    addMessage("assistant", error.message);
  } finally {
    setBusy(false);
  }
});

clearButton.addEventListener("click", async () => {
  setBusy(true);
  try {
    await api("/api/documents", { method: "DELETE" });
    renderSources([], {});
    addMessage("assistant", "The document index has been cleared.");
    await refreshHealth();
  } catch (error) {
    addMessage("assistant", error.message);
  } finally {
    setBusy(false);
  }
});

evalButton.addEventListener("click", async () => {
  setBusy(true);
  evalOutput.textContent = "Running evaluation...";
  try {
    const result = await api("/api/evaluate", { method: "POST" });
    evalOutput.textContent = JSON.stringify(
      {
        cases: result.cases,
        average_fact_recall: result.average_fact_recall,
        citation_or_refusal_rate: result.citation_or_refusal_rate,
        unanswerable_refusal_accuracy: result.unanswerable_refusal_accuracy,
        average_latency_ms: result.average_latency_ms,
      },
      null,
      2
    );
    await refreshHealth();
  } catch (error) {
    evalOutput.textContent = error.message;
  } finally {
    setBusy(false);
  }
});

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  questionInput.value = "";
  addMessage("user", question);
  setBusy(true);
  try {
    const response = await api("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    addMessage("assistant", response.answer);
    renderSources(response.sources, response.diagnostics);
    await refreshHealth();
  } catch (error) {
    addMessage("assistant", error.message);
  } finally {
    setBusy(false);
  }
});

refreshIcons();
refreshHealth();
