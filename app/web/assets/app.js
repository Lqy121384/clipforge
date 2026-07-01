const state = {
  apiBase: localStorage.getItem("clipforge.apiBase") || "/api/v1",
  apiKey: localStorage.getItem("clipforge.apiKey") || "",
  tenantId: localStorage.getItem("clipforge.tenantId") || "default",
  collection: localStorage.getItem("clipforge.collection") || "main",
  imageBase64: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (state.apiKey) headers["X-API-Key"] = state.apiKey;
  headers["X-Tenant-ID"] = state.tenantId;
  const response = await fetch(`${state.apiBase}${path}`, { ...options, headers });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = body.detail || {};
    throw new Error(detail.message || detail[0]?.msg || `Request failed (${response.status})`);
  }
  return body;
}

function toast(message, error = false) {
  const node = $("#toast");
  node.textContent = message;
  node.className = `toast show${error ? " error" : ""}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (node.className = "toast"), 2800);
}

async function loadHealth() {
  try {
    const [health, model] = await Promise.all([api("/health/live"), api("/model")]);
    $("#serviceStatus").textContent = health.status === "ok" ? "Service operational" : "Degraded";
    $(".pulse").classList.remove("offline");
    $("#modelCompact").textContent = `${model.name} · ${model.vector_store} · ${model.dimension}d`;
  } catch {
    $("#serviceStatus").textContent = "Connection failed";
    $(".pulse").classList.add("offline");
    $("#modelCompact").textContent = "Check API settings";
  }
}

function setImage(file) {
  if (!file?.type.startsWith("image/")) return toast("Please choose an image file.", true);
  if (file.size > 10 * 1024 * 1024) return toast("Image must be smaller than 10 MB.", true);
  const reader = new FileReader();
  reader.onload = () => {
    state.imageBase64 = reader.result;
    $("#preview").src = reader.result;
    $("#dropzone").classList.add("has-image");
    $("#fileMeta").firstElementChild.textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
  };
  reader.readAsDataURL(file);
}

function clearImage() {
  state.imageBase64 = null;
  $("#preview").src = "";
  $("#imageInput").value = "";
  $("#dropzone").classList.remove("has-image");
  $("#fileMeta").firstElementChild.textContent = "No image selected";
}

function labels() {
  return $("#labels").value.split("\n").map((x) => x.trim()).filter(Boolean);
}

function updateLabelCount() {
  $("#labelCount").textContent = `${labels().length} labels`;
}

async function compare() {
  const candidates = labels();
  if (!state.imageBase64) return toast("Add an image first.", true);
  if (!candidates.length) return toast("Add at least one candidate label.", true);
  const button = $("#compareButton");
  button.disabled = true;
  button.querySelector("span").textContent = "Comparing…";
  try {
    const result = await api("/classifications/zero-shot", {
      method: "POST",
      body: JSON.stringify({
        labels: candidates,
        images: [state.imageBase64],
        top_k: candidates.length,
      }),
    });
    const ranked = result.results[0].predictions;
    const min = Math.min(...ranked.map((item) => item.score));
    const max = Math.max(...ranked.map((item) => item.score));
    $("#emptyResults").style.display = "none";
    $("#ranking").innerHTML = ranked.map((item, index) => {
      const relative = max === min ? 100 : 22 + ((item.score - min) / (max - min)) * 78;
      return `<div class="rank-item">
        <span class="rank-number">${String(index + 1).padStart(2, "0")}</span>
        <span class="rank-label" title="${escapeHtml(item.label)}">${escapeHtml(item.label)}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${relative}%"></div></div>
        <span class="rank-score">${(item.probability * 100).toFixed(1)}%</span>
      </div>`;
    }).join("");
    $("#latency").textContent = `${result.duration_ms.toFixed(1)} MS · ${result.prompt_count} PROMPTS · ${result.model}`;
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.disabled = false;
    button.querySelector("span").textContent = "Run comparison";
  }
}

function escapeHtml(value) {
  const node = document.createElement("div");
  node.textContent = value;
  return node.innerHTML;
}

function parseIndexItems() {
  return $("#indexItems").value.split("\n").map((line) => {
    const split = line.indexOf("|");
    if (split < 1) return null;
    const id = line.slice(0, split).trim();
    const text = line.slice(split + 1).trim();
    return id && text ? { id, text, metadata: { source: "console" } } : null;
  }).filter(Boolean);
}

async function indexItems() {
  const items = parseIndexItems();
  if (!items.length) return toast("Use the format: id | text", true);
  const button = $("#indexButton");
  button.disabled = true;
  try {
    const path = items.length > 64
      ? "/jobs/index/text"
      : `/collections/${encodeURIComponent(state.collection)}/items/text`;
    const payload = items.length > 64 ? { collection: state.collection, items } : { items };
    const result = await api(path, { method: "POST", body: JSON.stringify(payload) });
    toast(result.id
      ? `Background job queued · ${result.id.slice(0, 8)}`
      : `${result.affected} records indexed · ${result.index_size} total`);
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.disabled = false;
  }
}

async function search(feedback = null) {
  const query = $("#searchQuery").value.trim();
  if (!query) return toast("Enter a search query.", true);
  const button = $("#searchButton");
  button.disabled = true;
  try {
    const result = await api(`/collections/${encodeURIComponent(state.collection)}/search/interactive`, {
      method: "POST",
      body: JSON.stringify({
        query_type: "text",
        query,
        limit: 10,
        ...(feedback ? { feedback } : {}),
      }),
    });
    $("#searchResults").innerHTML = result.hits.length
      ? result.hits.map((hit, index) => `<div class="hit">
          <span class="hit-rank">${String(index + 1).padStart(2, "0")}</span>
          <div><strong>${escapeHtml(hit.id)}</strong><small>${escapeHtml(hit.metadata._text || "")}</small></div>
          <span class="hit-score">${hit.score.toFixed(4)}</span>
        </div>`).join("")
      : '<div class="search-empty">No matching records found.</div>';
    renderUncertainty(result);
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.disabled = false;
  }
}

function renderUncertainty(result) {
  const panel = $("#uncertaintyPanel");
  const uncertainty = result.uncertainty;
  const confidence = Math.round(uncertainty.confidence * 100);
  const status = uncertainty.needs_clarification ? "Intent is ambiguous" : "Model is confident";
  panel.className = "uncertainty-panel show";
  panel.innerHTML = `<div class="confidence-card${uncertainty.needs_clarification ? " uncertain" : ""}">
    <span class="confidence-label"><i class="confidence-dot"></i>${status}</span>
    <span class="confidence-metrics">CONF ${confidence}% · MARGIN ${uncertainty.margin.toFixed(3)} · ENTROPY ${uncertainty.normalized_entropy.toFixed(3)}</span>
  </div>`;

  if (result.clarification) {
    panel.innerHTML += `<div class="clarification-card">
      <strong>Help the model understand your intent</strong>
      <p>${escapeHtml(result.clarification.question)}</p>
      <div class="clarification-options">
        ${result.clarification.options.map((option, index) => `<button class="clarification-option" data-feedback-id="${escapeHtml(option.id)}" data-option-index="${index}">
          <small>OPTION ${String.fromCharCode(65 + index)} · ${option.modality.toUpperCase()}</small>
          <span>${escapeHtml(option.label)}</span>
        </button>`).join("")}
      </div>
    </div>`;
    panel.querySelectorAll(".clarification-option").forEach((button) => {
      button.addEventListener("click", () => {
        const positive = button.dataset.feedbackId;
        const negative = result.clarification.options
          .map((option) => option.id)
          .filter((id) => id !== positive);
        search({ positive_ids: [positive], negative_ids: negative });
      });
    });
  } else if (result.feedback_applied) {
    panel.innerHTML += `<div class="feedback-note">FEEDBACK APPLIED · QUERY DRIFT ${result.query_drift.toFixed(3)}</div>`;
  }
}

$$(".nav-item[data-view]").forEach((button) => {
  button.addEventListener("click", () => {
    $$(".nav-item").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    $$(".view").forEach((view) => view.classList.remove("active"));
    $(`#${button.dataset.view}View`).classList.add("active");
    $("#pageTitle").textContent =
      button.dataset.view === "playground" ? "Compare meaning, not pixels." : "Search at the speed of thought.";
  });
});

$("#imageInput").addEventListener("change", (event) => setImage(event.target.files[0]));
$("#clearImage").addEventListener("click", clearImage);
$("#labels").addEventListener("input", updateLabelCount);
$("#loadExample").addEventListener("click", () => {
  $("#labels").value = "a studio product photograph\na vibrant landscape\na technical diagram\na candid portrait\nan abstract illustration";
  updateLabelCount();
});
$("#compareButton").addEventListener("click", compare);
$("#indexButton").addEventListener("click", indexItems);
$("#searchButton").addEventListener("click", search);
$("#searchQuery").addEventListener("keydown", (event) => event.key === "Enter" && search());
document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") compare();
});

const dropzone = $("#dropzone");
["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => {
  event.preventDefault(); dropzone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => {
  event.preventDefault(); dropzone.classList.remove("dragging");
}));
dropzone.addEventListener("drop", (event) => setImage(event.dataTransfer.files[0]));

const dialog = $("#settingsDialog");
$("#settingsButton").addEventListener("click", () => {
  $("#apiBase").value = state.apiBase;
  $("#apiKey").value = state.apiKey;
  $("#tenantId").value = state.tenantId;
  $("#collectionName").value = state.collection;
  dialog.showModal();
});
$("#saveSettings").addEventListener("click", (event) => {
  event.preventDefault();
  state.apiBase = $("#apiBase").value.replace(/\/$/, "") || "/api/v1";
  state.apiKey = $("#apiKey").value;
  state.tenantId = $("#tenantId").value.trim() || "default";
  state.collection = $("#collectionName").value.trim() || "main";
  localStorage.setItem("clipforge.apiBase", state.apiBase);
  localStorage.setItem("clipforge.apiKey", state.apiKey);
  localStorage.setItem("clipforge.tenantId", state.tenantId);
  localStorage.setItem("clipforge.collection", state.collection);
  $("#tenantBadge").textContent = `TENANT · ${state.tenantId}`;
  dialog.close();
  loadHealth();
  toast("Connection settings saved.");
});

loadHealth();
updateLabelCount();
$("#tenantBadge").textContent = `TENANT · ${state.tenantId}`;
