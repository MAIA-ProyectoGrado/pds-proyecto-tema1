/* Tablero SCIF: lee el catálogo y el estado de la API y muestra solo lo que ella devuelve. */

const API = (window.SCIF_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const README_MODELS_URL = "https://github.com/MAIA-ProyectoGrado/pds-proyecto-tema1#modelos";
const README_LLM_URL = "https://github.com/MAIA-ProyectoGrado/pds-proyecto-tema1#modelos-de-lenguaje";

/* ── Esquema de anotación del proyecto (no es salida del modelo) ── */
const DEFINITIONS = {
  Application: "Emplea un método, herramienta o dato del trabajo citado sin modificarlo.",
  Background: "Aporta contexto general, antecedentes o traza la historia de un campo.",
  Basis: "Constituye el punto de partida intelectual o metodológico del trabajo.",
  Comparison: "Contrasta métodos, datos o resultados con los del trabajo citado.",
  Evidence: "Respalda empíricamente una afirmación, hipótesis o decisión de diseño.",
  Further_Reading: "Remite al lector a literatura adicional o complementaria.",
  Gap: "Señala un vacío de investigación que justifica el trabajo actual.",
  Identification_of_the_Originator: "Reconoce la fuente original de una idea, método o teoría.",
  Modification_Improvement: "Adapta, extiende o mejora un método del trabajo citado.",
};

/* ── Ejemplos: entradas precargadas (contexto, sección y artículo citado real).
      Son solo valores para los campos; el resultado siempre lo produce el modelo. ── */
const EXAMPLES = [
  {
    ctx: "Although several citation function taxonomies have been proposed for the computer science domain [7], no publicly available corpus aligns the citation context with passage-level evidence inside the cited document.",
    section: "Introduction",
    arxiv: "1904.01608",   // Cohan et al., Structural Scaffolds for Citation Intent Classification
  },
  {
    ctx: "We extend SciBERT [4] with an additional classification head and fine-tune it on our corpus to improve accuracy.",
    section: "Methods",
    arxiv: "1903.10676",   // Beltagy et al., SciBERT
  },
  {
    ctx: "The masked language modelling objective was first introduced by Devlin et al. (2019).",
    section: "Introduction",
    arxiv: "1810.04805",   // Devlin et al., BERT
  },
];

/* ── Secciones canónicas con que /retrieve etiqueta los pasajes ── */
const SECTION_NAMES = {
  Abstract: "Abstract", Introduccion: "Introduction", Trabajo_relacionado: "Related Work",
  Metodos: "Methods", Experimentos: "Experiments", Resultados: "Results",
  Discusion: "Discussion", Conclusion: "Conclusion", Cuerpo_generico: "Body",
  Otra: "Other", Desconocida: "Unknown",
};
const secName = (s) => SECTION_NAMES[s] || s;

/* ── Qué leyó un modelo de lenguaje del artículo citado (campo key_sentences.source) ── */
const LLM_SOURCES = {
  "arxiv-abstract": "Título y abstract en arXiv",
  "arxiv-title": "Título en arXiv (sin abstract)",
  "text-abstract": "Abstract del texto pegado",
  "text": "Inicio del texto pegado",
};

/* ── De dónde salieron las oraciones clave (campo key_sentences.source) ── */
const KEY_SOURCES = {
  "arxiv-abstract": "del abstract en arXiv",
  "arxiv-title": "del título en arXiv",
  "text-abstract": "del abstract del texto pegado",
  "text": "del texto pegado",
};

const UNKNOWN_SECTION = "";   // valor de «Otra / no sé»: se envía sin sección
const HEALTH_POLL_MS = 3000;
const HEALTH_POLL_MAX = 60;

let META = null;    // respuesta de GET /models
let busy = false;
let runId = 0;      // consulta en curso; descarta pasajes de consultas anteriores
let healthPolls = 0;

/* ═══ Tema ═══ */
const root = document.documentElement;
const bL = document.getElementById("tLight");
const bD = document.getElementById("tDark");
function setTheme(t) {
  root.setAttribute("data-theme", t);
  bL.setAttribute("aria-pressed", String(t === "light"));
  bD.setAttribute("aria-pressed", String(t === "dark"));
}
bL.addEventListener("click", () => setTheme("light"));
bD.addEventListener("click", () => setTheme("dark"));
setTheme(root.getAttribute("data-theme")
  || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));

/* ═══ Utilidades ═══ */
const $ = (id) => document.getElementById(id);

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function fmtMs(v) {
  if (v == null) return "no aplica";
  return v < 1 ? "<1 ms" : `${Math.round(v).toLocaleString("es")} ms`;
}
const readmeLink = () =>
  `<a href="${README_MODELS_URL}" target="_blank" rel="noopener">Ver cómo descargarlos</a>`;
const llmReadmeLink = () =>
  `<a href="${README_LLM_URL}" target="_blank" rel="noopener">Ver cómo activarlos</a>`;
const linkFor = (m) => (m && m.backend === "bedrock" ? llmReadmeLink() : readmeLink());
const fmtUsd = (v, digits) => `USD ${v.toFixed(digits)}`;

function showBanner(html) {
  const b = $("banner");
  b.innerHTML = html;
  b.hidden = false;
}
function hideBanner() { $("banner").hidden = true; }

/** Retira el resultado anterior. Se llama en cada fallo: dejar en pantalla una
 *  predicción vieja junto a un texto nuevo la haría pasar por la respuesta actual. */
function invalidate() {
  $("results").hidden = true;
  $("empty").hidden = false;
  hideRetrieval();
}

function setApiState(text, cls, title) {
  const el = $("apiState");
  el.textContent = text;
  el.className = cls || "";
  el.title = title || "";
}

/** nginx responde 502 o 504, sin detalle de la API, cuando el servicio de inferencia no contesta. */
function apiUnreachable(r, body) {
  return (r.status === 502 || r.status === 504) && !body.detail;
}

function modelInfo(id) { return META.models.find((m) => m.id === id); }
function modelName(id) { const m = modelInfo(id); return m ? m.name : id; }

/* ═══ Arranque: catálogo y estado de la API ═══ */
async function boot() {
  $("apiUrlLabel").textContent = API;
  $("docsLink").href = `${API}/docs`;
  try {
    const r = await fetch(`${API}/models`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    META = await r.json();
  } catch (err) {
    invalidate();
    setApiState("API no disponible", "down");
    showBanner(
      `<b>No se pudo contactar la API</b> en <code>${escapeHtml(API)}</code>. ` +
      `Revisa que el servicio de inferencia esté en marcha (con Docker, <code>docker compose ps</code>; ` +
      `sin Docker, <code>uvicorn src.api.main:app --port 8000</code> desde la raíz del repo) ` +
      `y recarga esta página. El tablero no muestra predicciones sin la API.`
    );
    $("empty").hidden = true;
    return;
  }
  hideBanner();
  renderSections();
  renderModels();
  renderGlossary();
  renderDistSkeleton();
  renderAvailability();
  refreshHealth();
}

/** Estado de la cabecera según /health. Mientras el modelo por defecto se carga
 *  en memoria, vuelve a consultar cada pocos segundos. */
async function refreshHealth() {
  let h;
  try {
    const r = await fetch(`${API}/health`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    h = await r.json();
  } catch (err) {
    setApiState("API no disponible", "down");
    return;
  }
  const def = h.models[h.default_model] || {};
  if (def.available && !def.loaded && healthPolls < HEALTH_POLL_MAX) {
    setApiState(`cargando ${modelName(h.default_model)}…`, "wait",
      "El modelo se está cargando en memoria; la primera clasificación puede tardar unos segundos más.");
    healthPolls += 1;
    setTimeout(refreshHealth, HEALTH_POLL_MS);
  } else if (h.status === "ok") {
    setApiState("API conectada", "live");
  } else {
    setApiState("servicio parcial", "wait", "Falta algún componente; el aviso de arriba explica cuál.");
  }
}

function renderSections() {
  $("section").innerHTML =
    META.sections.map((s) =>
      `<option value="${escapeHtml(s)}"${s === "Introduction" ? " selected" : ""}>${escapeHtml(s)}</option>`
    ).join("") +
    `<option value="${UNKNOWN_SECTION}">Otra / no sé</option>`;
}

/** Selector de modelos. Queda marcado el modelo por defecto si está instalado; si no,
 *  el primero que sí lo esté, para que el tablero funcione con lo que haya. */
function renderModels() {
  const def = modelInfo(META.default_model);
  const chosen = def && def.available ? def : META.models.find((m) => m.available);
  $("models").innerHTML = META.models.map((m) => {
    const badge = m.recommended ? '<span class="badge">Recomendado</span>' : "";
    const missing = m.available ? "" :
      `<span class="missing"><b>No disponible en este servidor.</b> ${escapeHtml(m.unavailable_reason)} ${linkFor(m)}</span>`;
    const cost = m.backend === "bedrock" && m.cost_per_query_usd != null
      ? `<span class="cost">≈ ${fmtUsd(m.cost_per_query_usd, 5)} por consulta · ${fmtMs(m.typical_latency_ms)} típicos</span>`
      : "";
    return `
      <label class="model">
        <input type="radio" name="model" value="${escapeHtml(m.id)}"
               ${chosen && m.id === chosen.id ? "checked" : ""} ${m.available ? "" : "disabled"}>
        <span>
          <span class="kind">${escapeHtml(m.kind)}</span>
          <span class="name">${escapeHtml(m.name)}${badge}</span>
          <span class="reads">Lee: ${escapeHtml(m.reads)}</span>
          ${cost}
          ${missing}
        </span>
      </label>`;
  }).join("");
  $("run").disabled = !chosen;
}

/** Avisos sobre lo que falta en este servidor: modelo recomendado, oraciones clave y pasajes. */
function renderAvailability() {
  const rec = META.models.find((m) => m.recommended);
  const available = META.models.filter((m) => m.available);
  const parts = [];

  if (!available.length) {
    showBanner(
      `<b>No hay ningún clasificador instalado en este servidor.</b> ` +
      META.models.map((m) => escapeHtml(m.unavailable_reason)).join(" ") + ` ${readmeLink()}`
    );
  } else if (rec && !rec.available) {
    const fallback = available[0];
    parts.push(
      `<b>${escapeHtml(rec.name)} no está instalado en este servidor.</b> Se clasifica con ` +
      `${escapeHtml(fallback.name)}, que lee: ${escapeHtml(fallback.reads.toLowerCase())}. ` +
      `${escapeHtml(rec.unavailable_reason)} ${readmeLink()}`
    );
  } else if (rec && rec.reads_passages && !META.key_sentences_available) {
    parts.push(
      `<b>Las oraciones clave del abstract no están disponibles en este servidor:</b> falta ` +
      `<code>models/minilm-l6-v2</code>. ${escapeHtml(rec.name)} clasifica sin el artículo citado. ` +
      readmeLink()
    );
  }

  const n = $("notice");
  n.innerHTML = parts.join("<br>");
  n.hidden = !parts.length;

  const rt = $("retrTag");
  rt.textContent = META.retrieval_available ? "pasajes del artículo" : "pasajes no disponibles";
  rt.className = META.retrieval_available ? "" : "down";
}

function renderGlossary() {
  $("glossary").innerHTML = META.labels.map((lab) => `
    <div style="--cc:${colorFor(lab)}">
      <dt><i></i>${escapeHtml(META.label_display[lab])}</dt>
      <dd>${escapeHtml(DEFINITIONS[lab] || "")}</dd>
    </div>`).join("");
}

function colorFor(label) {
  const i = META.labels.indexOf(label);
  return `var(--c${(i < 0 ? 0 : i) + 1})`;
}

function renderDistSkeleton() {
  $("dist").innerHTML = META.labels.map((lab) => `
    <div class="drow" data-label="${escapeHtml(lab)}" style="--cc:${colorFor(lab)}">
      <span class="dname"><i></i><span title="${escapeHtml(META.label_display[lab])}">${escapeHtml(META.label_display[lab])}</span></span>
      <span class="dtrack"><span class="dbar"></span></span>
      <span class="dval"></span>
    </div>`).join("");
}

/* ═══ Entrada ═══ */
const ctxEl = $("ctx");
function updateCounter() {
  const w = ctxEl.value.trim().split(/\s+/).filter(Boolean).length;
  $("wc").textContent = `${w} ${w === 1 ? "palabra" : "palabras"}`;
  const m = ctxEl.value.match(/\[\d+\]|\(\d{4}\)|et al\./);
  $("mk").textContent = m ? `marcador ${m[0]}` : "sin marcador";
}
ctxEl.addEventListener("input", updateCounter);

document.querySelectorAll(".examples button").forEach((b) => {
  b.addEventListener("click", () => {
    const ex = EXAMPLES[+b.dataset.ex];
    ctxEl.value = ex.ctx;
    $("section").value = ex.section;
    // el ejemplo trae su artículo citado, así que se muestra el flujo completo
    if (!$("citedText").hidden) $("togglePaste").click();
    $("arxiv").value = ex.arxiv;
    updateCounter();
    ctxEl.focus();
  });
});

/* ═══ Artículo citado: id de arXiv o texto pegado ═══ */
$("togglePaste").addEventListener("click", () => {
  const ta = $("citedText"), inp = $("arxiv");
  ta.hidden = !ta.hidden;
  $("togglePaste").textContent = ta.hidden ? "Pegar el texto en su lugar" : "Usar un id de arXiv en su lugar";
  if (!ta.hidden) { inp.value = ""; ta.focus(); } else { ta.value = ""; inp.focus(); }
});

function citedDocument() {
  const id = $("arxiv").value.trim();
  const txt = $("citedText").hidden ? "" : $("citedText").value.trim();
  if (txt) return { cited_text: txt };
  if (id) return { arxiv_id: id };
  return null;
}

/* ═══ Clasificar ═══ */
$("run").addEventListener("click", classify);
ctxEl.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") classify();
});

async function classify() {
  if (busy || !META) return;
  const text = ctxEl.value.trim();
  if (!text) {
    invalidate();
    showBanner("<b>Falta el contexto de cita.</b> Escribe o pega una o dos oraciones en inglés.");
    return;
  }
  const selected = document.querySelector('input[name="model"]:checked');
  if (!selected) {
    showBanner(`<b>No hay ningún clasificador instalado en este servidor.</b> ${readmeLink()}`);
    return;
  }

  const run = ++runId;
  busy = true;
  $("run").disabled = true;
  $("run").lastChild.textContent = " Clasificando…";
  hideBanner();

  const doc = citedDocument();
  const section = $("section").value || null;
  const sent = { context: text, section, doc };
  const wantsPassages = Boolean(doc) && META.retrieval_available;

  if (!doc) hideRetrieval();
  else if (!META.retrieval_available) showRetrievalUnavailable();
  else showRetrievalLoading(doc);

  // Primero la clasificación, que espera solo al abstract. Los pasajes se piden
  // cuando ella termina: leer el PDF en paralelo le resta CPU al modelo y alarga
  // la clasificación.
  try {
    const r = await fetch(`${API}/predict`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ citation_context: text, rhetorical_section: section, model: selected.value, ...(doc || {}) }),
    });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) {
      invalidate();
      if (apiUnreachable(r, body)) {
        setApiState("API no disponible", "down");
        showBanner(`<b>La API no responde (${r.status}).</b> No se muestra ninguna predicción. ` +
                   `Revisa que el servicio esté en marcha y vuelve a intentarlo.`);
      } else {
        const extra = r.status === 503 ? ` ${linkFor(modelInfo(selected.value))}` : "";
        showBanner(`<b>La API rechazó la petición (${r.status}).</b> ${escapeHtml(body.detail || "")}${extra}`);
      }
      return;
    }
    render(body, sent, wantsPassages ? "en curso…" : (doc ? "no disponible" : "no aplica"));
  } catch (err) {
    invalidate();
    setApiState("API no disponible", "down");
    showBanner(
      `<b>Se perdió la conexión con la API.</b> No se muestra ninguna predicción: ` +
      `el tablero solo presenta resultados que provengan del modelo.`
    );
    return;
  } finally {
    busy = false;
    $("run").disabled = false;
    $("run").lastChild.textContent = " Clasificar";
  }

  if (wantsPassages) await loadPassages(run, text, doc);
}

/** Pasajes del artículo completo. Si mientras tanto se lanzó otra consulta, el
 *  resultado se descarta para no mezclarlo con la clasificación nueva. */
async function loadPassages(run, text, doc) {
  let r, body;
  try {
    r = await fetch(`${API}/retrieve`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ citation_context: text, ...doc, top_k: 3 }),
    });
    body = await r.json().catch(() => ({}));
  } catch (err) {
    if (run === runId) showRetrievalError("Se perdió la conexión con la API mientras se buscaban los pasajes.", "");
    return;
  }
  if (run !== runId) return;
  if (!r.ok) {
    showRetrievalError(apiUnreachable(r, body)
      ? `La API no respondió mientras se buscaban los pasajes (${r.status}).`
      : `No se pudieron buscar los pasajes (${r.status}).`, body.detail || "");
    return;
  }
  // Un fallo al pintar no es un fallo de red: se informa como lo que es.
  try {
    renderRetrieval(body);
  } catch (err) {
    console.error("renderRetrieval:", err);
    showRetrievalError("No se pudieron mostrar los pasajes del artículo.", String(err));
  }
}

/* ═══ Resultado de la clasificación ═══ */
function render(res, sent, retrievalState) {
  $("results").hidden = false;
  $("empty").hidden = true;
  setApiState("API conectada", "live");

  const color = colorFor(res.predicted_label);
  $("vName").textContent = res.predicted_label_display;
  $("vName").style.color = color;
  $("vDef").textContent = DEFINITIONS[res.predicted_label] || "";
  $("mChip").textContent = `${modelName(res.model_id)} · ${res.model_version}`;

  // Los modelos de lenguaje dan un puntaje por función; el tablero lo muestra normalizado
  const isLlm = Boolean(res.llm);
  $("vEyebrow").textContent = isLlm ? "clase de mayor puntaje" : "clase de mayor probabilidad";
  $("gLbl").textContent = isLlm ? "PUNTAJE" : "CONFIANZA";
  $("distEyebrow").textContent = isLlm
    ? "puntajes del modelo, normalizados a suma 1" : "probabilidades devueltas por el modelo";
  $("readEyebrow").textContent = isLlm ? `prompt del enfoque ${res.llm.approach}` : "entrada del clasificador";
  $("ioSummary").textContent = isLlm
    ? `Mensaje enviado al modelo (va después de las instrucciones del enfoque ${res.llm.approach}: ${res.llm.approach_name})`
    : "Entrada exacta enviada al modelo";

  const ks = res.key_sentences;
  const tag = $("readTag");
  const basis = res.rhetorical_section ? "el contexto y la sección" : "el contexto";
  if (res.passages_used) {
    tag.textContent = "Clasificado con las oraciones clave del abstract";
    tag.className = "readtag used";
  } else if (ks.status === "abstract_used") {
    tag.textContent = ks.source === "arxiv-title"
      ? "Clasificado con el título del artículo citado"
      : "Clasificado con el título y el abstract del artículo citado";
    tag.className = "readtag used";
  } else if (ks.status === "not_read_by_model") {
    tag.textContent = `Clasificado con ${basis}: este modelo no lee el artículo citado`;
    tag.className = "readtag";
  } else {
    tag.textContent = `Clasificado solo con ${basis}`;
    tag.className = "readtag";
  }

  const arc = $("gArc");
  arc.style.stroke = color;
  arc.setAttribute("stroke-dashoffset", (352 * (1 - res.confidence)).toFixed(1));
  $("gTxt").textContent = res.confidence.toFixed(2);

  // Aviso de ambigüedad: lo marca el propio predictor (margen menor que SCIF_AMBIGUITY_THRESHOLD).
  // En los modelos de lenguaje el margen se mide sobre sus puntajes, sin normalizar.
  const entries = Object.entries(isLlm ? res.llm.scores : res.probabilities).sort((a, b) => b[1] - a[1]);
  const al = $("alert");
  if (res.ambiguous && entries.length > 1) {
    const d = (entries[0][1] - entries[1][1]).toFixed(2);
    $("alertTxt").innerHTML =
      `El modelo marcó este caso como <b>ambiguo</b>: la diferencia entre ` +
      `<b>${escapeHtml(META.label_display[entries[0][0]])}</b> y <b>${escapeHtml(META.label_display[entries[1][0]])}</b> ` +
      `es de ${d}. Candidato a revisión humana.`;
    al.hidden = false;
  } else {
    al.hidden = true;
  }

  document.querySelectorAll(".drow").forEach((row) => {
    const lab = row.dataset.label;
    const p = res.probabilities[lab];
    row.classList.toggle("top", lab === res.predicted_label);
    row.querySelector(".dbar").style.width = p == null ? "0%" : `${p * 100}%`;
    row.querySelector(".dval").textContent = p == null ? "" : p.toFixed(3);
  });

  renderRead(res, sent);
  renderTelemetry(res, retrievalState);
}

/** Panel «Lo que leyó el modelo»: contexto, sección y oraciones clave, o por qué no las hubo. */
function renderRead(res, sent) {
  const ks = res.key_sentences;
  let paper;
  if (ks.status === "used") {
    const cited = ks.cited_paper || {};
    const ref = cited.id ? `arXiv:${escapeHtml(cited.id)}` : "";
    const title = cited.title ? `<span class="ks-title">${escapeHtml(cited.title)}</span>` : "";
    paper = `
      <p class="ks-src">${ks.sentences.length} oraciones clave ${escapeHtml(KEY_SOURCES[ks.source] || "")}
        ${ref ? `· <span class="ks-id">${ref}</span>` : ""}</p>
      ${title}
      <ol class="ks">${ks.sentences.map((s) => `
        <li>
          <span class="rank">${s.rank}</span>
          <span class="ks-text">${escapeHtml(s.text)}</span>
          <span class="sim" title="Similitud coseno con el contexto (MiniLM)">
            <span class="simbar"><i style="width:${Math.max(0, Math.round(s.similarity * 100))}%"></i></span>
            <span class="simval">${s.similarity.toFixed(3)}</span>
          </span>
        </li>`).join("")}
      </ol>
      ${ks.detail ? `<p class="ks-note">${escapeHtml(ks.detail)}</p>` : ""}`;
  } else if (ks.status === "abstract_used") {
    const cited = ks.cited_paper || {};
    const ref = cited.id ? `arXiv:${escapeHtml(cited.id)}` : "";
    paper = `
      <p class="ks-src">${escapeHtml(LLM_SOURCES[ks.source] || "")}
        ${ref ? `· <span class="ks-id">${ref}</span>` : ""}</p>
      ${cited.title ? `<span class="ks-title">${escapeHtml(cited.title)}</span>` : ""}
      ${ks.abstract ? `<p class="ks-abstract">${escapeHtml(ks.abstract)}</p>` : ""}
      ${ks.detail ? `<p class="ks-note">${escapeHtml(ks.detail)}</p>` : ""}`;
  } else if (ks.status === "failed") {
    paper = `<p class="ks-note warn">${escapeHtml(ks.detail)}</p>`;
  } else if (ks.status === "not_provided") {
    paper = `<p class="ks-note">${escapeHtml(ks.detail)} Indica su ID de arXiv o pega su abstract
             para que el modelo lo lea.</p>`;
  } else {
    paper = `<p class="ks-note">${escapeHtml(ks.detail)}</p>`;
  }

  $("readBody").innerHTML = `
    <div class="rd-row"><span class="rd-l">Contexto</span><p>${escapeHtml(sent.context)}</p></div>
    <div class="rd-row"><span class="rd-l">Sección</span>
      <p>${res.rhetorical_section ? escapeHtml(res.rhetorical_section) : "Sin sección (Otra / no sé)"}</p></div>
    <div class="rd-row"><span class="rd-l">Artículo citado</span><div>${paper}</div></div>`;

  $("ioText").innerHTML = highlightInput(res.input_text);
}

/** La entrada exacta, con el separador de cada modelo resaltado. */
function highlightInput(s) {
  for (const sep of [" [SEP] ", " [SEC] "]) {
    const i = s.indexOf(sep);
    if (i >= 0) {
      return `${escapeHtml(s.slice(0, i))}<span class="sep">${sep}</span>${escapeHtml(s.slice(i + sep.length))}`;
    }
  }
  return escapeHtml(s);
}

function renderTelemetry(res, retrievalState) {
  const ks = res.key_sentences;
  const readsPaper = ["used", "abstract_used", "failed"].includes(ks.status);
  const lat = res.latency_ms;
  const cell = (label, value, id) =>
    `<div class="tc"${id ? ` id="${id}"` : ""}><span class="l">${label}</span><span class="v">${value}</span></div>`;
  const model = cell("Modelo", `<span style="font-size:12px">${escapeHtml(modelName(res.model_id))} · ${escapeHtml(res.model_version)}</span>`);
  const paper = cell("Artículo citado", readsPaper ? fmtMs(lat.cited_paper) : "no aplica");
  const tail = cell("Total", fmtMs(lat.total)) + cell("Pasajes del artículo", escapeHtml(retrievalState), "teleRetr");
  if (res.llm) {
    const l = res.llm;
    $("tele").innerHTML = model + paper +
      cell("Clasificación (Bedrock)", fmtMs(lat.inference)) +
      cell("Costo de la consulta", fmtUsd(l.cost_usd, 5)) +
      cell("Gasto de hoy", `${fmtUsd(l.spent_today_usd, 3)} <span style="font-size:11px">de ${l.daily_budget_usd}</span>`) +
      tail;
    return;
  }
  $("tele").innerHTML = model + paper +
    cell("Oraciones clave", ks.status === "used" ? fmtMs(lat.key_sentences) : "no aplica") +
    cell("Clasificación", fmtMs(lat.inference)) +
    tail;
}

function setRetrievalTelemetry(text) {
  const c = $("teleRetr");
  if (c) c.querySelector(".v").textContent = text;
}

/* ═══ Pasajes del artículo completo ═══ */
function hideRetrieval() {
  $("pChunks").hidden = true;
  $("pMap").hidden = true;
}

function retrievalState(cls, html) {
  $("pChunks").hidden = false;
  $("pMap").hidden = true;
  $("paperMeta").hidden = true;
  $("chunks").innerHTML = "";
  const st = $("rstate");
  st.className = cls;
  st.innerHTML = html;
  st.hidden = false;
}

function showRetrievalLoading(doc) {
  $("chunksEyebrow").textContent = "pasajes del artículo · SciBERT";
  retrievalState("rstate", doc.arxiv_id
    ? `<span class="spin"></span>Descargando <b>arXiv:${escapeHtml(doc.arxiv_id)}</b>, dividiéndolo en fragmentos y comparándolos con la cita…`
    : `<span class="spin"></span>Dividiendo el texto en fragmentos y comparándolos con la cita…`);
}

function showRetrievalUnavailable() {
  const scibert = modelInfo("scibert");
  retrievalState("rstate",
    `<b>Los pasajes del artículo completo no están disponibles en este servidor:</b> se buscan con SciBERT. ` +
    `${escapeHtml(scibert ? scibert.unavailable_reason : "")} ${readmeLink()}`);
}

function showRetrievalError(title, detail) {
  retrievalState("rstate err",
    `<b>${escapeHtml(title)}</b> ${escapeHtml(detail)}<br>` +
    `La clasificación de arriba no depende de este paso y sigue siendo válida.`);
  setRetrievalTelemetry("falló");
}

function renderRetrieval(res) {
  $("rstate").hidden = true;
  $("pChunks").hidden = false;

  const p = res.paper;
  const meta = $("paperMeta");
  if (p.source === "text") {
    meta.innerHTML = `<span class="pid">texto pegado por el usuario</span>` +
      `<span class="ps"><b>${res.n_words.toLocaleString("es")} palabras</b><b>${res.n_chunks} fragmentos</b></span>`;
  } else {
    const authors = p.authors.length > 3 ? `${p.authors.slice(0, 3).join(", ")} et al.` : p.authors.join(", ");
    meta.innerHTML =
      `<span class="pid">arXiv:${escapeHtml(p.id)}</span>` +
      `<span class="pt">${escapeHtml(p.title || "")}</span>` +
      `<span class="pa">${escapeHtml(authors)}${p.year ? " · " + escapeHtml(p.year) : ""}</span>` +
      `<span class="ps"><b>${res.n_words.toLocaleString("es")} palabras</b><b>${res.n_chunks} fragmentos</b>` +
      `<b>${res.cached ? "en caché" : fmtMs(res.latency_ms.download) + " de descarga"}</b></span>`;
  }
  meta.hidden = false;

  $("chunksEyebrow").textContent = `pasajes del artículo · SciBERT · ${fmtMs(res.latency_ms.total)}`;
  $("chunks").innerHTML = res.chunks.map((c) => `
    <article class="chunk">
      <div class="ch-head">
        <span class="rank">${c.rank}</span>
        <span class="secbadge">${escapeHtml(secName(c.section))}</span>
        <span class="pos">fragmento ${c.index + 1} de ${res.n_chunks} · ${c.n_words} palabras</span>
        <span class="sim"><span class="simbar"><i style="width:${Math.round(c.cosine * 100)}%"></i></span>
        <span class="simval">${c.cosine.toFixed(3)}</span></span>
      </div>
      <p>${escapeHtml(c.text)}</p>
    </article>`).join("");

  $("pMap").hidden = false;
  drawMap(res);
  setRetrievalTelemetry(res.cached ? `${fmtMs(res.latency_ms.total)} (en caché)` : fmtMs(res.latency_ms.total));
}

function drawMap(res) {
  const W = 900, H = 190, padT = 26, padB = 34, padL = 4, padR = 4;
  const scores = res.scores, n = scores.length, bw = (W - padL - padR) / n, plotH = H - padT - padB;
  const topIdx = new Map(res.chunks.map((c) => [c.index, c.rank]));
  const lo = Math.min(...scores), hi = Math.max(...scores);
  const y = (v) => padT + plotH * (1 - (hi > lo ? (v - lo) / (hi - lo) : 0.5));
  let g = "";

  // bandas por sección (cambios consecutivos en res.sections)
  let start = 0, band = 0;
  for (let i = 1; i <= n; i++) {
    if (i === n || res.sections[i] !== res.sections[start]) {
      const x = padL + start * bw, w = (i - start) * bw;
      if (band % 2 === 0) g += `<rect x="${x}" y="${padT}" width="${w}" height="${plotH}" fill="var(--surface-2)"/>`;
      g += `<line x1="${x}" y1="${padT}" x2="${x}" y2="${H - padB + 6}" stroke="var(--rule)" stroke-width="1"/>`;
      if (w > 54) g += `<text x="${x + 5}" y="${H - padB + 19}" font-family="'DM Mono', monospace" font-size="9.5" fill="var(--ink-4)">${escapeHtml(secName(res.sections[start]))}</text>`;
      start = i; band++;
    }
  }

  scores.forEach((v, i) => {
    const h = Math.max(2, padT + plotH - y(v)), x = padL + i * bw + bw * 0.16, w = bw * 0.68, top = padT + plotH - h;
    const isTop = topIdx.has(i);
    g += `<rect x="${x.toFixed(1)}" y="${top.toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" ` +
         `fill="${isTop ? "var(--retr)" : "var(--sunken)"}" stroke="${isTop ? "none" : "var(--rule)"}" stroke-width=".8">` +
         `<title>fragmento ${i + 1} · ${escapeHtml(secName(res.sections[i]))} · coseno ${v.toFixed(3)}</title></rect>`;
    if (isTop) {
      g += `<rect x="${(x + w / 2 - 8).toFixed(1)}" y="${(top - 19).toFixed(1)}" width="16" height="15" fill="var(--retr)"/>` +
           `<text x="${(x + w / 2).toFixed(1)}" y="${(top - 11).toFixed(1)}" text-anchor="middle" dominant-baseline="central" ` +
           `font-family="'DM Mono', monospace" font-size="10" fill="var(--surface)">${topIdx.get(i)}</text>`;
    }
  });
  g += `<line x1="${padL}" y1="${padT + plotH}" x2="${W - padR}" y2="${padT + plotH}" stroke="var(--ink-3)" stroke-width="1.2"/>`;
  g += `<text x="${W - padR}" y="${padT - 8}" text-anchor="end" font-family="'DM Mono', monospace" font-size="9" fill="var(--ink-4)">coseno de ${lo.toFixed(2)} a ${hi.toFixed(2)}</text>`;
  $("map").innerHTML = g;
  $("mapEyebrow").textContent = `${n} fragmentos · ${new Set(res.sections).size} secciones`;
}

updateCounter();
boot();
