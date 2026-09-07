/* Dashboard SCIF */

const API = (window.SCIF_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

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

/* ── Textos de ejemplo: solo entradas para el textarea ── */
const EXAMPLE_TEXTS = [
  "Although several citation function taxonomies have been proposed for the computer science domain [7], no publicly available corpus aligns the citation context with passage-level evidence inside the cited document.",
  "We build our citation encoder on top of SciBERT [4] and keep its in-domain vocabulary unchanged so that scientific subword units are preserved throughout fine-tuning.",
  "For a comprehensive review of contextual citation analysis, see Jones (2019).",
];

let META = null; // respuesta de GET /models
let busy = false;

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
}

function setApiState(text, cls) {
  const el = $("apiState");
  el.textContent = text;
  el.className = cls || "";
}

/* ═══ Arranque: leer el catálogo de la API ═══ */
async function boot() {
  $("apiUrlLabel").textContent = API;
  try {
    const r = await fetch(`${API}/models`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    META = await r.json();
  } catch (err) {
    invalidate();
    setApiState("API no disponible", "down");
    showBanner(
      `<b>No se pudo contactar la API</b> en <code>${API}</code>. ` +
      `Levántala con <code>uvicorn src.api.main:app --port 8000</code> desde la raíz del repo ` +
      `y recarga esta página. El tablero no muestra predicciones sin la API.`
    );
    $("empty").hidden = true;
    return;
  }
  setApiState("API conectada", "live");
  hideBanner();
  renderSections();
  renderModels();
  renderDistSkeleton();
  $("run").disabled = false;
}

function renderSections() {
  $("section").innerHTML = META.sections
    .map((s) => `<option value="${s}"${s === "Introduction" ? " selected" : ""}>${s}</option>`)
    .join("");
}

function renderModels() {
  $("models").innerHTML = META.models.map((m) => {
    const checked = m.id === META.default_model && m.available;
    const metric = m.metric_verified && m.f1_macro_val !== null
      ? `<span class="metric">F1-macro val ${m.f1_macro_val.toFixed(3)} · verificado</span>`
      : `<span class="metric unverified">Sin métrica verificada para este artefacto</span>`;
    const badge = m.recommended ? '<span class="badge">Recomendado</span>' : "";
    const unavailable = m.available ? "" : ' — <span style="color:var(--warn)">pesos no encontrados</span>';
    return `
      <label class="model" title="${m.metric_note.replace(/"/g, "&quot;")}">
        <input type="radio" name="model" value="${m.id}" ${checked ? "checked" : ""} ${m.available ? "" : "disabled"}>
        <span>
          <span class="kind">${m.kind}</span>
          <span class="name">${m.name}${badge}${unavailable}</span>
          ${metric}
        </span>
      </label>`;
  }).join("");
}

function colorFor(label) {
  const i = META.labels.indexOf(label);
  return `var(--c${(i < 0 ? 0 : i) + 1})`;
}

function renderDistSkeleton() {
  $("dist").innerHTML = META.labels.map((lab) => `
    <div class="drow" data-label="${lab}" style="--cc:${colorFor(lab)}">
      <span class="dname"><i></i><span title="${META.label_display[lab]}">${META.label_display[lab]}</span></span>
      <span class="dtrack"><span class="dbar"></span></span>
      <span class="dval">—</span>
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
    ctxEl.value = EXAMPLE_TEXTS[+b.dataset.ex];
    updateCounter();
    ctxEl.focus();
  });
});

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
    showBanner("<b>No hay ningún clasificador disponible.</b> Revisa que los pesos estén en <code>models/</code>.");
    return;
  }

  busy = true;
  $("run").disabled = true;
  $("run").lastChild.textContent = " Clasificando…";
  hideBanner();

  try {
    const r = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        citation_context: text,
        rhetorical_section: $("section").value,
        model: selected.value,
      }),
    });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) {
      invalidate();
      showBanner(`<b>La API rechazó la petición (${r.status}).</b> ${body.detail || ""}`);
      return;
    }
    render(body);
  } catch (err) {
    invalidate();
    setApiState("API no disponible", "down");
    showBanner(
      `<b>Se perdió la conexión con la API.</b> No se muestra ninguna predicción: ` +
      `el tablero solo presenta resultados que provengan del modelo.`
    );
  } finally {
    busy = false;
    $("run").disabled = false;
    $("run").lastChild.textContent = " Clasificar";
  }
}

/* ═══ Render de la respuesta ═══ */
function render(res) {
  $("results").hidden = false;
  $("empty").hidden = true;
  setApiState("API conectada", "live");

  const color = colorFor(res.predicted_label);

  $("vName").textContent = res.predicted_label_display;
  $("vName").style.color = color;
  $("vDef").textContent = DEFINITIONS[res.predicted_label] || "";
  $("mChip").textContent = modelName(res.model_id);

  const arc = $("gArc");
  arc.style.stroke = color;
  arc.setAttribute("stroke-dashoffset", (352 * (1 - res.confidence)).toFixed(1));
  $("gTxt").textContent = res.confidence.toFixed(2);

  // Aviso de ambigüedad: viene del propio predictor (margen < 0.10)
  const entries = Object.entries(res.probabilities).sort((a, b) => b[1] - a[1]);
  const al = $("alert");
  if (res.ambiguous && entries.length > 1) {
    const d = (entries[0][1] - entries[1][1]).toFixed(2);
    $("alertTxt").innerHTML =
      `El modelo marcó este caso como <b>ambiguo</b>: la diferencia entre ` +
      `<b>${META.label_display[entries[0][0]]}</b> y <b>${META.label_display[entries[1][0]]}</b> ` +
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
    row.querySelector(".dval").textContent = p == null ? "—" : p.toFixed(3);
  });

  const [ctxPart, secPart] = splitInput(res.input_text);
  $("ioText").innerHTML = `${escapeHtml(ctxPart)}<span class="sec"> [SEC] ${escapeHtml(secPart)}</span>`;

  const meta = META.models.find((m) => m.id === res.model_id) || {};
  $("tele").innerHTML = `
    <div class="tc"><span class="l">Modelo</span><span class="v" style="font-size:12px">${modelName(res.model_id)}</span></div>
    <div class="tc"><span class="l">Latencia</span><span class="v">${res.latency_ms} ms</span></div>
    <div class="tc"><span class="l">Confianza</span><span class="v">${res.confidence.toFixed(3)}</span></div>
    <div class="tc"><span class="l">Ambiguo</span><span class="v" style="color:var(--${res.ambiguous ? "warn" : "ok"})">${res.ambiguous ? "Sí" : "No"}</span></div>`;
  $("prov").textContent = meta.metric_note || "";
}

function splitInput(s) {
  const i = s.lastIndexOf(" [SEC] ");
  return i < 0 ? [s, ""] : [s.slice(0, i), s.slice(i + 7)];
}
function modelName(id) {
  const m = META.models.find((x) => x.id === id);
  return m ? m.name : id;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

updateCounter();
boot();
