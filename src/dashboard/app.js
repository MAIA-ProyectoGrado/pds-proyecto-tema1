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
    section: "Method",
    arxiv: "1903.10676",   // Beltagy et al., SciBERT
  },
  {
    ctx: "The masked language modelling objective was first introduced by Devlin et al. (2019).",
    section: "Introduction",
    arxiv: "1810.04805",   // Devlin et al., BERT
  },
];

/* ── Nombres legibles de las secciones canónicas que devuelve la API ── */
const SECTION_NAMES = {
  Abstract: "Abstract", Introduccion: "Introduction", Trabajo_relacionado: "Related Work",
  Metodos: "Methods", Experimentos: "Experiments", Resultados: "Results",
  Discusion: "Discussion", Conclusion: "Conclusion", Cuerpo_generico: "Body",
  Otra: "Other", Desconocida: "Unknown",
};
const secName = (s) => SECTION_NAMES[s] || s;

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
  hideRetrieval();
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
  const rt = $("retrTag");
  rt.textContent = META.retrieval_available ? "Top-3" : "Top-3 no disponible";
  rt.className = META.retrieval_available ? "" : "down";
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
    const badge = m.recommended ? '<span class="badge">Recomendado</span>' : "";
    const unavailable = m.available ? "" : ' — <span style="color:var(--warn)">pesos no encontrados</span>';
    return `
      <label class="model">
        <input type="radio" name="model" value="${m.id}" ${checked ? "checked" : ""} ${m.available ? "" : "disabled"}>
        <span>
          <span class="kind">${m.kind}</span>
          <span class="name">${m.name}${badge}${unavailable}</span>
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

/* ═══ Documento citado: id de arXiv o texto pegado ═══ */
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
    showBanner("<b>No hay ningún clasificador disponible.</b> Revisa que los pesos estén en <code>models/</code>.");
    return;
  }

  busy = true;
  $("run").disabled = true;
  $("run").lastChild.textContent = " Clasificando…";
  hideBanner();

  const doc = citedDocument();
  const payload = { citation_context: text, rhetorical_section: $("section").value };

  // La recuperación tarda segundos (descarga + embeddings); la clasificación, milisegundos.
  // Se lanzan en paralelo y cada panel muestra su propio estado.
  if (doc) showRetrievalLoading(doc);
  else hideRetrieval();

  const predictP = fetch(`${API}/predict`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...payload, model: selected.value }),
  });
  const retrieveP = doc ? fetch(`${API}/retrieve`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...payload, ...doc, top_k: 3 }),
  }) : null;
  if (retrieveP) retrieveP.catch(() => {});   // si la clasificación falla, no dejar un rechazo huérfano

  try {
    const r = await predictP;
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
    return;
  } finally {
    busy = false;
    $("run").disabled = false;
    $("run").lastChild.textContent = " Clasificar";
  }

  if (retrieveP) {
    let r, body;
    try {
      r = await retrieveP;
      body = await r.json().catch(() => ({}));
    } catch (err) {
      showRetrievalError("Se perdió la conexión con la API durante la recuperación.", "");
      return;
    }
    if (!r.ok) {
      showRetrievalError(`La API rechazó la recuperación (${r.status}).`, body.detail || "");
      return;
    }
    // Un fallo al pintar no es un fallo de red: se informa como lo que es.
    try {
      renderRetrieval(body);
    } catch (err) {
      console.error("renderRetrieval:", err);
      showRetrievalError("No se pudieron mostrar los pasajes recuperados.", String(err));
    }
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

  $("tele").innerHTML = `
    <div class="tc"><span class="l">Modelo</span><span class="v" style="font-size:12px">${modelName(res.model_id)}</span></div>
    <div class="tc"><span class="l">Latencia</span><span class="v">${res.latency_ms} ms</span></div>
    <div class="tc"><span class="l">Confianza</span><span class="v">${res.confidence.toFixed(3)}</span></div>
    <div class="tc"><span class="l">Ambiguo</span><span class="v" style="color:var(--${res.ambiguous ? "warn" : "ok"})">${res.ambiguous ? "Sí" : "No"}</span></div>`;
}

/* ═══ Recuperación de pasajes ═══ */
function hideRetrieval() {
  $("pChunks").hidden = true;
  $("pMap").hidden = true;
}

function showRetrievalLoading(doc) {
  $("pChunks").hidden = false;
  $("pMap").hidden = true;
  $("paperMeta").hidden = true;
  $("chunks").innerHTML = "";
  const st = $("rstate");
  st.className = "rstate";
  st.innerHTML = doc.arxiv_id
    ? `<span class="spin"></span>Descargando <b>arXiv:${escapeHtml(doc.arxiv_id)}</b>, segmentando y calculando similitudes con SciBERT…`
    : `<span class="spin"></span>Segmentando el texto y calculando similitudes con SciBERT…`;
  st.hidden = false;
}

function showRetrievalError(title, detail) {
  $("pChunks").hidden = false;
  $("pMap").hidden = true;
  $("paperMeta").hidden = true;
  $("chunks").innerHTML = "";
  const st = $("rstate");
  st.className = "rstate err";
  st.innerHTML = `<b>${escapeHtml(title)}</b> ${escapeHtml(detail)}<br>` +
    `La clasificación de arriba no depende de este paso y sigue siendo válida.`;
  st.hidden = false;
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
      `<span class="pa">${escapeHtml(authors)}${p.year ? " · " + p.year : ""}</span>` +
      `<span class="ps"><b>${res.n_words.toLocaleString("es")} palabras</b><b>${res.n_chunks} fragmentos</b>` +
      `<b>${res.cached ? "caché" : Math.round(res.latency_ms.download) + " ms descarga"}</b></span>`;
  }
  meta.hidden = false;

  $("chunksEyebrow").textContent = `similitud coseno · SciBERT · ${res.latency_ms.total} ms`;
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
  g += `<text x="${W - padR}" y="${padT - 8}" text-anchor="end" font-family="'DM Mono', monospace" font-size="9" fill="var(--ink-4)">coseno ${lo.toFixed(2)} – ${hi.toFixed(2)}</text>`;
  $("map").innerHTML = g;
  $("mapEyebrow").textContent = `${n} fragmentos · ${new Set(res.sections).size} secciones`;
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
