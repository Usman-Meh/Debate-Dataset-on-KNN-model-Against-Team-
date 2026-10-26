/* KNN Admission Predictor — vanilla JS for the three pages. */
"use strict";

const APP = window.APP || {};
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const charts = {};

const COLORS = {
  primary: "#1f63d6", navy: "#14254a", academic: "#1f63d6", reviewer: "#e08a1e",
  ok: "#18794e", error: "#b42318", muted: "#98a2b3", train: "rgba(20, 37, 74, 0.35)",
};

/* ---------------------------------------------------------------- helpers */

async function api(url, options = {}) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (err) {
    throw new Error("Could not reach the server. Is it still running?");
  }
  let body = null;
  try { body = await response.json(); } catch (err) { /* non-JSON body */ }
  if (!response.ok) {
    throw new Error((body && body.detail) || `Request failed (${response.status}).`);
  }
  return body;
}

const postJSON = (url, data) =>
  api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

const escapeHtml = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const num = (v, digits = 3) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(digits));
const pct = (v, digits = 1) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(digits)}%`);
const signed = (v, digits = 3) => (v >= 0 ? "+" : "") + Number(v).toFixed(digits);
const fmtCell = (v) => (v === null || v === undefined ? "" : typeof v === "number" ? +v.toFixed(4) : v);
const fmtDate = (iso) => (iso ? iso.replace("T", " ") : "—");

function show(el, visible = true) { if (el) el.hidden = !visible; }

function showError(el, message) {
  if (!el) return;
  el.textContent = message || "";
  el.hidden = !message;
}

function setLoading(button, loading, label) {
  if (!button) return;
  if (loading) {
    button.dataset.label = button.textContent;
    button.textContent = label || "Working…";
  } else if (button.dataset.label) {
    button.textContent = button.dataset.label;
  }
  button.disabled = loading;
  button.classList.toggle("loading", loading);
}

function renderTable(table, columns, rows, { cellClass, rowClass, format } = {}) {
  // Right-align a header when its first data cell is numeric, so headers line up with values.
  const first = rows[0] || {};
  const headClass = (c) => ((cellClass ? cellClass(c, first[c], first) : (typeof first[c] === "number" ? "num" : "")) || "")
    .split(" ").includes("num") ? "num" : "";
  const head = `<thead><tr>${columns.map((c) => `<th class="${headClass(c)}">${escapeHtml(c)}</th>`).join("")}</tr></thead>`;
  const body = rows.map((row, i) => {
    const cells = columns.map((c) => {
      const value = format ? format(c, row[c], row) : fmtCell(row[c]);
      const cls = cellClass ? cellClass(c, row[c], row) : (typeof row[c] === "number" ? "num" : "");
      return `<td class="${cls || ""}">${escapeHtml(value)}</td>`;
    }).join("");
    return `<tr class="${rowClass ? rowClass(row, i) || "" : ""}">${cells}</tr>`;
  }).join("");
  table.innerHTML = head + `<tbody>${body}</tbody>`;
}

function makeChart(id, config) {
  const canvas = document.getElementById(id);
  if (!canvas || typeof Chart === "undefined") return null;
  if (charts[id]) charts[id].destroy();
  config.options = Object.assign({ responsive: true, maintainAspectRatio: false, animation: { duration: 250 } }, config.options);
  charts[id] = new Chart(canvas, config);
  return charts[id];
}

function setupDropzone(zone, input, onFile) {
  if (!zone || !input) return;
  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  input.addEventListener("change", () => { if (input.files[0]) onFile(input.files[0]); input.value = ""; });
  ["dragenter", "dragover"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("drag"); }));
  zone.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]); });
}

const MAX_UPLOAD = 5 * 1024 * 1024;
function checkFile(file) {
  if (!/\.(csv|xlsx|xls)$/i.test(file.name)) return "Unsupported file format. Please upload a .csv, .xlsx or .xls file.";
  if (file.size > MAX_UPLOAD) return "File is too large. The maximum size is 5 MB.";
  if (file.size === 0) return "The selected file is empty.";
  return null;
}

function fillSheetPicker(wrapper, select, sheets, current) {
  show(wrapper, sheets && sheets.length > 1);
  if (!sheets || sheets.length <= 1) return;
  select.innerHTML = sheets.map((s) => `<option ${s === current ? "selected" : ""}>${escapeHtml(s)}</option>`).join("");
}

function scatterConfig(actual, predicted, extra = []) {
  const points = actual.map((a, i) => ({ x: a, y: predicted[i] }));
  const all = actual.concat(predicted);
  const lo = Math.max(0, Math.floor(Math.min(...all) * 10) / 10);
  const hi = Math.min(1, Math.ceil(Math.max(...all) * 10) / 10);
  return {
    type: "scatter",
    data: {
      datasets: [
        ...extra,
        { label: "Test students", data: points, backgroundColor: "rgba(31, 99, 214, 0.65)", pointRadius: 3.5 },
        { label: "Perfect prediction", data: [{ x: lo, y: lo }, { x: hi, y: hi }], showLine: true, pointRadius: 0,
          borderColor: COLORS.muted, borderDash: [6, 4], borderWidth: 1.5 },
      ],
    },
    options: {
      scales: {
        x: { min: lo, max: hi, title: { display: true, text: "Actual Chance of Admit" } },
        y: { min: lo, max: hi, title: { display: true, text: "Predicted Chance of Admit" } },
      },
      plugins: { tooltip: { callbacks: { label: (c) => `actual ${pct(c.parsed.x)}, predicted ${pct(c.parsed.y)}` } } },
    },
  };
}

function updateModelStrip(meta) {
  let strip = $(".model-strip");
  if (!strip) {
    strip = document.createElement("div");
    strip.className = "model-strip";
    $(".container").prepend(strip);
  }
  const cfg = meta.config;
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
  strip.innerHTML = `<span class="dot ok"></span> Active model: <strong>K = ${cfg.k}</strong>
    · ${cap(cfg.metric)} · ${cap(cfg.weights)} weights · ${meta.features.length} features
    · Test R² <strong>${num(meta.metrics.test.r2)}</strong> · trained ${escapeHtml(fmtDate(meta.created_at))}`;
}

function medianProfile() {
  const stats = (APP.model && APP.model.feature_stats) || {};
  const defaults = { "GRE Score": 317, "TOEFL Score": 107, "University Rating": 3, "SOP": 3.5, "LOR": 3.5, "CGPA": 8.56, "Research": 1 };
  const profile = {};
  for (const f of APP.features) {
    let v = stats[f] ? stats[f].median : defaults[f];
    if (f === "GRE Score" || f === "TOEFL Score" || f === "University Rating" || f === "Research") v = Math.round(v);
    else if (f === "SOP" || f === "LOR") v = Math.round(v * 2) / 2;
    else v = Math.round(v * 100) / 100;
    profile[f] = v;
  }
  return profile;
}

/* Validate a value against the server's feature ranges; returns an error string or null. */
function validateFeature(feature, raw) {
  const [lo, hi, integer] = APP.ranges[feature];
  if (raw === "" || raw === null || raw === undefined) return `${feature} is required.`;
  const v = Number(raw);
  if (!Number.isFinite(v)) return `${feature} must be a number.`;
  if (v < lo || v > hi) return `${feature} must be between ${lo} and ${hi}.`;
  if (integer && !Number.isInteger(v)) return `${feature} must be a whole number.`;
  return null;
}

/* ---------------------------------------------------------------- Train page */

function initTrain() {
  const state = { uploadId: null, sheet: null, validRows: 0, valid: false };
  const uploadError = $("#upload-error");
  const trainBtn = $("#train-btn");
  const kInput = $("#k");
  const testRange = $("#test-percent");

  function trainCount() {
    const n = state.validRows;
    return n - Math.ceil(n * Number(testRange.value) / 100);
  }

  function refreshTrainState() {
    const k = Number(kInput.value);
    const nTrain = trainCount();
    const features = $$("#feature-checks input:checked");
    let problem = null;
    if (!state.valid) problem = "Upload a valid dataset to enable training.";
    else if (!Number.isInteger(k) || k < 1) problem = "K must be a whole number of at least 1.";
    else if (k > nTrain) problem = `K cannot exceed the ${nTrain} training samples.`;
    else if (!features.length) problem = "Select at least one feature.";
    trainBtn.disabled = Boolean(problem);
    $("#train-status").textContent = problem || `Ready: ${nTrain} training / ${state.validRows - nTrain} testing samples.`;
    const hint = $("#k-hint");
    hint.textContent = state.valid ? `Valid range: 1 – ${nTrain} (training samples available).`
      : "Number of similar students averaged for each prediction.";
    hint.classList.toggle("error", Boolean(state.valid && (k < 1 || k > nTrain || !Number.isInteger(k))));
    kInput.classList.toggle("invalid", hint.classList.contains("error"));
  }

  function renderUpload(d) {
    state.uploadId = d.upload_id;
    state.sheet = d.sheet;
    state.validRows = d.validation.valid_rows;
    state.valid = d.validation.valid;
    show($("#upload-result"));
    $("#up-filename").textContent = d.filename + (d.sheet ? ` · ${d.sheet}` : "");
    $("#up-rows").textContent = d.rows;
    $("#up-cols").textContent = d.columns;
    $("#up-valid").textContent = d.validation.valid_rows;
    fillSheetPicker($("#sheet-picker"), $("#sheet-select"), d.sheets, d.sheet);

    const v = d.validation;
    const issues = v.issues.length ? `<ul>${v.issues.map((i) => `<li>${escapeHtml(i)}</li>`).join("")}</ul>` : "";
    let cls = "ok";
    let title = "Dataset is valid: all seven admission features and the Chance of Admit target were found.";
    if (!v.valid) { cls = "error"; title = "This dataset cannot be used for training yet."; }
    else if (v.invalid_rows) { cls = "warn"; }
    $("#validation-box").innerHTML = `<div class="alert ${cls}"><strong>${escapeHtml(title)}</strong>${issues}</div>`;

    $("#column-chips").innerHTML = d.column_names.map((c) => {
      const cls2 = APP.features.includes(c) ? "ok" : c === "Chance of Admit" ? "target" : "";
      return `<span class="chip ${cls2}">${escapeHtml(c)}</span>`;
    }).join("");
    renderTable($("#missing-table"), d.column_names, [d.missing_values], {
      cellClass: (c, val) => (val > 0 ? "missing num" : "num"),
    });
    renderTable($("#preview-table"), d.column_names, d.preview);
    refreshTrainState();
  }

  async function upload(file) {
    showError(uploadError, checkFile(file));
    if (checkFile(file)) return;
    const form = new FormData();
    form.append("file", file);
    $("#dropzone p").textContent = `Uploading ${file.name}…`;
    try {
      renderUpload(await api("/api/upload", { method: "POST", body: form }));
    } catch (err) {
      showError(uploadError, err.message);
      state.valid = false;
      show($("#upload-result"), false);
      refreshTrainState();
    } finally {
      $("#dropzone p").innerHTML = "<strong>Drag &amp; drop</strong> another file here, or <span class=\"link\">browse</span>";
    }
  }

  setupDropzone($("#dropzone"), $("#file-input"), upload);

  $("#sheet-select").addEventListener("change", async (e) => {
    showError(uploadError, null);
    try {
      renderUpload(await postJSON("/api/upload/sheet", { upload_id: state.uploadId, sheet: e.target.value }));
    } catch (err) {
      showError(uploadError, err.message);
    }
  });

  testRange.addEventListener("input", () => {
    $("#test-label").textContent = `${testRange.value}%`;
    $("#train-label").textContent = `${100 - Number(testRange.value)}%`;
    refreshTrainState();
  });
  kInput.addEventListener("input", refreshTrainState);
  $$("#feature-checks input").forEach((cb) => cb.addEventListener("change", refreshTrainState));

  $("#train-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (trainBtn.disabled) return;
    showError($("#train-error"), null);
    setLoading(trainBtn, true, "Training…");
    try {
      const result = await postJSON("/api/train", {
        source: "upload",
        upload_id: state.uploadId,
        sheet: state.sheet,
        k: Number(kInput.value),
        test_percent: Number(testRange.value),
        metric: $("#metric").value,
        weights: $("#weights").value,
        features: $$("#feature-checks input:checked").map((cb) => cb.value),
      });
      APP.model = result.model;
      renderResults(result.model, result.scatter, result.train_scatter, false);
      updateModelStrip(result.model);
      $("#results-card").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      showError($("#train-error"), err.message);
    } finally {
      setLoading(trainBtn, false);
      refreshTrainState();
    }
  });

  function renderResults(meta, scatter, trainScatter, existing) {
    show($("#results-card"));
    const m = meta.metrics;
    $("#m-train-r2").textContent = num(m.train.r2);
    $("#m-test-r2").textContent = num(m.test.r2);
    $("#m-train-mae").textContent = num(m.train.mae);
    $("#m-test-mae").textContent = num(m.test.mae);
    $("#m-train-rmse").textContent = num(m.train.rmse);
    $("#m-test-rmse").textContent = num(m.test.rmse);
    $("#r-ntrain").textContent = meta.n_train;
    $("#r-ntest").textContent = meta.n_test;
    $("#r-k").textContent = meta.config.k;
    $("#r-created").textContent = fmtDate(meta.created_at);
    const parts = [];
    if (existing) parts.push(`Showing the currently active model (trained on ${meta.source_filename}).`);
    parts.push(`Split ${meta.config.train_percent}% / ${meta.config.test_percent}% · ${meta.config.metric} distance · ${meta.config.weights} weights · features: ${meta.features.join(", ")}.`);
    if (meta.dropped_rows) parts.push(`${meta.dropped_rows} invalid row(s) were skipped.`);
    $("#r-dropped").textContent = parts.join(" ");
    const trainSet = {
      label: "Training students", data: trainScatter.actual.map((a, i) => ({ x: a, y: trainScatter.predicted[i] })),
      backgroundColor: COLORS.train, pointRadius: 2,
    };
    makeChart("train-scatter", scatterConfig(scatter.actual, scatter.predicted, [trainSet]));
  }

  // Show the persisted model so users see it survives restarts.
  if (APP.model) {
    api("/api/evaluation")
      .then((ev) => renderResults(ev.model, ev.scatter, ev.train_scatter, true))
      .catch(() => {});
  }
  refreshTrainState();
}

/* ---------------------------------------------------------------- Predict page */

const FORM_IDS = {
  "GRE Score": "gre_score", "TOEFL Score": "toefl_score", "University Rating": "university_rating",
  "SOP": "sop", "LOR": "lor", "CGPA": "cgpa", "Research": "research",
};

function initPredict() {
  const form = $("#predict-form");
  if (!form) return; // no model trained yet

  $("#fill-example").addEventListener("click", () => {
    const profile = medianProfile();
    for (const [feature, id] of Object.entries(FORM_IDS)) {
      $(`#${id}`).value = String(profile[feature]);
      $(`#${id}`).classList.remove("invalid");
    }
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const errors = [];
    const payload = {};
    for (const [feature, id] of Object.entries(FORM_IDS)) {
      const input = $(`#${id}`);
      const problem = validateFeature(feature, input.value);
      input.classList.toggle("invalid", Boolean(problem));
      if (problem) errors.push(problem);
      else payload[id] = Number(input.value);
    }
    if (errors.length) {
      showError($("#predict-error"), errors.join(" "));
      return;
    }
    showError($("#predict-error"), null);
    const button = form.querySelector("button[type=submit]");
    setLoading(button, true, "Predicting…");
    try {
      const r = await postJSON("/api/predict", payload);
      show($("#predict-empty"), false);
      show($("#predict-result"));
      $("#predict-value").textContent = `${r.percentage.toFixed(1)}%`;
      $("#predict-bar").style.width = `${Math.max(0, Math.min(100, r.percentage))}%`;
      const k = APP.model ? APP.model.config.k : "K";
      $("#predict-context").textContent =
        `Average Chance of Admit of the ${k} most similar students in the training data (${APP.model.config.weights} weighting).`;
    } catch (err) {
      showError($("#predict-error"), err.message);
    } finally {
      setLoading(button, false);
    }
  });

  // Bulk predictions
  let batchFile = null;
  async function runBatch(file, sheet) {
    showError($("#batch-error"), checkFile(file));
    if (checkFile(file)) return;
    batchFile = file;
    const data = new FormData();
    data.append("file", file);
    if (sheet) data.append("sheet", sheet);
    const zoneText = $("#batch-dropzone p");
    zoneText.textContent = `Predicting ${file.name}…`;
    try {
      renderBatch(await api("/api/predict-batch", { method: "POST", body: data }));
    } catch (err) {
      showError($("#batch-error"), err.message);
      show($("#batch-result"), false);
    } finally {
      zoneText.innerHTML = "<strong>Drag &amp; drop</strong> another file here, or <span class=\"link\">browse</span>";
    }
  }

  function renderBatch(d) {
    show($("#batch-result"));
    fillSheetPicker($("#batch-sheet-picker"), $("#batch-sheet-select"), d.sheets, d.sheet);
    $("#b-filename").textContent = d.filename + (d.sheet ? ` · ${d.sheet}` : "");
    $("#b-total").textContent = d.total;
    $("#b-success").textContent = d.successful;
    $("#b-invalid").textContent = d.invalid_count;
    $("#b-invalid").classList.toggle("ok-text", d.invalid_count === 0);
    $("#download-csv").href = `/api/predict-batch/${d.result_id}/download?format=csv`;
    $("#download-xlsx").href = `/api/predict-batch/${d.result_id}/download?format=xlsx`;
    $("#b-truncated").textContent = d.truncated ? `(showing first 1,000 of ${d.total} rows — download for all)` : "";

    show($("#invalid-box"), d.invalid_count > 0);
    renderTable($("#invalid-table"), ["Row", "Problems"],
      d.invalid.map((r) => ({ Row: r.row, Problems: r.errors.join("; ") })));

    const pred = "Predicted_Chance_of_Admit";
    const rows = d.rows.map((r, i) => ({ "#": i + 1, ...r }));
    renderTable($("#batch-table"), ["#", ...d.columns], rows, {
      format: (c, v) => (c === pred ? (v === null ? "invalid" : pct(v)) : fmtCell(v)),
      cellClass: (c, v) => (c === pred ? "pred num" : typeof v === "number" ? "num" : ""),
      rowClass: (r) => (r[pred] === null ? "bad" : ""),
    });
  }

  setupDropzone($("#batch-dropzone"), $("#batch-input"), (file) => runBatch(file, null));
  $("#batch-sheet-select").addEventListener("change", (e) => batchFile && runBatch(batchFile, e.target.value));
}

/* ---------------------------------------------------------------- Evaluation page */

function initEvaluation() {
  if (!APP.model) return;
  const errorBox = $("#eval-error");

  Promise.all([api("/api/evaluation"), api("/api/k-analysis")])
    .then(([ev, k]) => {
      renderPerformance(ev);
      renderImportance(ev);
      renderK(k, ev.model);
      renderErrors(ev);
    })
    .catch((err) => showError(errorBox, err.message));

  initSensitivity();
  initExperiments();
}

function renderPerformance(ev) {
  const { train, test, gap } = ev.comparison;
  const rows = [
    ["R² (higher is better)", train.r2, test.r2, gap.r2],
    ["MAE (lower is better)", train.mae, test.mae, gap.mae],
    ["RMSE (lower is better)", train.rmse, test.rmse, gap.rmse],
  ];
  $("#perf-table tbody").innerHTML = rows.map(([name, a, b, g]) =>
    `<tr><td>${name}</td><td class="num">${num(a)}</td><td class="num">${num(b)}</td><td class="num">${signed(g)}</td></tr>`).join("");
  $("#perf-notes").innerHTML = ev.notes.map((n) => `<p>${escapeHtml(n)}</p>`).join("");

  const bar = (id, label, values, max) => makeChart(id, {
    type: "bar",
    data: { labels: ["Training", "Testing"], datasets: [{ label, data: values, backgroundColor: [COLORS.navy, COLORS.primary], maxBarThickness: 70 }] },
    options: { plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, max } } },
  });
  bar("r2-chart", "R²", [train.r2, test.r2], 1);
  bar("mae-chart", "MAE", [train.mae, test.mae]);
}

function renderImportance(ev) {
  const items = ev.feature_importance;
  makeChart("importance-chart", {
    type: "bar",
    data: {
      labels: items.map((i) => i.feature),
      datasets: [{
        label: "Increase in test MAE when shuffled",
        data: items.map((i) => i.importance),
        backgroundColor: items.map((i) => (i.group === "academic" ? COLORS.academic : COLORS.reviewer)),
      }],
    },
    options: {
      indexAxis: "y",
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (c) => `MAE +${num(c.parsed.x, 4)} (± ${num(items[c.dataIndex].std, 4)})` } },
      },
      scales: { x: { title: { display: true, text: "Increase in test MAE when the feature is shuffled" } } },
    },
  });
  const positive = items.filter((i) => i.importance > 0);
  const total = positive.reduce((s, i) => s + i.importance, 0);
  if (total > 0) {
    const share = (group) => positive.filter((i) => i.group === group).reduce((s, i) => s + i.importance, 0) / total;
    $("#importance-summary").innerHTML =
      `Most influential feature: <strong>${escapeHtml(items[0].feature)}</strong>. Academic features account for ` +
      `<strong>${pct(share("academic"), 0)}</strong> of total importance; reviewer-influenced features for ` +
      `<strong>${pct(share("reviewer"), 0)}</strong>.`;
  }
}

function renderK(k, meta) {
  $("#k-folds").textContent = k.folds;
  $("#k-current").textContent = k.current_k;
  $("#k-best").textContent = k.best_k;
  const pointColor = k.k_values.map((v) => (v === k.best_k ? COLORS.ok : v === k.current_k ? COLORS.reviewer : COLORS.primary));
  const pointSize = k.k_values.map((v) => (v === k.best_k || v === k.current_k ? 6 : 3));
  makeChart("k-chart", {
    type: "line",
    data: {
      labels: k.k_values,
      datasets: [{ label: "Average cross-validation MAE", data: k.cv_mae_mean, borderColor: COLORS.primary,
        pointBackgroundColor: pointColor, pointBorderColor: pointColor, pointRadius: pointSize, tension: 0.15 }],
    },
    options: {
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (c) => `CV MAE ${num(c.parsed.y, 4)} (± ${num(k.cv_mae_std[c.dataIndex], 4)})` } },
      },
      scales: { x: { title: { display: true, text: "K (number of neighbours)" } },
                y: { title: { display: true, text: "Cross-validation MAE" } } },
    },
  });
  let text = `The lowest average validation MAE (${num(k.best_cv_mae, 4)}) was at <strong>K = ${k.best_k}</strong>.`;
  if (k.current_k_cv_mae !== null) text += ` The current K = ${k.current_k} scores ${num(k.current_k_cv_mae, 4)}.`;
  if (k.best_k === k.k_values[k.k_values.length - 1]) {
    text += " The best K is at the edge of the tested range, so slightly larger values may perform similarly.";
  }
  text += " <span class=\"muted\">Green = recommended, orange = current.</span>";
  $("#k-summary").innerHTML = text;

  const button = $("#retrain-btn");
  if (k.best_k !== k.current_k) {
    button.textContent = `Retrain with K = ${k.best_k}`;
    show(button);
    button.addEventListener("click", async () => {
      if (!window.confirm(`Train a new model with K = ${k.best_k}? It will replace the active model (same data, split and settings).`)) return;
      setLoading(button, true, "Retraining…");
      try {
        await postJSON("/api/train", {
          source: "active_model", k: k.best_k, test_percent: meta.config.test_percent,
          metric: meta.config.metric, weights: meta.config.weights, features: meta.features,
        });
        window.location.reload();
      } catch (err) {
        $("#retrain-status").textContent = err.message;
        setLoading(button, false);
      }
    });
  } else {
    $("#retrain-status").textContent = "The active model already uses the recommended K.";
  }
}

function renderErrors(ev) {
  const e = ev.errors;
  $("#err-mae").textContent = `${(e.mean_absolute_error * 100).toFixed(1)} pts`;
  $("#err-median").textContent = `${(e.median_absolute_error * 100).toFixed(1)} pts`;
  $("#err-bias").textContent = `${signed(e.mean_signed_error * 100, 1)} pts`;
  $("#err-max").textContent = `${(e.max_absolute_error * 100).toFixed(1)} pts`;
  $("#err-over5").textContent = e.over_5_points;
  $("#err-over10").textContent = e.over_10_points;
  makeChart("eval-scatter", scatterConfig(ev.scatter.actual, ev.scatter.predicted));

  const select = $("#errors-limit");
  select.options[1].textContent = `All ${e.ranked.length} test students`;
  const draw = () => {
    const rows = select.value === "all" ? e.ranked : e.ranked.slice(0, 10);
    renderTable($("#errors-table"), ["Student", "Actual", "Predicted", "Absolute Error"], rows.map((r) => ({
      Student: `#${r.student}`, Actual: num(r.actual, 2), Predicted: num(r.predicted, 2), "Absolute Error": num(r.abs_error, 3),
    })), { cellClass: (c) => (c === "Student" ? "" : "num") });
  };
  select.onchange = draw;
  draw();
}

/* ---------------------------------------------------------------- Evaluation experiments */

const pts = (v, digits = 1) => `${(v * 100).toFixed(digits)} pts`;

/* Wire a "Run" button to an API call; results are only computed when the user asks. */
function runExperiment(buttonId, url, errorId, resultId, render) {
  const button = $(`#${buttonId}`);
  if (!button) return;
  button.addEventListener("click", async () => {
    showError($(`#${errorId}`), null);
    setLoading(button, true, "Running…");
    try {
      render(await api(url));
      show($(`#${resultId}`));
    } catch (err) {
      showError($(`#${errorId}`), err.message);
    } finally {
      setLoading(button, false);
    }
  });
}

function renderSubjective(r) {
  const a = r.models.all;
  const b = r.models.academic;
  $("#subj-mean").textContent = pts(r.mean_abs_change);
  $("#subj-over5").textContent = `${r.over_5_points} of ${r.n_test}`;
  $("#subj-max").textContent = pts(r.max_abs_change);
  $("#subj-table").innerHTML = `
    <thead><tr><th>Test metric</th><th class="num">Model A<br><span class="muted small">all 7 features</span></th>
      <th class="num">Model B<br><span class="muted small">academic only</span></th></tr></thead>
    <tbody>
      <tr><td>MAE</td><td class="num">${num(a.mae, 4)}</td><td class="num">${num(b.mae, 4)}</td></tr>
      <tr><td>R²</td><td class="num">${num(a.r2, 3)}</td><td class="num">${num(b.r2, 3)}</td></tr>
      <tr><td>RMSE</td><td class="num">${num(a.rmse, 4)}</td><td class="num">${num(b.rmse, 4)}</td></tr>
    </tbody>`;
  renderTable($("#subj-changes"), ["Student", "Actual", "Model A", "Model B", "Change"], r.largest_changes.map((c) => ({
    Student: `#${c.student}`, Actual: pct(c.actual), "Model A": pct(c.all_features), "Model B": pct(c.academic_only),
    Change: `${signed(c.change * 100, 1)} pts`,
  })), { cellClass: (c) => (c === "Student" ? "" : "num") });
  const cfg = r.config;
  $("#subj-note").textContent = `Both models: K = ${cfg.k}, ${cfg.metric} distance, ${cfg.weights} weights, `
    + `${r.n_train} training / ${r.n_test} test students, each with its own StandardScaler fitted on the training rows. `
    + (r.matches_active_model ? "Model A reproduces the active model's predictions exactly." : "");
}

function renderConsistency(r) {
  const t = r.totals;
  $("#cons-tests").textContent = t.tests;
  $("#cons-dec").textContent = t.decreases;
  $("#cons-dec2").textContent = t.decreases_over_2;
  $("#cons-dec5").textContent = t.decreases_over_5;
  $("#cons-max").textContent = pts(t.largest_decrease);
  $("#cons-students").textContent = `${t.unique_students} of ${r.n_test}`;
  $("#cons-summary").innerHTML = `Academic improvements (GRE, TOEFL, CGPA, Research) lowered the prediction in `
    + `<strong>${t.academic_decreases}</strong> test(s), affecting <strong>${t.academic_unique_students}</strong> unique student(s). `
    + `<span class="muted">Counts are feature-change events; one student can appear in several.</span>`;
  renderTable($("#cons-features"), ["Feature", "Change", "Tests", "Unexpected Decreases", "Largest Decrease"],
    r.features.map((f) => ({
      Feature: f.feature, Change: f.step, Tests: f.tests, "Unexpected Decreases": f.decreases,
      "Largest Decrease": f.decreases ? pts(f.largest_decrease) : "—",
    })),
    { cellClass: (c) => (["Tests", "Unexpected Decreases", "Largest Decrease"].includes(c) ? "num" : ""),
      rowClass: (row) => (APP.academic.includes(row.Feature) && row["Unexpected Decreases"] > 0 ? "flag" : "") });
  const scope = $("#cons-examples-scope");
  const drawExamples = () => {
    const examples = scope.value === "academic" ? r.academic_examples : r.examples;
    show($("#cons-examples-empty"), !examples.length);
    renderTable($("#cons-examples"), ["Student", "Feature", "Original Value", "Improved Value", "Original Prediction", "Modified Prediction", "Change"],
      examples.map((e) => ({
        Student: `#${e.student}`, Feature: e.feature, "Original Value": e.original_value, "Improved Value": e.improved_value,
        "Original Prediction": pct(e.original_prediction), "Modified Prediction": pct(e.modified_prediction),
        Change: `${signed(e.change * 100, 1)} pts`,
      })),
      { cellClass: (c) => (["Student", "Feature"].includes(c) ? "" : "num"),
        rowClass: (row, i) => (examples[i].group === "academic" ? "flag" : "") });
  };
  scope.onchange = drawExamples;
  drawExamples();
}

function renderAtypical(r) {
  const fmtP = (p) => (p < 0.001 ? "< 0.001" : num(p, 3));
  $("#atyp-table").innerHTML = `
    <thead><tr><th>Group</th><th class="num">Students</th><th class="num">Avg distance</th><th class="num">Avg abs error</th></tr></thead>
    <tbody>
      <tr><td>Typical (80%)</td><td class="num">${r.typical.count}</td><td class="num">${num(r.typical.mean_distance, 2)}</td><td class="num">${pts(r.typical.mean_abs_error)}</td></tr>
      <tr><td>Atypical (20%)</td><td class="num">${r.atypical.count}</td><td class="num">${num(r.atypical.mean_distance, 2)}</td><td class="num">${pts(r.atypical.mean_abs_error)}</td></tr>
    </tbody>`;
  let text = r.error_ratio !== null
    ? `Error ratio (atypical ÷ typical): <strong>${num(r.error_ratio, 2)}×</strong>. `
    : "";
  text += r.spearman_rho !== null
    ? `Spearman correlation between neighbour distance and absolute error: <strong>ρ = ${num(r.spearman_rho, 3)}</strong> (p ${r.spearman_p < 0.001 ? "" : "= "}${fmtP(r.spearman_p)}).`
    : "Spearman correlation is not available for this data (too few students or constant values).";
  $("#atyp-summary").innerHTML = text;

  const point = (s) => ({ x: s.avg_distance, y: s.abs_error, student: s.student });
  makeChart("atyp-chart", {
    type: "scatter",
    data: {
      datasets: [
        { label: "Typical", data: r.students.filter((s) => s.group === "typical").map(point), backgroundColor: "rgba(31, 99, 214, 0.55)", pointRadius: 3 },
        { label: "Atypical", data: r.students.filter((s) => s.group === "atypical").map(point), backgroundColor: COLORS.reviewer, pointRadius: 4 },
      ],
    },
    options: {
      scales: {
        x: { title: { display: true, text: `Avg distance to ${r.k} nearest training students` } },
        y: { title: { display: true, text: "Absolute error" }, ticks: { callback: (v) => pct(v, 0) } },
      },
      plugins: { tooltip: { callbacks: { label: (c) => `#${c.raw.student}: distance ${num(c.parsed.x, 2)}, error ${pts(c.parsed.y)}` } } },
    },
  });

  const select = $("#nn-student");
  select.innerHTML = `<option value="" selected disabled>Choose a student (most atypical first)</option>` + r.students.map((s) =>
    `<option value="${s.student}">#${s.student} — ${s.group}, distance ${num(s.avg_distance, 2)}, error ${pts(s.abs_error)}</option>`).join("");
}

async function loadNeighbors(studentId) {
  showError($("#nn-error"), null);
  try {
    const r = await api(`/api/experiments/neighbors/${encodeURIComponent(studentId)}`);
    const short = { "GRE Score": "GRE", "TOEFL Score": "TOEFL", "University Rating": "Univ", "Research": "Res" };
    const features = Object.keys(r.features);
    const cols = ["Student", ...features.map((f) => short[f] || f), "Distance", "Chance of Admit"];
    const studentRow = { Student: `#${r.student} (test)`, Distance: "—", "Chance of Admit": `${pct(r.actual)} actual` };
    features.forEach((f) => { studentRow[short[f] || f] = r.features[f]; });
    const rows = [studentRow, ...r.neighbors.map((n) => {
      const row = { Student: `#${n.student}`, Distance: num(n.distance, 3), "Chance of Admit": pct(n.chance_of_admit) };
      features.forEach((f) => { row[short[f] || f] = n[f]; });
      return row;
    })];
    renderTable($("#nn-table"), cols, rows, { rowClass: (row, i) => (i === 0 ? "selected" : "") });
    $("#nn-summary").innerHTML = `Student #${r.student}: actual <strong>${pct(r.actual)}</strong>, predicted `
      + `<strong>${pct(r.predicted)}</strong> (${r.weights} average of the ${r.k} training students below). `
      + `Average neighbour distance: <strong>${num(r.avg_distance, 3)}</strong>.`;
    show($("#nn-result"));
  } catch (err) {
    showError($("#nn-error"), err.message);
  }
}

function initExperiments() {
  runExperiment("subj-run", "/api/experiments/subjective", "subj-error", "subj-result", renderSubjective);
  runExperiment("cons-run", "/api/experiments/consistency", "cons-error", "cons-result", renderConsistency);
  runExperiment("atyp-run", "/api/experiments/atypical", "atyp-error", "atyp-result", renderAtypical);
  const nn = $("#nn-student");
  if (nn) nn.addEventListener("change", () => nn.value && loadNeighbors(nn.value));

  const cutoffForm = $("#cutoff-form");
  if (cutoffForm) {
    cutoffForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const value = Number($("#cutoff").value);
      const out = $("#cutoff-result");
      if (!(value > 0 && value < 100)) { out.textContent = "Enter a cutoff between 1 and 99."; return; }
      try {
        const r = await api(`/api/experiments/cutoff?cutoff=${value / 100}`);
        out.innerHTML = `At a ${value}% cutoff, <strong>${r.disagreements}</strong> of ${r.n_test} test students fall on opposite sides: `
          + `${r.recorded_above_predicted_below} recorded at or above but predicted below, `
          + `${r.recorded_below_predicted_above} recorded below but predicted at or above. `
          + `<span class="muted">(Recorded ≥ cutoff: ${r.actual_above}; predicted ≥ cutoff: ${r.predicted_above}.)</span>`;
      } catch (err) {
        out.textContent = err.message;
      }
    });
  }
}

function initSensitivity() {
  const form = $("#sens-form");
  const featureSelect = $("#sens-feature");
  const newInput = $("#sens-new");
  const inputs = $$("[data-feature]", form);
  const profile = medianProfile();
  inputs.forEach((el) => { el.value = String(profile[el.dataset.feature]); });

  const STEP = { "GRE Score": 5, "TOEFL Score": 3, "University Rating": 1, "SOP": 0.5, "LOR": 0.5, "CGPA": 0.2, "Research": 1 };
  function suggestNewValue() {
    const f = featureSelect.value;
    const current = Number(inputs.find((el) => el.dataset.feature === f).value);
    const [lo, hi] = APP.ranges[f];
    let v = f === "Research" ? 1 - current : current + STEP[f];
    if (v > hi) v = current - STEP[f];
    newInput.value = String(Math.round(Math.max(lo, v) * 100) / 100);
    newInput.min = lo;
    newInput.max = hi;
    newInput.step = f === "CGPA" ? "0.01" : (f === "SOP" || f === "LOR") ? "0.5" : "1";
  }
  featureSelect.addEventListener("change", suggestNewValue);
  suggestNewValue();

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const errors = [];
    const payload = {};
    inputs.forEach((el) => {
      const f = el.dataset.feature;
      const problem = validateFeature(f, el.value);
      el.classList.toggle("invalid", Boolean(problem));
      if (problem) errors.push(problem); else payload[FORM_IDS[f]] = Number(el.value);
    });
    const newProblem = validateFeature(featureSelect.value, newInput.value);
    newInput.classList.toggle("invalid", Boolean(newProblem));
    if (newProblem) errors.push(`New value: ${newProblem}`);
    if (errors.length) { showError($("#sens-error"), errors.join(" ")); return; }
    showError($("#sens-error"), null);

    const button = form.querySelector("button[type=submit]");
    setLoading(button, true, "Running…");
    try {
      const r = await postJSON("/api/sensitivity", { profile: payload, feature: featureSelect.value, new_value: Number(newInput.value) });
      renderSensitivity(r);
    } catch (err) {
      showError($("#sens-error"), err.message);
    } finally {
      setLoading(button, false);
    }
  });
}

function renderSensitivity(r) {
  show($("#sens-result"));
  const fmtVal = (v) => (r.feature === "Research" ? (v ? "Yes" : "No") : v);
  const deltaCls = r.change > 1e-9 ? "delta-up" : r.change < -1e-9 ? "delta-down" : "";
  $("#sens-table tbody").innerHTML = `
    <tr><td><strong>${escapeHtml(r.feature)}</strong></td><td>${escapeHtml(fmtVal(r.original_value))}</td><td>${escapeHtml(fmtVal(r.modified_value))}</td></tr>
    <tr><td><strong>KNN prediction</strong></td><td>${pct(r.original_prediction)}</td><td>${pct(r.modified_prediction)}</td></tr>
    <tr><td><strong>Change</strong></td><td></td><td class="${deltaCls}">${signed(r.change * 100, 1)} pts</td></tr>`;
  const box = $("#sens-message");
  box.className = `alert ${r.academic_violation ? "error" : r.counterintuitive ? "warn" : "info"}`;
  box.innerHTML = (r.academic_violation ? "<strong>Inconsistent: </strong>" : "") + escapeHtml(r.message);

  const sw = r.sweep;
  $("#sweep-title").textContent = `Prediction as ${r.feature} varies (all else fixed)`;
  const marker = (value, color, label) => {
    const i = sw.values.findIndex((v) => Math.abs(v - value) < 1e-9);
    return { label, data: [{ x: value, y: i >= 0 ? sw.predictions[i] : null }], type: "scatter", backgroundColor: color, pointRadius: 6 };
  };
  makeChart("sweep-chart", {
    type: "line",
    data: {
      datasets: [
        { label: "Predicted chance", data: sw.values.map((v, i) => ({ x: v, y: sw.predictions[i] })),
          borderColor: COLORS.primary, pointRadius: 2, stepped: false, tension: 0 },
        marker(r.original_value, COLORS.navy, "Original"),
        marker(r.modified_value, r.counterintuitive ? COLORS.error : COLORS.ok, "Modified"),
      ],
    },
    options: {
      parsing: true,
      scales: {
        x: { type: "linear", title: { display: true, text: r.feature } },
        y: { title: { display: true, text: "Predicted chance" }, ticks: { callback: (v) => pct(v, 0) } },
      },
      plugins: { tooltip: { callbacks: { label: (c) => `${r.feature} ${c.parsed.x}: ${pct(c.parsed.y)}` } } },
    },
  });
  let note = `The original and modified points are recomputed with the same trained pipeline. `;
  if (sw.drops > 0) {
    note += `Across this range the prediction goes DOWN ${sw.drops} time(s) as ${r.feature} increases`;
    note += r.group === "academic" ? " — a non-monotonic response to an academic qualification." : ".";
  } else {
    note += `Across this range the prediction never decreases as ${r.feature} increases.`;
  }
  $("#sweep-note").textContent = note;
}

/* ---------------------------------------------------------------- boot */

document.addEventListener("DOMContentLoaded", () => {
  if (typeof Chart !== "undefined") {
    Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    Chart.defaults.color = "#475467";
  }
  const page = document.body.dataset.page;
  if (page === "train") initTrain();
  else if (page === "predict") initPredict();
  else if (page === "evaluation") initEvaluation();
});
