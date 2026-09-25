// Venlix Agent Console - vanilla JS, no build step.
// All server/LLM-provided text is inserted with textContent (or the escaping
// Markdown renderer below), never raw innerHTML.

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "style") el.style.cssText = value;
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else if (key === "html") el.innerHTML = value; // only used with md() output
    else el.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === undefined || child === null || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

async function api(path, options = {}) {
  const init = { ...options, headers: { "Content-Type": "application/json", ...(options.headers || {}) } };
  if (init.body && typeof init.body !== "string") init.body = JSON.stringify(init.body);
  const res = await fetch(path, init);
  let data = null;
  try { data = await res.json(); } catch (e) { /* empty body */ }
  if (!res.ok) {
    const detail = data && data.detail;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail || res.statusText));
  }
  return data;
}

function toast(message) {
  const el = h("div", { class: "toast", role: "status" }, message);
  document.body.append(el);
  setTimeout(() => el.remove(), 3500);
}

const LABELS = {
  resolved_auto: "Auto-resolved",
  auto_route: "Driver reassignment",
  customer_contact: "Customer contact",
  customer_declined: "Customer declined",
  awaiting_customer: "Awaiting customer",
  escalated_timeout: "LLM timeout",
  agent_error: "Agent error",
  rerouted: "Re-routed",
  openrouter: "OpenRouter",
  gemini: "Gemini",
};

const fmt = {
  int: (n) => Math.round(Number(n) || 0).toLocaleString(),
  ms: (n) => {
    const v = Number(n) || 0;
    if (v >= 1000) return `${(v / 1000).toFixed(v >= 10000 ? 0 : 1)} s`;
    return v < 10 ? `${v.toFixed(1)} ms` : `${Math.round(v)} ms`;
  },
  pct: (n) => `${Math.round((Number(n) || 0) * 100)}%`,
  inr: (n) => `₹${Math.round(Number(n) || 0).toLocaleString()}`,
  risk: (n) => (Number(n) || 0).toFixed(2),
  time: (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23" }),
  label: (s) => {
    if (LABELS[s]) return LABELS[s];
    const text = String(s || "none").replace(/_/g, " ");
    return text.charAt(0).toUpperCase() + text.slice(1);
  },
};

// ---------------------------------------------------------------------------
// Markdown (escape first, then a small safe subset)
// ---------------------------------------------------------------------------
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function mdInline(s) {
  return s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\s][^*]*?)\*(?!\w)/g, "$1<em>$2</em>")
    .replace(/(^|\s)_([^_\s][^_]*?)_(?=\s|$|[.,;:!?)])/g, "$1<em>$2</em>");
}
function md(src) {
  const lines = String(src || "").replace(/\r/g, "").split("\n");
  const out = [];
  let para = [];
  let list = null;
  const flushPara = () => { if (para.length) { out.push(`<p>${para.map((l) => mdInline(escapeHtml(l))).join("<br>")}</p>`); para = []; } };
  const flushList = () => { if (list) { out.push(`<${list.type}>${list.items.map((i) => `<li>${mdInline(escapeHtml(i))}</li>`).join("")}</${list.type}>`); list = null; } };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^```/.test(line.trim())) {
      flushPara(); flushList();
      const code = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i].trim())) code.push(lines[i++]);
      out.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }
    if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
      flushPara(); flushList();
      const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => mdInline(escapeHtml(c.trim())));
      const head = cells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(cells(lines[i++]));
      i--;
      out.push(`<div class="table-wrap"><table><thead><tr>${head.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
      continue;
    }
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) { flushPara(); flushList(); out.push(`<h4>${mdInline(escapeHtml(heading[2]))}</h4>`); continue; }
    const bullet = line.match(/^\s*(?:[-*•])\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (bullet || numbered) {
      flushPara();
      const type = bullet ? "ul" : "ol";
      if (!list || list.type !== type) { flushList(); list = { type, items: [] }; }
      list.items.push((bullet || numbered)[1]);
      continue;
    }
    if (!line.trim()) { flushPara(); flushList(); continue; }
    flushList();
    para.push(line);
  }
  flushPara(); flushList();
  return out.join("");
}

// ---------------------------------------------------------------------------
// Tooltip & charts (single-series horizontal bars; value at the tip)
// ---------------------------------------------------------------------------
const tooltip = $("#tooltip");
function showTooltip(target, value, label, event) {
  tooltip.replaceChildren(h("strong", {}, value), h("span", {}, label));
  tooltip.hidden = false;
  const rect = target.getBoundingClientRect();
  const x = event && event.clientX ? event.clientX : rect.left + rect.width / 2;
  const y = event && event.clientY ? event.clientY : rect.top;
  const tw = tooltip.offsetWidth;
  tooltip.style.left = `${Math.min(window.innerWidth - tw - 8, Math.max(8, x + 12))}px`;
  tooltip.style.top = `${Math.max(8, y - tooltip.offsetHeight - 10)}px`;
}
function hideTooltip() { tooltip.hidden = true; }

const tableMode = {};
function renderBars(containerId, entries, { format = fmt.int, empty = "No data yet.", valueLabel = "Value", categoryLabel = "Category" } = {}) {
  const container = document.getElementById(containerId);
  const rows = [...entries].sort((a, b) => b.value - a.value);
  if (!rows.length) { container.replaceChildren(h("p", { class: "chart-empty" }, empty)); return; }
  if (tableMode[containerId]) {
    container.replaceChildren(h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, categoryLabel), h("th", { class: "num" }, valueLabel))),
      h("tbody", {}, rows.map((r) => h("tr", {}, h("td", {}, r.label), h("td", { class: "num" }, format(r.value))))))));
    return;
  }
  const max = Math.max(...rows.map((r) => r.value), 1);
  container.replaceChildren(h("div", { class: "bars", role: "list" }, rows.map((r) => {
    const row = h("div", { class: "bar-row", tabindex: "0", role: "listitem", "aria-label": `${r.label}: ${format(r.value)}` },
      h("span", { class: "bar-label", title: r.label }, r.label),
      h("span", { class: "bar-track" },
        h("span", { class: "bar", style: `width: calc((100% - 4.5em) * ${r.value / max})` }),
        h("span", { class: "bar-value" }, format(r.value))));
    row.addEventListener("pointermove", (e) => showTooltip(row, format(r.value), r.label, e));
    row.addEventListener("pointerleave", hideTooltip);
    row.addEventListener("focus", () => showTooltip(row, format(r.value), r.label));
    row.addEventListener("blur", hideTooltip);
    return row;
  })));
}
$$("[data-table-toggle]").forEach((btn) => btn.addEventListener("click", () => {
  const id = btn.dataset.tableToggle;
  tableMode[id] = !tableMode[id];
  btn.textContent = tableMode[id] ? "Show chart" : "Show table";
  renderAll();
}));

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  cases: [],
  stats: null,
  health: null,
  llm: null,
  selected: null,
  chat: [],
  activity: [],
  run: null,
};

function statusBadge(status) {
  if (status === "resolved") return h("span", { class: "badge good" }, "Resolved");
  if (status === "escalated") return h("span", { class: "badge warning" }, "Escalated");
  if (status === "failed" || status === "error") return h("span", { class: "badge critical" }, "Failed");
  return h("span", { class: "badge neutral" }, fmt.label(status || "pending"));
}

// ---------------------------------------------------------------------------
// Tabs & theme
// ---------------------------------------------------------------------------
function selectTab(name) {
  if (!document.getElementById(`tab-${name}`)) name = "overview";
  $$(".tabs [role=tab]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  $$(".tab-panel").forEach((p) => { p.hidden = p.id !== `tab-${name}`; });
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
  if (name === "system") refreshLlm();
  if (name === "copilot") setTimeout(() => $("#chat-input").focus(), 0);
}
$$(".tabs [role=tab]").forEach((b) => b.addEventListener("click", () => selectTab(b.dataset.tab)));
$$("[data-goto]").forEach((b) => b.addEventListener("click", () => selectTab(b.dataset.goto)));
window.addEventListener("hashchange", () => selectTab(location.hash.slice(1)));

$("#theme-toggle").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("venlix-theme", root.dataset.theme); } catch (e) { /* storage unavailable */ }
});

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------
async function refreshCases() {
  try {
    state.cases = (await api("/api/cases")).cases;
    renderAll();
  } catch (e) { console.warn(e); }
}
async function refreshStats() {
  try {
    const bars = $$(".bars");
    bars.forEach((b) => b.classList.add("refreshing"));
    state.stats = await api("/api/stats");
    renderAll();
  } catch (e) { console.warn(e); }
}
async function refreshHealth() {
  try {
    state.health = await api("/api/health");
    renderHealth();
  } catch (e) {
    $("#provider-pill .pill-text").textContent = "Server unreachable";
    $("#provider-pill .dot").className = "dot critical";
  }
}
async function refreshLlm() {
  try {
    state.llm = await api("/api/llm/metrics?recent=40");
    renderSystem();
  } catch (e) { console.warn(e); }
}
let statsTimer = null;
function scheduleStats() {
  clearTimeout(statsTimer);
  statsTimer = setTimeout(() => { refreshStats(); if (!$("#tab-system").hidden) refreshLlm(); }, 250);
}

// ---------------------------------------------------------------------------
// Rendering: overview
// ---------------------------------------------------------------------------
function kpi(label, value, sub) {
  return h("div", { class: "card kpi" }, h("div", { class: "label" }, label), h("div", { class: "value" }, value), sub ? h("div", { class: "sub" }, sub) : null);
}

function renderOverview() {
  const agent = state.stats ? state.stats.agent : null;
  const totals = agent ? agent.totals : { cases: 0, resolved: 0, escalated: 0 };
  const savings = agent ? agent.savings : { time_min: 0, cost_inr: 0, fuel_inr: 0 };
  $("#kpis").replaceChildren(
    kpi("Cases processed", fmt.int(totals.cases), `${fmt.int(totals.resolved)} resolved automatically`),
    kpi("Resolution rate", totals.cases ? fmt.pct(agent.resolution_rate) : "–", "resolved without a human"),
    kpi("Escalated to humans", fmt.int(totals.escalated), "fraud, declines, unclear replies"),
    kpi("Cost saved", fmt.inr(savings.cost_inr), `plus ${fmt.inr(savings.fuel_inr)} fuel`),
    kpi("Time saved", `${fmt.int(savings.time_min)} min`, "avoided failed attempts"),
    kpi("Avg case time", agent && agent.avg_case_ms ? fmt.ms(agent.avg_case_ms) : "–", agent && agent.p95_case_ms ? `p95 ${fmt.ms(agent.p95_case_ms)}` : "end-to-end per case"),
  );
  const toEntries = (obj) => Object.entries(obj || {}).map(([k, v]) => ({ label: fmt.label(k), value: v }));
  renderBars("chart-failure", toEntries(agent && agent.by_failure_type), { empty: "Run cases to see failure types.", valueLabel: "Cases", categoryLabel: "Failure type" });
  renderBars("chart-outcome", toEntries(agent && agent.by_outcome), { empty: "Run cases to see outcomes.", valueLabel: "Cases", categoryLabel: "Outcome" });

  const recent = state.cases.slice(0, 6);
  $("#recent-cases").replaceChildren(recent.length
    ? h("div", { class: "table-wrap" }, caseTable(recent, { compact: true }))
    : h("p", { class: "muted" }, "No cases yet. Press “Run sample cases” to watch the agent work."));

  $("#activity").replaceChildren(...(state.activity.length
    ? state.activity.map((a) => h("li", {}, h("time", {}, fmt.time(a.ts)), h("span", {}, a.node || a.text)))
    : [h("li", {}, h("time", {}, ""), h("span", { class: "muted" }, "Waiting for events…"))]));
}

function addActivity(text, node) {
  state.activity.unshift({ ts: Date.now() / 1000, text, node });
  state.activity = state.activity.slice(0, 60);
  if (!$("#tab-overview").hidden) renderOverview();
}

// ---------------------------------------------------------------------------
// Rendering: cases
// ---------------------------------------------------------------------------
function filteredCases() {
  const status = $("#case-filter").value;
  const q = $("#case-search").value.trim().toLowerCase();
  return state.cases.filter((c) => {
    if (status !== "all" && c.status !== status) return false;
    if (!q) return true;
    const hay = [c.delivery_id, (c.customer || {}).name, c.failure_type, c.final_outcome, c.resolution_path].join(" ").toLowerCase();
    return hay.includes(q);
  });
}

function caseTable(cases, { compact = false } = {}) {
  const head = compact
    ? ["Case", "Failure type", "Outcome", "Status"]
    : ["Case", "Customer", "Risk", "Failure type", "Outcome", "Status", "Time"];
  return h("table", {},
    h("thead", {}, h("tr", {}, head.map((t) => h("th", { class: t === "Risk" || t === "Time" ? "num" : null }, t)))),
    h("tbody", {}, cases.map((c) => {
      const cells = compact
        ? [h("td", { class: "id" }, c.delivery_id), h("td", {}, fmt.label(c.failure_type)), h("td", {}, fmt.label(c.final_outcome)), h("td", {}, statusBadge(c.status))]
        : [
          h("td", { class: "id" }, c.delivery_id),
          h("td", {}, (c.customer || {}).name || "–"),
          h("td", { class: "num" }, fmt.risk(c.risk_score)),
          h("td", {}, fmt.label(c.failure_type)),
          h("td", {}, fmt.label(c.final_outcome)),
          h("td", {}, statusBadge(c.status)),
          h("td", { class: "num" }, c.duration_ms != null ? fmt.ms(c.duration_ms) : "–"),
        ];
      const row = h("tr", { class: `clickable${state.selected === c.delivery_id ? " selected" : ""}`, tabindex: "0" }, cells);
      const open = () => { state.selected = c.delivery_id; selectTab("cases"); renderCases(); };
      row.addEventListener("click", open);
      row.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
      return row;
    })));
}

function renderCases() {
  const cases = filteredCases();
  $("#cases-table").replaceChildren(cases.length
    ? caseTable(cases)
    : h("p", { class: "muted" }, state.cases.length ? "No cases match the filter." : "No cases yet. Run sample cases from the Overview tab, or create one with “New case”."));
  const selected = state.cases.find((c) => c.delivery_id === state.selected);
  if (selected) renderDetail(selected);
}

function fact(k, v) { return h("div", { class: "fact" }, h("div", { class: "k" }, k), h("div", { class: "v" }, v)); }

function renderDetail(c) {
  const box = $("#case-detail");
  const customer = c.customer || {};
  const intent = c.customer_intent;
  const calls = (c.llm_metadata && c.llm_metadata.calls) || [];
  const factors = (c.risk_factors || []).filter((f) => f.impact != null);

  const summaryTarget = h("div", {}, c.decision_summary ? h("div", { class: "summary-box" }, c.decision_summary) : null);
  const explainBtn = h("button", { class: "btn", type: "button" }, c.decision_summary ? "Re-explain decision" : "Explain this decision");
  explainBtn.addEventListener("click", async () => {
    explainBtn.disabled = true; explainBtn.textContent = "Explaining…";
    try {
      const res = await api(`/api/cases/${encodeURIComponent(c.delivery_id)}/summary`, { method: "POST" });
      c.decision_summary = res.summary;
      summaryTarget.replaceChildren(h("div", { class: "summary-box" }, res.summary));
    } catch (e) { toast(`Could not explain: ${e.message}`); }
    explainBtn.disabled = false; explainBtn.textContent = "Re-explain decision";
  });

  box.replaceChildren(
    h("div", { class: "detail-head" }, h("h2", {}, c.delivery_id), statusBadge(c.status)),
    h("div", { class: "muted" }, c.resolution_detail || fmt.label(c.final_outcome)),
    h("div", { class: "facts" },
      fact("Customer", customer.name || "–"),
      fact("Risk score", fmt.risk(c.risk_score)),
      fact("Failure type", `${fmt.label(c.failure_type)}${c.failure_confidence != null ? ` (${fmt.pct(c.failure_confidence)} of risk weight)` : ""}`),
      fact("Resolution path", fmt.label(c.resolution_path)),
      fact("Outcome", fmt.label(c.final_outcome)),
      fact("Source", fmt.label(c.source)),
      fact("Savings", `${fmt.int((c.savings || {}).time_min)} min · ${fmt.inr((c.savings || {}).cost_inr)}`),
      fact("Processing time", c.duration_ms != null ? fmt.ms(c.duration_ms) : "–"),
    ),
    h("div", { style: "margin-top:10px" }, explainBtn, summaryTarget),
    c.customer_message ? h("div", {},
      h("h3", {}, "Customer conversation"),
      h("div", { class: "bubble-meta right" }, "Agent SMS"),
      h("div", { class: "bubble out" }, c.customer_message),
      c.customer_reply ? h("div", { class: "bubble-meta" }, "Customer reply (simulated)") : null,
      c.customer_reply ? h("div", { class: "bubble in" }, c.customer_reply) : null,
      intent ? h("div", { class: "chips", style: "margin-top:8px" },
        h("span", { class: "chip" }, `wants reschedule: ${intent.wants_reschedule ? "yes" : "no"}`),
        h("span", { class: "chip" }, `declined: ${intent.declined ? "yes" : "no"}`),
        intent.new_slot ? h("span", { class: "chip" }, `slot: ${intent.new_slot}`) : null) : null,
    ) : null,
    factors.length ? h("div", {}, h("h3", {}, "Risk factors (ML impact)"), h("div", { id: "detail-factors" })) : null,
    h("h3", {}, "Agent trace"),
    h("ol", { class: "timeline" }, (c.trace || []).map((t) => h("li", { class: t.error ? "error" : null },
      h("div", { class: "node" }, `${t.node}${t.ms != null ? ` · ${fmt.ms(t.ms)}` : ""}`),
      h("div", {}, t.action),
      t.error ? h("pre", {}, t.error) : null))),
    calls.length ? h("div", {},
      h("h3", {}, "LLM calls"),
      h("div", { class: "table-wrap" }, h("table", {},
        h("thead", {}, h("tr", {}, ["Task", "Provider", "Model", "Latency", "OK"].map((t) => h("th", { class: t === "Latency" ? "num" : null }, t)))),
        h("tbody", {}, calls.map((call) => h("tr", {},
          h("td", {}, fmt.label(call.task)), h("td", {}, call.provider), h("td", {}, call.model || "–"),
          h("td", { class: "num" }, fmt.ms(call.latency_ms)), h("td", {}, call.ok ? "yes" : `no: ${call.error || ""}`))))))) : null,
    c.problem_prompt ? h("details", { style: "margin-top:14px" }, h("summary", {}, "Prompt sent to the LLM"), h("pre", {}, c.problem_prompt)) : null,
  );
  if (factors.length) {
    renderBars("detail-factors", factors.map((f) => ({ label: f.factor, value: Number(f.impact) || 0 })), { valueLabel: "Impact", categoryLabel: "Factor" });
  }
}

$("#case-filter").addEventListener("change", renderCases);
$("#case-search").addEventListener("input", renderCases);
$("#clear-cases").addEventListener("click", async () => {
  if (!confirm("Delete all stored cases?")) return;
  await api("/api/cases", { method: "DELETE" });
  state.selected = null;
  $("#case-detail").replaceChildren(h("p", { class: "muted" }, "Select a case to see what the agent did."));
  await Promise.all([refreshCases(), refreshStats()]);
});

// New case dialog
const PRESETS = {
  "Gate access": { factors: "High Gate Wait Time: 90\nVisitor Pass Pending: 40", action: "Notify Security Gate" },
  "Customer unreachable": { factors: "Customer Response Time: 92\nCustomer Unavailable: 30", action: "Offer Reschedule" },
  "Wrong address": { factors: "Low Address Confidence: 88\nPrevious Failed Deliveries: 20", action: "Verify Address" },
  "Traffic delay": { factors: "Heavy Traffic Congestion: 85\nAdverse Weather: 50", action: "Reassign Driver" },
  "Fraud signals": { factors: "Suspicious Order Pattern: 95\nDrop Distance Anomaly: 60", action: "Hold For Verification" },
};
const caseForm = $("#case-form");
$("#case-presets").replaceChildren(...Object.entries(PRESETS).map(([name, p]) => h("button", {
  class: "chip", type: "button",
  onclick: () => { caseForm.risk_factors.value = p.factors; caseForm.recommended_action.value = p.action; },
}, name)));
$("#new-case").addEventListener("click", () => $("#case-dialog").showModal());
$("#case-cancel").addEventListener("click", () => $("#case-dialog").close());
caseForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = caseForm;
  const factors = f.risk_factors.value.split("\n").map((l) => l.trim()).filter(Boolean).map((l) => {
    const m = l.match(/^(.*?)(?::\s*(\d+(?:\.\d+)?))?$/);
    return { factor: m[1].trim(), impact: m[2] ? Math.min(100, Number(m[2])) : 50 };
  }).filter((x) => x.factor);
  const body = {
    customer_name: f.customer_name.value.trim() || "Customer",
    phone: f.phone.value.trim() || null,
    risk_score: Number(f.risk_score.value) || 0,
    proposed_slot: f.proposed_slot.value.trim() || null,
    risk_factors: factors,
    recommended_actions: f.recommended_action.value.trim() ? [f.recommended_action.value.trim()] : [],
  };
  const submit = $("#case-submit");
  submit.disabled = true; submit.textContent = "Running…";
  try {
    const result = await api("/api/cases", { method: "POST", body });
    $("#case-dialog").close();
    state.selected = result.delivery_id;
    await refreshCases();
    selectTab("cases");
  } catch (err) { toast(`Case failed: ${err.message}`); }
  submit.disabled = false; submit.textContent = "Run through agent";
});

// ---------------------------------------------------------------------------
// Runs
// ---------------------------------------------------------------------------
async function startRun(source) {
  const buttons = [$("#run-sample"), $("#run-backend")];
  buttons.forEach((b) => { b.disabled = true; });
  try {
    const run = await api("/api/runs", { method: "POST", body: { source, concurrency: Number($("#concurrency").value) } });
    state.run = run;
    renderRun();
    if (!run.total) { buttons.forEach((b) => { b.disabled = false; }); }
  } catch (e) {
    toast(`Could not start run: ${e.message}`);
    buttons.forEach((b) => { b.disabled = false; });
  }
}
function renderRun() {
  const run = state.run;
  if (!run) return;
  const src = run.source === "backend" ? (run.backend_live === false ? "backend (sample data: backend unreachable)" : "backend") : "sample";
  let text = `${fmt.label(run.status)}: ${run.completed}/${run.total} ${src} cases`;
  if (run.skipped_low_risk) text += ` · ${run.skipped_low_risk} low-risk skipped`;
  if (run.duration_ms != null) text += ` · ${fmt.ms(run.duration_ms)} total`;
  $("#run-status").textContent = text;
  if (run.status !== "running") [$("#run-sample"), $("#run-backend")].forEach((b) => { b.disabled = false; });
}
$("#run-sample").addEventListener("click", () => startRun("sample"));
$("#run-backend").addEventListener("click", () => startRun("backend"));

// ---------------------------------------------------------------------------
// Live updates (WebSocket with reconnect)
// ---------------------------------------------------------------------------
let wsRetry = 0;
function connectWs() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  const setLive = (ok) => {
    $("#live-pill .dot").className = `dot ${ok ? "good" : "critical"}`;
    $("#live-pill").title = ok ? "Live updates connected" : "Live updates disconnected - reconnecting";
  };
  ws.onopen = () => { wsRetry = 0; setLive(true); };
  ws.onclose = () => { setLive(false); setTimeout(connectWs, Math.min(10000, 500 * 2 ** wsRetry++)); };
  ws.onmessage = (msg) => {
    let ev;
    try { ev = JSON.parse(msg.data); } catch (e) { return; }
    switch (ev.event) {
      case "case_started":
        addActivity(`Agent picked up ${ev.delivery_id} (risk ${fmt.risk(ev.risk_score)})`);
        break;
      case "case_completed": {
        const c = ev.case;
        state.cases = [c, ...state.cases.filter((x) => x.delivery_id !== c.delivery_id)];
        const line = h("span", {}, `${c.delivery_id}: `, statusBadge(c.status), ` ${fmt.label(c.final_outcome)} via ${fmt.label(c.resolution_path)} in ${fmt.ms(c.duration_ms)}`);
        addActivity(null, line);
        if (!$("#tab-cases").hidden) renderCases();
        scheduleStats();
        break;
      }
      case "run_started":
      case "run_progress":
      case "run_completed":
        state.run = ev.run;
        renderRun();
        if (ev.event === "run_completed") { addActivity(`Run finished: ${ev.run.completed} cases in ${fmt.ms(ev.run.duration_ms)}`); scheduleStats(); }
        break;
      case "cases_cleared":
        state.cases = [];
        renderAll();
        break;
      default:
        break;
    }
  };
}

// ---------------------------------------------------------------------------
// Copilot chat
// ---------------------------------------------------------------------------
const SUGGESTIONS = [
  "What's today's failure rate?",
  "Show high-risk deliveries",
  "Which cases did the agent escalate, and why?",
  "Tell me about HERO-001",
  "What is 17% of 2400?",
  "Draft a polite apology email to Alex Rivera for the missed delivery",
];
$("#suggestions").replaceChildren(...SUGGESTIONS.map((s) => h("button", { class: "chip", type: "button", onclick: () => sendChat(s) }, s)));

function chatMeta(info) {
  const parts = [];
  if (info.provider) parts.push(info.provider === "offline" ? "Offline engine" : `${fmt.label(info.provider)} · ${info.model}`);
  if (info.latency_ms != null) parts.push(fmt.ms(info.latency_ms) + (info.first_token_ms != null && info.provider !== "offline" ? ` (first token ${fmt.ms(info.first_token_ms)})` : ""));
  if (info.apis_called && info.apis_called.length) parts.push(`data: ${info.apis_called.join(", ")}`);
  return parts.join(" · ");
}

function renderChat() {
  const log = $("#chat-log");
  if (!state.chat.length) {
    log.replaceChildren(h("div", { class: "chat-empty" },
      h("h2", {}, "Ask the Operations Copilot"),
      h("p", {}, "It pulls live deliveries, reports, predictions and the agent's own cases when your question needs them, and answers general questions too.")));
    $("#suggestions").hidden = false;
    return;
  }
  $("#suggestions").hidden = true;
  log.replaceChildren(...state.chat.map((m) => {
    if (m.role === "user") return h("div", { class: "msg user" }, m.content);
    const body = h("div", { class: m.pending ? "cursor" : null, html: md(m.content) });
    return h("div", { class: `msg assistant${m.error ? " error" : ""}` }, body, m.meta ? h("div", { class: "meta" }, m.meta) : null);
  }));
  log.scrollTop = log.scrollHeight;
}

let chatBusy = false;
async function sendChat(text) {
  const message = (text || "").trim();
  if (!message || chatBusy) return;
  chatBusy = true;
  $("#chat-send").disabled = true;
  const history = state.chat.filter((m) => !m.error && !m.pending).map((m) => ({ role: m.role, content: m.content }));
  state.chat.push({ role: "user", content: message });
  const reply = { role: "assistant", content: "", pending: true, meta: "" };
  state.chat.push(reply);
  renderChat();
  $("#chat-input").value = "";
  try {
    if ($("#stream-toggle").checked) await streamChat(message, history, reply);
    else {
      const res = await api("/api/chat", { method: "POST", body: { message, history } });
      reply.content = res.answer;
      reply.meta = chatMeta(res);
    }
  } catch (e) {
    reply.error = true;
    reply.content = reply.content || `Sorry, I couldn't answer: ${e.message}`;
  }
  reply.pending = false;
  renderChat();
  chatBusy = false;
  $("#chat-send").disabled = false;
  scheduleStats();
}

async function streamChat(message, history, reply) {
  const res = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  });
  if (!res.ok || !res.body) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail; } catch (e) { /* ignore */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let apis = [];
  let frame = null;
  const paint = () => { frame = null; renderChat(); };
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      if (!chunk.startsWith("data: ")) continue;
      const ev = JSON.parse(chunk.slice(6));
      if (ev.type === "meta") apis = ev.apis_called || [];
      else if (ev.type === "delta") { reply.content += ev.text; if (!frame) frame = requestAnimationFrame(paint); }
      else if (ev.type === "done") reply.meta = chatMeta({ ...ev, apis_called: apis });
      else if (ev.type === "error") { reply.error = true; reply.content += (reply.content ? "\n\n" : "") + `Error: ${ev.message}`; }
    }
  }
}

$("#chat-form").addEventListener("submit", (e) => { e.preventDefault(); sendChat($("#chat-input").value); });
$("#chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); sendChat($("#chat-input").value); }
});
$("#chat-clear").addEventListener("click", () => { state.chat = []; renderChat(); });

// ---------------------------------------------------------------------------
// Exception analyzer
// ---------------------------------------------------------------------------
const NOTE_EXAMPLES = [
  "Gate code 4821 not working, security guard won't let me in, customer phone off",
  "Box is crushed on one side and leaking, customer refused to accept it",
  "Road closed due to flooding near the flyover, cannot reach the drop point",
  "Nobody answering at the door, neighbour says customer moved out last week",
];
$("#note-examples").replaceChildren(...NOTE_EXAMPLES.map((n) => h("button", { class: "chip", type: "button", onclick: () => { $("#note-input").value = n; } }, n.length > 42 ? `${n.slice(0, 40)}…` : n)));
$("#note-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const note = $("#note-input").value.trim();
  if (!note) return;
  const btn = $("#note-submit");
  btn.disabled = true; btn.textContent = "Analyzing…";
  try {
    const r = await api("/api/exceptions/analyze", { method: "POST", body: { note } });
    $("#note-result").replaceChildren(
      h("h2", {}, "Root cause"),
      h("p", {}, r.root_cause),
      h("h3", {}, "Suggested solution"),
      h("p", {}, r.suggested_solution),
      h("h3", {}, `Confidence ${fmt.pct(r.confidence)}`),
      h("div", { class: "meter", role: "meter", "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": String(Math.round(r.confidence * 100)) },
        h("span", { style: `width:${Math.round(r.confidence * 100)}%` })),
      h("div", { class: "muted", style: "font-size:12px" }, `Retrieval ${fmt.ms(r.retrieval_ms)} · LLM ${fmt.ms(r.llm_ms)}`),
      h("h3", {}, "Similar historical cases"),
      h("ul", { class: "refs" }, (r.references || []).map((ref) => h("li", {},
        h("div", {}, `“${ref.note}”`),
        h("div", { class: "muted", style: "font-size:12.5px;margin-top:4px" }, `${ref.category ? `${ref.category} · ` : ""}${ref.root_cause}`)))),
    );
  } catch (err) { toast(`Analysis failed: ${err.message}`); }
  btn.disabled = false; btn.textContent = "Analyze note";
});

// ---------------------------------------------------------------------------
// System / health
// ---------------------------------------------------------------------------
function renderHealth() {
  const llm = state.health && state.health.llm;
  if (!llm) return;
  const active = llm.active_provider;
  const pillText = $("#provider-pill .pill-text");
  const dot = $("#provider-pill .dot");
  if (active === "gemini") { pillText.textContent = `Gemini · ${llm.gemini.model}`; dot.className = "dot good"; }
  else if (active === "openrouter") { pillText.textContent = `OpenRouter · ${llm.openrouter.models[0]}`; dot.className = "dot good"; }
  else if (active === "offline") { pillText.textContent = "Offline mode"; dot.className = "dot warning"; }
  else { pillText.textContent = "No LLM configured"; dot.className = "dot critical"; }
  $("#offline-banner").hidden = active !== "offline";
  renderSystem();
}

function renderSystem() {
  const llm = state.health && state.health.llm;
  const backend = state.health && state.health.backend;
  if (llm) {
    const rows = [
      ["1", "Google Gemini", llm.gemini.configured ? (llm.gemini.forced_failure ? "Configured (forced off for testing)" : "Configured") : "No GEMINI_API_KEY", llm.gemini.model],
      ["2", "OpenRouter / OpenAI-compatible", llm.openrouter.configured ? "Configured" : "No OPENROUTER_API_KEY", `${llm.openrouter.models.join(", ")} @ ${llm.openrouter.base_url}`],
      ["3", "Offline engine", llm.offline_fallback ? "Enabled" : "Disabled (ALLOW_MOCK_FALLBACK=false)", "data-grounded, no key needed"],
    ];
    $("#provider-chain").replaceChildren(
      h("div", { class: "table-wrap" }, h("table", {},
        h("thead", {}, h("tr", {}, ["#", "Provider", "Status", "Model"].map((t) => h("th", {}, t)))),
        h("tbody", {}, rows.map((r) => h("tr", {}, r.map((v, i) => h("td", {}, i === 2 ? h("span", { class: `badge ${/^Configured$|^Enabled/.test(v) ? "good" : "neutral"}` }, v) : v))))))),
      h("p", { class: "muted", style: "margin:12px 0 0" }, `Answering first: ${fmt.label(llm.active_provider || "none")}. `,
        backend ? `Backend ${backend.url}: ${backend.live === true ? "live" : backend.live === false ? "unreachable, so sample data is used" : "not contacted yet"}.` : ""),
    );
  }
  const m = state.llm || (state.stats && state.stats.llm);
  if (!m) return;
  const providers = Object.entries(m.providers || {});
  const okCalls = m.total_calls - m.errors;
  const avg = providers.reduce((acc, [, p]) => acc + p.avg_ms * (p.calls - p.errors), 0) / Math.max(1, okCalls);
  $("#llm-kpis").replaceChildren(
    kpi("LLM calls", fmt.int(m.total_calls), "since server start"),
    kpi("Errors", fmt.int(m.errors), m.total_calls ? `${fmt.pct(m.errors / m.total_calls)} of calls (fell back)` : "none yet"),
    kpi("Avg latency", okCalls ? fmt.ms(avg) : "–", "successful calls"),
    kpi("Providers used", fmt.int(providers.length), providers.map(([n]) => fmt.label(n)).join(", ") || "none yet"),
  );
  renderBars("chart-latency", providers.filter(([, p]) => p.calls > p.errors).map(([name, p]) => ({ label: fmt.label(name), value: p.avg_ms })),
    { format: fmt.ms, empty: "No LLM calls yet.", valueLabel: "Average latency", categoryLabel: "Provider" });
  const recent = (state.llm && state.llm.recent) || [];
  $("#llm-calls").replaceChildren(recent.length
    ? h("table", {},
      h("thead", {}, h("tr", {}, ["Time", "Task", "Provider", "Model", "Latency", "Result"].map((t) => h("th", { class: t === "Latency" ? "num" : null }, t)))),
      h("tbody", {}, recent.map((c) => h("tr", {},
        h("td", { class: "num" }, fmt.time(c.ts)), h("td", {}, fmt.label(c.task)), h("td", {}, fmt.label(c.provider)),
        h("td", {}, c.model || "–"), h("td", { class: "num" }, fmt.ms(c.latency_ms)),
        h("td", {}, c.ok ? h("span", { class: "badge good" }, c.streamed ? "OK (streamed)" : "OK") : h("span", { class: "badge critical", title: c.error || "" }, `Failed: ${(c.error || "").slice(0, 80)}`))))))
    : h("p", { class: "muted" }, "No LLM calls yet."));
}
$("#refresh-llm").addEventListener("click", refreshLlm);

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
function renderAll() {
  renderOverview();
  renderCases();
  renderSystem();
}

selectTab(location.hash.slice(1) || "overview");
renderChat();
renderAll();
refreshHealth();
refreshCases();
refreshStats();
connectWs();
setInterval(() => { if (document.visibilityState === "visible") { refreshHealth(); refreshStats(); } }, 15000);
