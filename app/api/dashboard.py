"""Web dashboard: GET /dashboard (and / redirects here).

The server sends a static page; the browser loads everything from the public REST API
(GET /patients with its last_name / phone_number / date_of_birth filters, and GET /calls), so the
dashboard is also a live demonstration of the API. Validation errors come back in the API's
{"data": null, "error": {...}} envelope and are shown inline.

Rendering uses DOM text nodes only (never innerHTML): names and transcripts came from callers and
are untrusted. No external scripts or stylesheets.
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter(include_in_schema=False)


@router.get("/")
def root():
    return RedirectResponse("/dashboard")


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(PAGE)


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Patient Intake</title>
<style>
:root {
  --bg: #f6f7f9; --surface: #ffffff; --text: #16191d; --muted: #5d6673; --border: #e2e5ea;
  --accent: #1f6feb; --ok-bg: #e6f4ea; --ok-fg: #1e6b34; --warn-bg: #fdf1dc; --warn-fg: #8a5300;
  --info-bg: #e8effc; --info-fg: #1f4f99; --caller: #f1f3f6; --agent: #eaf2ff; --notice: #fdecec;
  --on-accent: #ffffff;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115; --surface: #171a20; --text: #e7e9ee; --muted: #98a1ae; --border: #2a2f38;
    --accent: #6ea8ff; --ok-bg: #173323; --ok-fg: #7fd69a; --warn-bg: #3a2c12; --warn-fg: #f3c06b;
    --info-bg: #182a47; --info-fg: #9cc2ff; --caller: #1f232b; --agent: #1a2638; --notice: #3b1d1d;
    --on-accent: #0f1115;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 48px; }
header { display: flex; flex-wrap: wrap; gap: 12px; align-items: baseline; justify-content: space-between; }
h1 { font-size: 22px; margin: 0; }
h2 { font-size: 16px; margin: 32px 0 12px; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.muted { color: var(--muted); } .small { font-size: 13px; } .nowrap { white-space: nowrap; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-top: 20px; }
.stat { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
.stat b { display: block; font-size: 24px; }
form { display: flex; flex-wrap: wrap; gap: 8px; align-items: flex-end; }
label { display: flex; flex-direction: column; gap: 4px; font-size: 13px; color: var(--muted); }
input { font: inherit; color: var(--text); background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 10px; min-width: 150px; }
.button { font: inherit; border: 1px solid var(--accent); background: var(--accent); color: var(--on-accent);
  border-radius: 8px; padding: 8px 14px; cursor: pointer; }
.button.secondary { background: transparent; color: var(--accent); }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border); vertical-align: top; }
th { white-space: nowrap; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); font-weight: 600; }
tr:last-child td { border-bottom: 0; }
.empty { color: var(--muted); padding: 16px; text-align: center; }
.notice { background: var(--notice); border-radius: 8px; padding: 8px 12px; margin: 8px 0 0; }
.notice:empty { display: none; }
.request { font: 12px ui-monospace, SFMono-Regular, Consolas, monospace; color: var(--muted); margin-top: 8px; }
.call { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; margin-bottom: 8px; }
.call summary { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; padding: 12px 14px; cursor: pointer; }
.call-who { font-weight: 600; flex: 1; min-width: 140px; }
.pill { font-size: 12px; font-weight: 600; border-radius: 999px; padding: 2px 10px; }
.pill.ok { background: var(--ok-bg); color: var(--ok-fg); }
.pill.warn { background: var(--warn-bg); color: var(--warn-fg); }
.pill.info { background: var(--info-bg); color: var(--info-fg); }
.transcript { padding: 4px 14px 14px; display: flex; flex-direction: column; gap: 6px; }
.line { border-radius: 8px; padding: 8px 10px; max-width: 760px; }
.line.caller { background: var(--caller); }
.line.agent { background: var(--agent); }
.who { display: block; font-size: 12px; font-weight: 600; color: var(--muted); }
@media (max-width: 600px) {
  .stats { grid-template-columns: repeat(3, 1fr); gap: 8px; }
  .stat { padding: 10px; }
  .stat b { font-size: 20px; }
  form label { flex: 1 1 100%; }
  input { width: 100%; min-width: 0; }
}
</style>
</head>
<body>
<main>
  <header>
    <h1>Patient Intake</h1>
    <span class="small"><a href="/docs">REST API docs</a> · <a href="/patients">/patients</a> · <a href="/calls">/calls</a></span>
  </header>

  <section class="stats">
    <div class="stat"><b id="stat-patients">-</b><span class="muted small">Registered patients</span></div>
    <div class="stat"><b id="stat-calls">-</b><span class="muted small">Recent calls</span></div>
    <div class="stat"><b id="stat-registered">-</b><span class="muted small">Calls that registered someone</span></div>
  </section>

  <h2>Patients</h2>
  <form id="search">
    <label>Last name<input name="last_name" maxlength="50" autocomplete="off"></label>
    <label>Phone<input name="phone_number" maxlength="20" inputmode="tel" autocomplete="off"></label>
    <label>Date of birth<input name="date_of_birth" maxlength="10" placeholder="MM/DD/YYYY" autocomplete="off"></label>
    <button class="button" type="submit">Search</button>
    <button class="button secondary" type="button" id="clear">Clear</button>
    <button class="button secondary" type="button" id="refresh">Refresh</button>
  </form>
  <p class="notice" id="notice" role="alert"></p>
  <div class="request" id="request"></div>
  <div class="card" style="margin-top:8px">
    <table>
      <thead><tr><th>Name</th><th>Date of birth</th><th>Sex</th><th>Phone</th><th>Address</th><th>Insurance</th><th>Registered</th></tr></thead>
      <tbody id="patients"><tr><td colspan="7" class="empty">Loading…</td></tr></tbody>
    </table>
  </div>

  <h2>Recent calls</h2>
  <div id="calls"><p class="empty">Loading…</p></div>
</main>
<script>
"use strict";

const form = document.getElementById("search");
const FILTERS = ["last_name", "phone_number", "date_of_birth"];
let names = new Map(); // patient_id -> "First Last", for labelling calls

function el(tag, props, ...children) {
  const node = document.createElement(tag);
  Object.assign(node, props || {});
  for (const child of children) {
    if (child !== null && child !== undefined) node.append(child); // strings become text nodes
  }
  return node;
}

// Every request goes through the public REST API and its {data, error} envelope.
async function api(path) {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  const body = await response.json();
  if (body.error) {
    const details = (body.error.details || []).map(d => d.message).join("; ");
    throw new Error(details || body.error.message);
  }
  return body.data;
}

const phone = p => p ? `(${p.slice(0, 3)}) ${p.slice(3, 6)}-${p.slice(6)}` : "";
const utc = new Intl.DateTimeFormat("en-US", {
  month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false, timeZone: "UTC",
});
const when = iso => iso ? utc.format(new Date(iso)) + " UTC" : "";

function patientRow(p) {
  const address = p.address_line_1 + (p.address_line_2 ? `, ${p.address_line_2}` : "");
  const insurance = p.insurance_provider
    ? el("td", {}, p.insurance_provider, p.insurance_member_id ? el("span", { className: "muted", textContent: ` #${p.insurance_member_id}` }) : null)
    : el("td", {}, el("span", { className: "muted", textContent: "None" }));
  return el("tr", {},
    el("td", {},
      el("a", { href: `/patients/${p.patient_id}`, textContent: `${p.first_name} ${p.last_name}` }),
      el("div", { className: "muted small", textContent: p.preferred_language })),
    el("td", { textContent: p.date_of_birth }),
    el("td", { textContent: p.sex }),
    el("td", { className: "nowrap", textContent: phone(p.phone_number) }),
    el("td", {}, address, el("div", { className: "muted small", textContent: `${p.city}, ${p.state} ${p.zip_code}` })),
    insurance,
    el("td", { className: "nowrap muted", textContent: when(p.created_at) }));
}

function showPatients(patients) {
  const body = document.getElementById("patients");
  body.replaceChildren(...(patients.length
    ? patients.map(patientRow)
    : [el("tr", {}, el("td", { colSpan: 7, className: "empty", textContent: "No patients match." }))]));
}

const STATUS = {
  completed: ["Registered", "ok"], abandoned: ["Not completed", "warn"],
  resumed: ["Resumed later", "info"], in_progress: ["In progress", "info"],
};

function showCalls(calls) {
  const box = document.getElementById("calls");
  if (!calls.length) {
    box.replaceChildren(el("p", { className: "empty", textContent: "No calls yet. Call the number to see transcripts here." }));
    return;
  }
  box.replaceChildren(...calls.map(c => {
    const [label, tone] = STATUS[c.status] || [c.status, "info"];
    const who = names.get(c.patient_id) || phone(c.from_number) || "Unknown caller";
    const lines = c.transcript.length
      ? c.transcript.map(t => el("div", { className: `line ${t.speaker}` },
          el("span", { className: "who", textContent: t.speaker === "caller" ? "Caller" : "Sam" }), t.text))
      : [el("p", { className: "muted small", textContent: "No transcript was stored for this call." })];
    return el("details", { className: "call" },
      el("summary", {},
        el("span", { className: `pill ${tone}`, textContent: label }),
        el("span", { className: "call-who", textContent: who }),
        el("span", { className: "muted small nowrap", textContent: `${when(c.started_at)} · ${c.transcript.length} turns` })),
      el("div", { className: "transcript" }, ...lines));
  }));
}

function currentFilters() {
  const params = new URLSearchParams();
  for (const name of FILTERS) {
    const value = form.elements[name].value.trim();
    if (value) params.set(name, value);
  }
  return params;
}

let latestSearch = 0;

async function search() {
  const mine = ++latestSearch; // typing fires overlapping requests; only the newest may render
  const params = currentFilters();
  const path = "/patients" + (params.toString() ? `?${params}` : "");
  document.getElementById("request").textContent = `GET ${path}`;
  history.replaceState(null, "", params.toString() ? `?${params}` : location.pathname);
  try {
    const patients = await api(path);
    if (mine !== latestSearch) return;
    showPatients(patients);
    document.getElementById("notice").textContent = "";
  } catch (error) {
    if (mine !== latestSearch) return;
    showPatients([]);
    document.getElementById("notice").textContent = error.message;
  }
}

async function loadAll() {
  try {
    const [everyone, calls] = await Promise.all([api("/patients?limit=500"), api("/calls?limit=30")]);
    names = new Map(everyone.map(p => [p.patient_id, `${p.first_name} ${p.last_name}`]));
    document.getElementById("stat-patients").textContent = everyone.length;
    document.getElementById("stat-calls").textContent = calls.length;
    document.getElementById("stat-registered").textContent = calls.filter(c => c.status === "completed").length;
    showCalls(calls);
  } catch (error) {
    document.getElementById("notice").textContent = `Could not load data: ${error.message}`;
  }
  await search();
}

let debounce;
form.addEventListener("input", () => { clearTimeout(debounce); debounce = setTimeout(search, 350); });
form.addEventListener("submit", event => { event.preventDefault(); clearTimeout(debounce); search(); });
document.getElementById("clear").addEventListener("click", () => { form.reset(); search(); });
document.getElementById("refresh").addEventListener("click", loadAll);

// Restore filters from the URL so a filtered view can be shared or bookmarked.
const initial = new URLSearchParams(location.search);
for (const name of FILTERS) form.elements[name].value = initial.get(name) || "";
loadAll();
</script>
</body>
</html>
"""
