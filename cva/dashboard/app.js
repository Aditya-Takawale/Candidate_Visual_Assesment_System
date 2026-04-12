// CVA Dashboard — Vanilla JS (no build step required for demo)
// Connects to FastAPI WebSocket and renders live assessment data

// Derive base URL from current page — works on any port or proxy
const API_BASE = window.location.origin;
const WS_URL   = (window.location.protocol === "https:" ? "wss://" : "ws://")
               + window.location.host + "/ws";

// ── State ──────────────────────────────────────────────────────────────────
let state = {
  sessionId: null,
  status: "idle",           // idle | starting | running | stopped
  score: null,
  health: null,
  redFlags: [],
  shap: {},
  moduleScores: {},
  role: "developer",
};

let ws = null;

// ── DOM ────────────────────────────────────────────────────────────────────
document.getElementById("app").innerHTML = `
<div class="max-w-7xl mx-auto px-4 py-6 space-y-6">

  <!-- Header -->
  <div class="flex items-center justify-between">
    <div>
      <h1 class="text-2xl font-bold text-sky-400">CVA Assessment Dashboard</h1>
      <p class="text-slate-400 text-sm mt-0.5">Candidate Visual Assessment — Demo Mode</p>
    </div>
    <div class="flex items-center gap-3">
      <select id="role-select" class="bg-slate-700 border border-slate-600 rounded-lg px-3 py-1.5 text-sm text-slate-100">
        <option value="developer">Developer</option>
        <option value="sales">Sales</option>
        <option value="hr">HR</option>
      </select>
      <button id="btn-start" class="bg-sky-500 hover:bg-sky-600 text-white font-semibold px-4 py-2 rounded-lg text-sm transition-colors">
        Start Session
      </button>
      <button id="btn-stop" class="hidden bg-red-500 hover:bg-red-600 text-white font-semibold px-4 py-2 rounded-lg text-sm transition-colors">
        Stop Session
      </button>
    </div>
  </div>

  <!-- Session + Backend Banner -->
  <div id="session-banner" class="hidden card px-4 py-3 flex items-center gap-4 text-sm">
    <span class="text-slate-400">Session:</span>
    <span id="session-id-label" class="font-mono text-sky-300"></span>
    <span class="text-slate-400 ml-4">Backend:</span>
    <span id="backend-label" class="font-mono text-emerald-300"></span>
    <span id="warmup-badge" class="hidden ml-auto bg-amber-900/40 text-amber-300 border border-amber-700/50 rounded-full px-3 py-0.5 text-xs">
      ⏳ Warming up…
    </span>
  </div>

  <!-- Camera + Scores Row -->
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">

    <!-- Live Camera Feed -->
    <div class="card overflow-hidden flex flex-col lg:col-span-1">
      <div class="px-4 pt-4 pb-2 flex items-center justify-between">
        <h2 class="text-slate-300 font-semibold text-sm uppercase tracking-wider">Live Camera</h2>
        <span id="cam-status" class="text-xs text-slate-500">Connecting…</span>
      </div>
      <div class="relative bg-black flex-1 min-h-[240px]">
        <img id="camera-feed"
          src="/video_feed"
          class="w-full h-full object-cover"
          style="min-height:240px; display:block;"
          onload="document.getElementById('cam-status').textContent='Live'"
          onerror="document.getElementById('cam-status').textContent='Unavailable'"
        />
        <!-- Score overlay badge -->
        <div id="cam-score-badge" class="hidden absolute top-2 right-2 bg-black/60 rounded-lg px-3 py-1 text-lg font-bold font-mono"></div>
      </div>
    </div>

  <!-- Score + Modules sub-grid -->
  <div class="lg:col-span-2 grid grid-cols-1 sm:grid-cols-2 gap-6">

    <!-- Score Ring -->
    <div class="card p-6 flex flex-col items-center justify-center gap-4">
      <h2 class="text-slate-300 font-semibold text-sm uppercase tracking-wider">Final Score</h2>
      <div class="relative w-40 h-40">
        <svg viewBox="0 0 120 120" class="w-full h-full -rotate-90">
          <circle cx="60" cy="60" r="50" fill="none" stroke="#1e3a5f" stroke-width="12"/>
          <circle id="score-ring" cx="60" cy="60" r="50" fill="none"
            stroke="#0ea5e9" stroke-width="12" stroke-linecap="round"
            stroke-dasharray="314.16" stroke-dashoffset="314.16"
            class="score-ring"/>
        </svg>
        <div class="absolute inset-0 flex flex-col items-center justify-center">
          <span id="score-value" class="text-4xl font-bold text-sky-400">--</span>
          <span class="text-slate-400 text-xs mt-1">/ 100</span>
        </div>
      </div>
      <p id="score-reason" class="text-center text-slate-400 text-xs px-2"></p>
      <div id="role-badge" class="bg-slate-700 text-slate-300 rounded-full px-3 py-0.5 text-xs capitalize"></div>
    </div>

    <!-- Module Scores -->
    <div class="card p-6 space-y-4">
      <h2 class="text-slate-300 font-semibold text-sm uppercase tracking-wider">Module Scores</h2>
      <div id="module-bars" class="space-y-3">
        ${["identity","body_language","first_impression","grooming"].map(m => `
          <div>
            <div class="flex justify-between text-xs mb-1">
              <span class="text-slate-400 capitalize">${m.replace('_',' ')}</span>
              <span id="mod-${m}-val" class="text-slate-300 font-mono">--</span>
            </div>
            <div class="h-2 bg-slate-700 rounded-full overflow-hidden">
              <div id="mod-${m}-bar" class="h-full rounded-full transition-all duration-500 bg-sky-500" style="width:0%"></div>
            </div>
          </div>
        `).join("")}
      </div>
    </div>

    <!-- SHAP Breakdown -->
    <div class="card p-6 space-y-4">
      <h2 class="text-slate-300 font-semibold text-sm uppercase tracking-wider">Score Impact (SHAP)</h2>
      <div id="shap-bars" class="space-y-3">
        <p class="text-slate-500 text-xs">Waiting for data…</p>
      </div>
    </div>

  </div><!-- end score+modules sub-grid -->
  </div><!-- end camera+scores outer grid -->

  <!-- System Health + Red Flags -->
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">

    <!-- System Health -->
    <div class="card p-6 space-y-4">
      <h2 class="text-slate-300 font-semibold text-sm uppercase tracking-wider">System Health</h2>
      <div class="grid grid-cols-2 gap-3">
        <div class="bg-slate-700/50 rounded-lg p-3">
          <p class="text-slate-500 text-xs">FPS</p>
          <p id="fps-val" class="text-xl font-bold text-emerald-400 font-mono">--</p>
        </div>
        <div class="bg-slate-700/50 rounded-lg p-3">
          <p class="text-slate-500 text-xs">Queue Depth</p>
          <p id="queue-val" class="text-xl font-bold text-amber-400 font-mono">--</p>
        </div>
        <div class="bg-slate-700/50 rounded-lg p-3">
          <p class="text-slate-500 text-xs">Warmup Left</p>
          <p id="warmup-val" class="text-xl font-bold text-sky-400 font-mono">--</p>
        </div>
        <div class="bg-slate-700/50 rounded-lg p-3">
          <p class="text-slate-500 text-xs">Backend</p>
          <p id="backend-val" class="text-sm font-bold text-purple-400 font-mono">--</p>
        </div>
      </div>
      <div>
        <p class="text-slate-500 text-xs mb-2">Modules</p>
        <div id="module-badges" class="flex flex-wrap gap-2"></div>
      </div>
    </div>

    <!-- Red Flags Timeline -->
    <div class="card p-6 space-y-4">
      <div class="flex items-center justify-between">
        <h2 class="text-slate-300 font-semibold text-sm uppercase tracking-wider">Red Flags</h2>
        <span id="flag-count" class="bg-red-900/40 text-red-300 text-xs font-mono px-2 py-0.5 rounded-full">0</span>
      </div>
      <div id="flag-list" class="space-y-2 max-h-64 overflow-y-auto pr-1">
        <p class="text-slate-500 text-xs">No red flags detected.</p>
      </div>
    </div>
  </div>

  <!-- Connection Status Bar -->
  <div class="flex items-center gap-2 text-xs text-slate-500">
    <span id="ws-dot" class="w-2 h-2 rounded-full bg-slate-600"></span>
    <span id="ws-status">Disconnected</span>
  </div>
</div>
`;

// ── Helpers ────────────────────────────────────────────────────────────────

function setWsStatus(connected) {
  const dot = document.getElementById("ws-dot");
  const label = document.getElementById("ws-status");
  dot.className = `w-2 h-2 rounded-full ${connected ? "bg-emerald-400" : "bg-slate-600"}`;
  label.textContent = connected ? "Live — WebSocket connected" : "Disconnected";
}

function renderScore(score) {
  if (!score) return;
  const val = score.final_score ?? 0;
  document.getElementById("score-value").textContent = val.toFixed(0);
  document.getElementById("score-reason").textContent = score.score_reason || "";
  document.getElementById("role-badge").textContent = score.role || "";

  const offset = 314.16 * (1 - val / 100);
  document.getElementById("score-ring").style.strokeDashoffset = offset;

  const colour = val >= 75 ? "#4ade80" : val >= 50 ? "#fbbf24" : "#f87171";
  document.getElementById("score-ring").style.stroke = colour;
  document.getElementById("score-value").style.color = colour;

  const badge = document.getElementById("cam-score-badge");
  badge.textContent = val.toFixed(0);
  badge.style.color = colour;
  badge.classList.remove("hidden");

  const mods = score.module_scores || {};
  for (const [mod, s] of Object.entries(mods)) {
    const key = mod.toLowerCase();
    const el  = document.getElementById(`mod-${key}-val`);
    const bar = document.getElementById(`mod-${key}-bar`);
    if (el)  el.textContent = `${s.toFixed(0)}%`;
    if (bar) {
      bar.style.width = `${s}%`;
      bar.style.background = s >= 75 ? "#4ade80" : s >= 50 ? "#fbbf24" : "#f87171";
    }
  }

  renderShap(score.shap_breakdown || {});
  renderRedFlags(score.red_flags || []);
}

function renderShap(shap) {
  const container = document.getElementById("shap-bars");
  if (!Object.keys(shap).length) return;
  container.innerHTML = Object.entries(shap)
    .sort((a, b) => a[1] - b[1])
    .map(([k, v]) => {
      const colour = v >= 0 ? "#4ade80" : "#f87171";
      const pct    = Math.abs(v);
      const sign   = v >= 0 ? "+" : "";
      return `
        <div>
          <div class="flex justify-between text-xs mb-1">
            <span class="text-slate-400 capitalize">${k.replace(/_/g,' ')}</span>
            <span style="color:${colour}" class="font-mono">${sign}${v.toFixed(1)}%</span>
          </div>
          <div class="h-1.5 bg-slate-700 rounded-full overflow-hidden">
            <div class="h-full rounded-full" style="width:${Math.min(pct,50)*2}%; background:${colour}"></div>
          </div>
        </div>`;
    }).join("");
}

function renderRedFlags(flags) {
  if (!flags.length) return;
  state.redFlags = [...state.redFlags, ...flags].slice(-20);
  document.getElementById("flag-count").textContent = state.redFlags.length;
  const container = document.getElementById("flag-list");
  container.innerHTML = [...state.redFlags].reverse().map(f => {
    const colours = { high: "text-red-400", medium: "text-amber-400", low: "text-sky-400" };
    const colour  = colours[f.severity] || "text-slate-300";
    const ts      = f.timestamp ? new Date(f.timestamp * 1000).toLocaleTimeString() : "";
    return `
      <div class="flag-enter bg-slate-700/50 rounded-lg p-2.5 text-xs border border-slate-600/50">
        <div class="flex justify-between items-start">
          <span class="${colour} font-semibold capitalize">[${f.module}] ${f.severity}</span>
          <span class="text-slate-500 font-mono">${ts}</span>
        </div>
        <p class="text-slate-300 mt-0.5">${f.reason}</p>
      </div>`;
  }).join("");
}

function renderHealth(health) {
  if (!health) return;
  document.getElementById("fps-val").textContent     = (health.fps || 0).toFixed(1);
  document.getElementById("queue-val").textContent   = health.frame_queue_depth ?? "--";
  document.getElementById("warmup-val").textContent  = health.warmup_remaining  ?? "--";
  document.getElementById("backend-val").textContent = health.hardware_backend  || "--";

  const banner = document.getElementById("warmup-badge");
  if (health.warmup_remaining > 0) banner.classList.remove("hidden");
  else                              banner.classList.add("hidden");

  const container = document.getElementById("module-badges");
  const all = [
    ...(health.active_modules   || []).map(m => ({ m, cls: "badge-active",   label: "active" })),
    ...(health.skipped_modules  || []).map(m => ({ m, cls: "badge-skipped",  label: "skipped" })),
    ...(health.degraded_modules || []).map(m => ({ m, cls: "badge-degraded", label: "degraded" })),
  ];
  container.innerHTML = all.map(({ m, cls, label }) => `
    <span class="${cls} rounded-full px-2.5 py-0.5 text-xs capitalize">
      ${m.replace(/_/g,' ')} · ${label}
    </span>`).join("");
}

// ── WebSocket ──────────────────────────────────────────────────────────────

function connectWS() {
  ws = new WebSocket(WS_URL);
  ws.onopen  = () => setWsStatus(true);
  ws.onclose = () => { setWsStatus(false); setTimeout(connectWS, 3000); };
  ws.onerror = () => ws.close();
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === "score")  renderScore(msg.data);
    if (msg.type === "health") renderHealth(msg.data);
  };
}

// ── Session controls ───────────────────────────────────────────────────────

document.getElementById("btn-start").addEventListener("click", async () => {
  const role = document.getElementById("role-select").value;
  state.role = role;
  state.redFlags = [];
  document.getElementById("flag-list").innerHTML = '<p class="text-slate-500 text-xs">No red flags detected.</p>';
  document.getElementById("flag-count").textContent = "0";

  try {
    const res = await fetch(`${API_BASE}/session/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ candidate_id: "demo_candidate", role }),
    });
    const data = await res.json();
    state.sessionId = data.session_id;
    state.status = "running";

    document.getElementById("session-banner").classList.remove("hidden");
    document.getElementById("session-id-label").textContent = data.session_id;
    document.getElementById("backend-label").textContent    = data.hardware_backend;
    document.getElementById("btn-start").classList.add("hidden");
    document.getElementById("btn-stop").classList.remove("hidden");
    document.getElementById("role-badge").textContent = role;
  } catch (err) {
    alert(`Failed to start session: ${err.message}`);
  }
});

document.getElementById("btn-stop").addEventListener("click", async () => {
  try {
    await fetch(`${API_BASE}/session/stop`, { method: "POST" });
  } catch (_) {}
  state.status = "stopped";
  document.getElementById("btn-stop").classList.add("hidden");
  document.getElementById("btn-start").classList.remove("hidden");
  document.getElementById("session-banner").classList.add("hidden");
});

// ── Boot ───────────────────────────────────────────────────────────────────
connectWS();
