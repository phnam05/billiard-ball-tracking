/* Billiard Tracker -- the page.  Plain JavaScript, no build step, no CDN.
 *
 * One function per screen (library, set-up, run/results, runs, live, help),
 * picked by the address's #fragment.  Each returns a clean-up function that
 * stops its timers and streams when another screen is opened.
 */
"use strict";

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

const $ = (sel, root = document) => root.querySelector(sel);

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
      else if (k === "dataset") Object.assign(el.dataset, v);
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else if (k === "html") el.innerHTML = v;
      else if (v === true) el.setAttribute(k, "");
      else el.setAttribute(k, v);
    }
  }
  append(el, kids);
  return el;
}

function append(el, kids) {
  for (const kid of kids.flat(Infinity)) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.appendChild(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

function icon(name) {
  const paths = {
    play: "M7 5v14l11-7z",
    pause: "M7 5h3v14H7z M14 5h3v14h-3z",
    prev: "M15 6l-6 6 6 6",
    next: "M9 6l6 6-6 6",
    plus: "M12 5v14 M5 12h14",
    upload: "M12 16V4 M7 9l5-5 5 5 M5 20h14",
    folder: "M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z",
    film: "M4 5h16v14H4z M8 5v14 M16 5v14 M4 9h4 M4 15h4 M16 9h4 M16 15h4",
    x: "M6 6l12 12 M18 6L6 18",
    download: "M12 4v12 M7 11l5 5 5-5 M5 20h14",
    up: "M12 19V5 M6 11l6-6 6 6",
    target: "M12 3v3 M12 18v3 M3 12h3 M18 12h3 M12 8a4 4 0 1 0 0 8a4 4 0 1 0 0-8",
    refresh: "M20 11a8 8 0 1 0-2.3 5.7 M20 5v6h-6",
    stop: "M7 7h10v10H7z",
  };
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
  p.setAttribute("d", paths[name] || "");
  if (name === "play" || name === "pause" || name === "stop") {
    p.setAttribute("fill", "currentColor");
    p.setAttribute("stroke", "none");
  }
  svg.appendChild(p);
  return svg;
}

async function api(method, url, body, { quiet = false } = {}) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(url, opts);
  } catch (err) {
    const e = new Error("The app is not responding. Is `billiards app` still running?");
    if (!quiet) toast(e.message, "bad");
    throw e;
  }
  let data = null;
  const text = await res.text();
  try { data = text ? JSON.parse(text) : null; } catch { data = null; }
  if (!res.ok) {
    const e = new Error((data && data.error) || `${res.status} ${res.statusText}`);
    e.status = res.status;
    if (!quiet) toast(e.message, "bad");
    throw e;
  }
  return data;
}

function toast(message, kind = "") {
  const el = h("div", { class: `toast ${kind}` }, message);
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), kind === "bad" ? 7000 : 3500);
}

function lifecycle() {
  const timers = [], ends = [];
  const life = {
    alive: true,
    every(ms, fn) { const id = setInterval(() => life.alive && fn(), ms); timers.push(id); return id; },
    later(ms, fn) { const id = setTimeout(() => life.alive && fn(), ms); timers.push(id); return id; },
    on(target, ev, fn, opts) { target.addEventListener(ev, fn, opts); ends.push(() => target.removeEventListener(ev, fn, opts)); },
    onEnd(fn) { ends.push(fn); },
    end() {
      life.alive = false;
      timers.forEach((id) => { clearInterval(id); clearTimeout(id); });
      ends.forEach((fn) => { try { fn(); } catch (e) { console.error(e); } });
    },
  };
  return life;
}

function debounce(fn, ms) {
  let id = null;
  return (...args) => { clearTimeout(id); id = setTimeout(() => fn(...args), ms); };
}

function fmtClock(s) {
  if (s === null || s === undefined || !isFinite(s)) return "–";
  const neg = s < 0; s = Math.abs(s);
  const m = Math.floor(s / 60), sec = s - 60 * m;
  return `${neg ? "-" : ""}${m}:${sec.toFixed(2).padStart(5, "0")}`;
}
function fmtDur(s) {
  if (s === null || s === undefined || !isFinite(s)) return "–";
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)} s`;
  const m = Math.floor(s / 60), sec = Math.round(s - 60 * m);
  if (m < 60) return `${m} min ${sec ? sec + " s" : ""}`.trim();
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}
function fmtBytes(n) {
  if (!n && n !== 0) return "";
  const u = ["B", "KB", "MB", "GB"]; let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n < 10 && i ? 1 : 0)} ${u[i]}`;
}
function fmtWhen(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000), now = new Date();
  const sameDay = d.toDateString() === now.toDateString();
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  if (sameDay) return `today ${time}`;
  return `${d.toLocaleDateString([], { day: "numeric", month: "short" })} ${time}`;
}
function fmtFps(f) { return f ? `${(+f).toFixed(Math.abs(f - Math.round(f)) < 0.02 ? 0 : 1)} fps` : ""; }
function plural(n, word, many) { return `${n} ${n === 1 ? word : (many || word + "s")}`; }

const store = {
  get(key, fallback) { try { const v = localStorage.getItem("bt." + key); return v === null ? fallback : JSON.parse(v); } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem("bt." + key, JSON.stringify(value)); } catch { /* private window */ } },
};

// Speeds are measured in inches a second; km/h reads better to most people.
const UNITS = {
  "in/s": { f: 1, label: "in/s", digits: 0 },
  "km/h": { f: 0.09144, label: "km/h", digits: 1 },
};
function speedText(v) {
  const u = UNITS[store.get("units", "km/h")] || UNITS["km/h"];
  return `${(v * u.f).toFixed(u.digits)} ${u.label}`;
}

const EVENTS = {
  ball_struck: { name: "Struck", colour: "var(--ev-struck)" },
  collision: { name: "Contact", colour: "var(--ev-collision)" },
  cushion: { name: "Cushion", colour: "var(--ev-cushion)" },
  pot: { name: "Pot", colour: "var(--ev-pot)" },
};
function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function eventColour(type) { return cssVar(`--ev-${type === "ball_struck" ? "struck" : type}`) || "#fff"; }

function eventText(e) {
  const who = (e.labels && e.labels.length ? e.labels : (e.track_ids || []).map((t) => `#${t}`));
  const name = (l) => (l === "CUE" ? "cue ball" : /^\d+$/.test(l) ? `the ${l}` : l);
  switch (e.type) {
    case "collision": return `${name(who[0] || "?")} hit ${name(who[1] || "?")}`;
    case "cushion": return `${name(who[0] || "?")} off a cushion`;
    case "pot": return `${name(who[0] || "?")} potted`;
    case "ball_struck": return `${name(who[0] || "?")} struck${e.speed_in_s ? ` · ${speedText(e.speed_in_s)}` : ""}`;
    default: return e.type;
  }
}

function ballChip(label, colour) {
  const light = colour && luminance(colour) > 0.45;
  return h("span", { class: "ballchip", style: { background: colour || "#999", color: light ? "#111" : "#fff" } }, label);
}
function luminance(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex || "");
  if (!m) return 0.5;
  const n = parseInt(m[1], 16);
  return (0.299 * (n >> 16 & 255) + 0.587 * (n >> 8 & 255) + 0.114 * (n & 255)) / 255;
}

// Bright versions of the numbered balls, for drawing: the colours measured on
// camera are dull, and a diagram reads better with the colours people expect.
const BALL_COLOURS = {
  standard: { 1: "#f2c318", 2: "#1d4fd8", 3: "#d62a2a", 4: "#7b3fb0", 5: "#f07b16", 6: "#15894b", 7: "#7a2e1c", 8: "#111111" },
  // The set on the sample broadcasts: the 4 pink and the 5 purple.
  tv: { 1: "#f2c318", 2: "#1d4fd8", 3: "#d62a2a", 4: "#ef6fa6", 5: "#7b3fb0", 6: "#15894b", 7: "#7a2e1c", 8: "#111111" },
};
let ballSet = "standard";
function drawColour(track) {
  if (!track) return "#9aa0a6";
  if (track.type === "cue" || track.label === "CUE") return "#f4f1e6";
  if (track.label === "8" || track.type === "eight") return "#141414";
  const n = track.number;
  const palette = BALL_COLOURS[ballSet] || BALL_COLOURS.standard;
  if (n) return palette[n > 8 ? n - 8 : n] || track.colour;
  return track.colour || "#9aa0a6";
}

function statusPill(status) {
  const map = {
    done: ["ok", "done"], running: ["run", "tracking"], calibrating: ["run", "finding the table"],
    queued: ["", "waiting"], finishing: ["run", "finishing"], failed: ["bad", "failed"],
    cancelled: ["warn", "stopped"], tracking: ["ok", "tracking"], connecting: ["run", "connecting"],
    no_table: ["warn", "no table found"], ended: ["", "ended"], stopped: ["", "stopped"], idle: ["", "idle"],
    error: ["bad", "error"],
  };
  const [cls, text] = map[status] || ["", status || "?"];
  return h("span", { class: `pill ${cls}` }, text);
}

function modal(title, body, foot) {
  const back = h("div", { class: "modal-back" });
  const close = () => { back.remove(); document.removeEventListener("keydown", onKey); };
  const onKey = (e) => { if (e.key === "Escape") close(); };
  document.addEventListener("keydown", onKey);
  back.addEventListener("mousedown", (e) => { if (e.target === back) close(); });
  const box = h("div", { class: "modal panel", role: "dialog", "aria-label": title },
    h("div", { class: "modal-head" }, h("h2", null, title),
      h("button", { class: "btn icon ghost", onclick: close, "aria-label": "Close" }, icon("x"))),
    h("div", { class: "modal-body" }, body),
    foot ? h("div", { class: "modal-foot" }, foot) : null);
  back.appendChild(box);
  document.body.appendChild(back);
  return { close, box };
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

const ROUTES = [
  [/^#\/library\/?$/, viewLibrary, "library"],
  [/^#\/video\/([0-9a-f]{12})$/, viewSetup, "library"],
  [/^#\/run\/([\w-]+)(?:\?(.*))?$/, viewRun, "runs"],
  [/^#\/runs\/?$/, viewRuns, "runs"],
  [/^#\/live\/?$/, viewLive, "live"],
  [/^#\/help\/?$/, viewHelp, "help"],
];
let currentEnd = null;

function navigate() {
  if (currentEnd) { currentEnd(); currentEnd = null; }
  const main = $("#main");
  main.innerHTML = "";
  const hash = location.hash || "#/library";
  for (const [re, view, nav] of ROUTES) {
    const m = re.exec(hash);
    if (m) {
      document.querySelectorAll("[data-nav]").forEach((a) => a.classList.toggle("active", a.dataset.nav === nav));
      const life = lifecycle();
      currentEnd = () => life.end();
      try { view(main, life, ...m.slice(1)); } catch (err) { console.error(err); main.append(h("div", { class: "notice bad" }, String(err))); }
      main.focus({ preventScroll: true });
      window.scrollTo(0, 0);
      return;
    }
  }
  location.hash = "#/library";
}

// Sidebar: how many runs are going, and whether live is on.
async function pollStatus() {
  try {
    const s = await api("GET", "/api/status", undefined, { quiet: true });
    const badge = $("#runs-badge");
    badge.hidden = !s.active_runs;
    badge.textContent = s.active_runs || "";
    $("#live-dot").hidden = !["tracking", "calibrating", "connecting", "no_table"].includes(s.live);
    const foot = $("#sidebar-foot");
    foot.textContent = "";
    foot.title = `Workspace: ${s.workspace}`;
    const short = s.workspace.split(/[\\/]/).filter(Boolean).slice(-2).join("/");
    append(foot, [`v${s.version} · OpenCV ${s.opencv}`, h("br"), `workspace: …/${short}`]);
    if (!s.browser_playable) foot.append(h("br"), h("span", { style: { color: "var(--warn)" } }, "videos shown frame by frame: pip install imageio-ffmpeg"));
  } catch { /* the page says so when it matters */ }
}

// ---------------------------------------------------------------------------
// Adding footage
// ---------------------------------------------------------------------------

async function uploadFiles(files) {
  const list = Array.from(files).filter((f) => /\.(mp4|mov|mkv|avi|webm|m4v|mpe?g|wmv|ts|mts|m2ts|flv|3gp)$/i.test(f.name));
  if (!list.length) { toast("Those are not video files.", "bad"); return []; }
  const added = [];
  for (const file of list) {
    const note = h("div", { class: "toast" }, `Uploading ${file.name}… `, h("span", { class: "muted" }, "0%"));
    $("#toasts").appendChild(note);
    try {
      const res = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("PUT", `/api/upload?name=${encodeURIComponent(file.name)}`);
        xhr.upload.onprogress = (e) => { if (e.lengthComputable) note.lastChild.textContent = `${Math.round(100 * e.loaded / e.total)}%`; };
        xhr.onload = () => {
          let data = null; try { data = JSON.parse(xhr.responseText); } catch { /* */ }
          xhr.status < 300 ? resolve(data) : reject(new Error((data && data.error) || xhr.statusText));
        };
        xhr.onerror = () => reject(new Error("upload failed"));
        xhr.send(file);
      });
      added.push(res.added);
      toast(`Added ${file.name}`, "ok");
    } catch (err) {
      toast(`${file.name}: ${err.message}`, "bad");
    } finally {
      note.remove();
    }
  }
  return added;
}

function openAddDialog(onAdded) {
  const input = h("input", { type: "text", placeholder: "C:\\Videos\\pool   or   C:\\Videos\\match.mp4", style: { flex: "1" } });
  const list = h("div", { class: "browser-list" });
  const where = h("div", { class: "small muted ellipsis" });
  const addHere = h("button", { class: "btn sm", disabled: true }, icon("folder"), "Add this folder");
  let current = "";

  async function addPath(path) {
    const res = await api("POST", "/api/videos/add", { path });
    if (res.added_folder) toast(`Added the folder: ${plural(res.videos.length, "video")}`, "ok");
    else toast(`Added ${res.added.name}`, "ok");
    onAdded && onAdded(res);
  }

  async function browse(path) {
    let data;
    try { data = await api("GET", `/api/browse?path=${encodeURIComponent(path || "")}`); } catch { return; }
    current = data.path;
    store.set("browse", current);
    where.textContent = data.path || "This computer";
    addHere.disabled = !data.path;
    list.innerHTML = "";
    if (data.parent !== null) {
      list.append(h("div", { onclick: () => browse(data.parent) }, icon("up"), h("span", null, "Up")));
    }
    for (const d of data.dirs) {
      const name = data.roots ? d : d.split(/[\\/]/).filter(Boolean).pop();
      list.append(h("div", { onclick: () => browse(d), title: d }, icon("folder"), h("span", { class: "ellipsis" }, name)));
    }
    for (const v of data.videos) {
      list.append(h("div", { class: "vid" }, icon("film"),
        h("span", { class: "ellipsis", style: { flex: 1 } }, v.name),
        h("span", { class: "muted small" }, fmtBytes(v.size_bytes)),
        h("button", { class: "btn sm", onclick: () => addPath(v.path) }, "Add")));
    }
    if (!data.dirs.length && !data.videos.length) list.append(h("div", { class: "vid muted" }, "Nothing here."));
  }

  addHere.addEventListener("click", () => current && addPath(current));
  const addTyped = async () => { if (input.value.trim()) { await addPath(input.value.trim()); input.value = ""; } };
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") addTyped(); });

  const fileInput = h("input", { type: "file", accept: "video/*", multiple: true, hidden: true });
  fileInput.addEventListener("change", async () => {
    const added = await uploadFiles(fileInput.files);
    fileInput.value = "";
    if (added.length) { m.close(); onAdded && onAdded({ added: added[0] }); }
  });

  const m = modal("Add videos", [
    h("div", { class: "field" }, h("span", null, "A video on this computer"),
      h("div", { class: "btn-row" }, h("button", { class: "btn primary", onclick: () => fileInput.click() }, icon("upload"), "Choose video files…"),
        h("span", { class: "small muted" }, "or drop them anywhere on the page")), fileInput),
    h("div", { class: "field" }, h("span", null, "Or list a whole folder, or a file where it is, without copying it"),
      h("div", { class: "btn-row" }, input, h("button", { class: "btn", onclick: addTyped }, "Add"))),
    h("div", { class: "field" }, h("div", { class: "btn-row" }, where, h("span", { style: { flex: 1 } }), addHere), list),
    h("div", { class: "small muted" }, "A folder is scanned for videos each time the library opens (not its sub-folders)."),
  ], [h("button", { class: "btn", onclick: () => m.close() }, "Done")]);
  browse(store.get("browse", ""));
}

// ---------------------------------------------------------------------------
// Library
// ---------------------------------------------------------------------------

function viewLibrary(main, life) {
  const search = h("input", { type: "search", placeholder: "Filter by name", value: store.get("filter", "") });
  const folders = h("div", { class: "chips" });
  const grid = h("div", { class: "grid" });
  const guide = h("div");
  const addBtn = h("button", { class: "btn", onclick: () => openAddDialog(load) }, icon("plus"), "Add videos");
  let data = { videos: [], folders: [] };

  main.append(
    h("div", { class: "page-head" },
      h("div", null, h("h1", null, "Library"), h("div", { class: "sub" }, "Your videos. Pick one and track it; its results open when it is done.")),
      h("div", { class: "spacer" }), addBtn),
    guide,
    h("div", { class: "toolbar" }, search, folders),
    grid);

  search.addEventListener("input", () => { store.set("filter", search.value); render(); });
  life.onEnd(() => { window.onLibraryChanged = null; });
  window.onLibraryChanged = load;

  async function load() {
    try { data = await api("GET", "/api/videos"); } catch { return; }
    if (life.alive) render();
  }

  function render() {
    folders.innerHTML = "";
    for (const f of data.folders) {
      const short = f.split(/[\\/]/).filter(Boolean).slice(-2).join("/");
      folders.append(h("span", { class: "chip", title: f }, icon("folder"), short,
        /[\\/]uploads$/.test(f) ? null : h("button", {
          "aria-label": `Stop listing ${f}`, title: "Stop listing this folder",
          onclick: async () => { await api("POST", "/api/folders/remove", { path: f }); load(); },
        }, "×")));
    }
    renderGuide();
    addBtn.classList.toggle("primary", !data.videos.length);
    const q = search.value.trim().toLowerCase();
    const vids = data.videos.filter((v) => !q || v.name.toLowerCase().includes(q));
    grid.innerHTML = "";
    if (!data.videos.length) {
      grid.append(h("div", { class: "empty", style: { gridColumn: "1 / -1" } },
        h("div", { class: "big" }, "No videos yet"),
        h("div", null, "Drop a video file anywhere on this page, or press Add videos."),
        h("button", { class: "btn primary", onclick: () => openAddDialog(load) }, icon("plus"), "Add videos")));
      return;
    }
    if (!vids.length) grid.append(h("div", { class: "muted" }, "Nothing matches that filter."));
    for (const v of vids) grid.append(card(v));
  }

  // Where the user is in "add a video, track it, see the results", and what to do next.
  function renderGuide() {
    const vids = data.videos.filter((v) => !v.error);
    const running = vids.find((v) => v.active_run);
    const tracked = vids.some((v) => v.last_run);
    let step, text;
    if (!vids.length) { step = 1; text = "Start by adding a video: drop the file anywhere on this page, or press Add videos."; }
    else if (running) { step = 2; text = h("span", null, "Tracking ", h("b", null, running.name), ". Its results open when it is done; you can look around meanwhile."); }
    else if (!tracked) { step = 2; text = h("span", null, "Press ", h("b", null, "Track"), " on a video. It needs no set-up, and takes about as long as the video plays."); }
    else { step = 3; text = h("span", null, "Press ", h("b", null, "See results"), " on a video to watch it back with every shot, cushion and pot. Or press ", h("b", null, "Track"), " on one you have not tracked yet."); }
    const labels = ["Add a video", "Track it", "See the results"];
    guide.replaceChildren(h("div", { class: "guide panel" },
      h("ol", { class: "steps" }, labels.map((label, i) =>
        h("li", { class: i + 1 < step ? "done" : i + 1 === step ? "now" : "" }, h("span", { class: "n" }, i + 1 < step ? "✓" : String(i + 1)), label))),
      h("div", { class: "guide-text" }, text)));
  }

  function card(v) {
    // The picture does what the blue button does, except start a run: results
    // once there are some, the progress while it tracks, else the set-up preview.
    const target = v.active_run ? `#/run/${v.active_run.id}` : v.last_run ? `#/run/${v.last_run.id}` : `#/video/${v.id}`;
    const open = () => { location.hash = target; };
    const img = h("img", { src: `/api/videos/${v.id}/thumb.jpg`, alt: "", loading: "lazy" });
    const thumb = h("div", { class: "thumb", onclick: open, title: v.last_run ? "See the results" : v.active_run ? "Watch it being tracked" : "Check the table before tracking" }, img,
      v.duration_s ? h("span", { class: "badge" }, fmtClock(v.duration_s).replace(/\.\d+$/, "")) : null);
    img.addEventListener("error", () => { img.remove(); thumb.prepend(h("div", { class: "noimg" }, v.error ? "cannot be read" : "no preview")); });

    let status;
    if (v.error) status = h("span", { style: { color: "var(--bad)" } }, `⚠ ${v.error}`);
    else if (v.active_run) {
      const r = v.active_run;
      status = h("div", { class: "stack", style: { gap: "4px" } },
        h("span", null, statusPill(r.status), " ", r.status === "running" ? `${Math.round(100 * r.progress)}% · ${r.fps} fps` : ""),
        h("div", { class: "progress" }, h("i", { style: { width: `${100 * r.progress}%` } })));
    } else if (v.last_run) {
      const b = v.last_run.brief || {};
      const shots = (b.shots || []).length;
      status = h("span", null, "✓ tracked ", fmtWhen(v.last_run.finished), " · ", plural(b.tracks || 0, "ball"), " · ", plural(shots, "shot"));
    } else status = h("span", { class: "muted" }, v.has_settings ? "set up, not tracked yet" : "not tracked yet");

    const track = async () => {
      try {
        const job = await api("POST", "/api/runs", { video_id: v.id });
        location.hash = `#/run/${job.id}`;
      } catch { /* toast shown */ }
    };
    const forget = async () => {
      if (!confirm(`Remove ${v.name} from the library? The file itself is kept.`)) return;
      await api("POST", "/api/videos/forget", { id: v.id });
      load();
    };
    // One obvious next step per video; everything else is a quiet link.
    let main_, more;
    if (v.active_run) {
      main_ = h("a", { class: "btn primary", href: target }, "Watch progress");
      more = [];
    } else if (v.last_run) {
      main_ = h("a", { class: "btn primary", href: target }, icon("play"), "See results");
      more = [h("button", { class: "link", onclick: track }, "Track again"), h("a", { class: "link", href: `#/video/${v.id}` }, "Change set-up")];
    } else {
      main_ = h("button", { class: "btn primary", onclick: track, disabled: !!v.error }, "Track");
      more = v.error ? [] : [h("a", { class: "link", href: `#/video/${v.id}` }, "Check the table first")];
    }
    return h("div", { class: "card panel" }, thumb,
      h("div", { class: "card-body" },
        h("div", { class: "card-title", title: v.path, onclick: open }, v.name),
        h("div", { class: "card-meta" }, v.error ? v.folder : `${v.width}×${v.height} · ${fmtFps(v.fps)} · ${fmtBytes(v.size_bytes)}`),
        h("div", { class: "card-status" }, status),
        h("div", { class: "card-actions" }, main_,
          h("div", { class: "card-more" }, more,
            h("button", { class: "link quiet", title: "Remove from the library (the file is kept)", onclick: forget }, "Remove")))));
  }

  load();
  life.every(2500, () => { if (data.videos.some((v) => v.active_run)) load(); });
  life.every(15000, load);
}

// ---------------------------------------------------------------------------
// Set up: where the tracker thinks the table is, on a frame you pick
// ---------------------------------------------------------------------------

const PRESET_NAMES = {
  "pool-9ft": "Pool, 9 ft (100 × 50 in)", "pool-8ft": "Pool, 8 ft (88 × 44 in)", "pool-7ft": "Pool, 7 ft (78 × 39 in)",
  "snooker-12ft": "Snooker, 12 ft", "carom-10ft": "Carom, 10 ft (no pockets)",
};

function settingsForm(settings, onChange, { duration = null, compact = false, currentTime = null } = {}) {
  const set = (k, v) => { settings[k] = v; onChange(settings, k); };
  const sel = (key, options, attrs = {}) => {
    const s = h("select", { ...attrs, onchange: () => set(key, s.value) },
      options.map(([v, label]) => h("option", { value: v, selected: String(settings[key]) === String(v) }, label)));
    return s;
  };
  const numbersPresets = [["1-15", "1–15 (8-ball)"], ["1-9", "1–9 (9-ball)"], ["1-10", "1–10 (10-ball)"]];
  if (!numbersPresets.some(([v]) => v === settings.numbers)) numbersPresets.push([settings.numbers, settings.numbers]);
  const widths = [[960, "960 px (fastest)"], [1280, "1280 px"], [1920, "1920 px"], [0, "Full size (slowest)"]];
  if (!widths.some(([v]) => v === settings.max_width)) widths.push([settings.max_width, `${settings.max_width} px`]);

  const timeInput = (key) => {
    const i = h("input", { type: "number", min: 0, step: 0.1, value: settings[key] ?? "", placeholder: key === "end_s" ? "end" : "0", style: { width: "86px" },
      onchange: () => set(key, i.value === "" ? (key === "end_s" ? null : 0) : parseFloat(i.value)) });
    return i;
  };
  const kids = [
    h("label", { class: "field" }, h("span", null, "Table"), sel("preset", Object.keys(PRESET_NAMES).map((k) => [k, PRESET_NAMES[k]]))),
    h("div", { class: "fields two" },
      h("label", { class: "field" }, h("span", null, "Ball set"), sel("ball_set", [["auto", "Work it out"], ["standard", "Standard (4 purple, 5 orange)"], ["tv", "TV (4 pink, 5 purple)"]])),
      h("label", { class: "field" }, h("span", null, "Balls in play"), sel("numbers", numbersPresets))),
  ];
  if (!compact) {
    const start = timeInput("start_s"), end = timeInput("end_s");
    kids.push(h("div", { class: "field" }, h("span", null, "Part of the video to track (seconds)"),
      h("div", { class: "btn-row" }, start, "to", end,
        currentTime ? h("button", { class: "btn sm ghost", title: "Start at the frame shown", onclick: () => { start.value = currentTime().toFixed(1); set("start_s", parseFloat(start.value)); } }, "start here") : null,
        currentTime ? h("button", { class: "btn sm ghost", title: "End at the frame shown", onclick: () => { end.value = currentTime().toFixed(1); set("end_s", parseFloat(end.value)); } }, "end here") : null),
      duration ? h("span", { class: "hint" }, `The whole video is ${fmtDur(duration)}.`) : null));
  }
  kids.push(h("label", { class: "field" }, h("span", null, "Processing size"), sel("max_width", widths),
    h("span", { class: "hint" }, "Bigger finds small balls better; smaller is faster.")));
  if (!compact) {
    kids.push(h("label", { class: "check" }, h("input", { type: "checkbox", checked: settings.draw_trails, onchange: (e) => set("draw_trails", e.target.checked) }), "Draw the balls' trails on the video"));
    kids.push(h("label", { class: "check" }, h("input", { type: "checkbox", checked: settings.burn_in_panel, onchange: (e) => set("burn_in_panel", e.target.checked) }), "Put the diagram and status under the video, too"));
  }
  kids.push(h("label", { class: "check", title: "Balls resting against the far cushion appear past the table's edge. Looking there finds them, and pots into the far corners, but can also take a hand on the rail, or a black ball's shadow, for a ball." },
    h("input", { type: "checkbox", checked: settings.far_cushion, onchange: (e) => set("far_cushion", e.target.checked) }),
    "Look for balls against the far cushion ", h("span", { class: "muted small" }, "(experimental)")));
  return h("div", { class: "fields" }, kids);
}

/** A picture with the table drawn over it, and corners that can be dragged. */
function tableStage({ onCornersChange } = {}) {
  const img = h("img", { alt: "" });
  const canvas = h("canvas", { class: "overlay" });
  const loading = h("div", { class: "loading", hidden: true }, "Checking…");
  const el = h("div", { class: "stage" }, img, canvas, loading);
  const st = { report: null, size: null, corners: null, editing: false, show: { outline: true, balls: true }, drag: -1 };

  function toImage(ev) {
    const r = canvas.getBoundingClientRect();
    return [(ev.clientX - r.left) * canvas.width / r.width, (ev.clientY - r.top) * canvas.height / r.height];
  }
  canvas.addEventListener("pointerdown", (ev) => {
    if (!st.editing || !st.corners) return;
    const [x, y] = toImage(ev);
    const reach = 22 * canvas.width / canvas.getBoundingClientRect().width;
    let best = -1, bestD = reach;
    st.corners.forEach(([cx, cy], i) => { const d = Math.hypot(cx - x, cy - y); if (d < bestD) { best = i; bestD = d; } });
    if (best >= 0) { st.drag = best; canvas.setPointerCapture(ev.pointerId); ev.preventDefault(); }
  });
  canvas.addEventListener("pointermove", (ev) => {
    if (st.drag < 0) return;
    const [x, y] = toImage(ev);
    st.corners[st.drag] = [Math.max(0, Math.min(canvas.width, x)), Math.max(0, Math.min(canvas.height, y))];
    draw();
  });
  const up = () => { if (st.drag >= 0) { st.drag = -1; onCornersChange && onCornersChange(st.corners); } };
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", up);

  function draw() {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const k = canvas.width / Math.max(1, canvas.getBoundingClientRect().width || canvas.width); // image px per screen px
    const rep = st.report;
    const accent = cssVar("--accent") || "#3ea6ff";
    const poly = (pts, close = true) => { ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); if (close) ctx.closePath(); };
    if (rep && rep.ok && !st.editing) {
      if (st.show.outline) {
        if (rep.outline) { ctx.setLineDash([6 * k, 5 * k]); ctx.strokeStyle = "rgba(255,255,255,.55)"; ctx.lineWidth = 1.2 * k; poly(rep.outline); ctx.stroke(); ctx.setLineDash([]); }
        poly(rep.bed); ctx.fillStyle = "rgba(62,166,255,.10)"; ctx.fill(); ctx.strokeStyle = accent; ctx.lineWidth = 2 * k; ctx.stroke();
        ctx.strokeStyle = "rgba(255,110,110,.8)"; ctx.lineWidth = 1.2 * k;
        for (const p of rep.pockets || []) { poly(p); ctx.stroke(); }
      }
      if (st.show.balls) {
        ctx.strokeStyle = "#5dff9d"; ctx.lineWidth = 2 * k;
        for (const d of rep.detections || []) { ctx.beginPath(); ctx.arc(d.x, d.y, Math.max(3, d.r + 2 * k), 0, 2 * Math.PI); ctx.stroke(); }
      }
    }
    if (st.editing && st.corners) {
      poly(st.corners); ctx.fillStyle = "rgba(245,185,74,.12)"; ctx.fill(); ctx.strokeStyle = "#f5b94a"; ctx.lineWidth = 2 * k; ctx.stroke();
      st.corners.forEach(([x, y], i) => {
        ctx.beginPath(); ctx.arc(x, y, 9 * k, 0, 2 * Math.PI); ctx.fillStyle = "rgba(20,20,20,.75)"; ctx.fill();
        ctx.lineWidth = 2 * k; ctx.strokeStyle = "#f5b94a"; ctx.stroke();
        ctx.fillStyle = "#fff"; ctx.font = `${11 * k}px system-ui`; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(String(i + 1), x, y + 0.5 * k);
      });
    }
  }
  new ResizeObserver(() => draw()).observe(el);

  return {
    el,
    setImage(src, size) {
      return new Promise((resolve) => {
        img.onload = () => { canvas.width = size ? size[0] : img.naturalWidth; canvas.height = size ? size[1] : img.naturalHeight; st.size = [canvas.width, canvas.height]; draw(); resolve(); };
        img.onerror = () => resolve();
        img.src = src;
      });
    },
    setReport(rep) { st.report = rep; if (rep && rep.image_size) { canvas.width = rep.image_size[0]; canvas.height = rep.image_size[1]; } draw(); },
    edit(corners) {
      st.editing = true;
      canvas.classList.add("editing");
      const [w, hh] = st.size || [canvas.width, canvas.height];
      st.corners = corners ? corners.map((c) => c.slice()) : [[w * .2, hh * .25], [w * .8, hh * .25], [w * .9, hh * .85], [w * .1, hh * .85]];
      draw();
    },
    stopEditing() { st.editing = false; canvas.classList.remove("editing"); draw(); },
    get corners() { return st.corners; },
    get editing() { return st.editing; },
    show(key, on) { st.show[key] = on; draw(); },
    busy(on, text) { loading.hidden = !on; if (text) loading.textContent = text; },
  };
}

function viewSetup(main, life, vid) {
  let video = null, settings = null, report = null, t = 0;
  const stage = tableStage({ onCornersChange: () => {} });
  const slider = h("input", { type: "range", min: 0, max: 1, step: 0.04, value: 0, "aria-label": "Frame" });
  const timeLabel = h("span", { class: "mono small muted", style: { minWidth: "70px" } });
  const tablePanel = h("div", { class: "panel panel-pad stack" });
  const cornerPanel = h("div", { class: "panel panel-pad stack" });
  const formPanel = h("div", { class: "panel panel-pad stack" });
  const runsPanel = h("div", { class: "panel panel-pad stack" });
  const trackBtn = h("button", { class: "btn primary", disabled: true }, "Track this video");
  const lastLink = h("a", { class: "btn", hidden: true }, icon("play"), "See the last results");
  const title = h("h1", { class: "ellipsis" }, "…");
  let mask = false;
  const chips = h("div", { class: "chips" },
    toggleChip("Outline", true, (on) => stage.show("outline", on)),
    toggleChip("Balls", true, (on) => stage.show("balls", on)),
    toggleChip("Cloth mask", false, (on) => { mask = on; loadImage(); }));

  main.append(
    h("div", { class: "crumbs" }, h("a", { href: "#/library" }, "Library"), " / set up"),
    h("div", { class: "page-head" },
      h("div", { style: { minWidth: 0, flex: "1 1 380px" } }, title,
        h("div", { class: "sub" }, "Check that the blue outline sits on the playing surface, then press ", h("b", null, "Track this video"),
          ". If it is off, press ", h("b", null, "Place corners by hand"), " on the right.")),
      h("div", { class: "btn-row", style: { flexWrap: "nowrap" } }, lastLink, trackBtn)),
    h("div", { class: "split" },
      h("div", null, stage.el,
        h("div", { class: "stage-bar" }, slider, timeLabel,
          h("button", { class: "btn sm", onclick: () => check() }, icon("refresh"), "Check this frame")),
        h("div", { class: "stage-bar" }, chips, h("span", { class: "spacer" }),
          h("span", { class: "small muted" }, "Blue: the bed the tracker uses · dashed: the cloth's edge · red: pockets · green: balls it sees"))),
      h("div", { class: "stack" }, tablePanel, cornerPanel, formPanel, runsPanel)));

  function toggleChip(label, on, fn) {
    const c = h("span", { class: `chip toggle ${on ? "" : "off"}`, role: "button", tabindex: 0 }, label);
    const flip = () => { on = !on; c.classList.toggle("off", !on); fn(on); };
    c.addEventListener("click", flip);
    c.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); flip(); } });
    return c;
  }

  function loadImage() {
    const w = settings.max_width || 0;
    const src = mask ? `/api/videos/${vid}/mask.jpg?t=${t}` : `/api/videos/${vid}/frame.jpg?t=${t}&w=${w}`;
    return stage.setImage(src, report && report.image_size);
  }

  const saveSettings = debounce(async () => {
    try { settings = (await api("PUT", `/api/videos/${vid}/settings`, settings)).settings; } catch { return; }
    check();
  }, 350);

  async function check() {
    stage.busy(true, "Checking the table…");
    try {
      report = await api("POST", `/api/videos/${vid}/check`, { settings, t });
    } catch { stage.busy(false); return; }
    if (!life.alive) return;
    await loadImage();
    stage.setReport(report);
    stage.busy(false);
    renderTable();
    renderCorners();
  }

  function renderTable() {
    tablePanel.innerHTML = "";
    const r = report;
    tablePanel.append(h("h2", null, "The table"));
    if (!r) return;
    if (!r.ok) {
      tablePanel.append(
        h("div", { class: "status-line" }, h("span", { class: "pill bad" }, "not found")),
        h("div", { class: "notice bad" }, r.error),
        h("div", { class: "small muted" }, "Turn on the cloth mask to see what was taken for cloth. If the table is there, place its corners by hand."));
      return;
    }
    tablePanel.append(
      h("div", { class: "status-line" },
        r.manual ? h("span", { class: "pill ok" }, "corners set by hand") : h("span", { class: "pill ok" }, "found automatically"),
        h("span", { class: "small muted" }, r.manual ? "" : `in ${r.frames_used} of ${r.frames_attempted} sampled frames`)),
      h("dl", { class: "kv" },
        h("dt", null, "Scale"), h("dd", null, `${r.px_per_inch} px per inch`),
        h("dt", null, "A ball"), h("dd", null, `${(2 * r.ball_radius_px).toFixed(1)} px across (middle of the table)`),
        h("dt", null, "Cloth"), h("dd", null, h("span", { class: "swatch", style: { background: r.cloth.colour } }), ` ${Math.round(100 * r.cloth.coverage)}% of the bed looks like cloth here`),
        h("dt", null, "Balls seen"), h("dd", null, `${r.detections.length} on this frame`),
        r.camera ? [h("dt", null, "Camera"), h("dd", null, `${Math.round(r.camera.position_in[2])} in above the cloth`)] : null));
    for (const w of r.warnings) tablePanel.append(h("div", { class: "notice" }, w));
  }

  function renderCorners() {
    cornerPanel.innerHTML = "";
    cornerPanel.append(h("h2", null, "Corners"));
    if (stage.editing) {
      cornerPanel.append(
        h("div", { class: "notice info" }, "Drag the four handles onto the corners of the ", h("b", null, "playing surface"), ": where the cushion noses meet, not the outer edge of the rails. Any order."),
        h("div", { class: "btn-row" },
          h("button", { class: "btn primary", onclick: async () => { settings.corners = stage.corners.map(([x, y]) => [Math.round(x * 10) / 10, Math.round(y * 10) / 10]); stage.stopEditing(); saveSettings(); } }, "Use these corners"),
          h("button", { class: "btn", onclick: () => { stage.stopEditing(); renderCorners(); } }, "Cancel")));
      return;
    }
    if (settings.corners) {
      cornerPanel.append(h("div", { class: "small muted" }, "This video uses corners you placed."),
        h("div", { class: "btn-row" },
          h("button", { class: "btn", onclick: () => { stage.edit(settings.corners); renderCorners(); } }, icon("target"), "Adjust"),
          h("button", { class: "btn ghost", onclick: () => { settings.corners = null; saveSettings(); } }, "Find automatically again")));
    } else {
      cornerPanel.append(h("div", { class: "small muted" }, "Found from the cloth. If the blue outline is off, place the corners yourself."),
        h("button", { class: "btn", onclick: () => { stage.edit(report && report.ok ? report.corners : null); renderCorners(); } }, icon("target"), "Place corners by hand"));
    }
  }

  function renderForm() {
    formPanel.innerHTML = "";
    formPanel.append(h("h2", null, "Settings"),
      settingsForm(settings, () => saveSettings(), { duration: video.duration_s, currentTime: () => t }));
  }

  async function renderRuns() {
    let runs = [];
    try { runs = (await api("GET", "/api/runs", undefined, { quiet: true })).runs.filter((r) => r.video_id === vid); } catch { return; }
    runsPanel.innerHTML = "";
    runsPanel.append(h("h2", null, "Results of this video"));
    const done = runs.find((r) => r.status === "done");
    lastLink.hidden = !done;
    if (done) lastLink.href = `#/run/${done.id}`;
    if (!runs.length) { runsPanel.append(h("div", { class: "small muted" }, "Not tracked yet.")); return; }
    for (const r of runs.slice(0, 6)) {
      runsPanel.append(h("a", { href: `#/run/${r.id}`, class: "status-line", style: { color: "var(--text)" } },
        statusPill(r.status), h("span", { class: "small" }, fmtWhen(r.started || r.created)),
        h("span", { class: "small muted" }, r.brief ? `${plural(r.brief.tracks || 0, "ball")}, ${plural((r.brief.shots || []).length, "shot")}` : "")));
    }
  }

  slider.addEventListener("input", () => { t = parseFloat(slider.value); timeLabel.textContent = fmtClock(t); loadImage(); });
  slider.addEventListener("change", () => check());
  trackBtn.addEventListener("click", async () => {
    trackBtn.disabled = true;
    try {
      await api("PUT", `/api/videos/${vid}/settings`, settings);
      const job = await api("POST", "/api/runs", { video_id: vid });
      location.hash = `#/run/${job.id}`;
    } catch { trackBtn.disabled = false; }
  });

  (async () => {
    let res;
    try { res = await api("GET", `/api/videos/${vid}/settings`); } catch { main.append(h("div", { class: "notice bad" }, "This video is no longer in the library.")); return; }
    video = res.video; settings = res.settings;
    title.textContent = video.name;
    title.title = video.path;
    const dur = video.duration_s || 0;
    slider.max = String(Math.max(0.04, dur - 0.05));
    t = settings.start_s + 0.3 * ((settings.end_s ?? dur) - settings.start_s);
    slider.value = String(t);
    timeLabel.textContent = fmtClock(t);
    trackBtn.disabled = false;
    renderForm();
    renderRuns();
    check();
  })();
}

// ---------------------------------------------------------------------------
// Top-down diagram, timeline and speed chart
// ---------------------------------------------------------------------------

/** Where track ``tr`` is at ``frame``: an index into its arrays, or -1. */
function trackIndexAt(tr, frame, slack = 2) {
  const f = tr.f;
  if (!f.length || frame < f[0] - slack || frame > f[f.length - 1] + slack) return -1;
  let lo = 0, hi = f.length - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (f[mid] <= frame) lo = mid; else hi = mid - 1; }
  if (f[lo] > frame) return frame >= f[0] - slack ? 0 : -1;
  return frame - f[lo] <= slack ? lo : -1;
}

function labelAt(tr, frame) {
  let label = tr.label;
  for (const [f, l] of tr.labels || []) { if (f <= frame) label = l; else break; }
  return label;
}

function drawOverhead(canvas, table, tracks, frame, { fps = 30, selected = null, events = [], trailS = 2.0 } = {}) {
  const L = table.length_in, W = table.width_in, D = table.ball_diameter_in;
  const pad = 2.2 * D;
  const cssW = canvas.clientWidth || 600;
  const scale = cssW / (L + 2 * pad);
  const cssH = (W + 2 * pad) * scale;
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
    canvas.width = Math.round(cssW * dpr); canvas.height = Math.round(cssH * dpr); canvas.style.height = `${cssH}px`;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const X = (x) => (pad + (table.flip_x ? L - x : x)) * scale;
  const Y = (y) => (pad + (table.flip_y ? W - y : y)) * scale;
  // Rails, bed, pockets, head string, foot spot.
  ctx.fillStyle = "#3a2a1e"; roundRect(ctx, 0, 0, cssW, cssH, 10 * scale); ctx.fill();
  ctx.fillStyle = table.cloth_colour || "#2f6c8f"; ctx.fillRect(pad * scale, pad * scale, L * scale, W * scale);
  ctx.strokeStyle = "rgba(255,255,255,.25)"; ctx.lineWidth = 1; ctx.strokeRect(pad * scale, pad * scale, L * scale, W * scale);
  ctx.fillStyle = "#0b0b0b";
  for (const [px, py] of table.pockets || []) { ctx.beginPath(); ctx.arc(X(px), Y(py), 0.95 * D * scale, 0, 2 * Math.PI); ctx.fill(); }
  ctx.strokeStyle = "rgba(255,255,255,.28)"; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(X(0.25 * L), Y(0)); ctx.lineTo(X(0.25 * L), Y(W)); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = "rgba(255,255,255,.5)"; ctx.beginPath(); ctx.arc(X(0.75 * L), Y(W / 2), 2, 0, 2 * Math.PI); ctx.fill();

  const r = Math.max(3, 0.5 * D * scale);
  const trailFrames = Math.round(trailS * fps);
  if (selected) {
    const tr = tracks.find((t) => t.id === selected);
    if (tr) {
      ctx.strokeStyle = "rgba(255,255,255,.35)"; ctx.lineWidth = 1;
      ctx.beginPath(); tr.f.forEach((_, i) => (i ? ctx.lineTo(X(tr.x[i]), Y(tr.y[i])) : ctx.moveTo(X(tr.x[i]), Y(tr.y[i])))); ctx.stroke();
    }
  }
  const live = [];
  for (const tr of tracks) {
    const i = trackIndexAt(tr, frame);
    if (i < 0) continue;
    live.push([tr, i]);
    let j = i;
    while (j > 0 && tr.f[j - 1] >= frame - trailFrames) j--;
    if (i - j >= 1) {
      const colour = drawColour(tr);
      for (let k = j + 1; k <= i; k++) {
        ctx.globalAlpha = 0.15 + 0.75 * (k - j) / (i - j + 1);
        ctx.strokeStyle = colour; ctx.lineWidth = Math.max(1.5, r * 0.45);
        ctx.beginPath(); ctx.moveTo(X(tr.x[k - 1]), Y(tr.y[k - 1])); ctx.lineTo(X(tr.x[k]), Y(tr.y[k])); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
  }
  for (const [tr, i] of live) {
    const x = X(tr.x[i]), y = Y(tr.y[i]), colour = drawColour(tr);
    const label = labelAt(tr, frame);
    const striped = tr.number && tr.number > 8;
    ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI);
    if (tr.o[i]) {
      ctx.fillStyle = striped ? "#f4f1e6" : colour; ctx.fill();
      if (striped) { ctx.save(); ctx.clip(); ctx.fillStyle = colour; ctx.fillRect(x - r, y - r * 0.55, 2 * r, 1.1 * r); ctx.restore(); }
      ctx.strokeStyle = "rgba(0,0,0,.6)"; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.stroke();
    } else {
      ctx.strokeStyle = colour; ctx.lineWidth = 2; ctx.setLineDash([3, 2]); ctx.stroke(); ctx.setLineDash([]);
    }
    if (tr.id === selected) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, r + 3, 0, 2 * Math.PI); ctx.stroke(); }
    if (label && label !== "CUE" && r >= 5) {
      ctx.fillStyle = luminance(colour) > 0.5 || striped ? "#111" : "#fff";
      ctx.font = `700 ${Math.max(8, r * 0.95)}px system-ui`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(label.replace(/^#/, ""), x, y + 0.5);
    }
  }
  for (const e of events) {
    const age = (frame - e.frame) / fps;
    if (age < -0.05 || age > 0.8) continue;
    ctx.globalAlpha = Math.max(0, 1 - age / 0.8);
    ctx.strokeStyle = eventColour(e.type); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(X(e.x_in), Y(e.y_in), r * (1.4 + 2 * Math.max(0, age)), 0, 2 * Math.PI); ctx.stroke();
    ctx.globalAlpha = 1;
  }
}

function roundRect(ctx, x, y, w, hgt, rad) {
  ctx.beginPath(); ctx.moveTo(x + rad, y); ctx.arcTo(x + w, y, x + w, y + hgt, rad); ctx.arcTo(x + w, y + hgt, x, y + hgt, rad);
  ctx.arcTo(x, y + hgt, x, y, rad); ctx.arcTo(x, y, x + w, y, rad); ctx.closePath();
}

function sizeCanvas(canvas, cssH) {
  const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth || 600;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(cssH * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(cssH * dpr); }
  const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [ctx, w, cssH];
}

function drawTimeline(canvas, { first, last, frame, shots, events, hidden }) {
  const [ctx, w, hh] = sizeCanvas(canvas, 58);
  ctx.clearRect(0, 0, w, hh);
  const span = Math.max(1, last - first);
  const X = (f) => 8 + (w - 16) * (f - first) / span;
  ctx.fillStyle = cssVar("--panel-2"); ctx.fillRect(8, 22, w - 16, 14);
  for (const s of shots) {
    const a = X(s.start_frame), b = X(s.end_frame ?? last);
    ctx.fillStyle = cssVar("--accent-soft") || "rgba(62,166,255,.2)"; ctx.fillRect(a, 20, Math.max(2, b - a), 18);
    ctx.fillStyle = cssVar("--accent"); ctx.fillRect(a, 20, 2, 18);
    ctx.fillStyle = cssVar("--text-2"); ctx.font = "11px system-ui"; ctx.textBaseline = "top"; ctx.fillText(`Shot ${s.index}`, a + 5, 4);
  }
  const rows = { ball_struck: 40, collision: 44, cushion: 48, pot: 52 };
  for (const e of events) {
    if (hidden && hidden.has(e.type)) continue;
    ctx.fillStyle = eventColour(e.type);
    const x = X(e.frame), y = rows[e.type] || 46;
    if (e.type === "pot") { ctx.beginPath(); ctx.arc(x, y, 3.5, 0, 2 * Math.PI); ctx.fill(); }
    else ctx.fillRect(x - 1, y - 3, 2, 7);
  }
  const px = X(frame);
  ctx.fillStyle = cssVar("--text"); ctx.fillRect(px - 1, 2, 2, hh - 4);
  ctx.beginPath(); ctx.moveTo(px - 5, 2); ctx.lineTo(px + 5, 2); ctx.lineTo(px, 8); ctx.closePath(); ctx.fill();
}

function drawSpeeds(canvas, { tracks, first, last, frame, selected, fps }) {
  const [ctx, w, hh] = sizeCanvas(canvas, 130);
  ctx.clearRect(0, 0, w, hh);
  const u = UNITS[store.get("units", "km/h")] || UNITS["km/h"];
  let vmax = 10;
  for (const tr of tracks) for (const v of tr.v) if (v > vmax) vmax = v;
  vmax = Math.min(vmax, 400);
  const left = 38, bottom = hh - 18, top = 8;
  const X = (f) => left + (w - left - 8) * (f - first) / Math.max(1, last - first);
  const Y = (v) => bottom - (bottom - top) * Math.min(v, vmax) / vmax;
  ctx.strokeStyle = cssVar("--line"); ctx.lineWidth = 1; ctx.fillStyle = cssVar("--text-3"); ctx.font = "10.5px system-ui";
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let k = 0; k <= 3; k++) {
    const v = vmax * k / 3, y = Y(v);
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(w - 8, y); ctx.stroke();
    ctx.fillText((v * u.f).toFixed(0), left - 5, y);
  }
  ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; ctx.fillText(u.label, 4, hh - 4);
  const line = (tr, colour, width) => {
    ctx.strokeStyle = colour; ctx.lineWidth = width; ctx.beginPath();
    let prev = null;
    tr.f.forEach((f, i) => { const x = X(f), y = Y(tr.v[i]); if (prev !== null && f - prev > 3) ctx.moveTo(x, y); else if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); prev = f; });
    ctx.stroke();
  };
  for (const tr of tracks) if (tr.id !== selected) line(tr, "rgba(160,170,180,.28)", 1);
  const sel = tracks.find((t) => t.id === selected);
  if (sel) line(sel, drawColour(sel) === "#141414" ? "#aaa" : drawColour(sel), 2);
  const px = X(frame);
  ctx.fillStyle = cssVar("--text"); ctx.fillRect(px - 0.5, top, 1, bottom - top);
  ctx.fillStyle = cssVar("--text-3"); ctx.textAlign = "right";
  ctx.fillText(fmtClock((last - first) / fps), w - 8, hh - 4);
}

// ---------------------------------------------------------------------------
// A run: progress while it goes, the results when it is done
// ---------------------------------------------------------------------------

function viewRun(main, life, rid, query) {
  // ``#/run/<id>?t=12.5&tab=events&ball=3`` opens the results at that moment.
  const params = new URLSearchParams(query || "");
  let state = null, shown = null;
  const head = h("div");
  const body = h("div");
  main.append(h("div", { class: "crumbs" }, h("a", { href: "#/runs" }, "Results"), " / one video"), head, body);

  async function poll() {
    try { state = await api("GET", `/api/runs/${rid}`, undefined, { quiet: shown !== null }); } catch (e) {
      if (e.status === 404) { body.innerHTML = ""; body.append(h("div", { class: "notice bad" }, "This run no longer exists.")); life.end(); }
      return;
    }
    if (!life.alive) return;
    const active = ["queued", "calibrating", "running", "finishing"].includes(state.status);
    const mode = active ? "active" : state.status === "failed" ? "failed" : "done";
    if (mode !== shown) { shown = mode; body.innerHTML = ""; head.innerHTML = ""; (mode === "active" ? renderActive : mode === "failed" ? renderFailed : renderDone)(); }
    else if (mode === "active") updateActive();
  }

  // -- running -------------------------------------------------------------
  let act = null;
  function renderActive() {
    const pill = h("span");
    const bar = h("i", { style: { width: "0%" } });
    const line = h("div", { class: "small muted" });
    const img = h("img", { class: "live-img", alt: "The frames as they are tracked", src: `/api/runs/${rid}/preview.mjpg?ts=${Date.now()}` });
    const shot = h("div", { class: "notice info", hidden: true });
    const balls = h("div", { class: "stack", style: { gap: "6px" } });
    const feed = h("div", { class: "feed" });
    head.append(h("div", { class: "page-head" },
      h("div", null, h("h1", null, state.video_name), h("div", { class: "sub" }, "Tracking. The results open on this page when it finishes; you can leave it and come back.")),
      h("div", { class: "spacer" }), pill,
      h("button", { class: "btn danger", onclick: async () => { await api("POST", `/api/runs/${rid}/cancel`); toast("Stopping; what was tracked so far is kept."); } }, icon("stop"), "Stop")));
    body.append(h("div", { class: "results" },
      h("div", { class: "stack" },
        h("div", { class: "panel panel-pad stack", style: { gap: "8px" } }, h("div", { class: "progress" }, bar), line),
        h("div", { class: "player" }, img)),
      h("div", { class: "stack" }, shot,
        h("div", { class: "panel panel-pad" }, h("h3", null, "On the table"), balls),
        h("div", { class: "panel" }, h("div", { class: "panel-pad" }, h("h3", null, "Events so far")), feed))));
    life.onEnd(() => { img.src = ""; });
    act = { pill, bar, line, shot, balls, feed };
    updateActive();
  }
  function updateActive() {
    const s = state;
    act.pill.replaceChildren(statusPill(s.status));
    act.bar.style.width = `${100 * s.progress}%`;
    act.line.textContent = s.status === "running"
      ? `frame ${s.frames_done} of ${s.total_frames} · ${s.fps} fps · ${s.eta_s !== null ? fmtDur(s.eta_s) + " left" : ""}`
      : s.message;
    act.shot.hidden = !s.shot;
    act.shot.textContent = s.shot || "";
    act.balls.replaceChildren(...(s.balls || []).map((b) => h("div", { class: "status-line" },
      ballChip(b.label, drawColour(b)), h("span", { class: "small", style: { flex: 1 } }, b.state === "coasting" ? "unseen, predicted" : ""),
      h("span", { class: "small mono" }, speedText(b.speed)))));
    if (!(s.balls || []).length) act.balls.append(h("div", { class: "small muted" }, "No balls yet."));
    act.feed.replaceChildren(...(s.recent_events || []).slice().reverse().map((e) => h("div", { class: "ev" },
      h("span", { class: "t" }, fmtClock(e.t_s)), h("span", { class: "dot", style: { background: eventColour(e.type) } }), h("span", { class: "what" }, eventText(e)))));
  }

  // -- failed --------------------------------------------------------------
  function renderFailed() {
    head.append(h("div", { class: "page-head" }, h("h1", null, state.video_name), statusPill("failed")));
    const noTable = /calibration failed|cloth/i.test(state.error || "");
    body.append(h("div", { class: "stack", style: { maxWidth: "760px" } },
      h("div", { class: "notice bad" }, h("b", null, "It stopped: "), state.error || "unknown error"),
      noTable ? h("div", { class: "notice info" }, "The table was not found. Open the set-up, check the outline on a frame where the whole table is in view, and place the corners by hand if needed.") : null,
      h("div", { class: "btn-row" },
        h("a", { class: "btn primary", href: `#/video/${state.video_id}` }, "Set up this video"),
        h("button", { class: "btn", onclick: async () => { const job = await api("POST", "/api/runs", { video_id: state.video_id }); location.hash = `#/run/${job.id}`; } }, "Try again"),
        h("button", { class: "btn danger", onclick: async () => { await api("DELETE", `/api/runs/${rid}`); location.hash = "#/runs"; } }, "Delete")),
      state.traceback ? h("details", null, h("summary", { class: "muted small" }, "Details for a bug report"), h("pre", { class: "mono small", style: { whiteSpace: "pre-wrap" } }, state.traceback)) : null));
  }

  // -- done ----------------------------------------------------------------
  async function renderDone() {
    let viewer;
    try {
      const res = await fetch(`/api/runs/${rid}/viewer.json`);
      if (!res.ok) throw new Error("no results file");
      viewer = await res.json();
    } catch (e) {
      body.append(h("div", { class: "notice bad" }, "The results of this run could not be read: ", String(e.message)));
      return;
    }
    if (!life.alive) return;
    results(main, life, head, body, state, viewer, rid, params);
  }

  poll();
  life.every(700, () => { if (shown === "active" || shown === null) poll(); });
}

function results(main, life, head, body, state, viewer, rid, params = new URLSearchParams()) {
  const fps = viewer.fps || 30;
  ballSet = viewer.ball_set || "standard";
  const tracks = viewer.tracks || [];
  // Events name balls by track id; give them the names the balls ended up with.
  const byId = new Map(tracks.map((t) => [t.id, t]));
  const events = (viewer.events || []).slice().sort((a, b) => a.frame - b.frame).map((e) => ({
    ...e, labels: e.labels || (e.track_ids || []).map((id) => (byId.has(id) ? byId.get(id).label : `#${id}`)),
  }));
  const shots = viewer.shots || [];
  const allFrames = tracks.flatMap((t) => [t.first, t.last]);
  const startFrame = viewer.start_frame ?? 0;
  const first = startFrame;
  const last = Math.max(first + 1, ...allFrames, ...(events.length ? [events[events.length - 1].frame] : []),
    viewer.frame_count ? startFrame + (state.summary?.frames_processed ?? 0) - 1 : 0);
  const summary = state.summary || {};
  const counts = summary.events || {};
  let selected = null;
  let frame = first;
  const hidden = new Set(store.get("hiddenEvents", []));
  const tabBodies = {};
  let activeTab = params.get("tab") || store.get("tab", "shots");
  if (params.get("ball")) selected = +params.get("ball");

  // Header
  const tail = state.video_file ? state.video_file : null;
  const download = h("details", { class: "menu" },
    h("summary", { class: "btn sm" }, icon("download"), "Download"),
    h("div", { class: "menu-list panel" },
      tail ? h("a", { href: `/api/runs/${rid}/files/${tail}` }, "The tracked video") : null,
      h("a", { href: `/api/runs/${rid}/files/tracks.csv` }, "Every ball, every frame (CSV)"),
      h("a", { href: `/api/runs/${rid}/files/run.json` }, "Shots and events (JSON)")));
  life.on(document, "click", (e) => { if (!download.contains(e.target)) download.open = false; });
  head.append(h("div", { class: "page-head" },
    h("div", { style: { minWidth: 0 } }, h("h1", { class: "ellipsis" }, state.video_name),
      h("div", { class: "sub", title: state.message || "" }, statusPill(state.status), " ", fmtWhen(state.finished),
        " · press play, or click any event to jump to it")),
    h("div", { class: "spacer" }),
    unitsToggle(() => refreshAll()),
    h("div", { class: "btn-row" }, download,
      state.live ? null : h("a", { class: "btn sm", href: `#/video/${state.video_id}` }, "Change set-up"),
      h("button", { class: "btn sm ghost danger", onclick: async () => { if (confirm("Delete these results and their files?")) { await api("DELETE", `/api/runs/${rid}`); location.hash = "#/runs"; } } }, "Delete"))));

  const seconds = (last - first) / fps;
  body.append(h("div", { class: "statgrid", style: { marginBottom: "14px" } },
    stat(tracks.length, "balls tracked"), stat(shots.length, "shots"), stat(counts.pot || 0, "pots"),
    stat(counts.cushion || 0, "cushions"), stat(counts.collision || 0, "contacts"),
    stat(fmtDur(seconds), "tracked"),
    stat(viewer.clock && viewer.clock.retimed ? `${viewer.clock.source_fps} fps` : `${fps} fps`, viewer.clock && viewer.clock.retimed ? "recovered frame rate" : "frame rate")));

  // Player
  const playable = state.browser_playable && state.video_file;
  let video = null, frameImg = null, playing = false, fallbackTimer = null;
  const player = h("div", { class: "player" });
  if (playable) {
    video = h("video", { src: `/api/runs/${rid}/video`, preload: "auto", playsinline: true, muted: true });
    player.append(video);
  } else if (state.video_file) {
    frameImg = h("img", { class: "frame", alt: "Tracked frame" });
    player.append(frameImg, h("div", { class: "overlay-msg", style: { pointerEvents: "none", background: "none", placeItems: "end start", fontWeight: 500, fontSize: "12px" } }, "Shown frame by frame (install imageio-ffmpeg for smooth playback)"));
  } else player.append(h("div", { class: "overlay-msg", style: { position: "static", minHeight: "200px" } }, "This run has no video."));

  const playBtn = h("button", { class: "btn icon", title: "Play / pause (space)", "aria-label": "Play" }, icon("play"));
  const timeEl = h("span", { class: "time" });
  const rates = h("div", { class: "seg" }, [0.25, 0.5, 1, 2].map((r) => h("button", { class: r === 1 ? "on" : "", onclick: (e) => setRate(r, e.target) }, `${r}×`)));
  const timeline = h("canvas", { class: "timeline", "aria-label": "Timeline: shots and events; click to jump" });
  const overhead = h("canvas", { class: "overhead", "aria-label": "Top-down view of the table" });
  const speeds = h("canvas", { class: "speedchart", "aria-label": "Speed of each ball" });
  const speedTitle = h("span", { class: "small muted" }, "Click a ball to follow its speed");

  body.append(h("div", { class: "results" },
    h("div", { class: "stack" },
      h("div", null, player,
        h("div", { class: "controls" }, playBtn,
          h("button", { class: "btn icon", title: "Back one frame (←)", "aria-label": "Back one frame", onclick: () => step(-1) }, icon("prev")),
          h("button", { class: "btn icon", title: "Forward one frame (→)", "aria-label": "Forward one frame", onclick: () => step(1) }, icon("next")),
          rates, timeEl, h("span", { class: "spacer" }),
          h("button", { class: "btn sm ghost", title: "Copy a link to this moment", onclick: copyMoment }, "Link to this moment")),
        timeline,
        h("div", { class: "chips", style: { marginTop: "8px", alignItems: "center" } }, Object.entries(EVENTS).map(([type, ev]) => {
          const c = h("span", { class: `chip toggle ${hidden.has(type) ? "off" : ""}`, role: "button", tabindex: 0 },
            h("span", { class: "sw", style: { background: ev.colour } }), ev.name, h("b", null, String(events.filter((e) => e.type === type).length)));
          c.addEventListener("click", () => { hidden.has(type) ? hidden.delete(type) : hidden.add(type); c.classList.toggle("off"); store.set("hiddenEvents", [...hidden]); renderEvents(); refreshAll(); });
          return c;
        }), h("span", { class: "spacer", style: { flex: 1 } }),
        h("span", { class: "small muted" }, h("kbd", null, "space"), " play  ", h("kbd", null, "←"), " ", h("kbd", null, "→"), " frame  ", h("kbd", null, "["), " ", h("kbd", null, "]"), " event"))),
      viewer.table ? h("div", { class: "panel panel-pad" }, h("div", { class: "status-line", style: { marginBottom: "8px" } }, h("h3", null, "From above"), h("span", { class: "spacer" })), overhead) : null,
      h("div", { class: "panel panel-pad" }, h("div", { class: "status-line", style: { marginBottom: "6px" } }, h("h3", null, "Speed"), speedTitle), speeds)),
    sidePanel()));

  function stat(v, k) { return h("div", { class: "stat" }, h("div", { class: "v" }, String(v)), h("div", { class: "k" }, k)); }

  // Side panel: shots, events, balls, details
  function sidePanel() {
    const tabs = h("div", { class: "tabs", role: "tablist" });
    const wrap = h("div", { class: "tab-body" });
    const defs = [["shots", "Shots", shots.length], ["events", "Events", events.length], ["balls", "Balls", tracks.length], ["info", "Details", null]];
    const buttons = {};
    for (const [key, label, n] of defs) {
      buttons[key] = h("button", { role: "tab", onclick: () => select(key) }, label, n !== null ? h("span", { class: "n" }, String(n)) : null);
      tabs.append(buttons[key]);
      tabBodies[key] = h("div", { hidden: true });
      wrap.append(tabBodies[key]);
    }
    function select(key) {
      activeTab = key; store.set("tab", key);
      for (const k of Object.keys(buttons)) { buttons[k].classList.toggle("on", k === key); tabBodies[k].hidden = k !== key; }
    }
    renderShots(); renderEvents(); renderBalls(); renderInfo();
    select(buttons[activeTab] ? activeTab : "shots");
    return h("div", { class: "panel" }, tabs, wrap);
  }

  function renderShots() {
    const el = tabBodies.shots; el.innerHTML = "";
    if (!shots.length) { el.append(h("div", { class: "panel-pad muted" }, "No shots were recognised: nothing was struck, or the cue ball was not found.")); return; }
    for (const s of shots) {
      const text = (s.summary || "").replace(/^shot \d+:\s*/, "");
      el.append(h("div", { class: "shot", dataset: { index: s.index }, onclick: () => seek(s.start_frame - Math.round(0.3 * fps)) },
        h("div", { class: "status-line" }, h("b", null, `Shot ${s.index}`), h("span", { class: "t" }, `${fmtClock((s.start_frame - first) / fps)} – ${s.end_frame ? fmtClock((s.end_frame - first) / fps) : "end"}`)),
        h("div", { class: "line" }, text),
        h("div", { class: "parts" },
          s.peak_speed_in_s ? h("span", { class: "chip" }, "top ", h("b", null, speedText(s.peak_speed_in_s))) : null,
          (s.potted || []).map((p) => h("span", { class: "chip" }, "potted ", h("b", null, p))))));
    }
  }
  function renderEvents() {
    const el = tabBodies.events; if (!el) return; el.innerHTML = "";
    const shown = events.filter((e) => !hidden.has(e.type));
    if (!shown.length) { el.append(h("div", { class: "panel-pad muted" }, events.length ? "All event types are hidden." : "No events.")); return; }
    for (const e of shown) {
      el.append(h("div", { class: "ev", dataset: { frame: e.frame }, onclick: () => seek(e.frame - Math.round(0.4 * fps)) },
        h("span", { class: "t" }, fmtClock((e.frame - first) / fps)), h("span", { class: "dot", style: { background: eventColour(e.type) } }),
        h("span", { class: "what" }, eventText(e), e.type === "cushion" && e.rail_distance_in ? h("small", null, ` · ${e.rail_distance_in.toFixed(1)} in off the rail`) : null)));
    }
  }
  function renderBalls() {
    const el = tabBodies.balls; el.innerHTML = "";
    const rows = tracks.slice().sort((a, b) => (b.observed_frames - a.observed_frames));
    const table = h("table", { class: "list" }, h("thead", null, h("tr", null, h("th", null, "Ball"), h("th", { class: "num" }, "Top speed"), h("th", { class: "num" }, "Travelled"), h("th", { class: "num" }, "Seen"))));
    const tb = h("tbody");
    for (const tr of rows) {
      const span = Math.max(1, tr.last - tr.first + 1);
      const row = h("tr", { class: `click ${tr.id === selected ? "sel" : ""}`, onclick: () => { selected = selected === tr.id ? null : tr.id; renderBalls(); refreshAll(); } },
        h("td", null, ballChip(labelAt(tr, last), drawColour(tr)), " ", h("span", { class: "muted small" }, tr.type === "unknown" ? "" : tr.type)),
        h("td", { class: "num" }, speedText(tr.top_speed_in_s)),
        h("td", { class: "num" }, `${(tr.distance_in / 12).toFixed(1)} ft`),
        h("td", { class: "num" }, `${Math.round(100 * tr.observed_frames / span)}%`));
      tb.append(row);
    }
    table.append(tb);
    el.append(table, h("div", { class: "panel-pad small muted" }, "Seen: how much of its time on the table the ball was actually detected, rather than predicted. Click a ball to follow it."));
    const sel = tracks.find((t) => t.id === selected);
    speedTitle.replaceChildren(sel ? h("span", null, ballChip(labelAt(sel, last), drawColour(sel)), " top ", speedText(sel.top_speed_in_s)) : "Click a ball to follow its speed");
  }
  function renderInfo() {
    const el = tabBodies.info; el.innerHTML = "";
    const cal = summary.calibration || {}, tbl = cal.table || {}, vid = summary.video || {}, clock = summary.clock || viewer.clock || {};
    const set = state.settings || {};
    el.append(h("div", { class: "panel-pad stack" },
      h("dl", { class: "kv" },
        h("dt", null, "Video"), h("dd", null, `${vid.width}×${vid.height}, ${vid.fps} fps, ${fmtDur(vid.duration_s)}`),
        h("dt", null, "Frames tracked"), h("dd", null, `${summary.frames_processed ?? "?"}${summary.frames_repeated ? ` (${summary.frames_repeated} repeats replayed)` : ""}`),
        h("dt", null, "Speed"), h("dd", null, `${summary.processing_fps ?? "?"} fps, ${fmtDur(summary.wall_seconds)}`),
        h("dt", null, "Clock"), h("dd", null, clock.retimed ? `screen-recorded: ${clock.container_fps} fps file, ${clock.source_fps} fps scene, ${clock.skipped_source_frames} frames missed` : "the file's own"),
        h("dt", null, "Table"), h("dd", null, `${tbl.length_in} × ${tbl.width_in} in, ${(+tbl.mean_px_per_inch || 0).toFixed(1)} px/in`),
        h("dt", null, "Found in"), h("dd", null, cal.frames_used !== undefined ? `${cal.frames_used} of ${cal.frames_attempted} frames` : "–"),
        h("dt", null, "Tracks made"), h("dd", null, `${summary.tracks_created ?? "?"} (${summary.tracks_revived ?? 0} brought back)`),
        h("dt", null, "Camera cuts"), h("dd", null, `${summary.frames_view_lost ?? 0} frames without the table, ${summary.recalibrations ?? 0} re-calibrations`),
        summary.frames_skipped !== undefined ? [h("dt", null, "Skipped (live)"), h("dd", null, String(summary.frames_skipped))] : null,
        h("dt", null, "Settings"), h("dd", null, `${PRESET_NAMES[set.preset] || set.preset || "?"}, ${set.ball_set || "auto"} set, balls ${set.numbers || "1-15"}${set.corners ? ", corners by hand" : ""}`),
        h("dt", null, "Run"), h("dd", { class: "mono small" }, rid))));
  }

  function copyMoment() {
    const t = ((currentFrame() - first) / fps).toFixed(2);
    const url = `${location.origin}/#/run/${rid}?t=${t}${selected !== null ? `&ball=${selected}` : ""}`;
    (navigator.clipboard ? navigator.clipboard.writeText(url) : Promise.reject()).then(
      () => toast("Link copied", "ok"), () => prompt("Copy this link", url));
  }

  // Playback
  let pendingSeek = null;
  function currentFrame() {
    if (video && video.readyState >= 1 && pendingSeek === null) return first + Math.round((video.currentTime || 0) * fps);
    return frame;
  }
  function seek(f) {
    f = Math.max(first, Math.min(last, f));
    frame = f;
    if (video) {
      const t = Math.max(0, (f - first) / fps + 0.5 / fps);
      if (video.readyState >= 1) video.currentTime = t;
      else pendingSeek = t;
    }
    if (frameImg) showFallback();
    refreshAll();
  }
  function step(n) { pause(); seek(currentFrame() + n); }
  function setRate(r, btn) {
    [...rates.children].forEach((b) => b.classList.toggle("on", b === btn));
    if (video) video.playbackRate = r;
    fallbackRate = r;
  }
  let fallbackRate = 1;
  function showFallback() { if (frameImg) frameImg.src = `/api/runs/${rid}/frame.jpg?f=${frame - first}&w=1280`; }
  function play() {
    playing = true; playBtn.replaceChildren(icon("pause"));
    if (video) { if (currentFrame() >= last - 1) video.currentTime = 0; video.play().catch(() => {}); }
    else if (frameImg) {
      clearInterval(fallbackTimer);
      fallbackTimer = setInterval(() => { if (frame >= last) { pause(); return; } frame = Math.min(last, frame + Math.max(1, Math.round(fps / 8 * fallbackRate))); showFallback(); refreshAll(); }, 125);
    }
  }
  function pause() { playing = false; playBtn.replaceChildren(icon("play")); if (video) video.pause(); clearInterval(fallbackTimer); }
  playBtn.addEventListener("click", () => (playing ? pause() : play()));
  if (video) {
    video.addEventListener("click", () => (playing ? pause() : play()));
    video.addEventListener("pause", () => { playing = false; playBtn.replaceChildren(icon("play")); refreshAll(); });
    video.addEventListener("play", () => { playing = true; playBtn.replaceChildren(icon("pause")); });
    video.addEventListener("seeked", () => refreshAll());
    video.addEventListener("loadeddata", () => refreshAll());
    video.addEventListener("loadedmetadata", () => { if (pendingSeek !== null) { const t = pendingSeek; pendingSeek = null; video.currentTime = t; } });
    video.addEventListener("error", () => { player.append(h("div", { class: "overlay-msg" }, "The browser could not play this video. Download it, or install imageio-ffmpeg and track again.")); });
    const tick = (now, meta) => {
      if (!life.alive) return;
      if (meta && meta.mediaTime !== undefined) frame = first + Math.round(meta.mediaTime * fps);
      else frame = currentFrame();
      refreshAll(true);
      if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(tick);
    };
    if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(tick);
    else { const raf = () => { if (!life.alive) return; if (playing) { frame = currentFrame(); refreshAll(true); } requestAnimationFrame(raf); }; requestAnimationFrame(raf); }
  } else if (frameImg) showFallback();
  life.onEnd(() => { pause(); if (video) { video.removeAttribute("src"); video.load(); } });

  timeline.addEventListener("click", (e) => {
    const r = timeline.getBoundingClientRect();
    const f = first + (last - first) * Math.max(0, Math.min(1, (e.clientX - r.left - 8) / (r.width - 16)));
    seek(Math.round(f));
  });
  speeds.addEventListener("click", (e) => {
    const r = speeds.getBoundingClientRect();
    const f = first + (last - first) * Math.max(0, Math.min(1, (e.clientX - r.left - 38) / (r.width - 46)));
    seek(Math.round(f));
  });
  overhead.addEventListener("click", (e) => {
    if (!viewer.table) return;
    const t = viewer.table, D = t.ball_diameter_in, pad = 2.2 * D, r = overhead.getBoundingClientRect();
    const scale = r.width / (t.length_in + 2 * pad);
    let x = (e.clientX - r.left) / scale - pad, y = (e.clientY - r.top) / scale - pad;
    if (t.flip_x) x = t.length_in - x;
    if (t.flip_y) y = t.width_in - y;
    let best = null, bestD = 2.5 * D;
    for (const tr of tracks) { const i = trackIndexAt(tr, frame); if (i < 0) continue; const d = Math.hypot(tr.x[i] - x, tr.y[i] - y); if (d < bestD) { best = tr; bestD = d; } }
    selected = best ? best.id : null; renderBalls(); refreshAll();
  });

  life.on(document, "keydown", (e) => {
    if (e.target.matches("input, select, textarea")) return;
    if (e.key === " ") { e.preventDefault(); playing ? pause() : play(); }
    else if (e.key === "ArrowRight") { e.preventDefault(); e.shiftKey ? seek(currentFrame() + Math.round(fps)) : step(1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); e.shiftKey ? seek(currentFrame() - Math.round(fps)) : step(-1); }
    else if (e.key === "]" || e.key === "[") {
      const f = currentFrame(), shown = events.filter((ev) => !hidden.has(ev.type));
      const target = e.key === "]" ? shown.find((ev) => ev.frame - Math.round(0.4 * fps) > f + 1) : shown.slice().reverse().find((ev) => ev.frame - Math.round(0.4 * fps) < f - 1);
      if (target) { pause(); seek(target.frame - Math.round(0.4 * fps)); }
    } else if (e.key === "Home") seek(first);
    else if (e.key === "End") seek(last);
  });

  let lastMarked = { shot: null, ev: null };
  function refreshAll(fromPlayback = false) {
    if (!fromPlayback) frame = video ? currentFrame() : frame;
    timeEl.textContent = `${fmtClock((frame - first) / fps)} / ${fmtClock((last - first) / fps)} · frame ${frame}`;
    drawTimeline(timeline, { first, last, frame, shots, events, hidden });
    if (viewer.table) drawOverhead(overhead, viewer.table, tracks, frame, { fps, selected, events: events.filter((e) => !hidden.has(e.type)) });
    drawSpeeds(speeds, { tracks, first, last, frame, selected, fps });
    // The shot and event being played, highlighted in their lists.
    const shot = shots.find((s) => frame >= s.start_frame && frame <= (s.end_frame ?? last));
    if ((shot && shot.index) !== lastMarked.shot) {
      lastMarked.shot = shot && shot.index;
      tabBodies.shots.querySelectorAll(".shot").forEach((el) => el.classList.toggle("on", shot && +el.dataset.index === shot.index));
    }
    let evNow = null;
    for (const e of events) { if (e.frame <= frame + 1 && !hidden.has(e.type)) evNow = e; else if (e.frame > frame + 1) break; }
    const evKey = evNow ? evNow.frame + evNow.type : null;
    if (evKey !== lastMarked.ev) {
      lastMarked.ev = evKey;
      tabBodies.events.querySelectorAll(".ev").forEach((el) => el.classList.remove("on"));
      if (evNow) {
        const el = [...tabBodies.events.querySelectorAll(".ev")].reverse().find((x) => +x.dataset.frame <= frame + 1);
        if (el) { el.classList.add("on"); if (activeTab === "events" && playing) el.scrollIntoView({ block: "nearest" }); }
      }
    }
  }
  life.on(window, "resize", debounce(() => refreshAll(), 100));
  if (selected !== null) renderBalls();
  refreshAll();
  if (params.get("t")) seek(first + Math.round(parseFloat(params.get("t")) * fps));
}

function unitsToggle(onChange) {
  const current = store.get("units", "km/h");
  const seg = h("div", { class: "seg", title: "Speed units" }, Object.keys(UNITS).map((u) =>
    h("button", { class: u === current ? "on" : "", onclick: (e) => { store.set("units", u); [...seg.children].forEach((b) => b.classList.toggle("on", b === e.target)); onChange(); } }, u)));
  return seg;
}

// ---------------------------------------------------------------------------
// Runs
// ---------------------------------------------------------------------------

function viewRuns(main, life) {
  const list = h("div");
  main.append(h("div", { class: "page-head" },
    h("div", null, h("h1", null, "Results"), h("div", { class: "sub" }, "Every video you have tracked, newest first. Click one to watch it back.")),
    h("div", { class: "spacer" }), h("a", { class: "btn", href: "#/library" }, "Track another video")), list);
  let runs = [];
  async function load() {
    try { runs = (await api("GET", "/api/runs", undefined, { quiet: true })).runs; } catch { return; }
    if (!life.alive) return;
    list.innerHTML = "";
    if (!runs.length) { list.append(h("div", { class: "empty" }, h("div", { class: "big" }, "Nothing tracked yet"), h("a", { class: "btn primary", href: "#/library" }, "Go to the library"))); return; }
    const table = h("table", { class: "list panel", style: { overflow: "hidden" } },
      h("thead", null, h("tr", null, h("th", null, ""), h("th", null, "Video"), h("th", null, "When"), h("th", null, "Status"), h("th", null, "Result"), h("th", null, ""))));
    const tb = h("tbody");
    for (const r of runs) {
      const active = ["queued", "calibrating", "running", "finishing"].includes(r.status);
      const b = r.brief || {};
      const shotsN = (b.shots || []).length;
      tb.append(h("tr", { class: "click", onclick: (e) => { if (!e.target.closest("button, a")) location.hash = `#/run/${r.id}`; } },
        h("td", { style: { width: "92px" } }, h("img", { src: `/api/runs/${r.id}/files/thumb.jpg`, alt: "", style: { width: "80px", height: "45px", objectFit: "cover", borderRadius: "5px", display: "block", background: "#000" }, onerror: (e) => { e.target.style.visibility = "hidden"; } })),
        h("td", null, h("div", { class: "ellipsis", style: { maxWidth: "340px", fontWeight: 600 } }, r.video_name), r.error ? h("div", { class: "small", style: { color: "var(--bad)" } }, r.error) : null),
        h("td", { class: "small nowrap" }, fmtWhen(r.started || r.created)),
        h("td", null, statusPill(r.status), active ? h("div", { class: "progress", style: { marginTop: "6px", width: "120px" } }, h("i", { style: { width: `${100 * (r.progress || 0)}%` } })) : null),
        h("td", { class: "small" }, r.status === "done" || r.status === "cancelled" ? `${plural(b.tracks || 0, "ball")} · ${plural(shotsN, "shot")} · ${b.events ? (b.events.pot || 0) + " pots" : ""}` : active ? `${Math.round(100 * (r.progress || 0))}%` : ""),
        h("td", { class: "num" }, active
          ? h("button", { class: "btn sm danger", onclick: async () => { await api("POST", `/api/runs/${r.id}/cancel`); load(); } }, "Stop")
          : h("button", { class: "btn sm ghost", onclick: async () => { if (confirm("Delete these results and their files?")) { await api("DELETE", `/api/runs/${r.id}`); load(); } } }, "Delete"))));
    }
    table.append(tb);
    list.append(table);
  }
  load();
  life.every(1500, () => { if (runs.some((r) => ["queued", "calibrating", "running", "finishing"].includes(r.status))) load(); });
  life.every(10000, load);
}

// ---------------------------------------------------------------------------
// Live
// ---------------------------------------------------------------------------

function viewLive(main, life) {
  const cfg = store.get("live", { kind: "file", index: 0, url: "", video_id: "", speed: 1, start_s: 0, record: true });
  const settings = Object.assign({ preset: "pool-9ft", ball_set: "auto", numbers: "1-15", max_width: 960, corners: null }, store.get("liveSettings", {}));
  const save = () => { store.set("live", cfg); store.set("liveSettings", settings); };
  let state = { status: "idle" }, videos = [], cameras = null, streamKey = null;

  const sourceBox = h("div", { class: "stack", style: { gap: "10px" } });
  const kindSeg = h("div", { class: "seg" }, [["camera", "Camera"], ["url", "Stream"], ["file", "Video"]].map(([k, label]) =>
    h("button", { class: cfg.kind === k ? "on" : "", onclick: (e) => { cfg.kind = k; [...kindSeg.children].forEach((b) => b.classList.toggle("on", b === e.target)); save(); renderSource(); } }, label)));
  const startBtn = h("button", { class: "btn primary" }, icon("play"), "Start");
  const stopBtn = h("button", { class: "btn danger" }, icon("stop"), "Stop");
  const record = h("label", { class: "check" }, h("input", { type: "checkbox", checked: cfg.record, onchange: (e) => { cfg.record = e.target.checked; save(); } }), "Keep the session in Results");
  const cornersBox = h("div", { class: "stack", style: { gap: "8px" } });

  const img = h("img", { class: "live-img", alt: "The live picture, tracked" });
  const idle = h("div", { class: "overlay-msg" }, "Choose a source and press Start.");
  const statusLine = h("div", { class: "status-line" });
  const numbers = h("div", { class: "small muted" });
  const shotBox = h("div", { class: "notice info", hidden: true });
  const balls = h("div", { class: "stack", style: { gap: "6px" } });
  const feed = h("div", { class: "feed" });
  const shotsBox = h("div", { class: "stack", style: { gap: "6px" } });
  const overhead = h("canvas", { class: "overhead" });
  const after = h("div");

  main.append(h("div", { class: "page-head" },
    h("div", null, h("h1", null, "Live"), h("div", { class: "sub" }, "Track a table as it is played: a camera, a network stream, or a video replayed in real time.")),
    h("div", { class: "spacer" }), unitsToggle(() => {})),
    h("div", { class: "live" },
      h("div", { class: "stack" },
        h("div", { class: "panel panel-pad stack" }, h("h2", null, "Source"), kindSeg, sourceBox),
        h("div", { class: "panel panel-pad stack" }, h("h2", null, "Settings"),
          settingsForm(settings, () => save(), { compact: true }), record),
        h("div", { class: "btn-row" }, startBtn, stopBtn),
        h("div", { class: "panel panel-pad stack" }, h("h2", null, "Corners"), cornersBox)),
      h("div", { class: "stack" },
        h("div", { class: "player live" }, img, idle),
        h("div", { class: "panel panel-pad stack", style: { gap: "6px" } }, statusLine, numbers), after,
        h("div", { class: "panel panel-pad" }, h("h3", null, "From above"), h("div", { style: { marginTop: "8px" } }, overhead))),
      h("div", { class: "stack" }, shotBox,
        h("div", { class: "panel panel-pad" }, h("h3", null, "On the table"), balls),
        h("div", { class: "panel" }, h("div", { class: "panel-pad" }, h("h3", null, "Events")), feed),
        h("div", { class: "panel panel-pad" }, h("h3", null, "Shots"), shotsBox))));

  function renderSource() {
    sourceBox.innerHTML = "";
    if (cfg.kind === "camera") {
      const selEl = h("select", { onchange: () => { cfg.index = +selEl.value; save(); } });
      const fill = () => {
        selEl.innerHTML = "";
        if (!cameras) selEl.append(h("option", null, "Looking for cameras…"));
        else if (!cameras.length) selEl.append(h("option", null, "No camera found"));
        else cameras.forEach((c) => selEl.append(h("option", { value: c.index, selected: c.index === cfg.index }, `Camera ${c.index} (${c.width}×${c.height})`)));
      };
      fill();
      const find = async (refresh) => { cameras = null; fill(); try { cameras = (await api("GET", `/api/live/cameras${refresh ? "?refresh=1" : ""}`)).cameras; } catch { cameras = []; } fill(); };
      if (!cameras) find(false);
      sourceBox.append(h("label", { class: "field" }, h("span", null, "Camera"), selEl),
        h("button", { class: "btn sm", onclick: () => find(true) }, icon("refresh"), "Look again"),
        h("div", { class: "small muted" }, "Point it at the whole table, from as high as you can. Keep it still."));
    } else if (cfg.kind === "url") {
      const input = h("input", { type: "text", value: cfg.url, placeholder: "rtsp://…   http://…/video.mjpg", onchange: () => { cfg.url = input.value.trim(); save(); } });
      sourceBox.append(h("label", { class: "field" }, h("span", null, "Stream address"), input,
        h("span", { class: "hint" }, "Anything OpenCV can open: an IP camera (RTSP), an MJPEG stream, a phone camera app's address, or a video URL.")));
    } else {
      const selEl = h("select", { onchange: () => { cfg.video_id = selEl.value; save(); } });
      const fill = () => {
        selEl.innerHTML = "";
        if (!videos.length) selEl.append(h("option", { value: "" }, "No videos in the library"));
        for (const v of videos.filter((x) => !x.error)) selEl.append(h("option", { value: v.id, selected: v.id === cfg.video_id }, v.name));
        if (!cfg.video_id && videos.length) { cfg.video_id = videos[0].id; save(); }
      };
      fill();
      api("GET", "/api/videos", undefined, { quiet: true }).then((d) => { videos = d.videos; if (life.alive) fill(); }).catch(() => {});
      const speed = h("div", { class: "seg" }, [0.5, 1, 2].map((s) => h("button", { class: cfg.speed === s ? "on" : "", onclick: (e) => { cfg.speed = s; [...speed.children].forEach((b) => b.classList.toggle("on", b === e.target)); save(); } }, `${s}×`)));
      const start = h("input", { type: "number", min: 0, step: 0.5, value: cfg.start_s || 0, style: { width: "90px" }, onchange: () => { cfg.start_s = parseFloat(start.value) || 0; save(); } });
      sourceBox.append(h("label", { class: "field" }, h("span", null, "Video"), selEl),
        h("div", { class: "btn-row" }, h("span", { class: "small muted" }, "Speed"), speed, h("span", { class: "small muted" }, "from"), start, h("span", { class: "small muted" }, "s")),
        h("div", { class: "small muted" }, "Played at its own pace, the way a camera would deliver it: frames the tracker has no time for are skipped."));
    }
  }

  function renderCorners() {
    cornersBox.innerHTML = "";
    cornersBox.append(h("div", { class: "small muted" }, settings.corners ? "Using corners you placed." : "Found automatically from the first seconds."));
    const row = h("div", { class: "btn-row" });
    row.append(h("button", { class: "btn sm", disabled: !state.image_size, onclick: placeCorners }, icon("target"), "Place by hand"));
    if (settings.corners) row.append(h("button", { class: "btn sm ghost", onclick: () => { settings.corners = null; save(); renderCorners(); } }, "Automatic"));
    cornersBox.append(row);
  }

  function placeCorners() {
    const stage = tableStage();
    const m = modal("Place the table's corners", [
      h("div", { class: "small muted" }, "Drag the handles onto the four corners of the playing surface, where the cushion noses meet."),
      stage.el,
    ], [
      h("button", { class: "btn", onclick: () => m.close() }, "Cancel"),
      h("button", { class: "btn primary", onclick: () => {
        settings.corners = stage.corners.map(([x, y]) => [Math.round(x), Math.round(y)]);
        save(); renderCorners(); m.close();
        if (["tracking", "calibrating", "no_table", "connecting"].includes(state.status)) start();
      } }, "Use these corners"),
    ]);
    m.box.style.width = "min(1000px, 100%)";
    stage.setImage(`/api/live/snapshot.jpg?ts=${Date.now()}`, state.image_size).then(() => stage.edit(settings.corners || (state.table && state.table.corners_image) || null));
  }

  async function start() {
    const source = cfg.kind === "camera" ? { kind: "camera", index: cfg.index }
      : cfg.kind === "url" ? { kind: "url", url: cfg.url }
      : { kind: "file", video_id: cfg.video_id, speed: cfg.speed, start_s: cfg.start_s };
    if (cfg.kind === "url" && !cfg.url) { toast("Type the stream's address first.", "bad"); return; }
    if (cfg.kind === "file" && !cfg.video_id) { toast("Choose a video first.", "bad"); return; }
    startBtn.disabled = true;
    try { state = await api("POST", "/api/live/start", { source, settings, record: cfg.record }); } catch { startBtn.disabled = false; return; }
    streamKey = null;
    update();
  }
  startBtn.addEventListener("click", start);
  stopBtn.addEventListener("click", async () => { stopBtn.disabled = true; try { state = await api("POST", "/api/live/stop"); } finally { stopBtn.disabled = false; } update(); });

  let adopted = false;
  async function poll() {
    try { state = await api("GET", "/api/live", undefined, { quiet: true }); } catch { return; }
    if (!life.alive) return;
    if (!adopted && state.running && state.source) {
      // Opened while a session runs: show what it is running.
      adopted = true;
      const src = state.source;
      cfg.kind = src.kind;
      if (src.kind === "camera") cfg.index = src.index ?? 0;
      if (src.kind === "url") cfg.url = src.url || "";
      if (src.kind === "file") { cfg.video_id = src.video_id; cfg.speed = src.speed || 1; cfg.start_s = src.start_s || 0; }
      [...kindSeg.children].forEach((b, i) => b.classList.toggle("on", ["camera", "url", "file"][i] === cfg.kind));
      renderSource();
    }
    update();
  }

  function update() {
    const s = state;
    const on = s.running;
    if (s.ball_set) ballSet = s.ball_set;
    startBtn.disabled = false;
    startBtn.replaceChildren(icon("play"), on ? "Restart" : "Start");
    stopBtn.hidden = !on;
    const key = on ? String(s.started) : null;
    if (key !== streamKey) {
      streamKey = key;
      if (on) img.src = `/api/live/stream.mjpg?s=${s.started}`;
      else img.removeAttribute("src");
    }
    img.hidden = !img.getAttribute("src");
    idle.hidden = on || (s.status !== "idle" && !!img.getAttribute("src"));
    idle.textContent = s.status === "idle" ? "Choose a source and press Start." : s.message || "";
    statusLine.replaceChildren(statusPill(s.status), h("span", { class: "small" }, s.message || ""), ...(s.source_name ? [h("span", { class: "small muted" }, `· ${s.source_name}`)] : []));
    numbers.textContent = s.status === "idle" ? "" : `in ${s.fps_in ?? 0} fps · tracked ${s.fps_out ?? 0} fps · ${s.frames_processed ?? 0} frames · ${s.frames_skipped ?? 0} skipped`;
    shotBox.hidden = !s.shot; shotBox.textContent = s.shot || "";
    balls.replaceChildren(...(s.balls || []).map((b) => h("div", null,
      h("div", { class: "status-line" }, ballChip(b.label, drawColour(b)), h("span", { class: "small muted", style: { flex: 1 } }, b.state === "coasting" ? "unseen" : ""), h("span", { class: "small mono" }, speedText(b.speed))),
      h("div", { class: "speedbar" }, h("i", { style: { width: `${Math.min(100, b.speed / 1.5)}%`, background: drawColour(b) === "#141414" ? "#888" : drawColour(b) } })))));
    if (!(s.balls || []).length) balls.append(h("div", { class: "small muted" }, on ? "No balls yet." : "–"));
    feed.replaceChildren(...(s.recent_events || []).slice().reverse().map((e) => h("div", { class: "ev" },
      h("span", { class: "t" }, fmtClock(e.t_s)), h("span", { class: "dot", style: { background: eventColour(e.type) } }), h("span", { class: "what" }, eventText(e)))));
    shotsBox.replaceChildren(...(s.shots || []).slice().reverse().map((x) => h("div", { class: "small" }, x.summary)));
    if (!(s.shots || []).length) shotsBox.append(h("div", { class: "small muted" }, "–"));
    if (s.table && s.balls) {
      const tr = s.balls.map((b) => ({ id: b.id, f: [0], x: [b.x], y: [b.y], v: [b.speed], o: [b.state === "coasting" ? 0 : 1], label: b.label, number: b.number, type: b.type, colour: b.colour }));
      drawOverhead(overhead, s.table, tr, 0, { fps: 30 });
    }
    after.replaceChildren();
    if (!on && s.run_id && ["ended", "stopped"].includes(s.status)) {
      after.append(h("div", { class: "notice ok" }, "The session was saved. ", h("a", { href: `#/run/${s.run_id}` }, "Open the recording")));
    }
    if (s.status === "error") after.append(h("div", { class: "notice bad" }, s.error || s.message));
    if (s.status === "no_table") after.append(h("div", { class: "notice" }, "No table found yet. Make sure the whole playing surface is in the picture, or place its corners by hand."));
    renderCorners();
  }

  renderSource();
  renderCorners();
  poll();
  life.every(400, () => { if (state.running) poll(); });
  life.every(3000, () => { if (!state.running) poll(); });
  life.onEnd(() => { img.src = ""; });
}

// ---------------------------------------------------------------------------
// Help
// ---------------------------------------------------------------------------

function viewHelp(main) {
  main.append(h("div", { class: "page-head" }, h("h1", null, "Help")),
    h("div", { class: "help" },
      h("div", { class: "panel panel-pad" }, h("h2", null, "Getting started"),
        h("ol", null,
          h("li", null, h("b", null, "Add videos"), " in the Library, or drop video files anywhere on the page."),
          h("li", null, "Press ", h("b", null, "Track"), " on a video. It usually needs nothing else: the table, the cloth and the balls are found from the video itself."),
          h("li", null, "Watch it being tracked, or leave it: tracking carries on in the background, one video after another."),
          h("li", null, "Press ", h("b", null, "See results"), ". The tracked video plays beside a top-down view of the table, with every shot, contact, cushion and pot on a timeline you can click. Everything you have tracked is under ", h("b", null, "Results"), " on the left."))),
      h("div", { class: "panel panel-pad" }, h("h2", null, "When the table is not found, or the outline is off"),
        h("p", null, "Press ", h("b", null, "Check the table first"), " (or ", h("b", null, "Change set-up"), ") on the video. The blue outline is the playing surface the tracker uses, on a frame you choose with the slider."),
        h("ul", null,
          h("li", null, "Turn on the ", h("b", null, "cloth mask"), " to see what was taken for cloth. If it is not the table, the lighting or a banner of the same colour is the problem."),
          h("li", null, h("b", null, "Place corners by hand"), " and drag them onto the corners of the playing surface, where the cushion noses meet."),
          h("li", null, "Pick the right ", h("b", null, "table size"), ": positions and speeds are in inches, taken from it."),
          h("li", null, "If the video cuts away a lot (replays, crowd shots), set a start and end time around the play."))),
      h("div", { class: "panel panel-pad" }, h("h2", null, "Footage that tracks well"),
        h("ul", null,
          h("li", null, "The whole playing surface in view, from above or from behind an end rail, like a broadcast."),
          h("li", null, "A camera that stays still. Cuts and slow zooms are handled; constant panning is not."),
          h("li", null, "720p or better. Screen recordings of broadcasts work: their repeated frames are detected and the real frame rate recovered."),
          h("li", null, "Hard cases: a ball the colour of the cloth, balls pressed against the far cushion, and hands resting on the table."))),
      h("div", { class: "panel panel-pad" }, h("h2", null, "Live"),
        h("p", null, "Live tracks a camera (a webcam or USB camera), a network stream (RTSP, MJPEG, a phone camera app), or a library video replayed at its own pace to try things out. The table is found from the first few seconds. When tracking cannot keep up, frames are skipped rather than falling behind. Tick ", h("b", null, "Keep the session in Results"), " to watch it back later.")),
      h("div", { class: "panel panel-pad" }, h("h2", null, "Keyboard, on the results page"),
        h("table", null,
          h("tr", null, h("td", null, h("kbd", null, "space")), h("td", null, "play / pause")),
          h("tr", null, h("td", null, h("kbd", null, "←"), " ", h("kbd", null, "→")), h("td", null, "one frame back / forward (hold shift for a second)")),
          h("tr", null, h("td", null, h("kbd", null, "["), " ", h("kbd", null, "]")), h("td", null, "previous / next event")),
          h("tr", null, h("td", null, h("kbd", null, "Home"), " ", h("kbd", null, "End")), h("td", null, "start / end")))),
      h("div", { class: "panel panel-pad" }, h("h2", null, "The same from the command line"),
        h("pre", { class: "mono small", style: { whiteSpace: "pre-wrap", margin: 0 } },
          "billiards clip.mp4 -o tracked.mp4 --csv tracks.csv --json run.json\n" +
          "billiards clip.mp4 --table-corners x1,y1,x2,y2,x3,y3,x4,y4\n" +
          "billiards calibrate clip.mp4 --save-preview preview.png\n" +
          "billiards app --folder D:\\footage --port 8765")),
      h("div", { class: "panel panel-pad" }, h("h2", null, "Videos will not play in the browser?"),
        h("p", null, "The results video is written as H.264 when ", h("code", null, "imageio-ffmpeg"), " is installed (", h("code", null, "pip install imageio-ffmpeg"), "), or with whatever this computer's OpenCV can write that a browser plays. Without either, results are shown frame by frame."))));
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

(function init() {
  let depth = 0;
  const overlay = $("#drop-overlay");
  const hasFiles = (e) => e.dataTransfer && [...e.dataTransfer.types].includes("Files");
  window.addEventListener("dragenter", (e) => { if (!hasFiles(e)) return; depth++; overlay.hidden = false; e.preventDefault(); });
  window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) overlay.hidden = true; });
  window.addEventListener("drop", async (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault(); depth = 0; overlay.hidden = true;
    const added = await uploadFiles(e.dataTransfer.files);
    if (added.length) {
      if (window.onLibraryChanged) window.onLibraryChanged();
      else if (added.length === 1) location.hash = `#/video/${added[0].id}`;
      else location.hash = "#/library";
    }
  });
  window.addEventListener("hashchange", navigate);
  navigate();
  pollStatus();
  setInterval(pollStatus, 3000);
})();
