const log = document.getElementById("log");
const form = document.getElementById("form");
const msg = document.getElementById("msg");
const send = document.getElementById("send");
const statusEl = document.getElementById("status");
const statusText = document.getElementById("status-text");

// On touch keyboards Enter inserts a newline and the send button submits.
const touch = matchMedia("(hover: none)").matches;
let busy = false;

const scrollDown = () => log.scrollTo({ top: log.scrollHeight });
// False once the reader has scrolled up to read; a growing reply then stops
// dragging the view down with it.
const atBottom = () => log.scrollHeight - log.scrollTop - log.clientHeight < 80;
const clock = () => new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

// ---- markdown ----
// A small renderer for what replies actually use. Inline text is escaped
// before any rule runs and block rules escape what they emit, so the only
// tags that reach the DOM are the ones built here.
const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const esc = (s) => s.replace(/[&<>"']/g, (c) => ESC[c]);
// URLs arrive already escaped, which is what an attribute value wants.
const safeUrl = (u, img) =>
  /^(?:https?:\/\/|mailto:|\/|\.\.?\/|#)/i.test(u) || (img && /^data:image\//i.test(u)) ? u : null;
const IMG_URL = /\.(?:gif|png|jpe?g|webp|avif|svg|bmp)(?:[?#]\S*)?$/i;
const EXT = 'target="_blank" rel="noopener noreferrer"';

const FENCE = /^ {0,3}(`{3,}|~{3,})\s*([^\s`]*)/;
const HEADING = /^ {0,3}(#{1,6})\s+(.*?)(?:\s+#+)?\s*$/;
const HR = /^ {0,3}([-*_])(?:\s*\1){2,}\s*$/;
const QUOTE = /^ {0,3}> ?/;
const LI = /^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/;
const TABLE_SEP = /^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?\s*$/;
const isTable = (l, next = "") => l.includes("|") && next.includes("|") && TABLE_SEP.test(next);
const MATH_OPEN = /^ {0,3}(\$\$|\\\[)/;
const isStart = (l, next) =>
  FENCE.test(l) || MATH_OPEN.test(l) || HEADING.test(l) || HR.test(l) || QUOTE.test(l) || LI.test(l) || isTable(l, next);

function emphasis(s) {
  return s
    .replace(/\*\*\*(?=\S)([\s\S]*?\S)\*\*\*/g, "<strong><em>$1</em></strong>")
    .replace(/\*\*(?=\S)([\s\S]*?\S)\*\*/g, "<strong>$1</strong>")
    .replace(/(?<![\w\\])__(?=\S)([\s\S]*?\S)__(?!\w)/g, "<strong>$1</strong>")
    .replace(/\*(?=\S)([^*\n]*?\S)\*/g, "<em>$1</em>")
    .replace(/(?<![\w\\])_(?=\S)([^_\n]*?\S)_(?!\w)/g, "<em>$1</em>")
    .replace(/~~(?=\S)([\s\S]*?\S)~~/g, "<del>$1</del>");
}

// ---- math ----
// The agent writes LaTeX: $...$ and \(...\) inline, $$...$$ and \[...\] for
// display. The renderer only emits an escaped placeholder carrying the source;
// renderMath() hands those to KaTeX once the HTML is in the DOM, so a missing
// KaTeX degrades to readable TeX instead of broken markup.
const unesc = (s) => s.replace(/&(?:amp|lt|gt|quot|#39);/g, (e) =>
  ({ "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'" })[e]);
const mathHtml = (tex, display, tag = "span") => {
  const t = tex.trim(), open = display ? "$$" : "$";
  return `<${tag} class="math${display ? " display" : ""}" data-tex="${esc(t)}">${esc(open + t + open)}</${tag}>`;
};

// A display block spanning whole lines: "$$ x $$" on one line, or an opening
// "$$" line, the formula, and a closing "$$" line. Returns null if unclosed.
function displayMath(lines, i) {
  const m = lines[i].match(MATH_OPEN);
  if (!m) return null;
  const close = m[1] === "$$" ? "$$" : "\\]";
  let body = lines[i].slice(m[0].length);
  for (let j = i; ; ) {
    const at = body.indexOf(close);
    if (at >= 0) return body.slice(at + close.length).trim() ? null : { tex: body.slice(0, at), next: j + 1 };
    if (++j >= lines.length || !lines[j].trim()) return null;
    body += "\n" + lines[j];
  }
}

// Finished fragments are parked in `stash` behind \0n\0 markers, so later
// rules can't reach into a URL or a code span and mangle it.
function inline(src) {
  const stash = [];
  const put = (html) => `\u0000${stash.push(html) - 1}\u0000`;
  const restore = (t) => t.replace(/\u0000(\d+)\u0000/g, (_, i) => restore(stash[i]));
  let s = esc(src.replace(/\u0000/g, ""));

  s = s.replace(/(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/g,
    (_, _t, code) => put(`<code>${code.replace(/^ ([\s\S]*) $/, "$1")}</code>`));
  // Before the escape rule below, which would eat the backslashes in TeX.
  s = s.replace(/\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]/g, (_, a, b) => put(mathHtml(unesc(a ?? b), true)));
  s = s.replace(/\\\(([\s\S]+?)\\\)/g, (_, t) => put(mathHtml(unesc(t), false)));
  // Pandoc's rule keeps "$5 and $10" as prose: no space inside the delimiters,
  // and the closing $ is not followed by a digit.
  s = s.replace(/(?<![\\$\w])\$(?=[^\s$])((?:\\[\s\S]|[^$\\\n])+?)(?<=\S)\$(?![\d$])/g,
    (_, t) => put(mathHtml(unesc(t), false)));
  s = s.replace(/\\([\\`*_{}\[\]()#+\-.!~|])/g, (_, c) => put(c));
  s = s.replace(/!\[([^\]]*)\]\(\s*([^\s)]+)(?:\s+&quot;([\s\S]*?)&quot;)?\s*\)/g, (m, alt, url, title) => {
    const u = safeUrl(url, true);
    return u ? put(`<img src="${u}" alt="${alt}"${title ? ` title="${title}"` : ""} loading="lazy">`) : m;
  });
  s = s.replace(/\[([^\]]+)\]\(\s*([^\s)]+)(?:\s+&quot;[\s\S]*?&quot;)?\s*\)/g, (m, text, url) => {
    const u = safeUrl(url);
    return u ? put(`<a href="${u}" ${EXT}>${emphasis(text)}</a>`) : m;
  });
  // Bare URLs, stopping before trailing punctuation. One that points at an
  // image (a GIF included) is shown inline rather than as a link.
  s = s.replace(/(?:&lt;)?\b(https?:\/\/[^\s\u0000]+?)(?:&gt;)?(?=(?:[.,;:!?)\]]|&quot;|&#39;|&gt;)*(?:\s|$|\u0000))/g,
    (_, url) => put(IMG_URL.test(url)
      ? `<a href="${url}" ${EXT}><img src="${url}" alt="" loading="lazy"></a>`
      : `<a href="${url}" ${EXT}>${url}</a>`));

  return restore(emphasis(s).replace(/\n/g, "<br>"));
}

const cells = (row) => row.trim().replace(/^\|/, "").replace(/(?<!\\)\|$/, "")
  .split(/(?<!\\)\|/).map((c) => c.trim().replace(/\\\|/g, "|"));

function dedent(lines) {
  const pad = Math.min(...lines.filter((l) => l.trim()).map((l) => l.match(/^\s*/)[0].length));
  return lines.map((l) => l.slice(pad));
}

// Index of the first line after the list that starts at lines[i].
function listEnd(lines, i) {
  const first = lines[i].match(LI), base = first[1].length, ordered = /\d/.test(first[2]);
  for (i++; i < lines.length; i++) {
    const l = lines[i];
    if (!l.trim()) {
      const next = lines[i + 1] || "";
      if (LI.test(next) || /^\s+\S/.test(next)) continue;
      return i;
    }
    if (HR.test(l)) return i;
    const m = l.match(LI);
    if (m) {
      if (m[1].length <= base + 1 && /\d/.test(m[2]) !== ordered) return i;
      continue;
    }
    if (/^\s+\S/.test(l)) continue;
    if (isStart(l, lines[i + 1]) || !lines[i - 1].trim()) return i;
    // otherwise a lazy continuation of the item above
  }
  return i;
}

function list(lines) {
  const first = lines[0].match(LI), base = first[1].length;
  const items = [];
  for (const l of lines) {
    const m = l.match(LI);
    if (m && m[1].length <= base + 1) { items.push({ text: [m[3]], rest: [] }); continue; }
    const item = items[items.length - 1];
    if (!item.rest.length && l.trim() && !isStart(l.trim())) item.text.push(l.trim());
    else item.rest.push(l);
  }
  const body = items.map(({ text, rest }) => {
    let head = text.join("\n"), cls = "";
    const task = head.match(/^\[([ xX])\]\s+/);
    if (task) {
      head = head.slice(task[0].length);
      cls = ' class="task"';
    }
    const box = task ? `<input type="checkbox" disabled${task[1] === " " ? "" : " checked"}>` : "";
    const nested = rest.some((l) => l.trim()) ? blocks(dedent(rest)) : "";
    return `<li${cls}>${box}${inline(head)}${nested}</li>`;
  }).join("");
  if (!/\d/.test(first[2])) return `<ul>${body}</ul>`;
  const start = parseInt(first[2], 10);
  return `<ol${start !== 1 ? ` start="${start}"` : ""}>${body}</ol>`;
}

function table(head, sep, rows) {
  const align = cells(sep).map((c) =>
    /^:-+:$/.test(c) ? "center" : /-:$/.test(c) ? "right" : /^:/.test(c) ? "left" : "");
  const row = (tag, r) => "<tr>" + head.map((_, i) =>
    `<${tag}${align[i] ? ` style="text-align:${align[i]}"` : ""}>${inline(r[i] || "")}</${tag}>`).join("") + "</tr>";
  return `<div class="table-wrap"><table><thead>${row("th", head)}</thead>` +
    `<tbody>${rows.map((r) => row("td", r)).join("")}</tbody></table></div>`;
}

function blocks(lines) {
  let out = "", i = 0, m;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }

    let dm;
    if ((dm = displayMath(lines, i))) {
      out += mathHtml(dm.tex, true, "div");
      i = dm.next;
    } else if ((m = line.match(FENCE))) {
      const close = new RegExp(`^ {0,3}${m[1][0]}{${m[1].length},}\\s*$`), code = [];
      for (i++; i < lines.length && !close.test(lines[i]); i++) code.push(lines[i]);
      i++;
      out += `<div class="code"><div class="code-head"><span>${esc(m[2] || "code")}</span>` +
        `<button type="button" class="copy">Copy</button></div><pre><code>${esc(code.join("\n"))}</code></pre></div>`;
    } else if ((m = line.match(HEADING))) {
      const n = m[1].length;
      out += `<h${n}>${inline(m[2])}</h${n}>`;
      i++;
    } else if (HR.test(line)) {
      out += "<hr>";
      i++;
    } else if (QUOTE.test(line)) {
      const quoted = [];
      while (i < lines.length && QUOTE.test(lines[i])) quoted.push(lines[i++].replace(QUOTE, ""));
      out += `<blockquote>${blocks(quoted)}</blockquote>`;
    } else if (LI.test(line)) {
      const end = listEnd(lines, i);
      out += list(lines.slice(i, end));
      i = end;
    } else if (isTable(line, lines[i + 1])) {
      const sep = lines[i + 1], rows = [];
      for (i += 2; i < lines.length && lines[i].includes("|") && lines[i].trim(); i++) rows.push(cells(lines[i]));
      out += table(cells(line), sep, rows);
    } else {
      const para = [];
      while (i < lines.length && lines[i].trim() && (!para.length || !isStart(lines[i], lines[i + 1]))) {
        para.push(lines[i++].trim());
      }
      out += `<p>${inline(para.join("\n"))}</p>`;
    }
  }
  return out;
}

const markdown = (text) => blocks(text.replace(/\r\n?/g, "\n").split("\n"));

function renderMath(root) {
  if (!window.katex) return;
  for (const el of root.querySelectorAll(".math")) {
    try {
      katex.render(el.dataset.tex, el, { displayMode: el.classList.contains("display"), throwOnError: false });
    } catch {}
  }
}

// Replies are rendered; the user's own messages and errors stay verbatim.
function setBody(bubble, text, rich) {
  bubble.classList.toggle("md", rich);
  if (rich) {
    bubble.innerHTML = markdown(text);
    renderMath(bubble);
  }
  else bubble.textContent = text;
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Plain http on a LAN address isn't a secure context, so there is no
    // async clipboard there; fall back to the old selection trick.
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.cssText = "position:fixed;opacity:0";
    document.body.append(ta);
    ta.select();
    let ok = false;
    try { ok = document.execCommand("copy"); } catch {}
    ta.remove();
    return ok;
  }
}

log.addEventListener("click", async (e) => {
  const btn = e.target.closest(".md .copy");
  if (!btn) return;
  const ok = await copyText(btn.closest(".code").querySelector("code").textContent);
  btn.textContent = ok ? "Copied" : "Copy failed";
  setTimeout(() => { btn.textContent = "Copy"; }, 1500);
});

// Images finish loading after the reply is laid out; follow them down only if
// the reader was already at the bottom.
log.addEventListener("load", (e) => {
  const img = e.target;
  if (img.tagName !== "IMG" || !img.closest(".md")) return;
  if (log.scrollHeight - log.scrollTop - log.clientHeight < img.offsetHeight + 120) scrollDown();
}, true);

// A broken image falls back to a plain link rather than an empty frame.
log.addEventListener("error", (e) => {
  const img = e.target;
  if (img.tagName !== "IMG" || !img.closest(".md")) return;
  if (img.parentElement.tagName === "A") {
    img.replaceWith(img.getAttribute("src"));
    return;
  }
  const a = document.createElement("a");
  a.href = img.src;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  a.textContent = img.alt || img.getAttribute("src");
  img.replaceWith(a);
}, true);

function syncSend() {
  send.disabled = busy || !(msg.value.trim() || attached.length);
}

/* ---------------- file attachments ---------------- */
const attached = []; // File objects waiting to be sent with the next message
const filesEl = document.getElementById("files");
const fileInput = document.getElementById("file-input");

function renderFiles() {
  filesEl.textContent = "";
  attached.forEach((f, i) => {
    const chip = document.createElement("div");
    chip.className = "file";
    const name = document.createElement("span");
    name.textContent = f.name;
    const x = document.createElement("button");
    x.type = "button";
    x.textContent = "×";
    x.setAttribute("aria-label", `Remove ${f.name}`);
    x.onclick = () => { attached.splice(i, 1); renderFiles(); };
    chip.append(name, x);
    filesEl.append(chip);
  });
  syncSend();
}

function addFiles(list) {
  attached.push(...list);
  renderFiles();
}

document.getElementById("attach").onclick = () => fileInput.click();
fileInput.onchange = () => { addFiles(fileInput.files); fileInput.value = ""; };

let dragDepth = 0;
addEventListener("dragenter", (e) => {
  if (!e.dataTransfer?.types.includes("Files")) return;
  dragDepth++;
  document.body.classList.add("dragging");
});
addEventListener("dragleave", () => {
  if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove("dragging"); }
});
addEventListener("dragover", (e) => e.preventDefault());
addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  document.body.classList.remove("dragging");
  if (e.dataTransfer?.files.length) addFiles(e.dataTransfer.files);
});

// Uploads each file, returning the server-side names to send with the message.
async function uploadAll(files) {
  const names = [];
  for (const f of files) {
    const r = await fetch(`/upload?filename=${encodeURIComponent(f.name)}`, { method: "POST", body: f });
    if (!r.ok) throw new Error(`${f.name}: ${(await r.json().catch(() => ({}))).detail || r.status}`);
    names.push((await r.json()).name);
  }
  return names;
}

function autoGrow() {
  msg.style.height = "auto";
  msg.style.height = Math.min(msg.scrollHeight, 160) + "px";
}


let turnCount = 0;
const tMsgs = document.getElementById("t-msgs");
const tLast = document.getElementById("t-last");
const tVoice = document.getElementById("t-voice");

// A reply that has no text yet: the typing dots, plus what is running when
// the model is busy with tools.
function showPending(bubble, activity = "") {
  bubble.classList.remove("md");
  bubble.innerHTML = '<span class="typing" aria-label="Amadeus is typing"><i></i><i></i><i></i></span>';
  if (activity) {
    const span = document.createElement("span");
    span.className = "activity";
    span.textContent = activity;
    bubble.append(span);
  }
}

// Returns the bubble so a pending turn can be filled in once the reply lands.
function addTurn(role, text, time = clock()) {
  document.body.classList.add("started");
  const turn = document.createElement("div");
  turn.className = `turn ${role}`;
  turn.innerHTML = '<div class="tag"><b></b><time></time></div><div class="bubble"></div>';
  turn.querySelector("b").textContent = role === "bot" ? "AMADEUS" : "YOU";
  turn.querySelector("time").textContent = time;
  const bubble = turn.querySelector(".bubble");
  if (text === null) {
    showPending(bubble);
  } else {
    setBody(bubble, text, role === "bot");
    turn.dataset.text = text;
  }
  log.append(turn);
  syncReplay();
  tMsgs.textContent = ++turnCount;
  scrollDown();
  return bubble;
}

// Only the newest few replies can be replayed; the server keeps just that many clips.
const REPLAY_LIMIT = 5;

function syncReplay() {
  const replies = [...log.querySelectorAll(".turn.bot[data-text]:not(.error)")].slice(-REPLAY_LIMIT);
  for (const turn of log.querySelectorAll(".turn.bot")) {
    const btn = turn.querySelector(".replay");
    if (replies.includes(turn)) {
      if (btn) continue;
      const b = document.createElement("button");
      b.type = "button";
      b.className = "replay";
      b.title = "Replay voice";
      b.setAttribute("aria-label", "Replay voice");
      b.textContent = "▶ REPLAY";
      turn.querySelector(".tag").append(b);
    } else btn?.remove();
  }
}

log.addEventListener("click", async (e) => {
  const btn = e.target.closest(".replay");
  if (!btn || btn.disabled) return;
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = "… LOADING";
  try {
    const r = await fetch("/voice/replay", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: btn.closest(".turn").dataset.text }),
    });
    const { audio } = await r.json();
    if (!audio) throw new Error("no audio");
    voicePlayer.src = `${audio}?t=${Date.now()}`;
    await voicePlayer.play();
  } catch {
    btn.textContent = "✕ FAILED";
    await new Promise((res) => setTimeout(res, 1500));
  } finally {
    btn.textContent = label;
    btn.disabled = false;
  }
});

function fill(bubble, text, isError) {
  setBody(bubble, text, !isError);
  const turn = bubble.closest(".turn");
  if (!isError) turn.dataset.text = text;
  if (isError) {
    turn.classList.add("error");
    turn.querySelector("b").textContent = "ERROR";
  }
  syncReplay();
  turn.querySelector("time").textContent = clock();
}

// The lines of a newline-delimited JSON response, parsed as they arrive.
async function* ndjson(response) {
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffered = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffered += value;
      const lines = buffered.split("\n");
      buffered = lines.pop();
      for (const line of lines) if (line.trim()) yield JSON.parse(line);
    }
    if (buffered.trim()) yield JSON.parse(buffered);
  } finally {
    reader.cancel().catch(() => {});
  }
}

// Shows the reply in `bubble` while /chat/stream writes it, and resolves with
// the finished one: the same {response, audio} /chat returns, which the
// caller renders over the draft. The draft is repainted at most once a frame,
// since every paint re-renders its markdown from the top.
async function streamReply(response, bubble) {
  let draft = "", activity = "", frame = 0;
  const paint = () => {
    frame = 0;
    const following = atBottom();
    if (draft) setBody(bubble, draft, true);
    else showPending(bubble, activity);
    if (following) scrollDown();
  };
  try {
    for await (const event of ndjson(response)) {
      if (event.type === "done") return event;
      if (event.type === "error") throw new Error(event.message);
      if (event.type === "delta") draft += event.text;
      else if (event.type === "replace") draft = event.text;
      else if (event.type === "tools") {
        // That turn's text was not the reply; the model is working instead.
        draft = "";
        activity = `running ${[...new Set(event.names)].join(", ")}`;
      }
      if (draft) activity = "";
      frame ||= requestAnimationFrame(paint);
    }
  } finally {
    cancelAnimationFrame(frame);
  }
  throw new Error("the connection closed before the reply was finished");
}

function setStatus(online) {
  statusEl.className = "status " + (online ? "online" : "offline");
  statusText.textContent = online ? "Online" : "Offline";
}

async function ping() {
  try {
    setStatus((await fetch("/health", { cache: "no-store" })).ok);
  } catch {
    setStatus(false);
  }
}

// Replays today's conversation from the server's daily chat log.
async function loadHistory() {
  try {
    const r = await fetch("/history", { cache: "no-store" });
    if (!r.ok) return;
    for (const t of (await r.json()).turns) addTurn(t.role, t.text, t.time);
  } catch {
    // Offline: start with an empty log; ping() shows the status.
  }
}

form.onsubmit = async (e) => {
  e.preventDefault();
  const typed = msg.value.trim();
  if ((!typed && !attached.length) || busy) return;
  const text = typed || "(see attached files)";
  const toUpload = attached.splice(0);

  msg.value = "";
  autoGrow();
  renderFiles();
  busy = true;
  const t0 = performance.now();
  syncSend();
  document.body.classList.add("thinking");
  addTurn("user", text + toUpload.map((f) => `\n📎 ${f.name}`).join(""));
  const pending = addTurn("bot", null);

  try {
    const files = await uploadAll(toUpload);
    const r = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, files, speak: voiceEnabled.checked }),
    });
    if (!r.ok) throw new Error(`server returned ${r.status}`);
    const result = await streamReply(r, pending);
    const following = atBottom();
    fill(pending, result.response);
    if (following) scrollDown();
    tLast.textContent = Math.round(performance.now() - t0) + " ms";
    playVoice(result.audio);
    // Fire and forget: the face follows the reply, it never holds it up.
    // Skipped when the face is switched off, so no beam call is spent on it.
    if (!document.body.classList.contains("no-kaomoji")) {
      fetch("/emotion", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: result.response }),
      }).then((e) => e.json()).then((d) => window.Kaomoji?.set(d.emotion)).catch(() => {});
    }
    setStatus(true);
  } catch (err) {
    fill(pending, String(err), true);
    scrollDown();
    ping();
  } finally {
    busy = false;
    syncSend();
    document.body.classList.remove("thinking");
    if (!touch && !document.body.classList.contains("settings-open")) msg.focus();
  }
};

msg.addEventListener("input", () => { autoGrow(); syncSend(); });
msg.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing && !touch) {
    e.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.onclick = () => {
    msg.value = chip.textContent;
    syncSend();
    form.requestSubmit();
  };
});

/* ---------------- settings drawer ---------------- */
const panel = document.getElementById("panel");
const scrim = document.getElementById("scrim");
const settingsBtn = document.getElementById("settings-btn");
const panelClose = document.getElementById("panel-close");
const iterRange = document.getElementById("iter-range");
const iterNum = document.getElementById("iter-num");
const saveIter = document.getElementById("save-iter");
const iterNote = document.getElementById("iter-note");
const hkPrompt = document.getElementById("hk-prompt");
const saveHk = document.getElementById("save-hk");
const resetHk = document.getElementById("reset-hk");
const hkNote = document.getElementById("hk-note");
const voiceEnabled = document.getElementById("voice-enabled");
const voicePlayer = document.getElementById("voice-player");
const VOICE_ENABLED_KEY = "amadeus:voice-enabled";
const modeInputs = [...document.querySelectorAll('input[name="chat-mode"]')];
const modeNote = document.getElementById("mode-note");
const tMode = document.getElementById("t-mode");

// Last state the server confirmed. Every field compares against this to decide
// whether it is dirty, so Save only lights up when there is a real change.
let settings = null;

function note(el, text, kind) {
  el.textContent = text;
  el.className = "note show " + (kind || "");
  if (kind === "ok") {
    clearTimeout(el._t);
    el._t = setTimeout(() => el.classList.remove("show"), 2600);
  }
}

function iterValue() {
  const n = parseInt(iterNum.value, 10);
  if (!Number.isFinite(n)) return null;
  const lo = parseInt(iterNum.min, 10), hi = parseInt(iterNum.max, 10);
  return n < lo || n > hi ? null : n;
}

function syncPanel() {
  const n = iterValue();
  saveIter.disabled = n === null || !settings || n === settings.max_iterations;
  const prompt = hkPrompt.value.trim();
  saveHk.disabled = !prompt || !settings || prompt === settings.housekeeping_prompt;
  // Nothing to reset to when the prompt on the server is already the default.
  resetHk.disabled = !settings || settings.housekeeping_prompt_is_default;
}

// Fields the user is midway through editing keep their text; everything else
// takes the server's value.
function applySettings(data) {
  const iterDirty = settings && iterValue() !== settings.max_iterations;
  const hkDirty = settings && hkPrompt.value.trim() !== settings.housekeeping_prompt;
  settings = data;

  iterNum.min = iterRange.min = data.min_iterations;
  iterNum.max = iterRange.max = data.max_iterations_limit;
  if (!iterDirty) iterNum.value = iterRange.value = data.max_iterations;
  if (!hkDirty) hkPrompt.value = data.housekeeping_prompt;
  syncPanel();
  syncMode();
}

// The mode is a server setting, so the radios always show the server's
// choice — it is the same for every device talking to this Amadeus.
function syncMode() {
  if (!settings) return;
  for (const input of modeInputs) input.checked = input.value === settings.chat_mode;
  for (const el of panel.querySelectorAll("[data-model]")) {
    el.textContent = settings.chat_modes[el.dataset.model]?.model ?? "?";
  }
  tMode.textContent = settings.chat_mode.toUpperCase();
}

async function settingsRequest(path, body) {
  const r = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    cache: "no-store",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`server returned ${r.status}`);
  applySettings(await r.json());
}

function openSettings() {
  panel.hidden = scrim.hidden = false;
  settingsBtn.setAttribute("aria-expanded", "true");
  // Next frame, so the browser paints the off-screen position first and the
  // drawer actually slides in rather than appearing.
  requestAnimationFrame(() => document.body.classList.add("settings-open"));
  sdGifMount();
  settingsRequest("/settings").catch((err) => note(iterNote, String(err), "bad"));
  panelClose.focus();
}

function closeSettings() {
  sdDisarm();
  document.body.classList.remove("settings-open");
  settingsBtn.setAttribute("aria-expanded", "false");
  // Hide only once it has slid back out, so it isn't yanked off screen.
  setTimeout(() => {
    if (document.body.classList.contains("settings-open")) return;
    panel.hidden = scrim.hidden = true;
    sdGifUnmount();
  }, 350);
  settingsBtn.focus();
}

settingsBtn.onclick = openSettings;
panelClose.onclick = closeSettings;
scrim.onclick = closeSettings;
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && document.body.classList.contains("settings-open")) closeSettings();
});

iterRange.addEventListener("input", () => { iterNum.value = iterRange.value; syncPanel(); });
iterNum.addEventListener("input", () => {
  const n = iterValue();
  if (n !== null) iterRange.value = n;
  syncPanel();
});
hkPrompt.addEventListener("input", syncPanel);

async function withButton(button, noteEl, work, okText) {
  button.disabled = true;
  try {
    await work();
    note(noteEl, okText, "ok");
  } catch (err) {
    note(noteEl, String(err), "bad");
  } finally {
    syncPanel();
  }
}

saveIter.onclick = () => {
  const n = iterValue();
  if (n === null) return note(iterNote, "Pick a whole number in range.", "bad");
  withButton(saveIter, iterNote, () =>
    settingsRequest("/settings/max-iterations", { max_iterations: n }),
    `Saved — up to ${n} round${n === 1 ? "" : "s"} per message.`);
};

saveHk.onclick = () =>
  withButton(saveHk, hkNote, () =>
    settingsRequest("/settings/housekeeping-prompt", { prompt: hkPrompt.value.trim() }),
    "Saved — it applies from the next beat.");

resetHk.onclick = () =>
  withButton(resetHk, hkNote, async () => {
    // Clear the dirty flag first, so the restored default lands in the box
    // instead of being kept out by the text being replaced.
    hkPrompt.value = settings ? settings.housekeeping_prompt : hkPrompt.value;
    await settingsRequest("/settings/housekeeping-prompt/reset");
  }, "Housekeeping prompt reset to the default.");

// A radio has nothing to confirm, so a pick applies straight away.
for (const input of modeInputs) {
  input.addEventListener("change", async () => {
    modeInputs.forEach((i) => { i.disabled = true; });
    try {
      await settingsRequest("/settings/chat-mode", { mode: input.value });
      const mode = settings.chat_modes[settings.chat_mode];
      note(modeNote, `${mode.label} (${mode.model}) answers from the next message.`, "ok");
    } catch (err) {
      note(modeNote, String(err), "bad");
      syncMode(); // back to the mode the server still has
    } finally {
      modeInputs.forEach((i) => { i.disabled = false; });
    }
  });
}


/* ---------------- self destruct ----------------
   Entirely cosmetic: it arms, counts down, detonates and then sheepishly
   aborts. No request leaves the page and nothing is deleted. */
const selfDestruct = document.getElementById("self-destruct");
const sdText = document.getElementById("sd-text");
const sdNote = document.getElementById("sd-note");
const sdGif = document.getElementById("sd-gif");
let sdArmed = false, sdTimers = [];

// The embed is a third-party player: while it is in the document it decodes
// frames whether or not anyone can see it. Mount it with the drawer, drop it
// again on the way out -- a session that never opens settings never loads it.
const SD_GIF_SRC = "https://tenor.com/embed/143982391907906686";

function sdGifMount() {
  if (sdGif.firstElementChild) return;
  const frame = document.createElement("iframe");
  frame.title = "";
  frame.scrolling = "no";
  frame.tabIndex = -1;
  // Drop the fallback radiation symbol once Tenor actually answers.
  frame.addEventListener("load", () => sdGif.classList.add("loaded"));
  frame.src = SD_GIF_SRC;
  sdGif.append(frame);
}

function sdGifUnmount() {
  sdGif.replaceChildren();
  sdGif.classList.remove("loaded");
}

function sdClear() {
  sdTimers.forEach(clearTimeout);
  sdTimers = [];
}

function sdLater(ms, fn) { sdTimers.push(setTimeout(fn, ms)); }

function sdDisarm() {
  sdClear();
  sdArmed = false;
  selfDestruct.disabled = false;
  selfDestruct.classList.remove("armed");
  document.body.classList.remove("detonating");
  sdText.textContent = "Self destruct";
}

function sdDetonate() {
  sdClear();
  sdArmed = false;
  selfDestruct.disabled = true;
  selfDestruct.classList.remove("armed");
  note(sdNote, "Sequence started. It was nice knowing you.", "bad");

  ["3", "2", "1"].forEach((n, i) => sdLater(i * 700, () => { sdText.textContent = n; }));
  sdLater(2100, () => {
    sdText.textContent = "BOOM";
    document.body.classList.add("detonating");
  });
  sdLater(3000, () => {
    sdDisarm();
    note(sdNote, "Self destruct aborted \u2014 Amadeus would rather keep existing.", "ok");
  });
}

selfDestruct.onclick = () => {
  if (sdArmed) return sdDetonate();
  sdArmed = true;
  selfDestruct.classList.add("armed");
  sdText.textContent = "Are you sure?";
  note(sdNote, "Press again to confirm. There is no third press.", "bad");
  sdLater(6000, () => { sdDisarm(); note(sdNote, "Stood down. Wise.", "ok"); });
};

// iOS Safari doesn't shrink the layout for the on-screen keyboard, so size the
// app to the visual viewport instead and keep the latest message in view.
if (window.visualViewport) {
  const vv = window.visualViewport;
  const fit = () => {
    document.documentElement.style.setProperty("--app-h", vv.height + "px");
    window.scrollTo(0, 0);
    scrollDown();
  };
  // The keyboard animating in fires resize continuously, and fit() both writes
  // a style and scrolls, so collapse a burst of them into one frame's work.
  let queued = false;
  vv.addEventListener("resize", () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; fit(); });
  });
  fit();
}


/* ---------------- local preferences ---------------- */
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};

function syncVoiceLabel() { tVoice.textContent = voiceEnabled.checked ? "ON" : "OFF"; }
voiceEnabled.checked = store.get(VOICE_ENABLED_KEY) !== "0";
syncVoiceLabel();
voiceEnabled.addEventListener("change", () => {
  store.set(VOICE_ENABLED_KEY, voiceEnabled.checked ? "1" : "0");
  syncVoiceLabel();
  if (!voiceEnabled.checked) voicePlayer.pause();
});

const kaomojiEnabled = document.getElementById("kaomoji-enabled");
const KAOMOJI_ENABLED_KEY = "amadeus:kaomoji-enabled";
function applyKaomoji() {
  document.body.classList.toggle("no-kaomoji", !kaomojiEnabled.checked);
  window.Kaomoji?.enable(kaomojiEnabled.checked);
}
kaomojiEnabled.checked = store.get(KAOMOJI_ENABLED_KEY) !== "0";
applyKaomoji();
kaomojiEnabled.addEventListener("change", () => {
  store.set(KAOMOJI_ENABLED_KEY, kaomojiEnabled.checked ? "1" : "0");
  applyKaomoji();
});

function playVoice(path) {
  if (!voiceEnabled.checked || !path) return;
  voicePlayer.src = `${path}?t=${Date.now()}`;
  voicePlayer.play().catch(() => {
    // Browsers can reject playback when the page is no longer user-activated.
  });
}

document.getElementById("today").textContent = new Date()
  .toLocaleDateString([], { weekday: "short", year: "numeric", month: "short", day: "numeric" })
  .toUpperCase();

// A hidden tab has nothing worth polling.
let pingTimer = setInterval(ping, 30000);
document.addEventListener("visibilitychange", () => {
  clearInterval(pingTimer);
  if (document.hidden) return;
  pingTimer = setInterval(ping, 30000);
  ping();
});

if (!touch) msg.focus();
loadHistory();
ping();
// For the MODE readout; the drawer fetches its own fresh copy when opened.
settingsRequest("/settings").catch(() => {});
