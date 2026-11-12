const W = 20;
const bar = p => {
  const n = Math.round(Math.max(0, Math.min(100, p)) / 100 * W);
  return '[<span class="bar">' + '#'.repeat(n) + '</span><span class="dim">' + '.'.repeat(W - n) + '</span>]';
};
const pct = p => String(Math.round(p)).padStart(3) + '%';
const gb = b => (b / 2 ** 30).toFixed(1);

async function tick() {
  try {
    const s = await (await fetch('/api/stats', { cache: 'no-store' })).json();
    const mp = s.mem_total ? 100 * s.mem_used / s.mem_total : 0;
    document.getElementById('mon').innerHTML =
      'cpu ' + bar(s.cpu) + ' ' + pct(s.cpu) + '\n' +
      'mem ' + bar(mp) + ' ' + pct(mp) + '  <span class="dim">' + gb(s.mem_used) + '/' + gb(s.mem_total) + ' GB</span>';
  } catch (e) { /* server restarting; try again next tick */ }
}
tick();
setInterval(tick, 1000);

// Storage box: tabs "On-device" (local disks) and "Google" (Drive quota via edge-store).
const storage = { tab: 'device', device: null, google: null };
try { storage.tab = localStorage.getItem('storageTab') || 'device'; } catch (e) {}

function renderStorage() {
  const tabs = [['device', 'On-device'], ['google', 'Google']].map(([k, label]) =>
    '<span class="tab' + (storage.tab === k ? ' on' : '') + '" data-tab="' + k + '">' +
    (storage.tab === k ? '[' + label + ']' : ' ' + label + ' ') + '</span>').join(' ');
  let body = '<span class="dim">loading ...</span>';
  if (storage.tab === 'device' && storage.device) {
    const d = storage.device;
    const dw = Math.max(...d.disks.map(x => x.dev.length), 5);
    const row = (name, used, total, extra = '') => {
      const p = total ? 100 * used / total : 0;
      return name.padEnd(dw) + ' ' + bar(p) + ' ' + pct(p) +
        '  <span class="dim">' + gb(used) + '/' + gb(total) + ' GB' + extra + '</span>';
    };
    body = d.disks.map(x => row(x.dev, x.used, x.total)).join('\n') + '\n' +
      '<span class="dim">' + '-'.repeat(dw + W + 12) + '</span>\n' + row('total', d.used, d.total);
  } else if (storage.tab === 'google' && storage.google) {
    const g = storage.google;
    if (!g.ok) body = '<span class="bad">' + esc(g.error || 'unavailable') + '</span>';
    else {
      const p = g.limit ? 100 * g.usage / g.limit : 0;
      const other = Math.max(0, g.usage - g.usageInDrive - g.usageInDriveTrash);
      body = 'account <span class="dim">' + esc(g.account) + '</span>\n' +
        'quota   ' + (g.limit ? bar(p) + ' ' + pct(p) + '  <span class="dim">' + gb(g.usage) + '/' + gb(g.limit) + ' GB</span>'
                             : '<span class="dim">unlimited, ' + gb(g.usage) + ' GB used</span>') + '\n' +
        '<span class="dim">drive ' + gb(g.usageInDrive) + ' GB · trash ' + gb(g.usageInDriveTrash) +
        ' GB · gmail/photos ' + gb(other) + ' GB</span>';
    }
  }
  document.getElementById('disks').innerHTML = tabs + '\n\n' + body;
}

const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

document.getElementById('disks').addEventListener('click', e => {
  const t = e.target.closest('[data-tab]');
  if (!t) return;
  storage.tab = t.dataset.tab;
  try { localStorage.setItem('storageTab', storage.tab); } catch (e) {}
  renderStorage();
});

async function disks() {
  try { storage.device = await (await fetch('/api/storage', { cache: 'no-store' })).json(); renderStorage(); } catch (e) {}
}
async function gdrive() {
  try { storage.google = await (await fetch('/api/gdrive', { cache: 'no-store' })).json(); renderStorage(); } catch (e) {}
}
disks(); gdrive();
setInterval(disks, 10000);
setInterval(gdrive, 60000);

async function docker() {
  const el = document.getElementById('docker');
  try {
    const d = await (await fetch('/api/docker', { cache: 'no-store' })).json();
    if (!d.ok) { el.innerHTML = '<span class="dim">docker unavailable</span>'; return; }
    if (!d.containers.length) { el.innerHTML = '<span class="dim">no running containers</span>'; return; }
    const nw = Math.max(...d.containers.map(c => c.name.length));
    el.innerHTML = d.containers.map(c =>
      c.name.padEnd(nw) + '  cpu ' + (c.cpu.toFixed(1) + '%').padStart(6) +
      '  mem ' + bar(c.mem) + ' ' + pct(c.mem) + '  <span class="dim">' + c.usage + '</span>').join('\n');
  } catch (e) { /* retry next tick */ }
}
docker();
setInterval(docker, 3000);

// Inference box: tabs "Tailnet" (Windows/macOS peers + Ollama) and "OpenRouter" (account credits).
const inference = { tab: 'tailnet', data: null };
try { inference.tab = localStorage.getItem('inferenceTab') || 'tailnet'; } catch (e) {}

function renderInference() {
  const tabs = [['tailnet', 'Tailnet'], ['openrouter', 'OpenRouter']].map(([k, label]) =>
    '<span class="tab' + (inference.tab === k ? ' on' : '') + '" data-tab="' + k + '">' +
    (inference.tab === k ? '[' + label + ']' : ' ' + label + ' ') + '</span>').join(' ');
  let body = '<span class="dim">loading ...</span>';
  const d = inference.data;
  if (d && inference.tab === 'tailnet') {
    if (d.tailnet_error) body = '<span class="bad">' + esc(d.tailnet_error) + '</span>';
    else if (d.hosts.length) {
      const lw = Math.max(...d.hosts.map(h => h.label.length));
      body = d.hosts.map(h => {
        const head = h.label.padEnd(lw) + '  ';
        if (!h.found) return head + '<span class="bad">not on tailnet</span>';
        const ts = h.online ? '<span class="ok">online </span>' : '<span class="bad">offline</span>';
        const ol = h.ollama ? '<span class="ok">ollama up  </span> <span class="dim">' + (h.ms + 'ms').padStart(6) + '</span>'
                            : '<span class="' + (h.online ? 'warn' : 'dim') + '">ollama down</span>';
        return head + ts + '  ' + ol + (h.path ? '  <span class="dim">' + esc(h.path) + '</span>' : '');
      }).join('\n');
    }
  } else if (d && inference.tab === 'openrouter') {
    const o = d.openrouter;
    if (!o.ok) body = '<span class="bad">' + esc(o.error || 'unavailable') + '</span>';
    else {
      const left = o.total - o.used, p = o.total ? 100 * o.used / o.total : 0;
      body = 'credits ' + '<span class="' + (left > 1 ? 'ok' : 'warn') + '">$' + left.toFixed(2) + '</span> left\n' +
        'used    ' + bar(p) + ' ' + pct(p) + '  <span class="dim">$' + o.used.toFixed(2) + '/$' + o.total.toFixed(2) + '</span>';
    }
  }
  document.getElementById('inference').innerHTML = tabs + '\n\n' + body;
}

document.getElementById('inference').addEventListener('click', e => {
  const t = e.target.closest('[data-tab]');
  if (!t) return;
  inference.tab = t.dataset.tab;
  try { localStorage.setItem('inferenceTab', inference.tab); } catch (e) {}
  renderInference();
});

async function inferenceTick() {
  try { inference.data = await (await fetch('/api/inference', { cache: 'no-store' })).json(); renderInference(); } catch (e) {}
}
inferenceTick();
setInterval(inferenceTick, 10000);

async function tunnels() {
  const el = document.getElementById('tunnels');
  try {
    const d = await (await fetch('/api/tunnels', { cache: 'no-store' })).json();
    if (!d.ok) { el.innerHTML = '<span class="bad">' + (d.error || 'unavailable') + '</span>'; return; }
    if (!d.tunnels.length) { el.innerHTML = '<span class="dim">no tunnels</span>'; return; }
    const cls = { healthy: 'ok', degraded: 'warn', down: 'bad', inactive: 'dim' };
    const nw = Math.max(...d.tunnels.map(t => t.name.length));
    const sw = Math.max(...d.tunnels.map(t => t.status.length));
    el.innerHTML = d.tunnels.map(t =>
      t.name.padEnd(nw) + '  <span class="' + (cls[t.status] || 'dim') + '">' + t.status.padEnd(sw) + '</span>' +
      '  <span class="dim">' + String(t.conns).padStart(2) + ' conn' + (t.colos.length ? ' ' + t.colos.join(',') : '') + '</span>' +
      t.hosts.map(h => '\n' + ' '.repeat(nw + 2) + '<span class="dim">└ </span>' + h).join('')).join('\n');
  } catch (e) { /* retry next tick */ }
}
tunnels();
setInterval(tunnels, 30000);

// Logo ring: each box needs its top offset within .side to place its slice of the circle (see style.css).
const side = document.querySelector('.side');
const placeRings = () => side.querySelectorAll('.box').forEach(b => b.style.setProperty('--y', b.offsetTop + 'px'));
// watch every box: the column has a fixed height, so a box growing (e.g. a Storage tab switch) only moves the ones below
const ringWatch = new ResizeObserver(placeRings);
side.querySelectorAll('.box').forEach(b => ringWatch.observe(b));
placeRings();

// Chat pane: full history of the root agent (agents.db `root` table); Enter sends, Shift+Enter is a newline.
const log = document.getElementById('log'), chatIn = document.getElementById('chat-in');
const chatBtn = document.querySelector('#chat-form button');

function addMsg(role, text, cls = '', thinking = '') {
  const el = document.createElement('div');
  el.className = 'msg ' + role + (cls ? ' ' + cls : '');
  if (thinking) {  // the model's thoughts, above its reply
    const t = document.createElement('div');
    t.className = 'think';
    t.textContent = thinking.trim();
    el.appendChild(t);
  }
  el.insertAdjacentHTML('beforeend', '<span class="who">' + (role === 'user' ? 'you' : 'max') + '&gt; </span>');
  // paragraphs (blank-line separated) get a half-line gap instead of a full empty line
  text.trim().split(/\n\s*\n/).forEach((para, i) => {
    const p = document.createElement('span');
    if (i) p.className = 'para';
    p.textContent = para;
    el.appendChild(p);
  });
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

const grow = () => { chatIn.style.height = 'auto'; chatIn.style.height = chatIn.scrollHeight + 'px'; };
chatIn.addEventListener('input', grow);
chatIn.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); document.getElementById('chat-form').requestSubmit(); }
});

document.getElementById('chat-form').addEventListener('submit', async e => {
  e.preventDefault();
  const text = chatIn.value.trim();
  if (!text || chatBtn.disabled) return;
  const vm = text.match(/^\/voice(?:\s+(on|off))?\s*$/i);
  if (vm) {  // voice mode lives here (it needs the mic); the agent does STT -> turn -> TTS over /api/voice
    chatIn.value = ''; grow();
    const want = vm[1] ? vm[1].toLowerCase() === 'on' : !voice.on;
    want ? voiceStart() : voiceStop('voice mode off');
    return;
  }
  chatIn.value = ''; grow();
  chatBtn.disabled = true;
  const isCmd = text.startsWith('/');  // slash commands (e.g. /clear N) are handled by the agent, not the model
  addMsg('user', text);
  const pending = addMsg('assistant', isCmd ? '...' : 'thinking ...', 'dim');
  try {
    const r = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) });
    const raw = await r.text();
    let d;
    try { d = JSON.parse(raw); } catch (_) {
      // not from the console (e.g. a Cloudflare error/login page): show status + page title
      const title = (raw.match(/<title>([^<]*)<\/title>/i) || [])[1] || raw.slice(0, 80);
      d = { error: 'HTTP ' + r.status + ' ' + title.trim() };
    }
    pending.remove();
    if (d.error) addMsg('assistant', 'error: ' + d.error, 'err');
    else if (d.command) { await loadHistory(); addMsg('assistant', d.reply, 'dim'); }  // history changed: redraw it
    else addMsg('assistant', d.reply, '', d.thinking || '');
  } catch (err) {
    pending.remove();
    addMsg('assistant', 'error: ' + err.message, 'err');
  }
  chatBtn.disabled = false;
  chatIn.focus();
});

async function loadHistory() {
  try {
    const d = await (await fetch('/api/chat', { cache: 'no-store' })).json();
    log.textContent = '';
    d.messages.forEach(m => addMsg(m.role, m.content, '', m.thinking || ''));
  } catch (e) {}
}
loadHistory();

// Logs box: live agent logs (agents/logs/agents.jsonl via /api/logs SSE). Keeps the last 500 lines and follows the
// bottom unless you've scrolled up.
const logBox = document.querySelector('#logs .loglines');
let logFirst = true;
function addLog(raw) {
  let e;
  try { e = JSON.parse(raw); } catch (_) { return; }
  if (logFirst) { logBox.textContent = ''; logFirst = false; }
  const atBottom = logBox.scrollHeight - logBox.scrollTop - logBox.clientHeight < 8;
  const t = new Date(e.ts);
  const hms = isNaN(t) ? '--:--:--' : t.toTimeString().slice(0, 8);
  const fields = Object.entries(e.fields || {}).map(([k, v]) => k + '=' + (typeof v === 'string' ? v : JSON.stringify(v))).join(' ');
  const row = document.createElement('div');
  row.innerHTML = '<span class="dim">' + hms + '</span> <span class="lv-' + esc(e.level) + '">' + esc(e.level.padEnd(5)) +
    '</span> <span class="dim">' + esc(e.agent) + '</span> ' + esc(e.msg) + (fields ? ' <span class="dim">' + esc(fields) + '</span>' : '');
  logBox.appendChild(row);
  while (logBox.childElementCount > 500) logBox.firstElementChild.remove();
  if (atBottom) logBox.scrollTop = logBox.scrollHeight;
}
const logStream = new EventSource('/api/logs');
logStream.onmessage = m => addLog(m.data);
logStream.onopen = () => { logFirst = true; };  // on reconnect the server replays the tail, so start clean

// Voice mode (/voice, /voice on|off): mic -> 16 kHz s16le PCM over a WebSocket to the root agent (/api/voice), which
// runs STT, a streamed chat turn and TTS (Phonon "Russell") sentence by sentence, and streams 24 kHz PCM back.
// The mic is shut for the whole turn (from "heard" until Max's audio has finished, plus a tail): silence is sent
// instead, so the STT neither hears Max nor keeps a half utterance pending. Space while Max is busy cuts him off.
const voice = { on: false, ready: false, ws: null, ctx: null, mic: null, rate: 24000, playEnd: 0, quietUntil: 0,
  busy: false, drop: false, sources: [], live: null };
const chatBox = document.querySelector('.chat');
const MIC_WORKLET = `
class Mic16k extends AudioWorkletProcessor {
  constructor() { super(); this.step = sampleRate / 16000; this.t = 0; this.out = []; }
  process(inputs) {
    const ch = inputs[0][0];
    if (!ch) return true;
    while (this.t < ch.length) {  // linear-interpolation resample to 16 kHz
      const i = Math.floor(this.t), f = this.t - i, a = ch[i], b = i + 1 < ch.length ? ch[i + 1] : a;
      this.out.push(a + (b - a) * f);
      this.t += this.step;
    }
    this.t -= ch.length;
    if (this.out.length >= 1600) {  // ~100 ms frames
      const pcm = new Int16Array(this.out.length);
      this.out.forEach((v, k) => { pcm[k] = Math.max(-1, Math.min(1, v)) * 0x7fff; });
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
      this.out = [];
    }
    return true;
  }
}
registerProcessor('mic16k', Mic16k);`;

// true while Max's audio is (or was just) playing
const speaking = () => !!voice.ctx && voice.ctx.currentTime < Math.max(voice.playEnd + 0.5, voice.quietUntil);

async function voiceStart() {
  if (voice.on) return addMsg('assistant', 'voice mode is already on', 'dim');
  voice.on = true;
  voice.ctx = new AudioContext();  // created inside the Enter/click so the browser lets it play
  try {
    voice.mic = await navigator.mediaDevices.getUserMedia(
      { audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    await voice.ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([MIC_WORKLET], { type: 'application/javascript' })));
  } catch (e) {
    voiceStop();
    return addMsg('assistant', 'error: microphone unavailable: ' + e.message, 'err');
  }
  const node = new AudioWorkletNode(voice.ctx, 'mic16k');
  const mute = voice.ctx.createGain(); mute.gain.value = 0;  // keeps the worklet pulled without echoing the mic
  voice.ctx.createMediaStreamSource(voice.mic).connect(node).connect(mute).connect(voice.ctx.destination);

  const ws = voice.ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/api/voice');
  ws.binaryType = 'arraybuffer';
  node.port.onmessage = e => {
    if (!voice.ready || ws.readyState !== 1) return;
    const listening = !voice.busy && !speaking();
    ws.send(listening ? e.data : new ArrayBuffer(e.data.byteLength));  // zeros = silence while Max has the floor
  };
  ws.onmessage = e => typeof e.data === 'string' ? voiceEvent(JSON.parse(e.data)) : playPcm(e.data);
  ws.onclose = () => { if (voice.ws === ws && voice.on) voiceStop('voice mode ended: connection closed'); };
  chatBox.dataset.title = 'Max · voice';
  addMsg('assistant', 'voice mode: connecting ...', 'dim');
}

function voiceStop(note) {
  voice.on = voice.ready = voice.busy = false;
  if (voice.ws) { const ws = voice.ws; voice.ws = null; ws.close(); }
  if (voice.mic) voice.mic.getTracks().forEach(t => t.stop());
  if (voice.ctx) voice.ctx.close();
  voice.mic = voice.ctx = null; voice.playEnd = voice.quietUntil = 0; voice.sources = [];
  if (voice.live) { voice.live.el.remove(); voice.live = null; }
  chatBox.dataset.title = 'Max';
  if (note) addMsg('assistant', note, 'dim');
}

// Cut Max off: stop playback now, drop audio still in flight, and have the agent stop generating and speaking.
function voiceInterrupt() {
  voice.drop = true;
  voice.sources.forEach(s => { try { s.stop(); } catch (_) {} });
  voice.sources = [];
  voice.playEnd = 0;
  voice.quietUntil = voice.ctx.currentTime + 0.3;  // let the speaker's last sound die before listening again
  voice.busy = false;
  if (voice.ws && voice.ws.readyState === 1) voice.ws.send(JSON.stringify({ type: 'interrupt' }));
}

document.addEventListener('keydown', e => {
  if (e.code !== 'Space' || e.repeat || e.ctrlKey || e.metaKey || e.altKey || e.shiftKey) return;
  if (!voice.on || !(voice.busy || speaking())) return;
  e.preventDefault();
  voiceInterrupt();
});

// The reply being written: thoughts and text grow as deltas arrive; replaced by the finished message at the end.
function liveMsg() {
  const el = document.createElement('div');
  el.className = 'msg assistant';
  const think = document.createElement('div');
  think.className = 'think';
  think.style.display = 'none';
  const body = document.createElement('span');
  body.className = 'dim';
  body.textContent = 'thinking ...';
  el.appendChild(think);
  el.insertAdjacentHTML('beforeend', '<span class="who">max&gt; </span>');
  el.appendChild(body);
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return { el, think, body, started: false };
}

function follow(fn) {  // keep the log pinned to the bottom if it was
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
  fn();
  if (atBottom) log.scrollTop = log.scrollHeight;
}

function voiceEvent(ev) {
  const live = voice.live;
  if (ev.type === 'ready') { voice.ready = true; addMsg('assistant', 'voice mode on: listening. Space cuts Max off; /voice to stop', 'dim'); }
  else if (ev.type === 'heard') {
    voice.busy = true; voice.drop = false;
    addMsg('user', ev.text);
    voice.live = liveMsg();
  }
  else if (ev.type === 'thinking_delta' && live) follow(() => {
    live.think.style.display = '';
    live.think.textContent += live.think.textContent ? ev.text : ev.text.trimStart();
  });
  else if (ev.type === 'reply_delta' && live) follow(() => {
    if (!live.started) { live.started = true; live.body.className = ''; live.body.textContent = ''; }
    live.body.textContent += live.body.textContent ? ev.text : ev.text.trimStart();
  });
  else if (ev.type === 'speaking') voice.rate = ev.rate || 24000;
  else if (ev.type === 'reply') {
    if (live) { live.el.remove(); voice.live = null; }
    const el = addMsg('assistant', ev.text, '', ev.thinking || '');
    if (ev.interrupted) el.insertAdjacentHTML('beforeend', ' <span class="dim">[cut off]</span>');
  }
  else if (ev.type === 'done') voice.busy = false;
  else if (ev.type === 'error') {
    if (live) { live.el.remove(); voice.live = null; }
    voice.busy = false;
    addMsg('assistant', 'error: ' + ev.error, 'err');
  }
}

function playPcm(buf) {
  if (!voice.ctx || voice.drop) return;  // dropped: audio from a turn that was cut off
  const pcm = new Int16Array(buf), audio = voice.ctx.createBuffer(1, pcm.length, voice.rate), ch = audio.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 0x8000;
  const src = voice.ctx.createBufferSource();
  src.buffer = audio;
  src.connect(voice.ctx.destination);
  const at = Math.max(voice.ctx.currentTime + 0.05, voice.playEnd);  // queue chunks back to back
  src.start(at);
  voice.playEnd = at + audio.duration;
  voice.sources.push(src);
  src.onended = () => { voice.sources = voice.sources.filter(s => s !== src); };
}
