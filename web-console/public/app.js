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

// Chat pane: full history of the root agent (agents.db `root` table); Enter sends, Shift+Enter is a newline.
const log = document.getElementById('log'), chatIn = document.getElementById('chat-in');
const chatBtn = document.querySelector('#chat-form button');

// A tool call the model made: "$ command" and the start of its output.
function toolBlock(command, output) {
  const b = document.createElement('div');
  b.className = 'tool';
  const c = document.createElement('div');
  c.className = 'cmd';
  c.textContent = '$ ' + command;
  const o = document.createElement('div');
  b.append(c, o);
  b.setOutput = text => {
    const lines = String(text).trimEnd().split('\n');
    o.textContent = lines.slice(0, 12).join('\n') + (lines.length > 12 ? '\n[+' + (lines.length - 12) + ' lines]' : '');
  };
  if (output !== undefined) b.setOutput(output);
  return b;
}

// Sub-agents (reached through the blackboard): their work shows indented, as "<name>> what they said", in their own
// colour (Chronos white, Athena violet), instead of a "$ command" tool block.
const SUBAGENTS = ['athena', 'chronos'];
function subBlock(agent, request, text) {
  const b = document.createElement('div');
  b.className = 'sub ' + agent;
  if (request) {
    const q = document.createElement('div');
    q.className = 'req';
    q.textContent = 'max -> ' + agent + ': ' + request;
    b.appendChild(q);
  }
  const line = document.createElement('div');
  line.innerHTML = '<span class="who">' + agent + '&gt; </span>';
  const t = document.createElement('span');
  line.appendChild(t);
  b.appendChild(line);
  b.setOutput = out => { t.textContent = subText(agent, out); };
  if (text !== undefined) b.setOutput(text);
  return b;
}
// What a sub-agent said, from its tool output (which also carries notes meant for the model).
function subText(agent, out) {
  out = String(out || '').trim()
    .replace(/^[a-z]+ answered:\s*/i, '')
    .replace(/\s*\(What they fetched is now on the user's screen[^)]*\)/, '')
    .replace(/\s*Her answer describes emails:[\s\S]*$/, '');
  if (agent === 'chronos') return out.replace(/^Chronos answered:\s*/, '').replace(/\s*The schedule is now on the user's screen.*$/, '');
  if (agent === 'athena') return out.split('\n')[0].replace(/^Athena /, '').replace(/; the list is now on the user's screen.*$/, ' (in the Agent Workspace)');
  return out;
}

// One step of a turn, as stored: thinking (grey italics), text said before a tool call, a tool call, a sub-agent's
// work, or a sub-agent report handed to Max at the start of the turn.
// "agent: request", the label of a delegate call
const delegated = command => { const m = String(command || '').match(/^([a-z]+):\s*([\s\S]*)$/); return m ? [m[1], m[2]] : ['agent', command]; };
function stepNode(st) {
  if (st.type === 'tool' && st.name === 'delegate') { const [a, req] = delegated(st.command); return subBlock(a, req, st.output); }
  if (st.type === 'tool' && SUBAGENTS.includes(st.name)) return subBlock(st.name, st.command, st.output);
  if (st.type === 'tool') return toolBlock(st.command, st.output);
  if (st.type === 'report') {
    const agent = st.agent || 'agent';
    return subBlock(agent, '', String(st.text || '').replace(new RegExp('^' + agent + ':\\s*'), ''));
  }
  const d = document.createElement('div');
  d.className = st.type === 'thinking' ? 'think' : 'steptext';
  d.textContent = (st.text || '').trim();
  return d;
}

// A message. For replies, `extra` carries what happened before the answer: `steps` (in order), or for older rows
// `thinking` and `tools`.
function addMsg(role, text, cls = '', extra = {}) {
  const el = document.createElement('div');
  el.className = 'msg ' + role + (cls ? ' ' + cls : '');
  if (extra.steps) extra.steps.forEach(st => el.appendChild(stepNode(st)));
  else {
    if (extra.thinking) el.appendChild(stepNode({ type: 'thinking', text: extra.thinking }));
    (extra.tools || []).forEach(t => el.appendChild(toolBlock(t.command, t.output)));
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
  if (picker.open) return;  // the picker has the keyboard (its own handler below)
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); document.getElementById('chat-form').requestSubmit(); }
});

document.getElementById('chat-form').addEventListener('submit', async e => {
  e.preventDefault();
  const text = chatIn.value.trim();
  if (!text || chatBtn.disabled) return;
  if (/^\/model\s*$/i.test(text)) {  // the model picker lives here too (see pickerOpen)
    chatIn.value = ''; grow();
    pickerOpen();
    return;
  }
  const vm = text.match(/^\/voice(?:\s+(on|off))?\s*$/i);
  if (vm) {  // voice mode lives here (it needs the mic); the agent does STT -> turn -> TTS over /api/voice
    chatIn.value = ''; grow();
    const want = vm[1] ? vm[1].toLowerCase() === 'on' : !voice.on;
    want ? voiceStart() : voiceStop('voice mode off');
    return;
  }
  chatIn.value = ''; grow();
  chatBtn.disabled = true;
  addMsg('user', text);
  turnStart();
  try {
    const r = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) });
    if (!(r.headers.get('content-type') || '').includes('text/event-stream')) {
      // not the agent's stream (e.g. a Cloudflare error or login page): show status + page title
      const raw = await r.text(), title = (raw.match(/<title>([^<]*)<\/title>/i) || [])[1] || raw.slice(0, 80);
      throw new Error('HTTP ' + r.status + ' ' + title.trim());
    }
    if (!await readEvents(r, turnEvent)) throw new Error('connection lost; the turn carries on in the agent, reload to see it');
  } catch (err) {
    turnEvent({ type: 'error', error: err.message });
  }
  chatBtn.disabled = false;
  chatIn.focus();
});

// Read a Server-Sent Events response (one JSON event per "data:" line), calling onEvent for each. True if the stream
// ended with a final event (reply, note or error).
async function readEvents(r, onEvent) {
  const reader = r.body.getReader(), dec = new TextDecoder();
  let buf = '', ended = false;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return ended;
    buf += dec.decode(value, { stream: true });
    let cut;
    while ((cut = buf.indexOf('\n\n')) >= 0) {
      const data = buf.slice(0, cut).split('\n').filter(l => l.startsWith('data:')).map(l => l.slice(5).trimStart()).join('\n');
      buf = buf.slice(cut + 2);
      if (!data) continue;  // keep-alive
      const ev = JSON.parse(data);
      if (['reply', 'note', 'error'].includes(ev.type)) ended = true;
      onEvent(ev);
    }
  }
}

// Context meter above the input: tokens the conversation fills in the model's window, as of the last reply.
let lastCtx = null;  // the meter's last reading, so a model switch can redraw it with the new maximum
function showContext(c) {
  if (!c) return;
  lastCtx = c;
  const k = n => (n / 1000).toFixed(1) + 'k';
  const p = c.used == null ? 0 : Math.min(100, 100 * c.used / c.max);
  const label = c.used == null ? 'context <span class="dim">known after the next reply</span>'
    : 'context ' + Math.round(p) + '% <span class="dim">' + k(c.used) + ' / ' + k(c.max) + '</span>';
  document.getElementById('ctx').innerHTML = '<span class="lead"></span>[ ' + label + ' ]' +
    '<span class="track"><span class="fill" style="width:' + p.toFixed(1) + '%"></span></span>';
}

// `restore`: bring back the newest workspace view too (first load only: a reload after /clear leaves the Agent
// Workspace as it is, since clearing the chat isn't a request to change what's on screen)
async function loadHistory(restore = false) {
  try {
    const d = await (await fetch('/api/chat', { cache: 'no-store' })).json();
    log.textContent = '';
    showContext(d.context);
    d.messages.forEach(m => addMsg(m.role, m.content, '', m));
    // bring back the newest workspace view (e.g. the last email list) without switching tabs
    const views = restore ? d.messages.flatMap(m => (m.steps || []).map(s => s.view).filter(Boolean)) : [];
    if (views.length) renderView(views[views.length - 1], false);
  } catch (e) {}
}
loadHistory(true);

// Live pushes from the root agent (push.rs), one stream per tab: Max telling the user something on his own (e.g.
// about new mail from Athena) and sub-agent reports (Chronos' reminders). The tab also says whether it has focus:
// Max sends his own messages to Discord only when no console tab is open and focused.
const TAB = Math.random().toString(36).slice(2);
const focusedNow = () => document.hasFocus() && document.visibilityState === 'visible';
function sendPresence() {
  fetch('/api/presence', { method: 'POST', headers: { 'Content-Type': 'application/json' }, keepalive: true,
    body: JSON.stringify({ id: TAB, focused: focusedNow() }) }).catch(() => {});
}
function pushEvent(ev) {
  if (ev.type === 'proactive') follow(() => {
    addMsg('assistant', ev.text || '', '', ev);
    showContext(ev.context);
    (ev.steps || []).forEach(st => st.view && renderView(st.view, false));
  });
  else if (ev.type === 'report') follow(() => {
    const d = document.createElement('div');
    d.className = 'msg assistant';
    d.appendChild(subBlock(ev.agent || 'agent', '', ev.text || ''));
    log.appendChild(d);
  });
}
const push = new EventSource('/api/events?id=' + TAB + '&focused=' + (focusedNow() ? 1 : 0));
push.onmessage = e => { try { pushEvent(JSON.parse(e.data)); } catch (err) {} };
push.onopen = sendPresence;  // after a reconnect too, so the agent has this tab's current focus
window.addEventListener('focus', sendPresence);
window.addEventListener('blur', sendPresence);
document.addEventListener('visibilitychange', sendPresence);

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
  showPane(pane);  // the Chat tab shows "· voice"
  addMsg('assistant', 'voice mode: connecting ...', 'dim');
}

function voiceStop(note) {
  voice.on = voice.ready = voice.busy = false;
  if (voice.ws) { const ws = voice.ws; voice.ws = null; ws.close(); }
  if (voice.mic) voice.mic.getTracks().forEach(t => t.stop());
  if (voice.ctx) voice.ctx.close();
  voice.mic = voice.ctx = null; voice.playEnd = voice.quietUntil = 0; voice.sources = [];
  if (turnLive) { turnLive.el.remove(); turnLive = null; }
  showPane(pane);
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

// The turn being written (typed or spoken): steps appear in order as events arrive. Text streams in after "max>";
// if a tool call follows, that text was said on the way, so it moves up into the steps. The finished message
// (from the "reply" event) replaces it all.
let turnLive = null;

function turnStart() {
  const el = document.createElement('div');
  el.className = 'msg assistant';
  const steps = document.createElement('div');
  const body = document.createElement('span');
  body.className = 'dim';
  body.textContent = 'thinking ...';
  el.appendChild(steps);
  el.insertAdjacentHTML('beforeend', '<span class="who">max&gt; </span>');
  el.appendChild(body);
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  turnLive = { el, steps, body, started: false, cur: null, curType: null, lastTool: null };
}

function follow(fn) {  // keep the log pinned to the bottom if it was
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
  fn();
  if (atBottom) log.scrollTop = log.scrollHeight;
}

function turnEvent(ev) {
  const live = turnLive;
  const flushBody = () => {  // text so far was said before a step: move it into the steps
    if (live.started && live.body.textContent.trim()) live.steps.appendChild(stepNode({ type: 'text', text: live.body.textContent }));
    live.started = false; live.body.className = 'dim'; live.body.textContent = 'working ...';
  };
  if (ev.type === 'thinking_delta' && live) follow(() => {
    if (live.curType !== 'thinking') {
      flushBody();
      live.cur = stepNode({ type: 'thinking', text: '' }); live.curType = 'thinking';
      live.steps.appendChild(live.cur);
    }
    live.cur.textContent += live.cur.textContent ? ev.text : ev.text.trimStart();
  });
  else if (ev.type === 'reply_delta' && live) follow(() => {
    if (!live.started) { live.started = true; live.body.className = ''; live.body.textContent = ''; }
    live.curType = 'text';
    live.body.textContent += live.body.textContent ? ev.text : ev.text.trimStart();
  });
  else if (ev.type === 'tool_call' && live) follow(() => {
    flushBody();
    live.lastTool = ev.name === 'delegate' ? subBlock(...delegated(ev.command))
      : SUBAGENTS.includes(ev.name) ? subBlock(ev.name, ev.command) : toolBlock(ev.command);
    live.lastTool.setOutput('running ...');
    live.steps.appendChild(live.lastTool);
    live.cur = live.lastTool; live.curType = 'tool';
  });
  else if (ev.type === 'tool_result' && live && live.lastTool) follow(() => live.lastTool.setOutput(ev.output));
  else if (ev.type === 'view') renderView(ev.view, true);  // Athena's email list, Chronos' schedule -> Agent Workspace
  else if (ev.type === 'reply') {
    if (live) { live.el.remove(); turnLive = null; }
    const el = addMsg('assistant', ev.text, '', ev);
    showContext(ev.context);
    if (ev.interrupted) el.insertAdjacentHTML('beforeend', ' <span class="dim">[cut off]</span>');
  }
  else if (ev.type === 'note') {  // a slash command's result; it may have changed the history (/clear), so redraw it
    if (live) { live.el.remove(); turnLive = null; }
    loadHistory().then(() => addMsg('assistant', ev.text, 'dim'));
  }
  else if (ev.type === 'error') {
    if (live) { live.el.remove(); turnLive = null; }
    addMsg('assistant', 'error: ' + ev.error, 'err');
  }
}

function voiceEvent(ev) {
  if (ev.type === 'ready') { voice.ready = true; addMsg('assistant', 'voice mode on: listening. Space cuts Max off; /voice to stop', 'dim'); }
  else if (ev.type === 'heard') {
    voice.busy = true; voice.drop = false;
    addMsg('user', ev.text);
    turnStart();
  }
  else if (ev.type === 'speaking') voice.rate = ev.rate || 24000;
  else if (ev.type === 'done') voice.busy = false;
  else {
    if (ev.type === 'error') voice.busy = false;
    turnEvent(ev);
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

// Chat pane title: one pane today (the graph lives in the workspace on the left); kept as tabs for more panes later.
const PANES = [['chat', 'Chat']];
let pane = 'chat';
try { pane = localStorage.getItem('chatPane') || 'chat'; } catch (e) {}

function showPane(k) {
  pane = PANES.some(([p]) => p === k) ? k : 'chat';
  try { localStorage.setItem('chatPane', pane); } catch (e) {}
  document.getElementById('panetabs').innerHTML = '[ ' + PANES.map(([p, label]) =>
    '<span class="tab' + (p === pane ? ' on' : '') + '" data-pane="' + p + '">' + label +
    (p === 'chat' && voice.on ? ' · voice' : '') + '</span>').join('<span class="dim"> · </span>') + ' ]';
  PANES.forEach(([p]) => { document.getElementById('pane-' + p).hidden = p !== pane; });
}
document.getElementById('panetabs').addEventListener('click', e => {
  const t = e.target.closest('[data-pane]');
  showPane(t ? t.dataset.pane : PANES[(PANES.findIndex(([p]) => p === pane) + 1) % PANES.length][0]);
});

// /model: pick the model Max runs on, in the chat pane. Every model on every inference box (Ollama) plus OpenRouter's
// tool-capable ones (GET /api/models). ↑/↓ move, typing filters, Enter picks (or click); then ←/→ or drag sets the
// context length (powers of two up to the model's maximum) and Enter applies it (POST /api/model). Esc goes back.
const picker = { open: false, stage: 'list', items: [], filter: '', idx: 0, cur: null, chosen: null, steps: [], si: 0,
                 note: '', errors: [] };
const pickerEl = document.getElementById('picker');
const isCurrent = m => picker.cur && m.provider === picker.cur.provider && m.base === picker.cur.base && m.name === picker.cur.model;
const pickerVisible = () => {
  const f = picker.filter.toLowerCase();
  return picker.items.filter(m => !f || (m.host + ' ' + m.name).toLowerCase().includes(f));
};

async function pickerOpen() {
  Object.assign(picker, { open: true, stage: 'list', filter: '', idx: 0, items: [], errors: [], note: 'loading models ...' });
  log.hidden = true; pickerEl.hidden = false; chatIn.blur();
  renderPicker();
  try {
    const d = await (await fetch('/api/models', { cache: 'no-store' })).json();
    picker.cur = d.current && !d.current.error ? d.current : null;
    picker.items = d.groups.flatMap(g => g.models.map(m => ({ ...m, provider: g.provider, host: g.label, base: g.base })));
    picker.errors = d.groups.filter(g => g.error).map(g => g.label + ': ' + g.error);
    picker.note = picker.cur ? 'now: ' + picker.cur.model + ' on ' + picker.cur.host + ', context ' +
      picker.cur.num_ctx.toLocaleString() : '';
    picker.idx = Math.max(0, pickerVisible().findIndex(isCurrent));
  } catch (e) { picker.note = 'error: ' + e.message; }
  renderPicker();
}

function pickerClose(note) {
  picker.open = false; pickerEl.hidden = true; log.hidden = false;
  if (note) addMsg('assistant', note, 'dim');
  log.scrollTop = log.scrollHeight;
  chatIn.focus();
}

async function pickerChoose(m) {
  if (!m) return;
  Object.assign(picker, { chosen: m, stage: 'ctx', steps: [], note: "reading the model's context size ..." });
  renderPicker();
  let max = m.ctx || 0, caps = null;
  if (m.provider === 'ollama') {
    try {
      const d = await (await fetch('/api/models/info?base=' + encodeURIComponent(m.base) + '&model=' + encodeURIComponent(m.name))).json();
      if (d.error) throw new Error(d.error);
      max = d.ctx; caps = d.capabilities;
    } catch (e) { picker.note = "couldn't read its context size (" + e.message + '); assuming 32k'; }
  }
  if (!max) max = 32768;
  const steps = [];
  for (let v = 2048; v < max; v *= 2) steps.push(v);
  steps.push(max);
  const want = isCurrent(m) ? picker.cur.num_ctx : Math.min(max, 32768);
  picker.steps = steps;
  picker.si = steps.reduce((b, v, i) => Math.abs(v - want) < Math.abs(steps[b] - want) ? i : b, 0);
  if (caps) picker.note = caps.includes('tools') ? '' : "note: this model can't call tools, so Max can't use bash or Athena with it";
  else if (!picker.note.startsWith("couldn't")) picker.note = '';
  renderPicker();
}

async function pickerApply() {
  const m = picker.chosen, num_ctx = picker.steps[picker.si];
  picker.note = 'switching ...'; renderPicker();
  try {
    const d = await (await fetch('/api/model', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider: m.provider, host: m.host, base: m.base, model: m.name, num_ctx }) })).json();
    if (d.error) throw new Error(d.error);
    showContext({ used: lastCtx ? lastCtx.used : null, max: num_ctx });
    pickerClose(d.note);
  } catch (e) { picker.note = 'error: ' + e.message; renderPicker(); }
}

function renderPicker() {
  const hint = t => '<span class="dim">' + t + '</span>';
  if (picker.stage === 'list') {
    const vis = pickerVisible();
    picker.idx = Math.min(picker.idx, Math.max(0, vis.length - 1));
    let html = '<div class="phead">[ model ] ' + hint('↑↓ move · type to filter · Enter pick · Esc close') + '</div>' +
      '<div class="pfilter">filter&gt; ' + esc(picker.filter) + '<span class="caret">_</span></div>' +
      (picker.note ? '<div class="dim">' + esc(picker.note) + '</div>' : '');
    let group = null;
    vis.forEach((m, i) => {
      if (m.host !== group) {
        group = m.host;
        html += '<div class="pgroup">' + esc(m.host) + ' ' + hint(m.provider === 'openrouter' ? 'openrouter.ai' : 'ollama') + '</div>';
      }
      const det = m.provider === 'openrouter'
        ? [m.ctx ? Math.round(m.ctx / 1024) + 'k ctx' : '', m.price != null ? '$' + m.price + '/M in' : '',
           m.price_out != null ? '$' + m.price_out + '/M out' : ''].filter(Boolean).join('  ')
        : [m.params, m.quant, m.size ? (m.size / 1e9).toFixed(1) + ' GB' : ''].filter(Boolean).join('  ');
      html += '<div class="prow' + (i === picker.idx ? ' on' : '') + '" data-i="' + i + '"><span class="pm">' +
        (i === picker.idx ? '>' : ' ') + (isCurrent(m) ? '●' : ' ') + '</span><span class="pn">' + esc(m.name) +
        '</span>' + hint(esc(det)) + '</div>';
    });
    if (!vis.length && picker.items.length) html += '<div class="dim">no models match</div>';
    picker.errors.forEach(e => { html += '<div class="err">' + esc(e) + '</div>'; });
    pickerEl.innerHTML = html;
    const on = pickerEl.querySelector('.prow.on');
    if (on) on.scrollIntoView({ block: 'nearest' });
    return;
  }
  const m = picker.chosen, steps = picker.steps;
  let html = '<div class="phead">[ context ] ' + hint('←→ or drag · Enter apply · Esc back') + '</div>' +
    '<div class="pgroup">' + esc(m.name) + ' ' + hint('on ' + esc(m.host)) + '</div>';
  if (steps.length) {
    html += '<div class="pslider"><input type="range" id="pk-range" min="0" max="' + (steps.length - 1) + '" step="1" value="' +
      picker.si + '"></div><div><span id="pk-val">' + steps[picker.si].toLocaleString() + '</span> tokens ' +
      hint('of ' + steps[steps.length - 1].toLocaleString() + ' max') + '</div>';
    if (m.provider === 'openrouter') html += '<div class="dim">OpenRouter runs the model at its own context size; this sets the context meter.</div>';
  }
  if (picker.note) html += '<div class="dim">' + esc(picker.note) + '</div>';
  pickerEl.innerHTML = html;
  const r = document.getElementById('pk-range');
  if (r) r.addEventListener('input', () => {  // drag: update the number only, so the slider isn't redrawn mid-drag
    picker.si = +r.value;
    document.getElementById('pk-val').textContent = steps[picker.si].toLocaleString();
  });
}

document.addEventListener('keydown', e => {
  if (!picker.open) return;
  if (picker.stage === 'list') {
    const n = pickerVisible().length;
    if (e.key === 'ArrowDown') picker.idx = n ? (picker.idx + 1) % n : 0;
    else if (e.key === 'ArrowUp') picker.idx = n ? (picker.idx - 1 + n) % n : 0;
    else if (e.key === 'Enter') { e.preventDefault(); pickerChoose(pickerVisible()[picker.idx]); return; }
    else if (e.key === 'Escape') { e.preventDefault(); pickerClose(); return; }
    else if (e.key === 'Backspace') { picker.filter = picker.filter.slice(0, -1); picker.idx = 0; }
    else if (e.key.length === 1 && !e.ctrlKey && !e.metaKey && !e.altKey) { picker.filter += e.key; picker.idx = 0; }
    else return;
  } else {
    if (e.key === 'ArrowLeft') picker.si = Math.max(0, picker.si - 1);
    else if (e.key === 'ArrowRight') picker.si = Math.min(picker.steps.length - 1, picker.si + 1);
    else if (e.key === 'Enter') { e.preventDefault(); if (picker.steps.length) pickerApply(); return; }
    else if (e.key === 'Escape') { picker.stage = 'list'; picker.note = ''; }
    else return;
  }
  e.preventDefault();
  renderPicker();
});
pickerEl.addEventListener('click', e => {
  const row = e.target.closest('.prow');
  if (row) { picker.idx = +row.dataset.i; pickerChoose(pickerVisible()[picker.idx]); }
});

// Knowledge graph of context/ (GET /api/graph: {nodes:[{id,label,type,attrs,sources}], edges:[{from,to,rel}]}),
// laid out once with a small force simulation and drawn as SVG. Hover a node for its links; click to pin it.
let G = null, gPinned = null;
const TYPE_COLOR = { doc: 'fg', section: 'dim', service: 'accent', term: 'violet' };  // node types agents/kg emits
const CHAR_W = 6.7;  // px per character of an 11px monospace label

async function drawGraph() {
  const svg = document.getElementById('graph');
  if (!G) {
    try { G = await (await fetch('/api/graph', { cache: 'no-store' })).json(); }
    catch (e) { G = { error: e.message, nodes: [], edges: [] }; }
    G.byId = Object.fromEntries(G.nodes.map(n => [n.id, n]));
    G.nbrs = Object.fromEntries(G.nodes.map(n => [n.id, new Set([n.id])]));
    G.edges.forEach(e => { G.nbrs[e.from].add(e.to); G.nbrs[e.to].add(e.from); });
  }
  const W = svg.clientWidth, H = svg.clientHeight;
  if (!W || !H || G.drawn === W + 'x' + H) return;
  G.drawn = W + 'x' + H;
  layoutGraph(G.nodes, G.edges, W, H);
  const deg = id => G.nbrs[id].size - 1;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  svg.innerHTML = '<g class="edges">' +
    G.edges.map((e, i) => {
      const a = G.byId[e.from], b = G.byId[e.to];
      return '<line data-i="' + i + '" class="' + (e.rel === 'mentions' ? 'mentions' : '') + '" x1="' + a.x.toFixed(1) +
        '" y1="' + a.y.toFixed(1) + '" x2="' + b.x.toFixed(1) + '" y2="' + b.y.toFixed(1) + '"/>';
    }).join('') + '</g><g class="rels"></g>' +
    G.nodes.map(n => {
      const shape = n.type === 'doc' ? '<rect x="-4.5" y="-4.5" width="9" height="9" class="t-doc"/>'
        : '<circle r="' + (3.5 + Math.min(deg(n.id), 10) * 0.35).toFixed(1) + '" class="t-' + esc(n.type) + '"/>';
      return '<g class="node" data-id="' + esc(n.id) + '" transform="translate(' + n.x.toFixed(1) + ',' + n.y.toFixed(1) + ')">' +
        shape + '<text x="9" y="4">' + esc(n.label) + '</text></g>';
    }).join('');
  graphFocus(gPinned);
}

// 1) Fruchterman-Reingold with no walls (walls pile nodes up along the edges); "mentions" pull at quarter strength
//    so the document hubs don't drag everything into one clump. 2) Scale the result to fill the box. 3) Push apart
//    any labels that overlap, using each label's real width, and keep every label inside the box.
function layoutGraph(nodes, edges, W, H) {
  const n = nodes.length, idx = {}, k = Math.sqrt(W * H / Math.max(n, 1));
  nodes.forEach((v, i) => { idx[v.id] = i; v.x = Math.cos(2 * Math.PI * i / n) * W / 3; v.y = Math.sin(2 * Math.PI * i / n) * H / 3; });
  const springs = {};
  edges.forEach(e => {
    const key = [idx[e.from], idx[e.to]].sort((a, b) => a - b).join(',');
    springs[key] = Math.max(springs[key] || 0, e.rel === 'mentions' ? 0.25 : 1);
  });
  const pairs = Object.entries(springs).map(([key, w]) => [...key.split(',').map(Number), w]);
  let t = k;
  for (let it = 0; it < 600; it++) {
    const dx = new Float64Array(n), dy = new Float64Array(n);
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const x = nodes[i].x - nodes[j].x, y = nodes[i].y - nodes[j].y, d2 = Math.max(x * x + y * y, 0.01), f = k * k / d2;
      dx[i] += x * f; dy[i] += y * f; dx[j] -= x * f; dy[j] -= y * f;
    }
    pairs.forEach(([i, j, w]) => {
      const x = nodes[i].x - nodes[j].x, y = nodes[i].y - nodes[j].y, f = w * Math.hypot(x, y) / k;
      dx[i] -= x * f; dy[i] -= y * f; dx[j] += x * f; dy[j] += y * f;
    });
    nodes.forEach((v, i) => {
      dx[i] -= v.x * 0.02; dy[i] -= v.y * 0.02;  // gentle pull to the centre
      const d = Math.max(Math.hypot(dx[i], dy[i]), 0.01), step = Math.min(d, t);
      v.x += dx[i] / d * step; v.y += dy[i] / d * step;
    });
    t = Math.max(t * 0.99, 0.5);
  }
  // fit: nodes into the box, leaving room on the right for an average label
  const pad = 16, lw = v => 9 + v.label.length * CHAR_W, rightRoom = 120;
  const xs = nodes.map(v => v.x), ys = nodes.map(v => v.y);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  nodes.forEach(v => {
    v.x = pad + (v.x - x0) / Math.max(x1 - x0, 1) * (W - 2 * pad - rightRoom);
    v.y = pad + (v.y - y0) / Math.max(y1 - y0, 1) * (H - 2 * pad);
  });
  // de-overlap: sweep top to bottom; a label that hits one already placed drops just below it
  const box = v => [v.x - 6, v.x + lw(v), v.y - 9, v.y + 9];
  const hits = (A, B) => Math.min(A[1], B[1]) > Math.max(A[0], B[0]) && Math.min(A[3], B[3]) > Math.max(A[2], B[2]);
  const placed = [];
  nodes.forEach(v => { v.x = Math.min(W - pad - lw(v), Math.max(pad, v.x)); });
  [...nodes].sort((a, b) => a.y - b.y).forEach(v => {
    for (let hit; (hit = placed.find(p => hits(box(p), box(v)))); ) v.y = hit.y + 19;
    placed.push(v);
  });
  // if the sweep pushed the bottom past the box, squeeze everything back in (spacing permitting)
  const yMax = Math.max(...nodes.map(v => v.y));
  if (yMax > H - pad) nodes.forEach(v => { v.y = pad + (v.y - pad) * (H - 2 * pad) / (yMax - pad); });
}

// Highlight a node, its neighbours and its edges (with their relation names); null: everything normal.
function graphFocus(id) {
  const svg = document.getElementById('graph');
  svg.querySelectorAll('.node').forEach(g => {
    g.classList.toggle('faded', !!id && !G.nbrs[id].has(g.dataset.id));
    g.classList.toggle('on', g.dataset.id === id);
  });
  const rels = [];
  svg.querySelectorAll('line').forEach(l => {
    const e = G.edges[l.dataset.i], hot = !!id && (e.from === id || e.to === id);
    l.classList.toggle('hot', hot);
    l.classList.toggle('faded-line', !!id && !hot);
    if (hot && e.rel !== 'mentions') {
      const a = G.byId[e.from], b = G.byId[e.to];
      rels.push('<text class="rel" text-anchor="middle" x="' + ((a.x + b.x) / 2).toFixed(1) + '" y="' + ((a.y + b.y) / 2 - 3).toFixed(1) + '">' + esc(e.rel) + '</text>');
    }
  });
  const relLayer = svg.querySelector('.rels');
  if (relLayer) relLayer.innerHTML = rels.join('');

  const info = document.getElementById('ginfo');
  if (G.error) { info.innerHTML = '<span class="bad">' + esc(G.error) + '</span>'; return; }
  if (!id) {
    const types = [...new Set(G.nodes.map(n => n.type))];
    info.innerHTML = G.nodes.length + ' nodes, ' + G.edges.length + ' edges <span class="dim">from context/: hover a node, click to pin</span>\n' +
      types.map(t => '<span class="sw" style="color:var(--' + (TYPE_COLOR[t] || 'fg') + ')">' + (t === 'doc' ? '■' : '●') + '</span> ' + esc(t)).join('   ') +
      '   <span class="dim">- - mentions</span>';
    return;
  }
  const n = G.byId[id], lbl = x => esc(G.byId[x].label);
  const attrs = Object.entries(n.attrs || {}).map(([k, v]) => '  <span class="k">' + esc(k) + '</span> ' + esc(v)).join('\n');
  const out = G.edges.filter(e => e.from === id && e.rel !== 'mentions').map(e => '  → ' + esc(e.rel) + ' ' + lbl(e.to));
  const inn = G.edges.filter(e => e.to === id && e.rel !== 'mentions').map(e => '  ← ' + esc(e.rel) + ' ' + lbl(e.from));
  const mentions = G.edges.filter(e => e.from === id && e.rel === 'mentions').map(e => lbl(e.to));
  info.innerHTML = '<span class="sw" style="color:var(--' + (TYPE_COLOR[n.type] || 'fg') + ')">' + (n.type === 'doc' ? '■' : '●') + '</span> ' +
    '<span class="k">' + esc(n.label) + '</span> <span class="dim">[' + esc(n.type) + ']</span>\n' +
    (attrs ? attrs + '\n' : '') + out.concat(inn).join('\n') + (out.length + inn.length ? '\n' : '') +
    (n.sources.length ? '<span class="dim">in: ' + esc(n.sources.join(', ')) + '</span>' : '') +
    (mentions.length ? '<span class="dim">mentions: ' + mentions.join(', ') + '</span>' : '');
}

const graphSvg = document.getElementById('graph');
graphSvg.addEventListener('mouseover', e => { const g = e.target.closest('.node'); if (g) graphFocus(g.dataset.id); });
graphSvg.addEventListener('mouseout', e => { if (e.target.closest('.node')) graphFocus(gPinned); });
graphSvg.addEventListener('click', e => {
  const g = e.target.closest('.node');
  gPinned = g && g.dataset.id !== gPinned ? g.dataset.id : null;
  graphFocus(gPinned);
});
// Left column tabs: Monitor (the status boxes), Graph, Agents (each agent's status and context use) and Workspace
// (whatever Max's agents last fetched: Athena's email list, Chronos' schedule). Click a tab, or the row to cycle.
const SIDE_TABS = [['monitor', 'Monitor'], ['graph', 'Graph'], ['agents', 'Agents'], ['agent', 'Workspace']];
let sideTab = 'graph';
try { sideTab = localStorage.getItem('sideTab') || 'graph'; } catch (e) {}
if (sideTab === 'mail') sideTab = 'agent';  // the old Mail tab

function showSide(k) {
  sideTab = SIDE_TABS.some(([t]) => t === k) ? k : 'graph';
  try { localStorage.setItem('sideTab', sideTab); } catch (e) {}
  document.getElementById('sidetabs').innerHTML = '[ ' + SIDE_TABS.map(([t, label]) =>
    '<span class="tab' + (t === sideTab ? ' on' : '') + '" data-side="' + t + '">' + label + '</span>')
    .join('<span class="dim"> · </span>') + ' ]';
  document.querySelector('.side').classList.toggle('monitor', sideTab === 'monitor');
  document.querySelectorAll('.wview').forEach(v => { v.hidden = v.id !== 'view-' + sideTab; });
  if (sideTab === 'graph') drawGraph();  // drawn at its on-screen size, so only once visible
  if (sideTab === 'agents') agentsTick();
}

// A view from Max's turn (a `view` event, or a stored tool step on reload): render it into the Agent Workspace.
// `show` switches to that tab (live turns); a restored view just updates it.
function renderView(v, show) {
  if (!v) return;
  const html = v.kind === 'email.list' ? mailView(v) : v.kind === 'schedule' ? scheduleView(v) : null;
  if (html === null) return;
  document.getElementById('view-agent').innerHTML = html;
  if (show) showSide('agent');
}

// Chronos' schedule: jobs (reminders), then calendar events from today on.
function scheduleView(v) {
  const jobs = v.jobs || [], events = v.events || [];
  const cal = v.calendar === 'ok' ? '' : ' · <span class="bad">calendar ' + esc(v.calendar || 'not connected') + '</span>';
  const jobRows = jobs.map(j =>
    '<div class="row' + (j.enabled ? '' : ' off') + '"><span class="n">' + esc(j.id.replace('job-', '')) + '</span><div>' +
    '<div class="top"><span class="from">' + esc(j.name) + '</span><span class="when">' +
    (j.enabled ? (j.next ? 'next ' + esc(j.next) : '') : (j.last_run ? 'done ' + esc(j.last_run) : 'paused')) + '</span></div>' +
    '<div class="subj">' + (j.prompt ? '<span class="dim">written fresh each time:</span> ' + esc(j.prompt) +
      (j.last_message ? '<div class="prev">last: ' + esc(j.last_message) + '</div>' : '') : esc(j.message || '')) +
    '</div><div class="prev">' + esc(j.when) +
    (j.runs ? ' · ran ' + j.runs + 'x' : '') + '</div></div></div>').join('');
  let day = '';
  const evRows = events.map(e => {
    const d = String(e.start || '').split(',')[0].replace(' (all day)', '');
    const head = d !== day ? '<div class="day">' + esc(d) + '</div>' : '';
    day = d;
    const time = e.all_day ? 'all day' : String(e.start || '').split(', ')[1] || '';
    const end = !e.all_day && e.end ? ' - ' + (String(e.end).split(', ')[1] || '') : '';
    return head + '<div class="row' + (e.past ? ' off' : '') + '"><span class="n">' + (e.recurring ? '&#8635;' : '') + '</span><div>' +
      '<div class="top"><span class="subj">' + esc(e.title) + '</span><span class="when">' + esc(time + end) + '</span></div>' +
      '<div class="prev">' + esc(e.calendar || '') + (e.location ? ' · ' + esc(e.location) : '') + '</div></div></div>';
  }).join('');
  return '<div class="head"><span class="k">Schedule</span> · ' + jobs.filter(j => j.enabled).length + ' active jobs · ' +
    events.length + ' events in ' + (v.days || 14) + ' days · by ' + esc(v.by || 'chronos') + (v.at ? ' at ' + esc(v.at) : '') + cal + '</div>' +
    (v.jobs_error ? '<div class="empty bad">' + esc(v.jobs_error) + '</div>' : '') +
    '<div class="sect">Jobs</div>' + (jobRows || '<div class="empty">No jobs.</div>') +
    '<div class="sect">Calendar</div>' + (evRows || '<div class="empty">No events.</div>');
}

// Athena's email list.
function mailView(v) {
  const acct = v.account === 'gmail' ? 'Gmail' : v.account === 'both' ? 'Gmail + Outlook' : 'Outlook', emails = v.emails || [];
  const rows = emails.map((m, i) =>
    '<div class="row' + (m.unread ? ' unread' : '') + '"><span class="n">' + (i + 1) + '</span><div>' +
    '<div class="top"><span class="from">' + esc(m.from || '?') + '</span><span class="when">' +
    (m.attachment ? '[attachment] ' : '') + esc(m.when || '') + '</span></div>' +
    '<div class="subj">' + esc(m.subject || '(no subject)') + '</div>' +
    (m.preview ? '<div class="prev">' + esc(m.preview) + '</div>' : '') + '</div></div>').join('');
  return '<div class="head"><span class="k">' + acct + '</span> · ' + emails.length + ' newest' +
    (v.unread_only ? ' unread' : '') + ' · fetched by ' + esc(v.by || 'athena') + (v.at ? ' at ' + esc(v.at) : '') +
    ' · <span title="unread in bold, numbered in blue">' + emails.filter(m => m.unread).length + ' unread</span></div>' +
    (rows || '<div class="empty">No emails.</div>');
}
// Agents tab: every agent (Max, then the sub-agents on the board) with its model, whether it's running, and how much
// of its context window is in use: Max's whole conversation; a sub-agent's latest model call (each of its requests
// starts a fresh window). Refreshed every 3 s while the tab is shown.
function ago(sqlTime) {  // "YYYY-MM-DD HH:MM:SS" in UTC (SQLite CURRENT_TIMESTAMP) -> "3 min ago"
  if (!sqlTime) return '';
  const t = Date.parse(sqlTime.replace(' ', 'T') + (sqlTime.endsWith('Z') ? '' : 'Z'));
  if (isNaN(t)) return '';
  const s = Math.max(0, (Date.now() - t) / 1000);
  return s < 60 ? 'just now' : s < 3600 ? Math.round(s / 60) + ' min ago' : s < 86400 ? Math.round(s / 3600) + ' h ago' : Math.round(s / 86400) + ' d ago';
}
function agentsView(list) {
  const k = n => n >= 1000 ? (n / 1000).toFixed(1) + 'k' : String(n);
  return list.map(a => {
    const used = a.ctx_used, max = a.ctx_max || 0, p = used != null && max ? Math.min(100, 100 * used / max) : 0;
    const level = p > 85 ? 'bad' : p > 60 ? 'warn' : '';
    const ctx = used == null ? '<span class="dim">no model call yet</span>'
      : Math.round(p) + '% <span class="dim">' + k(used) + ' / ' + k(max) + ' tokens</span>';
    return '<div class="agent ' + esc(a.name) + '">' +
      '<div class="top"><span class="name">' + esc(a.name) + '</span><span class="' + (a.online ? 'ok' : 'bad') + '">' +
      (a.online ? '● online' : '○ offline') + '</span></div>' +
      '<div class="role">' + esc(a.role || '') + '</div>' +
      '<div class="ctx">context ' + ctx + '<span class="track"><span class="fill ' + level + '" style="width:' + p.toFixed(1) + '%"></span></span></div>' +
      '<div class="meta">' + esc(a.model || '') + ' · window: ' + esc(a.window || '') + (a.ctx_at ? ' · last used ' + ago(a.ctx_at) : '') + '</div>' +
      '</div>';
  }).join('');
}
async function agentsTick() {
  if (sideTab !== 'agents') return;
  try {
    const d = await (await fetch('/api/agents', { cache: 'no-store' })).json();
    document.getElementById('view-agents').innerHTML = d.agents ? agentsView(d.agents) : '<div class="empty bad">' + esc(d.error || 'no data') + '</div>';
  } catch (e) {}
}
setInterval(agentsTick, 3000);

// The left column's tab after (step 1) or before (step -1) the current one, wrapping around.
const sideStep = step => SIDE_TABS[(SIDE_TABS.findIndex(([s]) => s === sideTab) + step + SIDE_TABS.length) % SIDE_TABS.length][0];
document.getElementById('sidetabs').addEventListener('click', e => {
  const t = e.target.closest('[data-side]');
  showSide(t ? t.dataset.side : sideStep(1));
});
// Tab cycles the left column's tabs (Shift+Tab goes back), wherever the focus is; the chat input keeps its focus.
// Not while the /model picker is open (it has the keyboard).
document.addEventListener('keydown', e => {
  if (e.key !== 'Tab' || e.ctrlKey || e.metaKey || e.altKey || e.isComposing || picker.open) return;
  e.preventDefault();
  showSide(sideStep(e.shiftKey ? -1 : 1));
});

window.addEventListener('resize', () => { if (sideTab === 'graph') drawGraph(); });
showPane(pane);
showSide(sideTab);  // last: the graph code above must be defined first
