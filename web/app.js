window.onSpotifyWebPlaybackSDKReady = () => {
    const player = new Spotify.Player({
        name: 'Max',
        getOAuthToken: async (callback) => {
            const response = await fetch('/max/spotify/token');
            const data = await response.json();
            callback(data.access_token);
        },
        volume: 0.5
    });

    player.addListener('ready', ({ device_id }) => {
        fetch('/max/spotify/transfer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ device_id })
        });
    });

    player.connect();
};

document.addEventListener('DOMContentLoaded', () => {
    const inputField = document.querySelector('.chat-input-bar input');
    const messagesEl = document.querySelector('.messages');

    const MODEL_STORAGE_KEY = 'max-model';

    // --- Model picker (custom dropdown, so the popup can be rounded/sized to fit) ---

    const modelPicker = document.querySelector('.model-picker');
    const modelPickerButton = modelPicker.querySelector('.model-picker-button');
    const modelPickerLabel = modelPicker.querySelector('.model-picker-label');
    const modelPickerList = modelPicker.querySelector('.model-picker-list');
    let selectedModelKey = null;
    let modelOptions = [];

    function renderModelOptions() {
        modelPickerList.replaceChildren();
        modelOptions.forEach(({ key, label }) => {
            const optionEl = document.createElement('li');
            optionEl.className = 'model-picker-option' + (key === selectedModelKey ? ' selected' : '');
            optionEl.setAttribute('role', 'option');
            optionEl.setAttribute('aria-selected', String(key === selectedModelKey));
            optionEl.textContent = label;
            optionEl.addEventListener('click', () => selectModel(key));
            modelPickerList.appendChild(optionEl);
        });
    }

    function selectModel(key) {
        selectedModelKey = key;
        const selected = modelOptions.find((o) => o.key === key);
        modelPickerLabel.textContent = selected ? selected.label : '';
        localStorage.setItem(MODEL_STORAGE_KEY, key);
        renderModelOptions();
        closeModelPicker();
        loadContextUsage();
    }

    function openModelPicker() {
        modelPickerList.hidden = false;
        modelPickerButton.setAttribute('aria-expanded', 'true');
    }

    function closeModelPicker() {
        modelPickerList.hidden = true;
        modelPickerButton.setAttribute('aria-expanded', 'false');
    }

    modelPickerButton.addEventListener('click', (event) => {
        event.stopPropagation();
        if (modelPickerList.hidden) openModelPicker();
        else closeModelPicker();
    });

    document.addEventListener('click', closeModelPicker);

    async function loadModelOptions() {
        try {
            const response = await fetch('/max/models');
            const { default: defaultKey, options } = await response.json();
            modelOptions = options;
            const saved = localStorage.getItem(MODEL_STORAGE_KEY);
            selectedModelKey = options.some((o) => o.key === saved) ? saved : defaultKey;
            const selected = modelOptions.find((o) => o.key === selectedModelKey);
            modelPickerLabel.textContent = selected ? selected.label : '';
            renderModelOptions();
            loadContextUsage();
        } catch (error) {
            console.error('Error loading model options:', error);
        }
    }

    loadModelOptions();

    // --- Context usage bar ---

    const contextUsageFill = document.querySelector('.context-usage-fill');
    const contextUsageLabel = document.querySelector('.context-usage-label');

    async function loadContextUsage() {
        if (!selectedModelKey) return;
        try {
            const response = await fetch(`/max/context-usage?model=${encodeURIComponent(selectedModelKey)}`);
            const { percent_remaining, used_tokens, context_length } = await response.json();
            contextUsageFill.style.width = `${percent_remaining}%`;
            contextUsageFill.classList.toggle('warn', percent_remaining <= 50 && percent_remaining > 20);
            contextUsageFill.classList.toggle('critical', percent_remaining <= 20);
            contextUsageLabel.textContent = `${percent_remaining}% context remaining`;
            contextUsageLabel.title = `${used_tokens.toLocaleString()} / ${context_length.toLocaleString()} tokens (estimated)`;
        } catch (error) {
            console.error('Error loading context usage:', error);
        }
    }

    setInterval(loadContextUsage, 20000);

    function renderInlineMarkdown(text) {
        const escaped = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
        return escaped
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.+?)\*/g, '<em>$1</em>')
            .replace(/`(.+?)`/g, '<code>$1</code>');
    }

    function renderMarkdown(text) {
        return renderInlineMarkdown(text).replace(/\n/g, '<br>');
    }

    // Block-level renderer for the Editor tab's Preview — headers and lists on top of
    // renderMarkdown's inline formatting, since context/*.md files use both.
    function renderMarkdownDocument(text) {
        let html = '';
        let inList = false;
        const closeList = () => { if (inList) { html += '</ul>'; inList = false; } };
        text.split('\n').forEach((line) => {
            const headerMatch = line.match(/^(#{1,4})\s+(.*)/);
            const listMatch = line.match(/^[-*]\s+(.*)/);
            if (headerMatch) {
                closeList();
                const level = headerMatch[1].length;
                html += `<h${level}>${renderInlineMarkdown(headerMatch[2])}</h${level}>`;
            } else if (listMatch) {
                if (!inList) { html += '<ul>'; inList = true; }
                html += `<li>${renderInlineMarkdown(listMatch[1])}</li>`;
            } else if (line.trim() === '') {
                closeList();
            } else {
                closeList();
                html += `<p>${renderInlineMarkdown(line)}</p>`;
            }
        });
        closeList();
        return html;
    }

    function formatTimestamp(date) {
        return date.toLocaleString('en-US', {
            month: 'long',
            day: '2-digit',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            hour12: true
        });
    }

    function createMessageEl(author, timestamp) {
        const messageEl = document.createElement('div');
        messageEl.className = `message ${author === 'You' ? 'user' : 'assistant'}`;
        const avatar = author === 'Max'
            ? '<img class="avatar" src="/assets/max.png" alt="Max">'
            : '<img class="avatar" src="/assets/me.jpg" alt="You">';
        messageEl.innerHTML = `
            ${avatar}
            <div class="message-body">
                <span class="timestamp"></span>
                <span class="content"></span>
            </div>
        `;
        messageEl.querySelector('.timestamp').textContent = timestamp || formatTimestamp(new Date());
        messageEl.querySelector('.message-body').appendChild(createDeleteButton(messageEl));
        messagesEl.appendChild(messageEl);
        messagesEl.scrollTop = messagesEl.scrollHeight;
        return messageEl;
    }

    function createDeleteButton(messageEl) {
        const button = document.createElement('button');
        button.className = 'message-delete';
        button.title = 'Delete this message';
        button.setAttribute('aria-label', 'Delete this message');
        button.textContent = '\u00d7';
        button.addEventListener('click', async () => {
            const turnId = messageEl.dataset.turnId;
            if (!turnId) {
                // Never persisted (e.g. a failed send) — just drop it from the view.
                messageEl.remove();
                return;
            }
            if (button.classList.contains('confirming')) {
                button.disabled = true;
                try {
                    const response = await fetch(`/max/history/${turnId}`, { method: 'DELETE' });
                    const result = await response.json();
                    if (result.deleted) {
                        messageEl.remove();
                    } else {
                        button.disabled = false;
                        button.classList.remove('confirming');
                        console.error('Turn not found on server:', turnId);
                    }
                } catch (error) {
                    button.disabled = false;
                    button.classList.remove('confirming');
                    console.error('Error deleting turn:', error);
                }
                return;
            }
            // Two-step: first click arms, second click deletes. Cheaper than a modal and
            // still guards against a stray click destroying history.
            button.classList.add('confirming');
            button.textContent = 'delete?';
            setTimeout(() => {
                if (button.classList.contains('confirming')) {
                    button.classList.remove('confirming');
                    button.textContent = '\u00d7';
                }
            }, 3000);
        });
        return button;
    }

    function createCollapsible(className, headerText) {
        const wrapper = document.createElement('span');
        wrapper.className = className;
        const toggle = document.createElement('span');
        toggle.className = 'collapsible-toggle';
        const arrow = document.createElement('span');
        arrow.className = 'collapsible-arrow';
        arrow.textContent = '▶';
        toggle.appendChild(arrow);
        toggle.appendChild(document.createTextNode(' ' + headerText));
        const body = document.createElement('span');
        body.className = 'collapsible-body';
        body.style.display = 'none';
        toggle.addEventListener('click', () => {
            const collapsed = body.style.display === 'none';
            body.style.display = collapsed ? 'block' : 'none';
            arrow.textContent = collapsed ? '▼' : '▶';
        });
        wrapper.appendChild(toggle);
        wrapper.appendChild(body);
        return { wrapper, body };
    }

    function getReasoningEl(messageEl) {
        let reasoningEl = messageEl.querySelector('.reasoning');
        if (!reasoningEl) {
            reasoningEl = document.createElement('span');
            reasoningEl.className = 'reasoning';
            messageEl.querySelector('.content').before(reasoningEl);
        }
        return reasoningEl;
    }

    function appendToolCallBlock(reasoningEl, toolCall) {
        const toolCallEl = document.createElement('span');
        toolCallEl.className = 'tool-call';
        const callLabel = document.createElement('b');
        callLabel.textContent = 'Tool Call:';
        toolCallEl.appendChild(callLabel);
        toolCallEl.appendChild(document.createTextNode(
            ` ${toolCall.name}(${JSON.stringify(toolCall.arguments)})`
        ));
        toolCallEl.appendChild(document.createElement('br'));
        const resultLabel = document.createElement('b');
        resultLabel.textContent = 'Result:';
        toolCallEl.appendChild(resultLabel);
        toolCallEl.appendChild(document.createTextNode(
            ` ${JSON.stringify(toolCall.output)}`
        ));
        reasoningEl.appendChild(toolCallEl);
    }

    function collapseReasoning(messageEl) {
        const reasoningEl = messageEl.querySelector('.reasoning');
        if (!reasoningEl) return;
        const { wrapper, body } = createCollapsible('reasoning-collapsed', 'Thought process');
        while (reasoningEl.firstChild) {
            body.appendChild(reasoningEl.firstChild);
        }
        reasoningEl.replaceWith(wrapper);
    }

    function addMessage(author, content, thinking, timestamp, toolCalls, turnId) {
        const messageEl = createMessageEl(author, timestamp);
        if (turnId) messageEl.dataset.turnId = turnId;
        if (thinking || (toolCalls || []).length) {
            const reasoningEl = getReasoningEl(messageEl);
            if (thinking) {
                const thinkingEl = document.createElement('span');
                thinkingEl.className = 'thinking';
                thinkingEl.textContent = thinking;
                reasoningEl.appendChild(thinkingEl);
            }
            (toolCalls || []).forEach((toolCall) => appendToolCallBlock(reasoningEl, toolCall));
            collapseReasoning(messageEl);
        }
        messageEl.querySelector('.content').innerHTML = renderMarkdown(content);
        messagesEl.scrollTop = messagesEl.scrollHeight;
        return messageEl;
    }

    async function sendMessage() {
        const message = inputField.value.trim();
        if (!message) return;
        activateTab('chat');
        addMessage('You', message);
        inputField.value = '';

        const messageEl = createMessageEl('Max');
        try {
            const response = await fetch('/max/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_input: message, model: selectedModelKey })
            });
            await consumeStream(response, messageEl);
        } catch (error) {
            console.error('Error sending message:', error);
        }
    }

    async function consumeStream(response, messageEl) {
        let contentText = '';
        let liveThinkingEl = null;
        {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();
                for (const line of lines) {
                    if (!line.trim()) continue;
                    const event = JSON.parse(line);
                    if (event.type === 'thinking') {
                        if (!liveThinkingEl) {
                            liveThinkingEl = document.createElement('span');
                            liveThinkingEl.className = 'thinking';
                            getReasoningEl(messageEl).appendChild(liveThinkingEl);
                        }
                        liveThinkingEl.textContent += event.delta;
                    } else if (event.type === 'tool_call') {
                        appendToolCallBlock(getReasoningEl(messageEl), event);
                        liveThinkingEl = null;
                    } else if (event.type === 'content') {
                        contentText += event.delta;
                        messageEl.querySelector('.content').innerHTML = renderMarkdown(contentText);
                    } else if (event.type === 'done') {
                        collapseReasoning(messageEl);
                        messageEl.querySelector('.timestamp').textContent = event.turn.timestamp;
                        if (event.turn.id) messageEl.dataset.turnId = event.turn.id;
                        loadContextUsage();
                    } else if (event.type === 'error') {
                        console.error('Generation error:', event.message);
                        messageEl.querySelector('.content').textContent =
                            contentText || `[${event.message}]`;
                    }
                    messagesEl.scrollTop = messagesEl.scrollHeight;
                }
            }
        }
    }

    inputField.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            sendMessage();
        }
    });

    document.querySelector('.send-button').addEventListener('click', sendMessage);

    async function loadHistory() {
        try {
            const response = await fetch('/max/history');
            const turns = await response.json();
            turns.forEach((turn) => {
                addMessage(turn.role === 'user' ? 'You' : 'Max', turn.content, turn.thinking, turn.timestamp, turn.tool_calls, turn.id);
            });
        } catch (error) {
            console.error('Error loading history:', error);
        }
    }

    async function reattachIfGenerating() {
        // A reload mid-response used to abandon the reply. The server keeps generating, so
        // pick the stream back up and render the rest into a fresh bubble.
        try {
            const status = await fetch('/max/chat/active').then((r) => r.json());
            if (!status.active) return;
            const response = await fetch('/max/chat/attach');
            if (!response.ok || !response.body) return;
            const messageEl = createMessageEl('Max');
            await consumeStream(response, messageEl);
        } catch (error) {
            console.error('Error reattaching to generation:', error);
        }
    }

    loadHistory().then(reattachIfGenerating);

    // --- Schedule presets, shared with the form's dropdown ---

    const SCHEDULE_LABELS = {
        '0 9 * * *': 'Every day at 9:00 AM',
        '0 8-23 * * *': 'Every hour, 8 AM – 11 PM',
        '15 7 * * 1-5': 'Weekdays at 7:15 AM',
        '0 18 * * 1-5': 'Weekdays at 6:00 PM',
        '0 12 * * *': 'Every day at noon',
        '0 22 * * *': 'Every day at 10:00 PM',
        '0 10 * * 6': 'Saturdays at 10:00 AM',
        '0 10 1 * *': 'Monthly, 1st at 10:00 AM'
    };

    const DAY_NAMES = { 0: 'Sun', 1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat' };

    function describeSchedule(expr) {
        if (SCHEDULE_LABELS[expr]) return SCHEDULE_LABELS[expr];
        // Best-effort plain English for expressions typed by hand or written by Max.
        const parts = (expr || '').trim().split(/\s+/);
        if (parts.length !== 5) return expr;
        const [min, hour, dom, month, dow] = parts;
        if (!/^\d+$/.test(min) || !/^\d+$/.test(hour)) return expr;
        const time = formatClock(Number(hour), Number(min));
        if (dom === '*' && month === '*' && dow === '*') return `Every day at ${time}`;
        if (dom === '*' && month === '*' && /^\d(-\d)?$/.test(dow)) {
            const [from, to] = dow.split('-');
            const range = to ? `${DAY_NAMES[from]}–${DAY_NAMES[to]}` : DAY_NAMES[from];
            return `${range} at ${time}`;
        }
        if (/^\d+$/.test(dom) && month === '*') return `Monthly, day ${dom} at ${time}`;
        if (/^\d+$/.test(dom) && /^\d+$/.test(month)) {
            return `Yearly on ${month}/${dom} at ${time}`;
        }
        return expr;
    }

    function formatClock(hour, minute) {
        const suffix = hour < 12 ? 'AM' : 'PM';
        const h = hour % 12 === 0 ? 12 : hour % 12;
        return `${h}:${String(minute).padStart(2, '0')} ${suffix}`;
    }

    function describeRunAt(runAt) {
        const parsed = new Date(String(runAt).replace(' ', 'T'));
        if (isNaN(parsed)) return runAt;
        return parsed.toLocaleString('en-US', {
            month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit'
        });
    }

    function renderJobs(containerEl, jobs, detailFn) {
        containerEl.replaceChildren();
        if (!jobs || jobs.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'job-empty';
            empty.textContent = 'No jobs';
            containerEl.appendChild(empty);
            return;
        }
        jobs.forEach((job) => {
            const jobEl = document.createElement('div');
            jobEl.className = 'job';
            jobEl.innerHTML = `
                <span class="job-name"></span>
                <span class="job-detail"></span>
                <span class="job-actions">
                    <button class="job-edit" title="Edit">edit</button>
                    <button class="job-delete" title="Delete">\u00d7</button>
                </span>
            `;
            jobEl.querySelector('.job-name').textContent = job.name;
            jobEl.querySelector('.job-detail').textContent = detailFn(job);
            jobEl.title = job.prompt || '';
            jobEl.querySelector('.job-edit').addEventListener('click', () => openJobForm(job));
            jobEl.querySelector('.job-delete').addEventListener('click', (event) =>
                confirmDeleteJob(event.currentTarget, job)
            );
            containerEl.appendChild(jobEl);
        });
    }

    async function confirmDeleteJob(button, job) {
        if (!button.classList.contains('confirming')) {
            button.classList.add('confirming');
            button.textContent = 'delete?';
            setTimeout(() => {
                if (button.classList.contains('confirming')) {
                    button.classList.remove('confirming');
                    button.textContent = '\u00d7';
                }
            }, 3000);
            return;
        }
        button.disabled = true;
        try {
            const result = await fetch(`/max/jobs/${job.id}`, { method: 'DELETE' })
                .then((r) => r.json());
            if (result.ok) {
                loadJobs();
            } else {
                button.disabled = false;
                button.classList.remove('confirming');
                button.textContent = '\u00d7';
                console.error('Delete failed:', result.error);
            }
        } catch (error) {
            button.disabled = false;
            button.classList.remove('confirming');
            button.textContent = '\u00d7';
            console.error('Error deleting job:', error);
        }
    }

    // --- Add / edit form ---

    const jobForm = document.querySelector('.job-form');
    const presetSelect = jobForm.querySelector('.job-field-preset');
    const scheduleInput = jobForm.querySelector('.job-field-schedule');
    const runAtInput = jobForm.querySelector('.job-field-runat');
    const nameInput = jobForm.querySelector('.job-field-name');
    const promptInput = jobForm.querySelector('.job-field-prompt');
    const formError = jobForm.querySelector('.job-form-error');
    let editingJobId = null;

    function syncPresetFields() {
        const value = presetSelect.value;
        scheduleInput.hidden = value !== '__custom__';
        runAtInput.hidden = value !== '__once__';
    }

    presetSelect.addEventListener('change', syncPresetFields);

    function openJobForm(job) {
        editingJobId = job ? job.id : null;
        formError.textContent = '';
        nameInput.value = job ? job.name : '';
        promptInput.value = job ? job.prompt : '';
        if (job && job.run_at) {
            presetSelect.value = '__once__';
            runAtInput.value = String(job.run_at).replace(' ', 'T');
        } else if (job && job.schedule) {
            if (SCHEDULE_LABELS[job.schedule]) {
                presetSelect.value = job.schedule;
            } else {
                presetSelect.value = '__custom__';
                scheduleInput.value = job.schedule;
            }
        } else {
            presetSelect.selectedIndex = 0;
            scheduleInput.value = '';
            runAtInput.value = '';
        }
        syncPresetFields();
        jobForm.querySelector('.job-save').textContent = job ? 'Update' : 'Create';
        jobForm.hidden = false;
        nameInput.focus();
    }

    function closeJobForm() {
        jobForm.hidden = true;
        editingJobId = null;
        formError.textContent = '';
    }

    document.querySelector('.job-add-button').addEventListener('click', () => {
        if (jobForm.hidden) openJobForm(null);
        else closeJobForm();
    });

    jobForm.querySelector('.job-cancel').addEventListener('click', closeJobForm);

    jobForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const payload = {
            name: nameInput.value.trim(),
            prompt: promptInput.value.trim()
        };
        const preset = presetSelect.value;
        if (preset === '__once__') {
            // datetime-local gives "YYYY-MM-DDTHH:MM"; the API wants a space.
            payload.run_at = runAtInput.value.replace('T', ' ').slice(0, 16);
        } else if (preset === '__custom__') {
            payload.schedule = scheduleInput.value.trim();
        } else {
            payload.schedule = preset;
        }
        const saveButton = jobForm.querySelector('.job-save');
        saveButton.disabled = true;
        formError.textContent = '';
        try {
            const url = editingJobId ? `/max/jobs/${editingJobId}` : '/max/jobs';
            const result = await fetch(url, {
                method: editingJobId ? 'PUT' : 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            }).then((r) => r.json());
            if (result.ok) {
                closeJobForm();
                loadJobs();
            } else {
                formError.textContent = result.error || 'Save failed';
            }
        } catch (error) {
            formError.textContent = 'Save failed';
            console.error('Error saving job:', error);
        } finally {
            saveButton.disabled = false;
        }
    });

    async function loadJobs() {
        try {
            const response = await fetch('/max/jobs');
            const jobs = await response.json();
            renderJobs(document.querySelector('.scheduled-jobs'), jobs.scheduled, (job) => describeRunAt(job.run_at));
            renderJobs(document.querySelector('.cron-jobs'), jobs.cron, (job) => describeSchedule(job.schedule));
        } catch (error) {
            console.error('Error loading jobs:', error);
        }
    }

    loadJobs();
    // Skip the refresh while the form is open, so polling can't clobber what's being typed.
    setInterval(() => {
        if (jobForm.hidden) loadJobs();
    }, 5000);

    async function loadNowPlaying() {
        try {
            const response = await fetch('/max/spotify/now-playing');
            const data = await response.json();
            const albumArt = document.querySelector('.spotify-container .album-art');
            const trackName = document.querySelector('.spotify-container .track-name');
            const artistName = document.querySelector('.spotify-container .artist-name');
            const iconPlay = document.querySelector('.spotify-play-pause .icon-play');
            const iconPause = document.querySelector('.spotify-play-pause .icon-pause');

            if (data.track) {
                albumArt.src = data.album_art || '';
                trackName.textContent = data.track;
                artistName.textContent = (data.artists || []).join(', ');
            } else {
                albumArt.src = '';
                trackName.textContent = 'Nothing playing';
                artistName.textContent = ' ';
            }
            iconPlay.style.display = data.is_playing ? 'none' : 'block';
            iconPause.style.display = data.is_playing ? 'block' : 'none';
        } catch (error) {
            console.error('Error loading now-playing:', error);
        }
    }

    async function spotifyAction(endpoint) {
        try {
            await fetch(`/max/spotify/${endpoint}`, { method: 'POST' });
            setTimeout(loadNowPlaying, 300);
        } catch (error) {
            console.error('Error sending Spotify action:', error);
        }
    }

    const spotifyMenu = document.querySelector('.spotify-menu');
    document.querySelector('.spotify-menu-button').addEventListener('click', (e) => {
        e.stopPropagation();
        spotifyMenu.classList.toggle('open');
    });
    document.addEventListener('click', () => spotifyMenu.classList.remove('open'));

    document.querySelector('.spotify-prev').addEventListener('click', () => spotifyAction('prev'));
    document.querySelector('.spotify-next').addEventListener('click', () => spotifyAction('next'));
    document.querySelector('.spotify-play-pause').addEventListener('click', () => {
        const isPlaying = document.querySelector('.spotify-play-pause .icon-pause').style.display !== 'none';
        spotifyAction(isPlaying ? 'pause' : 'resume');
    });

    loadNowPlaying();
    setInterval(loadNowPlaying, 5000);

    // --- Logs ---

    const logsOutput = document.querySelector('.logs-output');
    const logsStatus = document.querySelector('.logs-status');
    const followToggle = document.querySelector('.logs-follow-toggle');
    const MAX_LOG_LINES = 2000;
    let logSource = null;

    function setLogStatus(text, state) {
        logsStatus.textContent = text;
        logsStatus.className = 'logs-status' + (state ? ' ' + state : '');
    }

    function scrollLogsToBottom() {
        logsOutput.scrollTop = logsOutput.scrollHeight;
    }

    function levelOf(line) {
        if (/\bERROR\b|\bCRITICAL\b|Traceback/.test(line)) return 'level-error';
        if (/\bWARNING\b/.test(line)) return 'level-warning';
        if (/\bDEBUG\b/.test(line)) return 'level-debug';
        return 'level-info';
    }

    function appendLogLine(text) {
        const wasAtBottom =
            logsOutput.scrollHeight - logsOutput.scrollTop - logsOutput.clientHeight < 40;
        const el = document.createElement('div');
        el.className = 'log-line ' + levelOf(text);
        el.textContent = text;
        logsOutput.appendChild(el);
        while (logsOutput.childElementCount > MAX_LOG_LINES) {
            logsOutput.removeChild(logsOutput.firstElementChild);
        }
        if (followToggle.checked && wasAtBottom) scrollLogsToBottom();
    }

    function openLogStream() {
        if (logSource) return;
        setLogStatus('connecting…');
        logSource = new EventSource('/max/logs/stream');
        logSource.onopen = () => setLogStatus('live', 'live');
        logSource.onmessage = (event) => {
            try {
                appendLogLine(JSON.parse(event.data).line);
            } catch (error) {
                console.error('Bad log line:', error);
            }
        };
        logSource.onerror = () => {
            // EventSource retries on its own; just reflect the state.
            setLogStatus('reconnecting…', 'error');
        };
    }

    function closeLogStream() {
        if (!logSource) return;
        logSource.close();
        logSource = null;
        setLogStatus('paused');
    }

    document.querySelector('.logs-clear').addEventListener('click', () => {
        logsOutput.replaceChildren();
    });

    followToggle.addEventListener('change', () => {
        if (followToggle.checked) scrollLogsToBottom();
    });

    // --- Context files (list on the left; editing happens in the chat container's editor pane) ---

    const contextFileList = document.querySelector('.context-file-list');
    const editorEmpty = document.querySelector('.editor-empty');
    const editorContent = document.querySelector('.editor-content');
    const editorFilename = document.querySelector('.editor-filename');
    const editorTokens = document.querySelector('.editor-tokens');
    const editorFilenameInput = document.querySelector('.editor-filename-input');
    const editorTextarea = document.querySelector('.editor-textarea');
    const editorPreview = document.querySelector('.editor-preview');
    const editorError = document.querySelector('.editor-error');
    const editorSave = document.querySelector('.editor-save');
    const editorSubtabs = document.querySelectorAll('.editor-subtab');
    const contextConfirmOverlay = document.querySelector('.context-confirm-overlay');
    const contextConfirmFilename = document.querySelector('.context-confirm-filename');
    let contextEditingFilename = null;
    let contextOriginalContent = '';
    let isCreatingNewFile = false;

    async function loadContextFiles() {
        try {
            const response = await fetch('/max/context-files');
            const { files } = await response.json();
            contextFileList.replaceChildren();
            if (!files || files.length === 0) {
                const empty = document.createElement('div');
                empty.className = 'context-empty';
                empty.textContent = 'No .md files in context/';
                contextFileList.appendChild(empty);
                return;
            }
            files.forEach((filename) => {
                const fileEl = document.createElement('div');
                fileEl.className = 'context-file';
                fileEl.innerHTML = '<span class="context-file-name"></span>';
                fileEl.querySelector('.context-file-name').textContent = filename;
                fileEl.addEventListener('click', () => openContextEditor(filename));
                contextFileList.appendChild(fileEl);
            });
        } catch (error) {
            console.error('Error loading context files:', error);
        }
    }

    function setEditorSubtab(name) {
        editorSubtabs.forEach((btn) => btn.classList.toggle('active', btn.dataset.subtab === name));
        editorTextarea.hidden = name !== 'markdown';
        editorPreview.hidden = name !== 'preview';
        if (name === 'preview') editorPreview.innerHTML = renderMarkdownDocument(editorTextarea.value);
    }

    editorSubtabs.forEach((btn) => btn.addEventListener('click', () => setEditorSubtab(btn.dataset.subtab)));

    function updateSaveVisibility() {
        editorSave.hidden = isCreatingNewFile
            ? editorFilenameInput.value.trim() === ''
            : editorTextarea.value === contextOriginalContent;
    }

    function estimateTokens(text) {
        return Math.max(1, Math.round((text || '').length / 4));
    }

    function formatTokenCount(count) {
        return (count >= 1000 ? `${(count / 1000).toFixed(1)}k` : String(count)) + ' tokens';
    }

    function updateEditorTokens() {
        editorTokens.textContent = formatTokenCount(estimateTokens(editorTextarea.value));
    }

    editorTextarea.addEventListener('input', () => {
        updateSaveVisibility();
        updateEditorTokens();
    });
    editorFilenameInput.addEventListener('input', updateSaveVisibility);

    async function openContextEditor(filename) {
        try {
            const response = await fetch(`/max/context-files/${encodeURIComponent(filename)}`);
            const result = await response.json();
            if (!result.ok) {
                console.error('Failed to load context file:', result.error);
                return;
            }
            isCreatingNewFile = false;
            contextEditingFilename = filename;
            contextOriginalContent = result.content;
            editorFilename.textContent = filename;
            editorFilename.hidden = false;
            editorFilenameInput.hidden = true;
            editorTextarea.value = result.content;
            editorTokens.textContent = formatTokenCount(result.tokens);
            editorError.textContent = '';
            editorSave.hidden = true;
            editorEmpty.hidden = true;
            editorContent.hidden = false;
            setEditorSubtab('preview');
            activateTab('editor');
        } catch (error) {
            console.error('Error opening context file:', error);
        }
    }

    function openNewContextFile() {
        isCreatingNewFile = true;
        contextEditingFilename = null;
        contextOriginalContent = '';
        editorFilename.hidden = true;
        editorFilenameInput.hidden = false;
        editorFilenameInput.value = '';
        editorTextarea.value = '';
        updateEditorTokens();
        editorError.textContent = '';
        editorSave.hidden = true;
        editorEmpty.hidden = true;
        editorContent.hidden = false;
        setEditorSubtab('markdown');
        activateTab('editor');
        editorFilenameInput.focus();
    }

    document.querySelector('.context-file-add').addEventListener('click', openNewContextFile);

    editorSave.addEventListener('click', () => {
        const filename = isCreatingNewFile ? editorFilenameInput.value.trim() : contextEditingFilename;
        if (!filename) return;
        if (isCreatingNewFile && !filename.endsWith('.md')) {
            editorError.textContent = 'Filename must end in .md';
            return;
        }
        contextConfirmFilename.textContent = filename;
        contextConfirmOverlay.hidden = false;
    });

    document.querySelector('.context-confirm-cancel').addEventListener('click', () => {
        contextConfirmOverlay.hidden = true;
    });

    document.querySelector('.context-confirm-ok').addEventListener('click', async () => {
        const filename = isCreatingNewFile ? editorFilenameInput.value.trim() : contextEditingFilename;
        const content = editorTextarea.value;
        const saveButton = document.querySelector('.context-confirm-ok');
        saveButton.disabled = true;
        try {
            const result = await fetch(`/max/context-files/${encodeURIComponent(filename)}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content })
            }).then((r) => r.json());
            if (result.ok) {
                if (isCreatingNewFile) {
                    isCreatingNewFile = false;
                    contextEditingFilename = filename;
                    editorFilename.textContent = filename;
                    editorFilename.hidden = false;
                    editorFilenameInput.hidden = true;
                    loadContextFiles();
                }
                contextOriginalContent = content;
                editorSave.hidden = true;
                contextConfirmOverlay.hidden = true;
            } else {
                editorError.textContent = result.error || 'Save failed';
                contextConfirmOverlay.hidden = true;
            }
        } catch (error) {
            editorError.textContent = 'Save failed';
            contextConfirmOverlay.hidden = true;
            console.error('Error saving context file:', error);
        } finally {
            saveButton.disabled = false;
        }
    });

    loadContextFiles();

    // --- Tabs ---

    const tabs = document.querySelectorAll('.tab');
    const panels = document.querySelectorAll('.tab-panel');

    function activateTab(name) {
        tabs.forEach((tab) => {
            const isActive = tab.dataset.tab === name;
            tab.classList.toggle('active', isActive);
            tab.setAttribute('aria-selected', String(isActive));
        });
        panels.forEach((panel) => {
            const isActive = panel.dataset.panel === name;
            panel.classList.toggle('active', isActive);
            panel.hidden = !isActive;
        });
        if (name === 'logs') {
            openLogStream();
            scrollLogsToBottom();
        } else {
            closeLogStream();
        }
        if (name === 'chat') inputField.focus();
    }

    tabs.forEach((tab) => tab.addEventListener('click', () => activateTab(tab.dataset.tab)));

    // Don't hold a journalctl process open when the tab isn't visible.
    window.addEventListener('beforeunload', closeLogStream);
});
