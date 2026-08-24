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

    function renderMarkdown(text) {
        const escaped = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
        return escaped
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.+?)\*/g, '<em>$1</em>')
            .replace(/`(.+?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
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
        messagesEl.appendChild(messageEl);
        messagesEl.scrollTop = messagesEl.scrollHeight;
        return messageEl;
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

    function addMessage(author, content, thinking, timestamp, toolCalls) {
        const messageEl = createMessageEl(author, timestamp);
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
    }

    async function sendMessage() {
        const message = inputField.value.trim();
        if (!message) return;
        addMessage('You', message);
        inputField.value = '';

        const messageEl = createMessageEl('Max');
        let contentText = '';
        let liveThinkingEl = null;

        try {
            const response = await fetch('/max/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_input: message })
            });
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
                    }
                    messagesEl.scrollTop = messagesEl.scrollHeight;
                }
            }
        } catch (error) {
            console.error('Error sending message:', error);
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
                addMessage(turn.role === 'user' ? 'You' : 'Max', turn.content, turn.thinking, turn.timestamp, turn.tool_calls);
            });
        } catch (error) {
            console.error('Error loading history:', error);
        }
    }

    loadHistory();

    function renderJobs(containerEl, jobs, detailFn) {
        containerEl.innerHTML = '';
        if (!jobs || jobs.length === 0) {
            containerEl.innerHTML = '<div class="job-empty">No jobs</div>';
            return;
        }
        jobs.forEach((job) => {
            const jobEl = document.createElement('div');
            jobEl.className = 'job';
            jobEl.innerHTML = `
                <span class="job-name"></span>
                <span class="job-detail"></span>
            `;
            jobEl.querySelector('.job-name').textContent = job.name;
            jobEl.querySelector('.job-detail').textContent = detailFn(job);
            containerEl.appendChild(jobEl);
        });
    }

    async function loadJobs() {
        try {
            const response = await fetch('/max/jobs');
            const jobs = await response.json();
            renderJobs(document.querySelector('.scheduled-jobs'), jobs.scheduled, (job) => `${job.run_at} — ${job.prompt}`);
            renderJobs(document.querySelector('.cron-jobs'), jobs.cron, (job) => `${job.schedule} — ${job.prompt}`);
        } catch (error) {
            console.error('Error loading jobs:', error);
        }
    }

    loadJobs();
    setInterval(loadJobs, 5000);

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
});
