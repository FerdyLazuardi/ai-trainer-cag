// ============================================================
// AUTO INJECT HTML ke body (untuk Moodle integration)
// ============================================================
(function injectChatWidget() {
    // Cek apakah sudah ada (hindari duplikat)
    if (document.getElementById("chat-toggle")) return;

    const html = `
        <button id="chat-toggle">
            <i id="chat-icon" class="fas fa-comment-dots"></i>
            <span id="chat-badge">1</span>
        </button>

        <div id="chat-box" class="animate__animated">
            <div id="chat-header">
                <div class="header-info">
                    <div class="online-dot"></div>
                    <div style="display:flex; flex-direction:column;">
                        <span>Ava AI Trainer</span>
                        <small style="font-size:11px; opacity:.8;">Biasanya membalas &lt; 1 menit</small>
                    </div>
                </div>
                <div style="display:flex; gap:22px; align-items:center;">
                    <i class="fas fa-trash-alt header-icon" onclick="clearChat()" title="Clear chat" style="cursor:pointer; font-size:14px; opacity:0.8;"></i>
                    <i class="fas fa-times header-icon" onclick="toggleChat()" title="Close chat" style="cursor:pointer; font-size:14px; opacity:0.8;"></i>
                </div>
            </div>
            <div id="chat-messages"></div>
            <div id="chat-input">
                <textarea id="prompt" rows="1" placeholder="Ketik pesan..." onkeydown="handleKey(event)"></textarea>
                <button class="send-btn" onclick="handleSendClick()">
                    <i class="fas fa-paper-plane"></i>
                </button>
            </div>
        </div>
    `;

    const wrapper = document.createElement("div");
    wrapper.innerHTML = html;
    document.body.appendChild(wrapper);
})();

// ============================================================
// INIT — tunggu DOM siap
// ============================================================
const chatBox = document.getElementById("chat-box");
const messages = document.getElementById("chat-messages");
const textarea = document.getElementById("prompt");
let introduced = false;
let isStreaming = false; // Prevent double-sends during streaming
let currentAbortController = null;

function setSendButtonState(streaming) {
    const btns = document.querySelectorAll(".send-btn");
    btns.forEach(btn => {
        const icon = btn.querySelector("i");
        if (icon) {
            if (streaming) {
                icon.className = "fas fa-stop"; // Stop icon
                btn.classList.add("cancel-mode");
            } else {
                icon.className = "fas fa-paper-plane"; // Send icon
                btn.classList.remove("cancel-mode");
            }
        }
    });
}

function handleSendClick() {
    if (isStreaming) {
        // Cancel the ongoing request
        if (currentAbortController) {
            currentAbortController.abort();
            currentAbortController = null;
        }
        isStreaming = false;
        setSendButtonState(false);
    } else {
        send();
    }
}

/* AKTIFKAN PULSE */
const toggleBtn = document.getElementById("chat-toggle");
if (toggleBtn) {
    toggleBtn.classList.add("pulse");
    toggleBtn.onclick = toggleChat;
}

// ============================================================
// TOGGLE CHAT
// ============================================================
function toggleChat() {
    const icon = document.getElementById("chat-icon");
    const toggleBtn = document.getElementById("chat-toggle");

    icon.classList.add("icon-animate");

    setTimeout(() => {
        if (chatBox.style.display === "none" || chatBox.style.display === "") {
            // ===== OPEN =====
            chatBox.style.display = "flex";
            chatBox.classList.remove("animate__fadeOutDown");
            chatBox.classList.add("animate__fadeInUp");

            // Sembunyikan bubble button saat chat dibuka
            toggleBtn.style.opacity = "0";
            toggleBtn.style.pointerEvents = "none";
            toggleBtn.classList.remove("pulse");

            const badge = document.getElementById("chat-badge");
            if (badge) badge.style.display = "none";

            if (!introduced) {
                loadHistory();
                introduced = true;
            }

        } else {
            // ===== CLOSE =====
            chatBox.classList.remove("animate__fadeInUp");
            chatBox.classList.add("animate__fadeOutDown");

            setTimeout(() => {
                chatBox.style.display = "none";
                // Tampilkan kembali bubble button saat chat ditutup
                toggleBtn.style.opacity = "1";
                toggleBtn.style.pointerEvents = "auto";
                toggleBtn.classList.add("pulse");
            }, 500);
        }

        icon.classList.remove("icon-animate");
    }, 200);
}

// ============================================================
// INTRO — sapa user pakai nama Moodle
// ============================================================
function showIntro() {
    const nama = (typeof MOODLE_USER_NAME !== 'undefined' && MOODLE_USER_NAME)
        ? MOODLE_USER_NAME.split(' ')[0]
        : 'A-Team';

    addAIResponse(
        `Hi **${nama}**! Aku **Ava**. ` +
        `Ada yang bisa aku bantu hari ini terkait materi Amarthapedia? 😊`
    );
}

async function loadHistory() {

    setTimeout(showIntro, 100);

    const baseUrl = (typeof API_BASE_URL !== 'undefined' && API_BASE_URL) ? API_BASE_URL : "";
    const headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true"
    };

    if (typeof MOODLE_JWT !== 'undefined' && MOODLE_JWT) {
        headers["Authorization"] = `Bearer ${MOODLE_JWT}`;
    }

    try {
        const sessionId = getSessionId();
        const res = await fetch(`${baseUrl}/api/v1/chat/history/${sessionId}`, {
            method: "GET",
            headers: headers
        });

        if (!res.ok) throw new Error("No history found");

        const history = await res.json();

        if (history && history.length > 0) {
            // Tunggu sedikit agar intro muncul duluan sebelum history
            setTimeout(() => {
                history.forEach(msg => {
                    const role = msg.role === 'user' ? 'user' : 'ai';
                    const content = msg.content || msg.text || "";
                    addMessage(content, role);
                });
            }, 300);
        }
    } catch (err) {
        console.error("Failed to load history:", err);
    }
}

async function clearChat() {
    if (confirm("Yakin mau hapus semua chat history?")) {
        const sessionId = getSessionId();
        const baseUrl = (typeof API_BASE_URL !== 'undefined' && API_BASE_URL) ? API_BASE_URL : "";
        const headers = {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true"
        };

        if (typeof MOODLE_JWT !== 'undefined' && MOODLE_JWT) {
            headers["Authorization"] = `Bearer ${MOODLE_JWT}`;
        }

        try {
            await fetch(`${baseUrl}/api/v1/chat/history/${sessionId}`, {
                method: 'DELETE',
                headers: headers
            });
            messages.innerHTML = '';
            introduced = false;
            showIntro();
        } catch (e) {
            console.error("Error clearing chat history:", e);
        }
    }
}

// ============================================================
// HELPERS
// ============================================================
function getTime() {
    return new Date().toLocaleTimeString("id-ID", { hour: "2-digit", minute: "2-digit" });
}

function addMessage(text, type) {
    const wrap = document.createElement("div");
    wrap.className = `msg ${type} animate__animated animate__zoomIn animate__faster`;

    const bubble = document.createElement("div");
    bubble.className = `bubble ${type}`;

    const formattedText = marked.parse(text);

    const tempDiv = document.createElement("div");
    tempDiv.innerHTML = formattedText;
    tempDiv.querySelectorAll("a").forEach(link => {
        link.setAttribute("target", "_blank");
        link.setAttribute("rel", "noopener noreferrer");
    });

    bubble.innerHTML = `
        <div class="content">${tempDiv.innerHTML}</div>
        <span class="time">${getTime()}</span>
    `;

    wrap.appendChild(bubble);
    messages.appendChild(wrap);
    messages.scrollTop = messages.scrollHeight;

    return wrap;
}

function showTyping() {
    const typingDiv = document.createElement("div");
    typingDiv.id = "typing-id";
    typingDiv.className = "msg ai animate__animated animate__fadeIn";
    typingDiv.innerHTML = `
        <div class="bubble ai">
            <div class="typing">
                <div class="dot"></div><div class="dot"></div><div class="dot"></div>
            </div>
        </div>
    `;
    messages.appendChild(typingDiv);
    messages.scrollTop = messages.scrollHeight;
}

function removeTyping() {
    const el = document.getElementById("typing-id");
    if (el) el.remove();
}

// Configure marked for safe, clean rendering
marked.setOptions({ breaks: true, gfm: true });

// ============================================================
// SESSION ID — pakai Moodle User ID kalau tersedia
// ============================================================
function getSessionId() {
    if (typeof MOODLE_USER_ID !== 'undefined' && MOODLE_USER_ID > 0) {
        const nama = (typeof MOODLE_USER_NAME !== 'undefined' && MOODLE_USER_NAME)
            ? MOODLE_USER_NAME.replace(/\s+/g, '_').toLowerCase()
            : "user";
        const dept = (typeof MOODLE_DEPT !== 'undefined' && MOODLE_DEPT)
            ? MOODLE_DEPT.toLowerCase()
            : "general";
        return `${nama}_${MOODLE_USER_ID}_${dept}`;
    }
    let sid = sessionStorage.getItem("ava_sid");
    if (!sid) {
        sid = "sid-" + Math.random().toString(36).substring(2, 9);
        sessionStorage.setItem("ava_sid", sid);
    }
    return sid;
}

function resetChat() {
    sessionStorage.removeItem("ava_sid");
    window.location.reload();
}

// ============================================================
// STREAMING BUBBLE HELPERS
// ============================================================

/**
 * Create an empty AI message bubble ready for streaming tokens into.
 * Returns { wrap, contentDiv, bubble } to allow progressive updates.
 */
function createStreamBubble() {
    let wrap = document.getElementById("typing-id");
    let bubble;

    const contentDiv = document.createElement("div");
    contentDiv.className = "content";

    const timeSpan = document.createElement("span");
    timeSpan.className = "time";
    timeSpan.textContent = getTime();

    if (wrap) {
        // Reuse existing bubble to prevent visual drop
        wrap.removeAttribute("id");
        bubble = wrap.querySelector(".bubble");
        if (bubble) {
            bubble.classList.add("streaming");
            const typingDiv = bubble.querySelector(".typing");
            if (typingDiv) typingDiv.remove();

            bubble.appendChild(contentDiv);
            bubble.appendChild(timeSpan);
        }
    } else {
        wrap = document.createElement("div");
        wrap.className = "msg ai";

        bubble = document.createElement("div");
        bubble.className = "bubble ai streaming";

        bubble.appendChild(contentDiv);
        bubble.appendChild(timeSpan);
        wrap.appendChild(bubble);
        messages.appendChild(wrap);
    }

    messages.scrollTop = messages.scrollHeight;

    return { wrap, contentDiv, bubble };
}

/**
 * Finalize the streaming bubble: remove cursor, remove streaming class.
 */
function finalizeStreamBubble(contentDiv, bubble, fullText) {
    bubble.classList.remove("streaming");

    // Final accurate render
    const tempDiv = document.createElement("div");
    tempDiv.innerHTML = marked.parse(fullText);
    tempDiv.querySelectorAll("a").forEach(link => {
        link.setAttribute("target", "_blank");
        link.setAttribute("rel", "noopener noreferrer");
    });
    contentDiv.innerHTML = tempDiv.innerHTML;
    messages.scrollTop = messages.scrollHeight;
}

// ============================================================
// SEND MESSAGE — True SSE Streaming
// ============================================================
async function send() {
    const text = textarea.value.trim();
    if (!text || isStreaming) return;

    const userMsgNode = addMessage(text, "user");
    textarea.value = "";
    textarea.style.height = "auto";

    showTyping();
    isStreaming = true;
    currentAbortController = new AbortController();
    setSendButtonState(true);
    let streamWrap = null;

    const baseUrl = (typeof API_BASE_URL !== 'undefined' && API_BASE_URL)
        ? API_BASE_URL
        : "";

    const headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true"
    };

    if (typeof MOODLE_JWT !== 'undefined' && MOODLE_JWT) {
        headers["Authorization"] = `Bearer ${MOODLE_JWT}`;
    }

    const body = JSON.stringify({
        query: text,
        conversation_id: getSessionId(),
        course_id: typeof MOODLE_COURSE_ID !== 'undefined' ? MOODLE_COURSE_ID : 0,
        course_name: typeof MOODLE_COURSE_NAME !== 'undefined' ? MOODLE_COURSE_NAME : ''
    });

    try {
        const res = await fetch(`${baseUrl}/api/v1/chat/stream`, {
            method: "POST",
            headers: headers,
            body: body,
            signal: currentAbortController.signal
        });

        if (!res.ok) {
            if (res.status === 429) {
                throw new Error("RATE_LIMIT");
            }
            throw new Error(`Server returned ${res.status}`);
        }

        // ── Init SSE reader ──
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        let contentDiv = null;
        let bubble = null;
        let _streamStarted = false;
        let _targetText = "";
        let _displayedText = "";
        let _streamActive = true;
        let _finalized = false;

        function startStreamBubble() {
            if (!_streamStarted) {
                const bubbleObj = createStreamBubble();
                streamWrap = bubbleObj.wrap;
                contentDiv = bubbleObj.contentDiv;
                bubble = bubbleObj.bubble;
                _streamStarted = true;
            }
        }

        function smoothStreamWorker() {
            if (!_streamStarted) {
                if (_streamActive) setTimeout(smoothStreamWorker, 20);
                return;
            }

            if (_displayedText.length < _targetText.length) {
                const remaining = _targetText.length - _displayedText.length;
                // Reveal chunk by chunk to catch up smoothly
                const chunkSize = Math.max(1, Math.floor(remaining / 4));
                _displayedText += _targetText.substring(_displayedText.length, _displayedText.length + chunkSize);

                // Add inline cursor BEFORE parsing markdown so it stays inside paragraph blocks <p>
                const renderText = _displayedText + '<span class="streaming-cursor-inline">▍</span>';

                const tempDiv = document.createElement("div");
                tempDiv.innerHTML = marked.parse(renderText);
                tempDiv.querySelectorAll("a").forEach(link => {
                    link.setAttribute("target", "_blank");
                    link.setAttribute("rel", "noopener noreferrer");
                });
                contentDiv.innerHTML = tempDiv.innerHTML;
                messages.scrollTop = messages.scrollHeight;
            }

            if (_streamActive || _displayedText.length < _targetText.length) {
                setTimeout(smoothStreamWorker, 20); // Fast but smooth 20ms frame
            } else if (!_finalized) {
                _finalized = true;
                finalizeStreamBubble(contentDiv, bubble, _targetText || "Wah, Ava bingung nih jawabnya. Coba tanya hal lain yuk! 😊");
            }
        }

        smoothStreamWorker();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            let currentEventType = "";

            for (const line of lines) {
                if (line.startsWith("event: ")) {
                    currentEventType = line.substring(7).trim();
                    continue;
                }

                if (line.startsWith("data: ")) {
                    const jsonStr = line.substring(6);

                    try {
                        const parsed = JSON.parse(jsonStr);

                        if (currentEventType === "resolved") {
                            // Biarkan UI natural, tidak perlu me-replace chat user dengan rewrite dari AI
                            currentEventType = "";
                            continue;
                        }

                        if (currentEventType === "done") {
                            startStreamBubble();
                            _streamActive = false;
                            currentEventType = "";
                            continue;
                        }

                        if (currentEventType === "error") {
                            startStreamBubble();
                            _targetText = "⚠️ Waduh, ada masalah saat memproses pertanyaan kamu. Coba lagi ya!";
                            _streamActive = false;
                            currentEventType = "";
                            continue;
                        }

                        if (parsed.token !== undefined) {
                            startStreamBubble();
                            _targetText += parsed.token;
                        }

                    } catch (parseErr) {
                        console.warn("SSE parse error:", parseErr, jsonStr);
                    }

                    currentEventType = "";
                    continue;
                }

                if (line.trim() === "") {
                    currentEventType = "";
                }
            }
        }

        // Signal stream logic is finished receiving
        _streamActive = false;



    } catch (err) {
        if (err.name === 'AbortError') {
            console.log("Request cancelled by user.");
            removeTyping();
            return;
        }
        console.error("Stream Error:", err);

        if (err.message === "RATE_LIMIT") {
            removeTyping();
            addMessage("⏳ Sabar dulu ya! Kamu udah ngelebihin batas 20 chat per menit. Tunggu sebentar lagi baru tanya lagi. 😊", "ai");
            return;
        }

        // ── Fallback: try non-streaming endpoint ──
        // Jangan removeTyping() dulu — biarkan dots tetap tampil selama fallback request
        // Kalau streaming sudah mulai (streamWrap ada), typing sudah digantikan bubble — tetap lanjut
        if (!streamWrap) {
            // Pastikan typing dots tetap ada / tampilkan ulang untuk fallback
            if (!document.getElementById("typing-id")) showTyping();
        }

        try {
            console.log("Falling back to non-streaming /chat endpoint...");
            const fallbackRes = await fetch(`${baseUrl}/api/v1/chat`, {
                method: "POST",
                headers: headers,
                body: body,
                signal: currentAbortController ? currentAbortController.signal : undefined
            });

            if (!fallbackRes.ok) {
                if (fallbackRes.status === 429) {
                    throw new Error("RATE_LIMIT");
                }
                throw new Error(`Fallback returned ${fallbackRes.status}`);
            }

            const data = await fallbackRes.json();
            removeTyping(); // hapus dots SETELAH response tiba

            const reply = data?.answer || "Wah, Ava bingung nih jawabnya. Coba tanya hal lain yuk! 😊";
            streamMessageFallback(reply, "ai");

        } catch (fallbackErr) {
            if (fallbackErr.name === 'AbortError') {
                console.log("Fallback request cancelled by user.");
                removeTyping();
                return;
            }
            console.error("Fallback also failed:", fallbackErr);
            removeTyping();
            if (fallbackErr.message === "RATE_LIMIT") {
                addMessage("⏳ Sabar dulu ya! Kamu udah ngelebihin batas 20 chat per menit. Tunggu sebentar lagi baru tanya lagi. 😊", "ai");
            } else {
                addMessage("⚠️ Waduh, koneksi ke server lagi bermasalah nih. Coba lagi ya!", "ai");
            }
        }
    } finally {
        isStreaming = false;
        setSendButtonState(false);
        currentAbortController = null;
    }
}

// ============================================================
// FALLBACK: Client-side simulated streaming (when SSE fails)
// ============================================================
function streamMessageFallback(fullText, type) {
    const wrap = document.createElement("div");
    wrap.className = `msg ${type}`;

    const bubble = document.createElement("div");
    bubble.className = `bubble ${type}`;

    const contentDiv = document.createElement("div");
    contentDiv.className = "content";

    const timeSpan = document.createElement("span");
    timeSpan.className = "time";
    timeSpan.textContent = getTime();

    bubble.appendChild(contentDiv);
    bubble.appendChild(timeSpan);
    wrap.appendChild(bubble);
    messages.appendChild(wrap);

    let currentText = "";
    let i = 0;

    const speed = Math.max(10, Math.min(30, 2000 / fullText.length));

    function typeChar() {
        if (i < fullText.length) {
            const chunkSize = Math.max(1, Math.floor(fullText.length / 100));
            currentText += fullText.substring(i, i + chunkSize);
            i += chunkSize;

            const renderText = currentText + '<span class="streaming-cursor-inline">▍</span>';
            const tempDiv = document.createElement("div");
            tempDiv.innerHTML = marked.parse(renderText);
            tempDiv.querySelectorAll("a").forEach(link => {
                link.setAttribute("target", "_blank");
                link.setAttribute("rel", "noopener noreferrer");
            });
            contentDiv.innerHTML = tempDiv.innerHTML;
            messages.scrollTop = messages.scrollHeight;
            setTimeout(typeChar, speed);
        } else {
            finalizeStreamBubble(contentDiv, bubble, fullText);
        }
    }

    typeChar();
    return wrap;
}

function addAIResponse(text) {
    streamMessageFallback(text, "ai");
}

function handleKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (!isStreaming) {
            handleSendClick();
        }
    }
}

// Auto resize textarea
const promptNode = document.getElementById("prompt");
if (promptNode) {
    promptNode.addEventListener("input", function () {
        this.style.height = "auto";
        this.style.height = (this.scrollHeight) + "px";
    });
}
