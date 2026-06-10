// ============================================================
// AUTO INJECT HTML ke body (untuk Moodle integration)
// ============================================================
(function injectChatWidget() {
    // Cek apakah sudah ada (hindari duplikat)
    if (document.getElementById("chat-toggle")) return;

    // Inject Inter font (widget hidup di halaman Moodle yang tidak punya font ini)
    if (!document.getElementById("ava-inter-font")) {
        const fontLink = document.createElement("link");
        fontLink.id = "ava-inter-font";
        fontLink.rel = "stylesheet";
        fontLink.href = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap";
        document.head.appendChild(fontLink);
    }

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
                <div style="display:flex; gap:16px; align-items:center;">
                    <label class="mentor-switch" title="Mode Mentoring (Socratic)">
                        <input type="checkbox" id="mentor-toggle" onchange="setMentoring(this.checked, true)">
                        <span class="mentor-switch-track">
                            <span class="mentor-switch-text">Mentor</span>
                            <span class="mentor-switch-thumb"></span>
                        </span>
                    </label>
                    <i class="fas fa-trash-alt header-icon" onclick="clearChat()" title="Clear chat" style="cursor:pointer; font-size:14px; opacity:0.8;"></i>
                    <i class="fas fa-times header-icon" onclick="toggleChat()" title="Close chat" style="cursor:pointer; font-size:14px; opacity:0.8;"></i>
                </div>
            </div>
            <div id="chat-messages"></div>
            <div id="chat-input">
                <button class="topics-btn" onclick="openSectionPanel()" title="Daftar topik">
                    <i class="fas fa-list-ul"></i>
                </button>
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
// ============================================================
// MENTORING MODE — single source of truth for the slider + chips
// ============================================================
// setMentoring(on, showMsg): sets the flag read by send() (mentoring_mode in
// the request body), syncs the header slider, and — when showMsg — injects an
// awareness message so the user SEES the mode change. Both on AND off respond,
// so the switch never flips silently.
function setMentoring(on, showMsg) {
    window.MENTORING_MODE = !!on;
    const cb = document.getElementById("mentor-toggle");
    if (cb) cb.checked = !!on;
    const sw = document.querySelector(".mentor-switch");
    if (sw) sw.classList.toggle("on", !!on);
    if (showMsg) {
        if (on) {
            addMessage(
                "Oke, aku pandu kamu belajar ya. Apa yang bikin kamu bingung, materi Amarthapedia atau soal kerjaan kamu di Amartha?",
                "ai"
            );
        } else {
            addMessage("Oke, mode Mentoring dimatiin. Balik ke jawaban langsung ya.", "ai");
        }
    }
}

// Welcome-screen chip: "Topik" → fetch the topic list straight from the
// instant endpoint (Postgres, no LLM) and render it client-side, so the list
// appears immediately instead of waiting on the chat pipeline's generate step.
async function chipTopik() {
    removeWelcome();
    const baseUrl = (typeof API_BASE_URL !== 'undefined' && API_BASE_URL) ? API_BASE_URL : "";
    const headers = { "ngrok-skip-browser-warning": "true" };
    if (typeof MOODLE_JWT !== 'undefined' && MOODLE_JWT) {
        headers["Authorization"] = `Bearer ${MOODLE_JWT}`;
    }
    try {
        const res = await fetch(`${baseUrl}/api/v1/chat/topics`, { method: "GET", headers });
        const data = await res.json();
        const topics = (data && data.topics) || [];
        if (topics.length) {
            const list = topics.map(t => `* ${t}`).join("\n");
            addMessage("Topik yang tersedia di Amarthapedia:\n\n" + list, "ai");
        } else {
            addMessage("Belum ada topik yang bisa aku tampilkan saat ini.", "ai");
        }
    } catch (e) {
        console.error("chipTopik failed:", e);
        addMessage("Aku belum bisa nampilin daftar topik sekarang. Coba lagi ya.", "ai");
    }
}

// Welcome-screen chip: "Mentoring" → flip the slider ON and show the canned
// guiding prompt instantly (client-side, no backend round-trip).
function chipMentoring() {
    setMentoring(true, true);
}

// ── Topic-list button → in-chatbox section/item picker ───────────────────────
// Fetches /chat/sections (deterministic, no LLM) and renders an accordion INSIDE
// #chat-box (clipped by overflow:hidden, like the delete modal). Click a section
// to expand its items; click an item to send "jelaskan tentang <item>" as a
// normal KNOWLEDGE query. Replaces the old fragile free-text section parsing.
function _esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

async function openSectionPanel() {
    if (document.getElementById("ava-section-panel")) return;  // already open
    const baseUrl = (typeof API_BASE_URL !== 'undefined' && API_BASE_URL) ? API_BASE_URL : "";
    const headers = { "ngrok-skip-browser-warning": "true" };
    if (typeof MOODLE_JWT !== 'undefined' && MOODLE_JWT) {
        headers["Authorization"] = `Bearer ${MOODLE_JWT}`;
    }
    let sections = {};
    try {
        const res = await fetch(`${baseUrl}/api/v1/chat/sections`, { method: "GET", headers });
        const data = await res.json();
        sections = (data && data.sections) || {};
    } catch (e) {
        console.error("openSectionPanel fetch failed:", e);
    }

    const overlay = document.createElement("div");
    overlay.id = "ava-section-panel";
    overlay.className = "ava-panel-overlay";

    const names = Object.keys(sections);
    let rows = "";
    if (names.length) {
        rows = names.map(sec => {
            const items = (sections[sec] || []).map(it =>
                `<button class="ava-panel-item" data-item="${_esc(it)}">${_esc(it)}</button>`
            ).join("");
            return `<div class="ava-panel-section">
                <button class="ava-panel-sec-head"><span>${_esc(sec)}</span><i class="fas fa-chevron-down"></i></button>
                <div class="ava-panel-items">${items}</div>
            </div>`;
        }).join("");
    } else {
        rows = `<div class="ava-panel-empty">Belum ada topik yang bisa ditampilkan.</div>`;
    }

    overlay.innerHTML = `
        <div class="ava-panel-card">
            <div class="ava-panel-head">
                <span>Daftar Topik</span>
                <i class="fas fa-times ava-panel-close" title="Tutup"></i>
            </div>
            <div class="ava-panel-body">${rows}</div>
        </div>`;

    const close = () => overlay.remove();
    overlay.querySelector(".ava-panel-close").onclick = close;
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    // Accordion: toggle a section open/closed.
    overlay.querySelectorAll(".ava-panel-sec-head").forEach(h => {
        h.onclick = () => h.parentElement.classList.toggle("open");
    });
    // Click an item → close panel + ask about it as a normal question.
    overlay.querySelectorAll(".ava-panel-item").forEach(b => {
        b.onclick = () => {
            const item = b.getAttribute("data-item");
            close();
            send(`jelaskan tentang ${item}`);
        };
    });
    chatBox.appendChild(overlay);
}

// ── Auto-hook: OFFER mentoring after a reflective question ───────────────────
// Detects a diagnostic/"how-should-I" question about the user's OWN work and
// (when mentoring is OFF) appends a soft OFFER below the answer. It never
// auto-activates — the user must click "Ya, pandu aku" to opt in (which turns
// the switch green). Frontend-only: the answer already streamed normally.
function _looksReflective(t) {
    if (!t) return false;
    const s = " " + t.toLowerCase() + " ";
    const why = /\b(kok|kenapa|mengapa|knp|ngapa|napa)\b/.test(s);
    const work = /\b(aku|ku|saya|mitra|nasabah|target|tim|point|portfolio|portofolio|repayment|tagih|ditagih|kabur|nunggak|setoran|angsuran)\b/.test(s);
    if (why && work) return true;
    // "gimana caranya aku ...", "gimana aku bisa/harus ..."
    if (/\b(gimana|gmn|gmana|bagaimana)\b[^.?!\n]{0,25}\b(aku|ku|saya)\b/.test(s)) return true;
    return false;
}

function removeMentorOffer() {
    const el = document.getElementById("ava-mentor-offer");
    if (el) el.remove();
}

// backendSuggest: the server's suggest_mentoring flag (true/false), or null/
// undefined when absent. When present it's AUTHORITATIVE (semantic affinity);
// the regex _looksReflective is only the fallback when the backend didn't send
// a signal (e.g. fallback/non-stream path or older backend).
function maybeOfferMentoring(userText, backendSuggest) {
    if (window.MENTORING_MODE) return;                        // already on
    if (document.getElementById("ava-mentor-offer")) return;  // one at a time
    const show = (backendSuggest === true || backendSuggest === false)
        ? backendSuggest
        : _looksReflective(userText);
    if (!show) return;
    window._lastReflectiveQ = userText;  // remembered for the accept handler
    const wrap = document.createElement("div");
    wrap.id = "ava-mentor-offer";
    wrap.className = "ava-mentor-offer animate__animated animate__fadeIn animate__faster";
    wrap.innerHTML =
        '<span class="ava-offer-text">Mau ngulik ini bareng? Aku bisa pandu kamu mikir step by step.</span>' +
        '<button class="ava-offer-btn" onclick="acceptMentoringOffer()"><i class="fas fa-graduation-cap"></i> Ya, pandu aku</button>';
    messages.appendChild(wrap);
    messages.scrollTop = messages.scrollHeight;
}

// Topic-streak auto-hook: if the user keeps asking about the SAME topic several
// turns in a row, that's a strong "I want to go deep here" signal — offer to
// coach them through it. Topic is taken from the answer's `sources` (each chunk
// carries its course/section title). Frontend-only: no backend change.
const _TOPIC_STREAK_THRESHOLD = 3;   // offer after this many same-topic turns
window._topicStreak = { topic: null, count: 0 };

function _dominantTopic(sources) {
    if (!sources || !sources.length) return null;
    // Most frequent title among the retrieved chunks (ties → first/highest-rank).
    const freq = {};
    let best = null, bestN = 0;
    for (const s of sources) {
        const t = (s && s.title) ? s.title.trim() : "";
        if (!t || t === "Unknown") continue;
        freq[t] = (freq[t] || 0) + 1;
        if (freq[t] > bestN) { bestN = freq[t]; best = t; }
    }
    return best;
}

function maybeOfferMentoringByTopicStreak(userText, sources) {
    if (window.MENTORING_MODE) return;                        // already on
    if (document.getElementById("ava-mentor-offer")) return;  // per-question hook already offered
    const topic = _dominantTopic(sources);
    if (!topic) { window._topicStreak = { topic: null, count: 0 }; return; }

    const st = window._topicStreak;
    if (st.topic === topic) {
        st.count += 1;
    } else {
        window._topicStreak = { topic: topic, count: 1 };
    }
    if (window._topicStreak.count < _TOPIC_STREAK_THRESHOLD) return;

    // Streak reached → offer, then reset so we don't nag every subsequent turn.
    window._topicStreak = { topic: null, count: 0 };
    window._lastReflectiveQ = userText;  // accept handler re-asks this in mentoring mode
    const wrap = document.createElement("div");
    wrap.id = "ava-mentor-offer";
    wrap.className = "ava-mentor-offer animate__animated animate__fadeIn animate__faster";
    wrap.innerHTML =
        '<span class="ava-offer-text">Kamu udah beberapa kali ngebahas <b>' + topic +
        '</b> nih. Mau aku pandu ngulik lebih dalam soal ini?</span>' +
        '<button class="ava-offer-btn" onclick="acceptMentoringOffer()"><i class="fas fa-graduation-cap"></i> Ya, pandu aku</button>';
    messages.appendChild(wrap);
    messages.scrollTop = messages.scrollHeight;
}
// message) and re-ask their last question in mentoring mode so the answer
// continues straight into Socratic coaching ON THAT topic — not a reset to
// "apa yang bikin kamu bingung?". skipBubble: the question is already shown
// above, don't duplicate it.
function acceptMentoringOffer() {
    removeMentorOffer();
    setMentoring(true, false);   // green slider, no canned message
    const q = window._lastReflectiveQ;
    if (q) {
        window._lastReflectiveQ = null;
        send(q, { skipBubble: true });
    } else {
        // No remembered question (shouldn't happen from the offer) — fall back
        // to the canned guiding prompt.
        addMessage("Oke, aku pandu kamu belajar ya. Apa yang lagi pengen kamu ulik?", "ai");
    }
}

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
// INTRO — welcome screen terpusat (sapa user pakai nama Moodle)
// ============================================================

/**
 * Greeting berbasis waktu lokal browser (dihitung fresh tiap dibuka).
 *   5–10  → Selamat pagi
 *   11–14 → Selamat siang
 *   15–17 → Selamat sore
 *   else  → Selamat malam
 */
function getGreeting() {
    const h = new Date().getHours();
    if (h >= 5 && h <= 10) return "Selamat pagi";
    if (h >= 11 && h <= 14) return "Selamat siang";
    if (h >= 15 && h <= 17) return "Selamat sore";
    return "Selamat malam";
}

/** Hapus welcome screen kalau ada (dipanggil saat pesan pertama muncul). */
function removeWelcome() {
    const el = document.getElementById("ava-welcome");
    if (el) el.remove();
}

function showIntro() {
    // Hindari duplikat welcome
    if (document.getElementById("ava-welcome")) return;
    // Welcome = empty-state ONLY. Kalau sudah ada bubble chat (mis. user klik
    // chip Topik/Mentoring lalu close+open), JANGAN munculkan welcome lagi di
    // bawah konten — itu bug "Selamat siang muncul lagi".
    if (messages.querySelector(".msg")) return;

    const nama = (typeof MOODLE_USER_NAME !== 'undefined' && MOODLE_USER_NAME)
        ? MOODLE_USER_NAME.split(' ')[0]
        : 'A-Team';

    const welcome = document.createElement("div");
    welcome.id = "ava-welcome";
    welcome.className = "animate__animated animate__fadeIn animate__faster";
    welcome.innerHTML = `
        <h2 class="ava-welcome-title">${getGreeting()}, ${nama}</h2>
        <p class="ava-welcome-subtitle">Ada yang bisa aku bantu hari ini terkait materi Amarthapedia?</p>
        <div class="ava-chips">
            <button class="ava-chip" onclick="chipTopik()">
                <i class="fas fa-layer-group"></i> Topik
            </button>
            <button class="ava-chip" onclick="chipMentoring()">
                <i class="fas fa-graduation-cap"></i> Mentoring
            </button>
        </div>
    `;

    messages.appendChild(welcome);
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

function clearChat() {
    // Custom in-chatbox confirm modal (replaces native confirm()).
    showConfirmModal(
        "Hapus semua chat?",
        "Riwayat percakapan ini akan dihapus dan tidak bisa dikembalikan.",
        doClearChat
    );
}

// In-chatbox confirm modal. Overlay is appended INTO #chat-box, which is
// position:fixed + overflow:hidden, so it's clipped to the chat window (not the
// whole page). Buttons resolve by calling onConfirm or just closing.
function showConfirmModal(title, body, onConfirm) {
    if (document.getElementById("ava-modal")) return;
    const overlay = document.createElement("div");
    overlay.id = "ava-modal";
    overlay.className = "ava-modal-overlay";
    overlay.innerHTML = `
        <div class="ava-modal-card">
            <h3 class="ava-modal-title">${title}</h3>
            <p class="ava-modal-body">${body}</p>
            <div class="ava-modal-actions">
                <button class="ava-modal-btn ava-modal-cancel">Batal</button>
                <button class="ava-modal-btn ava-modal-confirm">Hapus</button>
            </div>
        </div>
    `;
    const close = () => overlay.remove();
    overlay.querySelector(".ava-modal-cancel").onclick = close;
    overlay.querySelector(".ava-modal-confirm").onclick = () => { close(); onConfirm(); };
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    chatBox.appendChild(overlay);
}

async function doClearChat() {
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

// ============================================================
// HELPERS
// ============================================================
function addMessage(text, type) {
    removeWelcome();
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
    removeWelcome();
    let wrap = document.getElementById("typing-id");
    let bubble;

    const contentDiv = document.createElement("div");
    contentDiv.className = "content";

    if (wrap) {
        // Reuse existing bubble to prevent visual drop
        wrap.removeAttribute("id");
        bubble = wrap.querySelector(".bubble");
        if (bubble) {
            bubble.classList.add("streaming");
            const typingDiv = bubble.querySelector(".typing");
            if (typingDiv) typingDiv.remove();

            bubble.appendChild(contentDiv);
        }
    } else {
        wrap = document.createElement("div");
        wrap.className = "msg ai";

        bubble = document.createElement("div");
        bubble.className = "bubble ai streaming";

        bubble.appendChild(contentDiv);
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
async function send(presetText, opts) {
    // presetText: send this instead of the textarea value (used by the
    // "Ya, pandu aku" accept handler to re-ask the user's last question in
    // mentoring mode). opts.skipBubble: don't add a user bubble (the question
    // is already shown above, so re-asking shouldn't duplicate it).
    opts = opts || {};
    const text = (presetText != null ? presetText : textarea.value).trim();
    if (!text || isStreaming) return;

    removeMentorOffer();
    if (!opts.skipBubble) {
        addMessage(text, "user");
    }
    if (presetText == null) {
        textarea.value = "";
        textarea.style.height = "auto";
    }

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
        course_name: typeof MOODLE_COURSE_NAME !== 'undefined' ? MOODLE_COURSE_NAME : '',
        mentoring_mode: !!window.MENTORING_MODE
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
        let _suggestMentoring = null;  // backend auto-hook signal (set in done event)
        let _doneSources = null;       // sources[] from done event (for topic-streak hook)

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
                // Auto-hook: after the answer lands, offer mentoring. Backend
                // signal (_suggestMentoring) is authoritative when present;
                // regex _looksReflective is the fallback when it's absent.
                maybeOfferMentoring(text, _suggestMentoring);
                // Topic-streak hook: if the per-question hook didn't already
                // offer, and the user has asked about the SAME topic 3x in a
                // row, offer to go deeper on that topic.
                maybeOfferMentoringByTopicStreak(text, _doneSources);
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
                            if (parsed.suggest_mentoring !== undefined) _suggestMentoring = parsed.suggest_mentoring;
                            if (parsed.sources !== undefined) _doneSources = parsed.sources;
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
            maybeOfferMentoring(text);

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
    removeWelcome();
    const wrap = document.createElement("div");
    wrap.className = `msg ${type}`;

    const bubble = document.createElement("div");
    bubble.className = `bubble ${type}`;

    const contentDiv = document.createElement("div");
    contentDiv.className = "content";

    bubble.appendChild(contentDiv);
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
