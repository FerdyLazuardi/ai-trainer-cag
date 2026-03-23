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
                        <span>Peped AI Trainer</span>
                        <small style="font-size:11px; opacity:.8;">Biasanya membalas &lt; 1 menit</small>
                    </div>
                </div>
                <i class="fas fa-times" style="cursor:pointer" onclick="toggleChat()"></i>
            </div>
            <div id="chat-messages"></div>
            <div id="chat-input">
                <textarea id="prompt" rows="1" placeholder="Ketik pesan..." onkeydown="handleKey(event)"></textarea>
                <button class="send-btn" onclick="send()">
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

/* AKTIFKAN PULSE */
document.getElementById("chat-toggle").classList.add("pulse");
document.getElementById("chat-toggle").onclick = toggleChat;

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

            icon.classList.remove("fa-comment-dots");
            icon.classList.add("fa-times");

            toggleBtn.classList.remove("pulse");

            const badge = document.getElementById("chat-badge");
            if (badge) badge.style.display = "none";

            if (!introduced) {
                setTimeout(showIntro, 500);
                introduced = true;
            }

        } else {
            // ===== CLOSE =====
            chatBox.classList.remove("animate__fadeInUp");
            chatBox.classList.add("animate__fadeOutDown");
            setTimeout(() => { chatBox.style.display = "none"; }, 500);

            icon.classList.remove("fa-times");
            icon.classList.add("fa-comment-dots");

            toggleBtn.classList.add("pulse");
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
        `Hi **${nama}**! Aku **Peped**. ` +
        `Ada yang bisa aku bantu hari ini terkait materi Amarthapedia? 😊`
    );
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

// Configure marked
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
    let sid = sessionStorage.getItem("peped_sid");
    if (!sid) {
        sid = "sid-" + Math.random().toString(36).substring(2, 9);
        sessionStorage.setItem("peped_sid", sid);
    }
    return sid;
}

function resetChat() {
    sessionStorage.removeItem("peped_sid");
    window.location.reload();
}

// ============================================================
// SEND MESSAGE
// ============================================================
async function send() {
    const text = textarea.value.trim();
    if (!text) return;

    addMessage(text, "user");
    textarea.value = "";
    textarea.style.height = "auto";

    showTyping();

    // Tentukan base URL — pakai API_BASE_URL kalau di Moodle,
    // fallback ke "" (relative) kalau di test-ui
    const baseUrl = (typeof API_BASE_URL !== 'undefined' && API_BASE_URL)
        ? API_BASE_URL
        : "";

    try {
        const res = await fetch(`${baseUrl}/api/v1/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query: text,
                conversation_id: getSessionId(),
                course_id: typeof MOODLE_COURSE_ID !== 'undefined' ? MOODLE_COURSE_ID : 0,
                course_name: typeof MOODLE_COURSE_NAME !== 'undefined' ? MOODLE_COURSE_NAME : ''
            })
        });

        if (!res.ok) throw new Error(`Server returned ${res.status}`);

        const data = await res.json();
        removeTyping();

        const reply = data?.answer || "Wah, Peped bingung nih jawabnya. Coba tanya hal lain yuk! 😊";
        addAIResponse(reply);

    } catch (err) {
        console.error("Chat Error:", err);
        removeTyping();
        addMessage("⚠️ Waduh, koneksi ke server lagi bermasalah nih. Coba lagi ya!", "ai");
    }
}

function addAIResponse(text) {
    addMessage(text, "ai");
}

function handleKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        send();
    }
}

// Auto resize textarea
document.getElementById("prompt").addEventListener("input", function () {
    this.style.height = "auto";
    this.style.height = (this.scrollHeight) + "px";
});