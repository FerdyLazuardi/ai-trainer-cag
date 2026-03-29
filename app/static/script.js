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
                <div style="display:flex; gap:22px; align-items:center;">
                    <i class="fas fa-trash-alt header-icon" onclick="clearChat()" title="Clear chat" style="cursor:pointer; font-size:14px; opacity:0.8;"></i>
                    <i class="fas fa-times header-icon" onclick="toggleChat()" title="Close chat" style="cursor:pointer; font-size:14px; opacity:0.8;"></i>
                </div>
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
        `Hi **${nama}**! Aku **Peped**. ` +
        `Ada yang bisa aku bantu hari ini terkait materi Amarthapedia? 😊`
    );
}

async function loadHistory() {
    // Selalu tampilkan intro pertama kali
    setTimeout(showIntro, 100);

    const baseUrl = (typeof API_BASE_URL !== 'undefined' && API_BASE_URL) ? API_BASE_URL : "";
    const headers = { "Content-Type": "application/json" };
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
        const headers = { "Content-Type": "application/json" };
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

    const userMsgNode = addMessage(text, "user");
    textarea.value = "";
    textarea.style.height = "auto";

    showTyping();

    const baseUrl = (typeof API_BASE_URL !== 'undefined' && API_BASE_URL)
        ? API_BASE_URL
        : "";

    const headers = { "Content-Type": "application/json" };
    if (typeof MOODLE_JWT !== 'undefined' && MOODLE_JWT) {
        headers["Authorization"] = `Bearer ${MOODLE_JWT}`;
    }

    try {
        const res = await fetch(`${baseUrl}/api/v1/chat`, {
            method: "POST",
            headers: headers,
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

        // Ganti UI bubble user jika input di-"resolve" oleh AI (misal, input "1" menjadi "Apa itu Amartha?")
        if (data && data.resolved_query && data.resolved_query !== text) {
            const contentNode = userMsgNode.querySelector('.content');
            if (contentNode) {
                const tempDiv = document.createElement("div");
                tempDiv.innerHTML = marked.parse(data.resolved_query);
                tempDiv.querySelectorAll("a").forEach(link => {
                    link.setAttribute("target", "_blank");
                    link.setAttribute("rel", "noopener noreferrer");
                });
                contentNode.innerHTML = tempDiv.innerHTML;
            }
        }

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
const promptNode = document.getElementById("prompt");
if (promptNode) {
    promptNode.addEventListener("input", function () {
        this.style.height = "auto";
        this.style.height = (this.scrollHeight) + "px";
    });
}
