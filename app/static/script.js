const chatBox = document.getElementById("chat-box");
const messages = document.getElementById("chat-messages");
const textarea = document.getElementById("prompt");
let introduced = false;

/* AKTIFKAN PULSE */
document.getElementById("chat-toggle").classList.add("pulse");

document.getElementById("chat-toggle").onclick = toggleChat;

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

            // ❌ matikan pulse
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

            // 💬 hidupkan pulse lagi
            toggleBtn.classList.add("pulse");
        }

        icon.classList.remove("icon-animate");
    }, 200);
}

function showIntro() {
    addAIResponse("Hi A-Team! Aku **Peped**. Ada yang bisa aku bantu hari ini terkait materi Amarthapedia? 😊");
}

function getTime() {
    return new Date().toLocaleTimeString("id-ID", { hour: "2-digit", minute: "2-digit" });
}

function addMessage(text, type) {
    const wrap = document.createElement("div");
    wrap.className = `msg ${type} animate__animated animate__zoomIn animate__faster`;

    const bubble = document.createElement("div");
    bubble.className = `bubble ${type}`;
    
    const formattedText = marked.parse(text);

    // paksa semua link buka tab baru
    const tempDiv = document.createElement("div");
    tempDiv.innerHTML = formattedText;

    tempDiv.querySelectorAll("a").forEach(link => {
        link.setAttribute("target", "_blank");
        link.setAttribute("rel", "noopener noreferrer");
    });

    const finalHTML = tempDiv.innerHTML;


    bubble.innerHTML = `
        <div class="content">${finalHTML}</div>
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

// Configure marked options once
marked.setOptions({
    breaks: true,
    gfm: true
});

async function send() {
    const text = textarea.value.trim();
    if (!text) return;

    addMessage(text, "user");
    textarea.value = "";
    textarea.style.height = "auto";

    showTyping();

    try {
        const res = await fetch("/api/v1/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: text, conversation_id: "user-demo" })
        });

        if (!res.ok) {
            throw new Error(`Server returned ${res.status}`);
        }

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

textarea.addEventListener("input", function() {
    this.style.height = "auto";
    this.style.height = (this.scrollHeight) + "px";
});