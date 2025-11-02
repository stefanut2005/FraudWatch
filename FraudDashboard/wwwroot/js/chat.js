// =========================
// AI Analyst Chat Client
// =========================

const AI_API_URL = "http://localhost:8080/query";

const chatWidget = document.getElementById("chatWidget");
const chatToggle = document.getElementById("chatToggle");
const chatCollapseBtn = document.getElementById("chatCollapseBtn");
const chatBody = document.getElementById("chatBody");
const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const chatText = document.getElementById("chatText");

// ---- UI helpers ----
function appendBubble(text, who) {
    const bubble = document.createElement("div");
    bubble.classList.add("chat-bubble");

    if (who === "user") {
        bubble.classList.add("from-user");
        bubble.textContent = text;
    } else if (who === "ai") {
        bubble.classList.add("from-ai");
        // Allow HTML for formatted AI responses
        bubble.innerHTML = text;
    } else if (who === "thinking") {
        bubble.classList.add("from-ai", "thinking");
        bubble.textContent = text;
    }

    chatMessages.appendChild(bubble);

    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendUserChat(text) {
    appendBubble(text, "user");
}

function appendAIChat(text) {
    appendBubble(text, "ai");
}

let thinkingBubbleEl = null;
function showThinking() {
    thinkingBubbleEl = document.createElement("div");
    thinkingBubbleEl.classList.add("chat-bubble", "from-ai", "thinking");
    thinkingBubbleEl.textContent = "Analyzing your question...";
    chatMessages.appendChild(thinkingBubbleEl);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function hideThinking() {
    if (thinkingBubbleEl && thinkingBubbleEl.parentNode) {
        thinkingBubbleEl.parentNode.removeChild(thinkingBubbleEl);
    }
    thinkingBubbleEl = null;
}

// ---- backend call ----
async function callAiBackend(questionText) {
    const q = questionText.trim();
    if (!q) {
        return { error: "Empty question" };
    }

    try {
        const resp = await fetch(AI_API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question: q,
                summarize: true
            })
        });

        if (!resp.ok) {
            const detail = await resp.text();
            return { error: `Backend error (${resp.status}): ${detail}` };
        }

        const data = await resp.json();
        // data = { sql, rows, summary }

        let finalHTML = "";

        // Only display the summary - ignore sql and rows
        if (data.summary) {
            // Clean summary: remove any mentions of SQL, queries, or technical terms that might have leaked through
            let cleanSummary = data.summary.trim();
            // Remove common SQL-related phrases that LLM might have included
            cleanSummary = cleanSummary.replace(/SQL executed[:\s]*/gi, '');
            cleanSummary = cleanSummary.replace(/The query[^.]*\./gi, '');
            cleanSummary = cleanSummary.replace(/SELECT[^.]*\./gi, '');
            cleanSummary = cleanSummary.replace(/query[^.]*executed[^.]*\./gi, '');
            cleanSummary = cleanSummary.replace(/\n\n\n+/g, '\n\n'); // Clean up extra newlines
            
            finalHTML += "<div class='ai-summary'>" + escapeHtml(cleanSummary) + "</div>";
        } else {
            finalHTML = "<div class='ai-error'>(No answer returned)</div>";
        }

        return { ok: true, text: finalHTML };

    } catch (err) {
        return { error: "Request failed: " + err.message };
    }
}

// ---- send flow ----
async function sendChat(questionText) {
    // open widget if closed
    chatWidget.classList.add("open");
    chatBody.style.display = "flex";

    showThinking();
    const result = await callAiBackend(questionText);
    hideThinking();

    if (result.error) {
        appendAIChat("⚠ " + result.error);
    } else {
        appendAIChat(result.text);
    }
}

// ---- hook UI events ----
if (chatToggle) {
    chatToggle.addEventListener("click", () => {
        const isOpen = chatWidget.classList.contains("open");
        if (isOpen) {
            chatWidget.classList.remove("open");
            chatBody.style.display = "none";
            if (chatCollapseBtn) chatCollapseBtn.textContent = "▲";
        } else {
            chatWidget.classList.add("open");
            chatBody.style.display = "flex";
            if (chatCollapseBtn) chatCollapseBtn.textContent = "▼";
        }
    });
}

if (chatForm) {
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = chatText.value.trim();
        if (!text) return;

        appendUserChat(text);
        // Clear the input immediately after submitting so the UI feels responsive.
        chatText.value = "";
        await sendChat(text);
    });
}

// Helper to escape HTML for security
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// expose for other scripts (like askAiWhy in modal)
window.appendUserChat = appendUserChat;
window.sendChat = sendChat;