const text = document.getElementById("newsText");
const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");
const refreshBtn = document.getElementById("refreshBtn");
const result = document.getElementById("result");
const prediction = document.getElementById("prediction");
const confidence = document.getElementById("confidence");
const meterBar = document.getElementById("meterBar");
const errorBox = document.getElementById("error");
const historyBox = document.getElementById("history");

predictBtn.addEventListener("click", async () => {
  errorBox.textContent = "";
  const value = text.value.trim();

  if (value.length < 20) {
    errorBox.textContent = "Please enter at least 20 characters.";
    return;
  }

  predictBtn.disabled = true;
  predictBtn.textContent = "Analyzing...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text: value})
    });

    const data = await response.json();

    if (!response.ok) throw new Error(data.error || "Prediction failed.");

    result.classList.remove("hidden");
    prediction.textContent = data.prediction;
    confidence.textContent = `Confidence: ${data.confidence}%`;
    meterBar.style.width = `${data.confidence}%`;

    await loadHistory();
  } catch (error) {
    errorBox.textContent = error.message;
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Analyze News";
  }
});

clearBtn.addEventListener("click", () => {
  text.value = "";
  errorBox.textContent = "";
  result.classList.add("hidden");
});

refreshBtn.addEventListener("click", loadHistory);

async function loadHistory() {
  try {
    const response = await fetch("/history");
    const items = await response.json();

    if (!items.length) {
      historyBox.innerHTML = '<p class="muted">No predictions yet.</p>';
      return;
    }

    historyBox.innerHTML = items.map(item => {
      const cls = item.prediction === "FAKE" ? "fake" : "real";
      const safeText = escapeHtml(item.text);
      return `
        <div class="history-item">
          <span class="badge ${cls}">${escapeHtml(item.prediction)}</span>
          <span class="muted"> ${Number(item.confidence).toFixed(2)}%</span>
          <p class="history-text">${safeText}</p>
          <small class="muted">${escapeHtml(item.created_at)}</small>
        </div>
      `;
    }).join("");
  } catch {
    historyBox.innerHTML = '<p class="muted">Could not load history.</p>';
  }
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;",
    '"': "&quot;", "'": "&#039;"
  }[char]));
}

loadHistory();
