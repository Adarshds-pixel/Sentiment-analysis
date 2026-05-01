const reviewInput = document.getElementById('reviewInput');
const analyzeButton = document.getElementById('analyzeButton');
const clearButton = document.getElementById('clearButton');
const copyButton = document.getElementById('copyButton');
const predictionText = document.getElementById('predictionText');
const confidenceText = document.getElementById('confidenceText');
const statusPanel = document.getElementById('statusPanel');
const resultEmoji = document.getElementById('resultEmoji');
const historyList = document.getElementById('historyList');

const HISTORY_STORAGE_KEY = 'sentimentAI_history';

function renderHistory() {
    const stored = JSON.parse(localStorage.getItem(HISTORY_STORAGE_KEY) || '[]');
    historyList.innerHTML = '';

    if (!stored.length) {
        historyList.innerHTML = '<p class="empty-state">No history yet. Submit a prediction to populate history.</p>';
        return;
    }

    stored.slice().reverse().forEach((item) => {
        const card = document.createElement('div');
        card.className = 'history-item';
        card.innerHTML = `
            <div>
                <p>${item.text}</p>
                <small>${item.sentiment} • ${item.confidence}% confidence</small>
            </div>
        `;
        historyList.appendChild(card);
    });
}

function setLoading(isLoading) {
    if (isLoading) {
        statusPanel.innerHTML = '<span class="loading">Analyzing sentiment...</span>';
        analyzeButton.disabled = true;
        analyzeButton.textContent = 'Analyzing...';
    } else {
        analyzeButton.disabled = false;
        analyzeButton.textContent = 'Analyze sentiment';
    }
}

function updateResult(prediction, confidence) {
    predictionText.textContent = prediction;
    confidenceText.textContent = `${(confidence * 100).toFixed(0)}%`;
    copyButton.disabled = false;
    statusPanel.textContent = 'Prediction completed successfully.';
    resultEmoji.textContent = prediction.toLowerCase() === 'positive' ? '😊' : '😡';
}

function saveHistory(text, sentiment, confidence) {
    const stored = JSON.parse(localStorage.getItem(HISTORY_STORAGE_KEY) || '[]');
    stored.push({
        text,
        sentiment,
        confidence: (confidence * 100).toFixed(0),
        createdAt: new Date().toISOString(),
    });
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(stored.slice(-12)));
    renderHistory();
}

async function analyzeSentiment() {
    const text = reviewInput.value.trim();
    if (!text) {
        statusPanel.textContent = 'Please enter some text to analyze.';
        return;
    }

    setLoading(true);
    statusPanel.textContent = '';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text }),
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Unable to analyze sentiment.');
        }

        updateResult(result.prediction, result.confidence);
        saveHistory(text, result.prediction, result.confidence);
    } catch (error) {
        statusPanel.textContent = error.message;
    } finally {
        setLoading(false);
    }
}

analyzeButton.addEventListener('click', analyzeSentiment);
clearButton.addEventListener('click', () => {
    reviewInput.value = '';
    predictionText.textContent = 'No analysis yet';
    confidenceText.textContent = '--%';
    resultEmoji.textContent = '🌐';
    statusPanel.textContent = 'Input cleared. Ready for a new review.';
    copyButton.disabled = true;
});

copyButton.addEventListener('click', () => {
    const sentiment = predictionText.textContent;
    const confidence = confidenceText.textContent;
    const payload = `Sentiment: ${sentiment}\nConfidence: ${confidence}`;
    navigator.clipboard.writeText(payload).then(() => {
        statusPanel.textContent = 'Result copied to clipboard.';
    }).catch(() => {
        statusPanel.textContent = 'Copy not supported in this browser.';
    });
});

window.addEventListener('load', () => {
    renderHistory();
    copyButton.disabled = true;
});
