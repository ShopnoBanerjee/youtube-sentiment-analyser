// YouTube Live Chat Sentiment Analyzer - Popup Script

let isAnalyzing = false;
let messageHistory = [];
let sentimentStats = {
    total: 0,
    positive: 0,
    negative: 0,
    neutral: 0
};

// DOM elements
const statusDiv = document.getElementById('status');
const statusText = document.getElementById('status-text');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const openDashboardBtn = document.getElementById('open-dashboard-btn');
const reloadBtn = document.getElementById('reload-btn');
const totalMessagesEl = document.getElementById('total-messages');
const positiveCountEl = document.getElementById('positive-count');
const negativeCountEl = document.getElementById('negative-count');
const neutralCountEl = document.getElementById('neutral-count');
const messageListEl = document.getElementById('message-list');
const apiUrlInput = document.getElementById('api-url');
const autoStartCheckbox = document.getElementById('auto-start');
const showConfidenceCheckbox = document.getElementById('show-confidence');

// Initialize popup
document.addEventListener('DOMContentLoaded', function() {
    loadSettings();
    updateUI();
    checkServerStatus();

    // Set up event listeners
    startBtn.addEventListener('click', startAnalysis);
    stopBtn.addEventListener('click', stopAnalysis);
    reloadBtn.addEventListener('click', reloadTab);
    openDashboardBtn.addEventListener('click', openDashboard);
    apiUrlInput.addEventListener('change', saveSettings);
    autoStartCheckbox.addEventListener('change', saveSettings);
    showConfidenceCheckbox.addEventListener('change', saveSettings);

    // Check if we're on a YouTube page
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        const currentTab = tabs[0];
        if (currentTab && currentTab.url.includes('youtube.com/watch')) {
            statusDiv.className = 'status active';
            statusText.textContent = 'YouTube Live Stream Detected';
        }
    });
});

// Load settings from storage
function loadSettings() {
    chrome.storage.sync.get({
        apiUrl: 'http://localhost:8000',
        autoStart: true,
        showConfidence: true
    }, function(items) {
        apiUrlInput.value = items.apiUrl;
        autoStartCheckbox.checked = items.autoStart;
        showConfidenceCheckbox.checked = items.showConfidence;
    });
}

// Save settings to storage
function saveSettings() {
    const settings = {
        apiUrl: apiUrlInput.value,
        autoStart: autoStartCheckbox.checked,
        showConfidence: showConfidenceCheckbox.checked
    };

    chrome.storage.sync.set(settings, function() {
        console.log('Settings saved');
    });
}

// Check if the API server is running
async function checkServerStatus() {
    try {
        const response = await fetch(`${apiUrlInput.value}/health`);
        if (response.ok) {
            statusText.textContent += ' | Server: ✓ Connected';
        } else {
            statusText.textContent += ' | Server: ✗ Disconnected';
        }
    } catch (error) {
        statusText.textContent += ' | Server: ✗ Disconnected';
        console.error('Server check failed:', error);
    }
}

// Check if content script is loaded and inject if needed
function ensureContentScriptLoaded(tabId, callback) {
    chrome.tabs.sendMessage(tabId, { action: 'ping' }, function(response) {
        if (chrome.runtime.lastError) {
            console.log('Content script not loaded, injecting...');
            chrome.scripting.executeScript({
                target: { tabId: tabId },
                files: ['content.js']
            }).then(() => {
                console.log('Content script injected successfully');
                // Wait a moment for the script to initialize
                setTimeout(callback, 1000);
            }).catch(error => {
                console.error('Failed to inject content script:', error);
                showNotification('Error: Could not load extension on this page. Try refreshing the page.');
            });
        } else {
            console.log('Content script already loaded');
            callback();
        }
    });
}

// Start sentiment analysis
function startAnalysis() {
    if (isAnalyzing) return;

    // Send message to content script to start monitoring
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        if (tabs[0]) {
            // Check if we're on a YouTube page
            if (!tabs[0].url.includes('youtube.com/watch')) {
                showNotification('Error: Please navigate to a YouTube video page first');
                return;
            }

            isAnalyzing = true;
            updateUI();

            // Ensure content script is loaded, then start analysis
            ensureContentScriptLoaded(tabs[0].id, function() {
                chrome.tabs.sendMessage(tabs[0].id, {
                    action: 'startAnalysis',
                    apiUrl: apiUrlInput.value,
                    showConfidence: showConfidenceCheckbox.checked
                }, function(response) {
                    if (chrome.runtime.lastError) {
                        console.error('Error starting analysis:', chrome.runtime.lastError.message);
                        showNotification('Error: Could not start analysis - ' + chrome.runtime.lastError.message);
                        stopAnalysis();
                    } else {
                        console.log('Analysis started successfully');
                    }
                });
            });
        } else {
            showNotification('Error: No active tab found');
        }
    });
}

// Stop sentiment analysis
function stopAnalysis() {
    if (!isAnalyzing) return;

    isAnalyzing = false;
    updateUI();

    // Send message to content script to stop monitoring
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        if (tabs[0]) {
            chrome.tabs.sendMessage(tabs[0].id, {
                action: 'stopAnalysis'
            }, function(response) {
                if (chrome.runtime.lastError) {
                    console.error('Error stopping analysis:', chrome.runtime.lastError.message);
                    // Don't show notification for stop errors as it's less critical
                } else {
                    console.log('Analysis stopped successfully');
                }
            });
        }
    });
}

// Open dashboard in new tab
function openDashboard() {
    chrome.tabs.create({
        url: `${apiUrlInput.value}/dashboard`
    });
}

// Reload current tab to reinject content script
function reloadTab() {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        if (tabs[0]) {
            chrome.tabs.reload(tabs[0].id);
            showNotification('Page reloaded - extension will reinitialize');
        }
    });
}

// Update UI based on analysis state
function updateUI() {
    if (isAnalyzing) {
        startBtn.disabled = true;
        stopBtn.disabled = false;
        statusDiv.className = 'status active';
        statusText.textContent = 'Analysis Active';
    } else {
        startBtn.disabled = false;
        stopBtn.disabled = true;
        statusDiv.className = 'status inactive';
        statusText.textContent = 'Analysis Stopped';
    }
}

// Update sentiment statistics
function updateStats(stats) {
    sentimentStats = stats;
    totalMessagesEl.textContent = stats.total;
    positiveCountEl.textContent = stats.positive;
    negativeCountEl.textContent = stats.negative;
    neutralCountEl.textContent = stats.neutral;
}

// Add message to the display
function addMessage(message) {
    messageHistory.unshift(message);

    // Keep only last 10 messages
    if (messageHistory.length > 10) {
        messageHistory = messageHistory.slice(0, 10);
    }

    updateMessageDisplay();
}

// Update message display
function updateMessageDisplay() {
    messageListEl.innerHTML = '';

    messageHistory.forEach(msg => {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${msg.sentiment}`;

        const confidenceText = showConfidenceCheckbox.checked ?
            ` (${(msg.confidence * 100).toFixed(1)}%)` : '';

        messageDiv.innerHTML = `
            <strong>${msg.sentiment.toUpperCase()}</strong>${confidenceText}:
            ${msg.text.length > 50 ? msg.text.substring(0, 50) + '...' : msg.text}
        `;

        messageListEl.appendChild(messageDiv);
    });
}

// Show notification
function showNotification(message) {
    // Create a simple notification
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #dc3545;
        color: white;
        padding: 10px 15px;
        border-radius: 5px;
        z-index: 10000;
        font-size: 14px;
    `;
    notification.textContent = message;

    document.body.appendChild(notification);

    setTimeout(() => {
        document.body.removeChild(notification);
    }, 3000);
}

// Listen for messages from content script
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    if (request.action === 'updateStats') {
        updateStats(request.stats);
    } else if (request.action === 'addMessage') {
        addMessage(request.message);
    } else if (request.action === 'analysisError') {
        showNotification('Analysis Error: ' + request.error);
        stopAnalysis();
    }
});

// Handle extension installation/update
chrome.runtime.onInstalled.addListener(function() {
    // Set default settings
    chrome.storage.sync.set({
        apiUrl: 'http://localhost:8000',
        autoStart: true,
        showConfidence: true
    });
});
