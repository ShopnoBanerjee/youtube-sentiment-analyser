// YouTube Live Chat Sentiment Analyzer - Popup Script

// Type definitions
interface SentimentStats {
    total: number;
    positive: number;
    negative: number;
    neutral: number;
}

interface SentimentMessage {
    text: string;
    sentiment: string;
    confidence: number;
}

interface Settings {
    apiUrl: string;
    autoStart: boolean;
    showConfidence: boolean;
}

interface MessageData {
    action: string;
    stats?: SentimentStats;
    message?: SentimentMessage;
    messages?: SentimentMessage[];
    error?: string;
}

let isAnalyzing: boolean = false;
let messageHistory: SentimentMessage[] = [];
let popupSentimentStats: SentimentStats = {
    total: 0,
    positive: 0,
    negative: 0,
    neutral: 0
};
const statusDiv = document.getElementById('status') as HTMLDivElement | null;
const statusText = document.getElementById('status-text') as HTMLSpanElement | null;
const startBtn = document.getElementById('start-btn') as HTMLButtonElement | null;
const stopBtn = document.getElementById('stop-btn') as HTMLButtonElement | null;
const reloadBtn = document.getElementById('reload-btn') as HTMLButtonElement | null;
const totalMessagesEl = document.getElementById('total-messages') as HTMLSpanElement | null;
const positiveCountEl = document.getElementById('positive-count') as HTMLSpanElement | null;
const negativeCountEl = document.getElementById('negative-count') as HTMLSpanElement | null;
const neutralCountEl = document.getElementById('neutral-count') as HTMLSpanElement | null;
const messageListEl = document.getElementById('message-list') as HTMLDivElement | null;
const apiUrlInput = document.getElementById('api-url') as HTMLInputElement | null;
const autoStartCheckbox = document.getElementById('auto-start') as HTMLInputElement | null;
const showConfidenceCheckbox = document.getElementById('show-confidence') as HTMLInputElement | null;

// Connect to background script
const port: chrome.runtime.Port = chrome.runtime.connect({ name: 'popup' });

// Initialize popup
document.addEventListener('DOMContentLoaded', function() {
    loadSettings();
    loadStoredData();
    updateUI();
    checkServerStatus();

    // Set up event listeners with null checks
    if (startBtn) startBtn.addEventListener('click', startAnalysis);
    if (stopBtn) stopBtn.addEventListener('click', stopAnalysis);
    if (reloadBtn) reloadBtn.addEventListener('click', reloadTab);
    if (apiUrlInput) apiUrlInput.addEventListener('change', saveSettings);
    if (autoStartCheckbox) autoStartCheckbox.addEventListener('change', saveSettings);
    if (showConfidenceCheckbox) showConfidenceCheckbox.addEventListener('change', saveSettings);

    // Check if we're on a YouTube page
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs: chrome.tabs.Tab[]) {
        const currentTab = tabs[0];
        if (currentTab && currentTab.url && currentTab.url.includes('youtube.com/watch')) {
            if (statusDiv) statusDiv.className = 'status active';
            if (statusText) statusText.textContent = 'YouTube Live Stream Detected';
        }
    });
});

// Load settings from storage
function loadSettings(): void {
    chrome.storage.sync.get({
        apiUrl: 'http://localhost:8000',
        autoStart: true,
        showConfidence: true
    }, function(items: { [key: string]: any }) {
        if (apiUrlInput) apiUrlInput.value = items.apiUrl;
        if (autoStartCheckbox) autoStartCheckbox.checked = items.autoStart;
        if (showConfidenceCheckbox) showConfidenceCheckbox.checked = items.showConfidence;
    });
}

// Load stored sentiment data
function loadStoredData(): void {
    chrome.storage.local.get({
        sentimentStats: { total: 0, positive: 0, negative: 0, neutral: 0 },
        recentMessages: []
    }, function(items: { [key: string]: any }) {
        popupSentimentStats = items.sentimentStats;
        messageHistory = items.recentMessages;
        updateStats(popupSentimentStats);
        updateMessageDisplay();
    });
}

// Save settings to storage
function saveSettings(): void {
    if (!apiUrlInput || !autoStartCheckbox || !showConfidenceCheckbox) return;

    const settings: Settings = {
        apiUrl: apiUrlInput.value,
        autoStart: autoStartCheckbox.checked,
        showConfidence: showConfidenceCheckbox.checked
    };

    chrome.storage.sync.set(settings, function() {
        console.log('Settings saved');
    });
}

// Check if the API server is running
async function checkServerStatus(): Promise<void> {
    if (!apiUrlInput || !statusText) return;

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
function ensureContentScriptLoaded(tabId: number, callback: () => void): void {
    chrome.tabs.sendMessage(tabId, { action: 'ping' }, function(response: any) {
        if (chrome.runtime.lastError) {
            console.log('Content script not loaded, injecting...');
            chrome.scripting.executeScript({
                target: { tabId: tabId },
                files: ['content.js']
            }).then(() => {
                console.log('Content script injected successfully');
                // Wait a moment for the script to initialize
                setTimeout(callback, 1000);
            }).catch((error: any) => {
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
function startAnalysis(): void {
    if (isAnalyzing) return;

    // Send message to content script to start monitoring
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs: chrome.tabs.Tab[]) {
        const tab = tabs[0];
        if (tab && typeof tab.id === 'number') {
            const tabId = tab.id;
            // Check if we're on a YouTube page
            if (!tab.url?.includes('youtube.com/watch')) {
                showNotification('Error: Please navigate to a YouTube video page first');
                return;
            }

            isAnalyzing = true;
            updateUI();

            // Ensure content script is loaded, then start analysis
            ensureContentScriptLoaded(tabId, function() {
                if (!apiUrlInput || !showConfidenceCheckbox) return;

                chrome.tabs.sendMessage(tabId, {
                    action: 'startAnalysis',
                    apiUrl: apiUrlInput.value,
                    showConfidence: showConfidenceCheckbox.checked
                }, function(response: any) {
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
function stopAnalysis(): void {
    if (!isAnalyzing) return;

    isAnalyzing = false;
    updateUI();

    // Send message to content script to stop monitoring
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs: chrome.tabs.Tab[]) {
        const tab = tabs[0];
        if (tab && typeof tab.id === 'number') {
            const tabId = tab.id;
            chrome.tabs.sendMessage(tabId, {
                action: 'stopAnalysis'
            }, function(response: any) {
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

// Reload current tab to reinject content script
function reloadTab(): void {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs: chrome.tabs.Tab[]) {
        const tab = tabs[0];
        if (tab && typeof tab.id === 'number') {
            const tabId = tab.id;
            chrome.tabs.reload(tabId);
            showNotification('Page reloaded - extension will reinitialize');
        }
    });
}

// Update UI based on analysis state
function updateUI(): void {
    if (isAnalyzing) {
        if (startBtn) startBtn.disabled = true;
        if (stopBtn) stopBtn.disabled = false;
        if (statusDiv) statusDiv.className = 'status active';
        if (statusText) statusText.textContent = 'Analysis Active';
    } else {
        if (startBtn) startBtn.disabled = false;
        if (stopBtn) stopBtn.disabled = true;
        if (statusDiv) statusDiv.className = 'status inactive';
        if (statusText) statusText.textContent = 'Analysis Stopped';
    }
}

// Update sentiment statistics
function updateStats(stats: SentimentStats): void {
    popupSentimentStats = stats;
    if (totalMessagesEl) totalMessagesEl.textContent = stats.total.toString();
    if (positiveCountEl) positiveCountEl.textContent = stats.positive.toString();
    if (negativeCountEl) negativeCountEl.textContent = stats.negative.toString();
    if (neutralCountEl) neutralCountEl.textContent = stats.neutral.toString();
}

// Add message to the display
function addMessage(message: SentimentMessage): void {
    messageHistory.unshift(message);

    // Keep only last 10 messages
    if (messageHistory.length > 10) {
        messageHistory = messageHistory.slice(0, 10);
    }

    updateMessageDisplay();
}

// Update message display
function updateMessageDisplay(): void {
    if (!messageListEl) return;

    messageListEl.innerHTML = '';

    messageHistory.forEach(msg => {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${msg.sentiment}`;

        const confidenceText = showConfidenceCheckbox?.checked ?
            ` (${(msg.confidence * 100).toFixed(1)}%)` : '';

        messageDiv.innerHTML = `
            <strong>${msg.sentiment.toUpperCase()}</strong>${confidenceText}:
            ${msg.text.length > 50 ? msg.text.substring(0, 50) + '...' : msg.text}
        `;

        messageListEl.appendChild(messageDiv);
    });
}

// Show notification
function showNotification(message: string): void {
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

// Listen for messages from background via port
port.onMessage.addListener(function(request: MessageData) {
    if (request.action === 'loadData') {
        popupSentimentStats = request.stats || popupSentimentStats;
        messageHistory = request.messages || [];
        updateStats(popupSentimentStats);
        updateMessageDisplay();
    } else if (request.action === 'updateStats') {
        if (request.stats) updateStats(request.stats);
    } else if (request.action === 'addMessage') {
        if (request.message) addMessage(request.message);
    } else if (request.action === 'analysisError') {
        showNotification('Analysis Error: ' + (request.error || 'Unknown error'));
        stopAnalysis();
    } else if (request.action === 'analysisStarted') {
        console.log('Analysis started');
    } else if (request.action === 'analysisStopped') {
        isAnalyzing = false;
        updateUI();
        console.log('Analysis stopped');
    }
});

// Handle extension installation/update
chrome.runtime.onInstalled.addListener(function(details: chrome.runtime.InstalledDetails) {
    // Set default settings
    chrome.storage.sync.set({
        apiUrl: 'http://localhost:8000',
        autoStart: true,
        showConfidence: true
    });
});
