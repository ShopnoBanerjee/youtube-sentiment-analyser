// YouTube Live Chat Sentiment Analyzer - Background Script

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

interface MessageData {
    action: string;
    stats?: SentimentStats;
    message?: SentimentMessage;
    messages?: SentimentMessage[];
    error?: string;
    [key: string]: any; // For additional properties
}

interface Settings {
    apiUrl: string;
    autoStart: boolean;
    showConfidence: boolean;
}

// Handle extension installation
chrome.runtime.onInstalled.addListener(function(details: chrome.runtime.InstalledDetails) {
    if (details.reason === 'install') {
        console.log('YouTube Sentiment Analyzer extension installed');

        // Set default settings
        const defaultSettings: Settings = {
            apiUrl: 'http://localhost:8000',
            autoStart: true,
            showConfidence: true
        };
        chrome.storage.sync.set(defaultSettings);

        // Initialize stored data
        const defaultStats: SentimentStats = { total: 0, positive: 0, negative: 0, neutral: 0 };
        chrome.storage.local.set({
            sentimentStats: defaultStats,
            recentMessages: []
        });

        // Show welcome message
        chrome.tabs.create({
            url: chrome.runtime.getURL('welcome.html')
        });
    } else if (details.reason === 'update') {
        console.log('YouTube Sentiment Analyzer extension updated');
        // Initialize stored data if not exists
        chrome.storage.local.get(['sentimentStats'], function(result: { [key: string]: any }) {
            if (!result.sentimentStats) {
                const defaultStats: SentimentStats = { total: 0, positive: 0, negative: 0, neutral: 0 };
                chrome.storage.local.set({
                    sentimentStats: defaultStats,
                    recentMessages: []
                });
            }
        });
    }
});

// Handle messages from content scripts and popup
let popupPort: chrome.runtime.Port | null = null;
let currentStats: SentimentStats = { total: 0, positive: 0, negative: 0, neutral: 0 };
let recentMessages: SentimentMessage[] = [];

chrome.runtime.onConnect.addListener(function(port: chrome.runtime.Port) {
    if (port.name === 'popup') {
        popupPort = port;
        // Send current data to popup
        popupPort.postMessage({ action: 'loadData', stats: currentStats, messages: recentMessages });
        popupPort.onMessage.addListener(function(message: any) {
            // Handle messages from popup if needed
        });
        popupPort.onDisconnect.addListener(function() {
            popupPort = null;
        });
    }
});

chrome.runtime.onMessage.addListener(function(request: MessageData, sender: chrome.runtime.MessageSender, sendResponse: (response?: any) => void) {
    if (request.action === 'log') {
        console.log('[Content Script]', request.message);
    } else if (request.action === 'error') {
        console.error('[Content Script Error]', request.error);
    } else if (request.action === 'updateStats') {
        if (request.stats) {
            currentStats = request.stats;
            chrome.storage.local.set({ sentimentStats: currentStats });
            if (popupPort) {
                popupPort.postMessage(request);
            }
        }
    } else if (request.action === 'addMessage') {
        if (request.message) {
            recentMessages.unshift(request.message);
            if (recentMessages.length > 10) {
                recentMessages = recentMessages.slice(0, 10);
            }
            chrome.storage.local.set({ recentMessages: recentMessages });
            if (popupPort) {
                popupPort.postMessage(request);
            }
        }
    } else {
        // Relay other messages to popup if connected
        if (popupPort) {
            popupPort.postMessage(request);
        }
    }
});

// Handle tab updates to detect YouTube pages
chrome.tabs.onUpdated.addListener(function(tabId: number, changeInfo: chrome.tabs.TabChangeInfo, tab?: chrome.tabs.Tab) {
    if (changeInfo.status === 'complete' && tab?.url && tab.url.includes('youtube.com/watch')) {
        // Check if content script is already injected by trying to send a message
        chrome.tabs.sendMessage(tabId, { action: 'ping' }, function(response: any) {
            if (chrome.runtime.lastError) {
                // Content script not injected yet, so inject it
                chrome.scripting.executeScript({
                    target: { tabId: tabId },
                    files: ['content.js']
                }).catch((error: any) => {
                    console.log('Content script injection failed:', error);
                });
            } else {
                console.log('Content script already running on this tab');
            }
        });
    }
});

// Handle extension icon click
chrome.action.onClicked.addListener(function(tab?: chrome.tabs.Tab) {
    // Open popup programmatically if needed
    chrome.action.openPopup();
});

// Periodic health check of the API server
function checkServerHealth(): void {
    chrome.storage.sync.get(['apiUrl'], function(result: { [key: string]: any }) {
        const apiUrl = result.apiUrl || 'http://localhost:8000';

        fetch(`${apiUrl}/health`)
            .then((response: Response) => {
                if (response.ok) {
                    console.log('API server is healthy');
                } else {
                    console.warn('API server returned non-OK status:', response.status);
                }
            })
            .catch((error: any) => {
                console.warn('API server health check failed:', error);
            });
    });
}

// Check server health every 30 seconds
setInterval(checkServerHealth, 30000);

// Initial health check
checkServerHealth();

// Handle extension suspension
chrome.runtime.onSuspend.addListener(function() {
    console.log('Extension is being suspended');
});

// Error handling
self.addEventListener('error', function(event: ErrorEvent) {
    console.error('Background script error:', event.error);
});

self.addEventListener('unhandledrejection', function(event: PromiseRejectionEvent) {
    console.error('Background script unhandled rejection:', event.reason);
});
