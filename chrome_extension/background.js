// YouTube Live Chat Sentiment Analyzer - Background Script

// Handle extension installation
chrome.runtime.onInstalled.addListener(function(details) {
    if (details.reason === 'install') {
        console.log('YouTube Sentiment Analyzer extension installed');

        // Set default settings
        chrome.storage.sync.set({
            apiUrl: 'http://localhost:8000',
            autoStart: true,
            showConfidence: true
        });

        // Show welcome message
        chrome.tabs.create({
            url: chrome.runtime.getURL('welcome.html')
        });
    } else if (details.reason === 'update') {
        console.log('YouTube Sentiment Analyzer extension updated');
    }
});

// Handle messages from content scripts and popup
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    if (request.action === 'log') {
        console.log('[Content Script]', request.message);
    } else if (request.action === 'error') {
        console.error('[Content Script Error]', request.error);
    }
});

// Handle tab updates to detect YouTube pages
chrome.tabs.onUpdated.addListener(function(tabId, changeInfo, tab) {
    if (changeInfo.status === 'complete' && tab.url && tab.url.includes('youtube.com/watch')) {
        // Check if content script is already injected by trying to send a message
        chrome.tabs.sendMessage(tabId, { action: 'ping' }, function(response) {
            if (chrome.runtime.lastError) {
                // Content script not injected yet, so inject it
                chrome.scripting.executeScript({
                    target: { tabId: tabId },
                    files: ['content.js']
                }).catch(error => {
                    console.log('Content script injection failed:', error);
                });
            } else {
                console.log('Content script already running on this tab');
            }
        });
    }
});

// Handle extension icon click
chrome.action.onClicked.addListener(function(tab) {
    // Open popup programmatically if needed
    chrome.action.openPopup();
});

// Periodic health check of the API server
function checkServerHealth() {
    chrome.storage.sync.get(['apiUrl'], function(result) {
        const apiUrl = result.apiUrl || 'http://localhost:8000';

        fetch(`${apiUrl}/health`)
            .then(response => {
                if (response.ok) {
                    console.log('API server is healthy');
                } else {
                    console.warn('API server returned non-OK status:', response.status);
                }
            })
            .catch(error => {
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
self.addEventListener('error', function(event) {
    console.error('Background script error:', event.error);
});

self.addEventListener('unhandledrejection', function(event) {
    console.error('Background script unhandled rejection:', event.reason);
});
