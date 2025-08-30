// YouTube Live Chat Sentiment Analyzer - Content Script

let isMonitoring = false;
let observer = null;
let processedMessages = new Set();
let sentimentStats = {
    total: 0,
    positive: 0,
    negative: 0,
    neutral: 0
};
let apiUrl = 'http://localhost:8000';
let showConfidence = true;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('YouTube Sentiment Analyzer content script loaded');

    // Check if this is a YouTube watch page
    if (window.location.href.includes('youtube.com/watch')) {
        // Auto-start if enabled
        chrome.storage.sync.get(['autoStart'], function(result) {
            if (result.autoStart !== false) {
                setTimeout(startMonitoring, 3000); // Wait for page to load
            }
        });
    }
});

// Start monitoring live chat
function startMonitoring() {
    if (isMonitoring) return;

    console.log('Starting YouTube live chat monitoring...');
    isMonitoring = true;

    // Find the live chat container
    const chatContainer = findChatContainer();

    if (chatContainer) {
        console.log('Found chat container, setting up observer...');
        setupChatObserver(chatContainer);
        sendMessageToPopup({ action: 'analysisStarted' });
    } else {
        console.log('Chat container not found, will retry...');
        // Retry after a delay
        setTimeout(() => {
            const retryContainer = findChatContainer();
            if (retryContainer) {
                setupChatObserver(retryContainer);
                sendMessageToPopup({ action: 'analysisStarted' });
            } else {
                console.error('Could not find YouTube live chat container');
                sendMessageToPopup({
                    action: 'analysisError',
                    error: 'Could not find live chat. Make sure you are on a live stream.'
                });
            }
        }, 5000);
    }
}

// Stop monitoring
function stopMonitoring() {
    if (!isMonitoring) return;

    console.log('Stopping YouTube live chat monitoring...');
    isMonitoring = false;

    if (observer) {
        observer.disconnect();
        observer = null;
    }

    processedMessages.clear();
    sentimentStats = { total: 0, positive: 0, negative: 0, neutral: 0 };
    sendMessageToPopup({ action: 'analysisStopped' });
}

// Find the YouTube live chat container
function findChatContainer() {
    // Try different selectors for live chat
    const selectors = [
        '#chat-messages', // Live chat messages container
        '#chat #items', // Alternative chat container
        'yt-live-chat-renderer #items', // Polymer-based chat
        '[class*="live-chat"] [class*="message"]', // Generic live chat
        '#contents #chat-messages', // Another variation
    ];

    for (const selector of selectors) {
        const container = document.querySelector(selector);
        if (container) {
            console.log(`Found chat container with selector: ${selector}`);
            return container;
        }
    }

    // Try to find by traversing the DOM
    const chatFrame = document.querySelector('#chatframe');
    if (chatFrame) {
        try {
            const chatDoc = chatFrame.contentDocument || chatFrame.contentWindow.document;
            const chatContainer = chatDoc.querySelector('#chat-messages, #items');
            if (chatContainer) {
                console.log('Found chat container in iframe');
                return chatContainer;
            }
        } catch (e) {
            console.log('Could not access iframe content:', e);
        }
    }

    return null;
}

// Set up mutation observer for chat messages
function setupChatObserver(container) {
    observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                processNewMessages(mutation.addedNodes);
            }
        });
    });

    observer.observe(container, {
        childList: true,
        subtree: true
    });

    console.log('Chat observer set up successfully');

    // Process existing messages
    const existingMessages = container.querySelectorAll('[class*="message"], [class*="comment"]');
    if (existingMessages.length > 0) {
        processNewMessages(existingMessages);
    }
}

// Process new chat messages
function processNewMessages(nodes) {
    const messages = [];

    nodes.forEach(node => {
        if (node.nodeType === Node.ELEMENT_NODE) {
            // Try different selectors for message text
            const messageSelectors = [
                '[class*="message"], [class*="content"]',
                '[class*="text"]',
                '[class*="body"]'
            ];

            let messageText = null;
            for (const selector of messageSelectors) {
                const element = node.querySelector(selector);
                if (element) {
                    messageText = element.textContent || element.innerText;
                    break;
                }
            }

            // If no specific selector worked, try getting text from the node itself
            if (!messageText) {
                messageText = node.textContent || node.innerText;
            }

            // Clean up the message text
            if (messageText) {
                messageText = messageText.trim();

                // Skip if message is too short or too long
                if (messageText.length < 3 || messageText.length > 500) {
                    return;
                }

                // Create a unique ID for the message
                const messageId = generateMessageId(node, messageText);

                // Skip if already processed
                if (processedMessages.has(messageId)) {
                    return;
                }

                processedMessages.add(messageId);
                messages.push({
                    id: messageId,
                    text: messageText,
                    timestamp: Date.now()
                });
            }
        }
    });

    // Process messages in batches
    if (messages.length > 0) {
        analyzeMessages(messages);
    }
}

// Generate unique message ID
function generateMessageId(node, text) {
    // Use a combination of node attributes and text hash
    const nodeId = node.id || node.className || '';
    const textHash = hashString(text);
    return `${nodeId}_${textHash}`;
}

// Simple string hash function
function hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
}

// Analyze messages using the API
async function analyzeMessages(messages) {
    try {
        const texts = messages.map(msg => msg.text);

        const response = await fetch(`${apiUrl}/analyze-batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                texts: texts,
                max_length: 50
            })
        });

        if (!response.ok) {
            throw new Error(`API request failed: ${response.status}`);
        }

        const result = await response.json();

        // Update statistics and send to popup
        result.results.forEach((analysis, index) => {
            const message = messages[index];
            const sentiment = analysis.sentiment.toLowerCase();

            sentimentStats.total++;
            sentimentStats[sentiment]++;

            // Send message to popup
            sendMessageToPopup({
                action: 'addMessage',
                message: {
                    text: message.text,
                    sentiment: sentiment,
                    confidence: analysis.confidence
                }
            });
        });

        // Send updated stats to popup
        sendMessageToPopup({
            action: 'updateStats',
            stats: sentimentStats
        });

    } catch (error) {
        console.error('Error analyzing messages:', error);
        sendMessageToPopup({
            action: 'analysisError',
            error: error.message
        });
    }
}

// Send message to popup
function sendMessageToPopup(message) {
    chrome.runtime.sendMessage(message);
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    if (request.action === 'startAnalysis') {
        apiUrl = request.apiUrl;
        showConfidence = request.showConfidence;
        startMonitoring();
        sendResponse({ success: true });
    } else if (request.action === 'stopAnalysis') {
        stopMonitoring();
        sendResponse({ success: true });
    } else if (request.action === 'getStats') {
        sendResponse({
            isMonitoring: isMonitoring,
            stats: sentimentStats
        });
    } else if (request.action === 'ping') {
        // Respond to ping to indicate content script is running
        sendResponse({ pong: true });
    }
});

// Handle page navigation (YouTube SPA)
let currentUrl = window.location.href;
setInterval(() => {
    if (window.location.href !== currentUrl) {
        currentUrl = window.location.href;
        if (isMonitoring && !window.location.href.includes('youtube.com/watch')) {
            // Stop monitoring if navigated away from watch page
            stopMonitoring();
        } else if (!isMonitoring && window.location.href.includes('youtube.com/watch')) {
            // Auto-start if navigated to watch page
            chrome.storage.sync.get(['autoStart'], function(result) {
                if (result.autoStart !== false) {
                    setTimeout(startMonitoring, 2000);
                }
            });
        }
    }
}, 1000);

// Clean up on page unload
window.addEventListener('beforeunload', function() {
    if (observer) {
        observer.disconnect();
    }
});
