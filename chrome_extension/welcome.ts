// Welcome page script for YouTube Sentiment Analyzer extension

document.addEventListener('DOMContentLoaded', function() {
    // Add event listeners to buttons
    const openDashboardBtn = document.getElementById('open-dashboard-btn');
    const closeWelcomeBtn = document.getElementById('close-welcome-btn');

    if (openDashboardBtn) {
        openDashboardBtn.addEventListener('click', openDashboard);
    }

    if (closeWelcomeBtn) {
        closeWelcomeBtn.addEventListener('click', closeWelcome);
    }
});

function openDashboard() {
    chrome.tabs.create({ url: 'http://localhost:8000/dashboard' });
    window.close();
}

function closeWelcome() {
    window.close();
}
