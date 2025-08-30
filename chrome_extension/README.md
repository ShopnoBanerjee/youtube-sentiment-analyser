# YouTube Live Chat Sentiment Analyzer - Chrome Extension

A Chrome extension that provides real-time sentiment analysis of YouTube live chat messages using a FastAPI backend with MLflow model integration.

## Features

- 🎥 **Real-time Analysis**: Monitors YouTube live chat and analyzes sentiment in real-time
- 📊 **Interactive Dashboard**: Web-based dashboard with charts and statistics
- 🎯 **Confidence Scores**: Shows confidence levels for each sentiment prediction
- ⚡ **Auto-start**: Automatically starts analysis when visiting YouTube live streams
- 🔧 **Customizable**: Configurable API server URL and analysis settings
- 📱 **Modern UI**: Clean, responsive interface with real-time updates

## Installation

### 1. Install the Extension

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" in the top right
3. Click "Load unpacked" and select the `chrome_extension` folder
4. The extension should now appear in your extensions list

### 2. Start the API Server

Before using the extension, make sure the FastAPI server is running:

```bash
cd server
pip install -r requirements.txt
python main.py
```

The server should be running on `http://localhost:8000`

### 3. Configure the Extension

1. Click the extension icon in your browser toolbar
2. Configure the API server URL if different from default
3. Enable/disable auto-start and confidence display settings

## Usage

### Basic Usage

1. Navigate to any YouTube live stream
2. Click the extension icon and click "Start Analysis"
3. Watch real-time sentiment analysis of chat messages
4. Click "Open Dashboard" for detailed statistics and charts

### Dashboard Features

- **Sentiment Distribution**: Pie chart showing positive/negative/neutral message ratios
- **Real-time Statistics**: Live counters for each sentiment type
- **Recent Messages**: List of recent messages with sentiment labels
- **Time Series**: Chart showing sentiment trends over time

## Architecture

### Files Structure

```
chrome_extension/
├── manifest.json          # Extension manifest
├── popup.html            # Extension popup interface
├── popup.js              # Popup JavaScript logic
├── content.js            # Content script for YouTube pages
├── background.js         # Background service worker
├── welcome.html          # Welcome page for new users
└── icons/                # Extension icons
```

### API Endpoints

The extension communicates with the FastAPI server using these endpoints:

- `GET /health` - Health check
- `POST /analyze` - Single message analysis
- `POST /analyze-batch` - Batch message analysis
- `GET /dashboard` - Web dashboard
- `WebSocket /ws/dashboard` - Real-time dashboard updates

## Development

### Prerequisites

- Python 3.11+
- Node.js (for development tools)
- Chrome browser

### Setting up Development Environment

1. **Backend Setup**:
   ```bash
   cd server
   python -m venv venv
   venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. **Extension Development**:
   - Make changes to the extension files
   - Reload the extension in `chrome://extensions/`
   - Check the browser console for debugging

### Building and Testing

1. **Test the API Server**:
   ```bash
   cd server
   python main.py
   # Visit http://localhost:8000/docs for API documentation
   ```

2. **Test the Extension**:
   - Load the extension in developer mode
   - Navigate to a YouTube live stream
   - Open browser developer tools to see console logs

## Troubleshooting

### Common Issues

1. **Extension not loading**:
   - Make sure all files are in the correct directory structure
   - Check that manifest.json is valid JSON
   - Reload the extension in chrome://extensions

2. **API server connection failed**:
   - Ensure the FastAPI server is running on the correct port
   - Check that the API URL in extension settings is correct
   - Verify CORS settings allow the extension to connect

3. **Chat messages not being analyzed**:
   - Make sure you're on a YouTube live stream (not a regular video)
   - Check that the live chat is visible on the page
   - Look for errors in the browser console

4. **Model not loaded**:
   - Ensure the model artifacts exist in the experiments folder
   - Check that the server can access the model files
   - Verify PyTorch and dependencies are installed

### Debug Mode

Enable debug logging by opening the browser developer tools:
- Go to the YouTube page
- Open DevTools (F12)
- Check the Console tab for extension messages

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Check the troubleshooting section above
- Open an issue on GitHub
- Review the API documentation at `/docs` when the server is running
