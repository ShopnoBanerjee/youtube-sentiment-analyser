# YouTube Live Chat Sentiment Analyzer

A complete real-time sentiment analysis system for YouTube live chat messages, consisting of a FastAPI backend with MLflow model integration and a Chrome extension for seamless analysis.

## 🚀 Features

- **Real-time Analysis**: Monitor and analyze YouTube live chat sentiment in real-time
- **MLflow Integration**: Uses pre-trained PyTorch LSTM models registered in MLflow
- **Chrome Extension**: Browser extension for seamless YouTube integration
- **Interactive Dashboard**: Web-based dashboard with real-time charts and statistics
- **Batch Processing**: Efficient batch analysis for multiple messages
- **RESTful API**: Clean FastAPI endpoints for easy integration
- **WebSocket Support**: Real-time updates via WebSocket connections

## 📋 Prerequisites

- Python 3.11+
- Chrome browser
- PyTorch-compatible hardware (CPU/GPU)
- MLflow server access

## 🛠️ Installation & Setup

### 1. Backend Setup

```bash
# Navigate to server directory
cd server

# Install dependencies
pip install -e .

# Start the FastAPI server
python main.py
```

The server will start on `http://localhost:8000` with automatic API documentation at `http://localhost:8000/docs`.

### 2. Chrome Extension Setup

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top right)
3. Click "Load unpacked" and select the `chrome_extension` folder
4. The extension icon should appear in your toolbar

### 3. Model Setup

Ensure your MLflow model artifacts are in the correct location:
- `../experiments/best_model.pth` - PyTorch model weights
- `../experiments/tokenizer.pkl` - Text tokenizer
- `../experiments/label_encoder.pkl` - Label encoder

## 🎯 Usage

### Starting the System

1. **Start the API Server**:
   ```bash
   cd server
   python main.py
   ```

2. **Use the Chrome Extension**:
   - Navigate to any YouTube live stream
   - Click the extension icon
   - Click "Start Analysis"
   - View real-time sentiment analysis

3. **Access the Dashboard**:
   - Click "Open Dashboard" in the extension popup
   - Or visit `http://localhost:8000/dashboard` directly

### API Endpoints

#### REST Endpoints
- `GET /` - Root endpoint with system info
- `GET /health` - Health check and model status
- `POST /analyze` - Analyze single text message
- `POST /analyze-batch` - Analyze multiple messages
- `GET /dashboard` - Interactive web dashboard

#### WebSocket Endpoints
- `WebSocket /ws/dashboard` - Real-time dashboard updates

#### Example API Usage

```python
import requests

# Single message analysis
response = requests.post("http://localhost:8000/analyze",
    json={"text": "This is amazing!", "max_length": 50}
)
print(response.json())

# Batch analysis
response = requests.post("http://localhost:8000/analyze-batch",
    json={"texts": ["Great stream!", "This is terrible"], "max_length": 50}
)
print(response.json())
```

## 🏗️ Architecture

### Backend Components

- **FastAPI Server**: RESTful API with automatic OpenAPI documentation
- **PyTorch Model**: LSTM-based sentiment classifier
- **MLflow Integration**: Model versioning and artifact management
- **WebSocket Support**: Real-time communication
- **Dashboard**: Interactive web interface with Chart.js

### Chrome Extension Components

- **Popup Interface**: Control panel for analysis settings
- **Content Script**: Monitors YouTube live chat messages
- **Background Script**: Handles API communication and extension lifecycle
- **Real-time Updates**: WebSocket-based live data streaming

### Data Flow

1. Chrome extension monitors YouTube live chat
2. New messages are sent to FastAPI server for analysis
3. PyTorch model processes messages and returns sentiment predictions
4. Results are displayed in extension popup and web dashboard
5. Real-time statistics and charts update continuously

## 🔧 Configuration

### Server Configuration

The server automatically loads model artifacts from the experiments folder. You can modify parameters in `main.py`:

```python
# Model parameters
max_features = 5000  # Vocabulary size
maxlen = 50         # Maximum sequence length
embedding_dim = 64  # Embedding dimensions
```

### Extension Configuration

Configure the extension through the popup interface:
- **API Server URL**: Change the backend server address
- **Auto-start**: Automatically begin analysis on YouTube pages
- **Show Confidence**: Display confidence scores with predictions

## 📊 Dashboard Features

- **Sentiment Distribution**: Real-time pie chart of message sentiments
- **Live Statistics**: Counters for positive/negative/neutral messages
- **Recent Messages**: Scrollable list of recent chat messages with sentiment labels
- **Time Series Chart**: Sentiment trends over time
- **WebSocket Updates**: Live data streaming without page refresh

## 🐛 Troubleshooting

### Common Issues

1. **Server won't start**:
   - Check Python version (3.11+ required)
   - Ensure all dependencies are installed
   - Verify model artifacts exist

2. **Extension not working**:
   - Reload extension in `chrome://extensions`
   - Check browser console for errors
   - Ensure API server is running

3. **Model prediction errors**:
   - Verify model files are not corrupted
   - Check PyTorch installation
   - Ensure compatible Python versions

### Debug Mode

Enable debug logging:
- Server: Check console output when starting
- Extension: Open DevTools on YouTube page and check Console tab

## 🚀 Deployment

### Production Deployment

1. **Server Deployment**:
   ```bash
   # Using uvicorn with production settings
   uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

2. **Model Updates**:
   - Update model artifacts in experiments folder
   - Restart server to load new models
   - Use MLflow for model versioning

3. **Extension Updates**:
   - Update extension files
   - Reload extension in browser
   - Publish updates to Chrome Web Store

## 📈 Performance

- **Batch Processing**: Handles multiple messages efficiently
- **CPU Optimization**: Optimized for CPU-based inference
- **Memory Management**: Automatic cleanup to prevent memory leaks
- **Real-time Updates**: WebSocket-based streaming for live data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper testing
4. Update documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

- **API Documentation**: Visit `/docs` when server is running
- **Extension Logs**: Check browser DevTools console
- **Server Logs**: Monitor terminal output
- **GitHub Issues**: Report bugs and request features

---

**Note**: This system requires YouTube live streams with active chat. Regular YouTube videos without live chat will not work with the extension.