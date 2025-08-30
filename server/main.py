"""
FastAPI server for real-time YouTube live chat sentiment analysis
"""
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import mlflow
import pandas as pd
from collections import Counter
import logging
from datetime import datetime
import json
from pathlib import Path
from tokenizer import SimpleTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="YouTube Sentiment Analysis API",
    description="Real-time sentiment analysis for YouTube live chat",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Mount static files for dashboard
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for model and artifacts
model = None
tokenizer = None
label_encoder = None
device = torch.device('cpu')

class SentimentLSTM(nn.Module):
    """PyTorch LSTM Model for sentiment analysis"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout_rate, bidirectional=True):
        super(SentimentLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Calculate input dimension for fully connected layer
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Fully connected layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(lstm_output_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Use the last hidden state (for classification)
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]

        # Fully connected layers
        output = self.dropout(hidden)
        output = F.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.fc2(output)

        return output

class SentimentRequest(BaseModel):
    text: str
    max_length: int = 100  # Match original training parameter

class BatchSentimentRequest(BaseModel):
    texts: List[str]
    max_length: int = 100  # Match original training parameter

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    summary: Dict[str, Any]

def load_model_artifacts():
    """Load model and artifacts from experiments folder"""
    global model, tokenizer, label_encoder

    try:
        # Try to load tokenizer from pickle
        try:
            with open('../experiments/tokenizer.pkl', 'rb') as f:
                tokenizer = pickle.load(f)
            logger.info("Tokenizer loaded from pickle file")
        except (AttributeError, pickle.UnpicklingError) as e:
            logger.warning(f"Failed to load tokenizer from pickle: {e}")
            logger.info("Recreating tokenizer from training data...")

            # Recreate tokenizer from training data with CORRECT parameters
            try:
                df = pd.read_csv('../preprocessing_eda/data/processed/Preprocessed_Data.csv')
                tokenizer = SimpleTokenizer(max_features=10000)  # Match original: 10000
                tokenizer.fit_on_texts(df['clean_text'].dropna())
                logger.info("Tokenizer recreated successfully")
            except Exception as recreate_error:
                logger.error(f"Failed to recreate tokenizer: {recreate_error}")
                raise Exception("Could not load or recreate tokenizer")

        # Load label encoder
        with open('../experiments/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)

        # Load model
        model_path = '../experiments/best_model.pth'
        if os.path.exists(model_path):
            # Get model parameters - MATCH ORIGINAL TRAINING CONFIG
            vocab_size = len(tokenizer.word_to_index)
            embedding_dim = 128  # Match original: 128
            hidden_dim = 128     # Match original: 128
            num_layers = 1
            num_classes = len(label_encoder.classes_)
            dropout_rate = 0.3
            bidirectional = True

            model = SentimentLSTM(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout_rate=dropout_rate,
                bidirectional=False  # Match training config: False
            )

            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            logger.info("Model and artifacts loaded successfully")
        else:
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        raise

def preprocess_text(text: str, max_length: int = 100) -> torch.Tensor:  # Match original: 100
    """Preprocess text for model input"""
    # Tokenize
    tokens = tokenizer.texts_to_sequences([text])[0]

    # Pad or truncate
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    else:
        tokens = tokens + [0] * (max_length - len(tokens))

    return torch.tensor([tokens], dtype=torch.long).to(device)

def predict_sentiment(text: str, max_length: int = 50) -> Dict[str, Any]:
    """Predict sentiment for a single text"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Preprocess text
        input_tensor = preprocess_text(text, max_length)

        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_class = torch.argmax(outputs, dim=1).cpu().numpy()[0]

        # Get sentiment label and confidence
        sentiment = str(label_encoder.classes_[predicted_class])
        confidence = float(probabilities[predicted_class])

        # Create probabilities dict
        probs_dict = {str(label): float(prob) for label, prob in zip(label_encoder.classes_, probabilities)}

        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
            "probabilities": probs_dict
        }

    except Exception as e:
        logger.error(f"Error predicting sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model_artifacts()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "YouTube Sentiment Analysis API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "label_encoder_loaded": label_encoder is not None
    }

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """Analyze sentiment of a single text"""
    return predict_sentiment(request.text, request.max_length)

@app.post("/analyze-batch", response_model=BatchSentimentResponse)
async def analyze_batch_sentiment(request: BatchSentimentRequest):
    """Analyze sentiment of multiple texts"""
    results = []
    sentiment_counts = {}

    for text in request.texts:
        result = predict_sentiment(text, request.max_length)
        results.append(result)

        # Count sentiments
        sentiment = result["sentiment"]
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

    # Calculate summary statistics
    total_texts = len(results)
    summary = {
        "total_texts": total_texts,
        "sentiment_distribution": sentiment_counts,
        "sentiment_percentages": {
            sentiment: count / total_texts * 100
            for sentiment, count in sentiment_counts.items()
        }
    }

    return {"results": results, "summary": summary}

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard HTML"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>YouTube Live Chat Sentiment Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .header {
                background: #2c3e50;
                color: white;
                padding: 20px;
                text-align: center;
            }
            .dashboard-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                padding: 20px;
            }
            .chart-container {
                background: #f8f9fa;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .stats-container {
                background: #f8f9fa;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .stat-card {
                background: white;
                border-radius: 6px;
                padding: 15px;
                margin: 10px 0;
                box-shadow: 0 1px 5px rgba(0,0,0,0.1);
            }
            .sentiment-positive { border-left: 4px solid #28a745; }
            .sentiment-negative { border-left: 4px solid #dc3545; }
            .sentiment-neutral { border-left: 4px solid #ffc107; }
            .live-indicator {
                display: inline-block;
                width: 10px;
                height: 10px;
                background: #dc3545;
                border-radius: 50%;
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            .chat-messages {
                max-height: 400px;
                overflow-y: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                background: #f8f9fa;
            }
            .message {
                padding: 8px;
                margin: 5px 0;
                border-radius: 5px;
                font-size: 14px;
            }
            .message.positive { background: #d4edda; border-left: 4px solid #28a745; }
            .message.negative { background: #f8d7da; border-left: 4px solid #dc3545; }
            .message.neutral { background: #fff3cd; border-left: 4px solid #ffc107; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎥 YouTube Live Chat Sentiment Analysis</h1>
                <p><span class="live-indicator"></span> Live Analysis Active</p>
            </div>

            <div class="dashboard-grid">
                <div class="chart-container">
                    <h3>Sentiment Distribution</h3>
                    <canvas id="sentimentChart"></canvas>
                </div>

                <div class="stats-container">
                    <h3>Real-time Statistics</h3>
                    <div class="stat-card sentiment-positive">
                        <h4>Positive Messages</h4>
                        <div id="positive-count">0</div>
                    </div>
                    <div class="stat-card sentiment-negative">
                        <h4>Negative Messages</h4>
                        <div id="negative-count">0</div>
                    </div>
                    <div class="stat-card sentiment-neutral">
                        <h4>Neutral Messages</h4>
                        <div id="neutral-count">0</div>
                    </div>
                    <div class="stat-card">
                        <h4>Total Messages</h4>
                        <div id="total-count">0</div>
                    </div>
                </div>

                <div class="chart-container">
                    <h3>Sentiment Over Time</h3>
                    <canvas id="timeChart"></canvas>
                </div>

                <div class="stats-container">
                    <h3>Recent Messages</h3>
                    <div id="chat-messages" class="chat-messages"></div>
                </div>
            </div>
        </div>

        <script>
            let sentimentChart, timeChart;
            let sentimentData = { positive: 0, negative: 0, neutral: 0 };
            let timeData = [];
            let messageHistory = [];

            // Initialize charts
            function initCharts() {
                const sentimentCtx = document.getElementById('sentimentChart').getContext('2d');
                sentimentChart = new Chart(sentimentCtx, {
                    type: 'doughnut',
                    data: {
                        labels: ['Positive', 'Negative', 'Neutral'],
                        datasets: [{
                            data: [0, 0, 0],
                            backgroundColor: ['#28a745', '#dc3545', '#ffc107'],
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: { position: 'bottom' }
                        }
                    }
                });

                const timeCtx = document.getElementById('timeChart').getContext('2d');
                timeChart = new Chart(timeCtx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [
                            {
                                label: 'Positive',
                                data: [],
                                borderColor: '#28a745',
                                backgroundColor: 'rgba(40, 167, 69, 0.1)',
                                tension: 0.4
                            },
                            {
                                label: 'Negative',
                                data: [],
                                borderColor: '#dc3545',
                                backgroundColor: 'rgba(220, 53, 69, 0.1)',
                                tension: 0.4
                            },
                            {
                                label: 'Neutral',
                                data: [],
                                borderColor: '#ffc107',
                                backgroundColor: 'rgba(255, 193, 7, 0.1)',
                                tension: 0.4
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: { beginAtZero: true }
                        }
                    }
                });
            }

            // Update dashboard with new data
            function updateDashboard(data) {
                // Update sentiment counts
                sentimentData = data.sentiment_distribution || sentimentData;

                document.getElementById('positive-count').textContent = sentimentData.positive || 0;
                document.getElementById('negative-count').textContent = sentimentData.negative || 0;
                document.getElementById('neutral-count').textContent = sentimentData.neutral || 0;
                document.getElementById('total-count').textContent = data.total_messages || 0;

                // Update sentiment chart
                sentimentChart.data.datasets[0].data = [
                    sentimentData.positive || 0,
                    sentimentData.negative || 0,
                    sentimentData.neutral || 0
                ];
                sentimentChart.update();

                // Update time chart
                if (data.timestamp) {
                    const timeLabel = new Date(data.timestamp).toLocaleTimeString();
                    timeData.push({
                        time: timeLabel,
                        positive: sentimentData.positive || 0,
                        negative: sentimentData.negative || 0,
                        neutral: sentimentData.neutral || 0
                    });

                    // Keep only last 20 data points
                    if (timeData.length > 20) {
                        timeData.shift();
                    }

                    timeChart.data.labels = timeData.map(d => d.time);
                    timeChart.data.datasets[0].data = timeData.map(d => d.positive);
                    timeChart.data.datasets[1].data = timeData.map(d => d.negative);
                    timeChart.data.datasets[2].data = timeData.map(d => d.neutral);
                    timeChart.update();
                }

                // Update message history
                if (data.recent_messages) {
                    messageHistory = data.recent_messages.slice(-10); // Keep last 10 messages
                    updateMessageDisplay();
                }
            }

            function updateMessageDisplay() {
                const container = document.getElementById('chat-messages');
                container.innerHTML = '';

                messageHistory.forEach(msg => {
                    const messageDiv = document.createElement('div');
                    messageDiv.className = `message ${msg.sentiment.toLowerCase()}`;
                    messageDiv.innerHTML = `
                        <strong>${msg.sentiment}</strong> (${(msg.confidence * 100).toFixed(1)}%): ${msg.text}
                    `;
                    container.appendChild(messageDiv);
                });

                container.scrollTop = container.scrollHeight;
            }

            // WebSocket connection for real-time updates
            let ws;
            function connectWebSocket() {
                ws = new WebSocket('ws://localhost:8000/ws/dashboard');

                ws.onopen = function(event) {
                    console.log('Connected to dashboard WebSocket');
                };

                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };

                ws.onclose = function(event) {
                    console.log('Dashboard WebSocket closed, reconnecting...');
                    setTimeout(connectWebSocket, 1000);
                };

                ws.onerror = function(error) {
                    console.error('Dashboard WebSocket error:', error);
                };
            }

            // Initialize everything when page loads
            document.addEventListener('DOMContentLoaded', function() {
                initCharts();
                connectWebSocket();

                // Simulate some initial data (remove this in production)
                setTimeout(() => {
                    updateDashboard({
                        sentiment_distribution: { positive: 15, negative: 3, neutral: 8 },
                        total_messages: 26,
                        timestamp: new Date().toISOString(),
                        recent_messages: [
                            { text: "Great stream!", sentiment: "positive", confidence: 0.95 },
                            { text: "This is amazing!", sentiment: "positive", confidence: 0.89 },
                            { text: "Not sure about this", sentiment: "neutral", confidence: 0.67 }
                        ]
                    });
                }, 1000);
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# WebSocket endpoint for real-time dashboard updates
@app.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # In a real implementation, you'd receive data from the Chrome extension
            # For now, we'll just keep the connection alive
            data = await websocket.receive_text()
            # Process the data and send back analysis results
            await websocket.send_text(json.dumps({
                "status": "received",
                "data": data
            }))
    except WebSocketDisconnect:
        logger.info("Dashboard WebSocket disconnected")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
