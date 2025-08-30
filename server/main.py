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
        # Try to load tokenizer from JSON first, then fallback to pickle, then recreate
        tokenizer_json_path = '../experiments/tokenizer.json'
        tokenizer_pickle_path = '../experiments/tokenizer.pkl'

        if os.path.exists(tokenizer_json_path):
            # Load from JSON (preferred method)
            try:
                tokenizer = SimpleTokenizer.load(tokenizer_json_path)
                logger.info("Tokenizer loaded from JSON file")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer from JSON: {e}")
                tokenizer = None
        else:
            tokenizer = None

        # If JSON loading failed or file doesn't exist, try pickle
        if tokenizer is None and os.path.exists(tokenizer_pickle_path):
            try:
                with open(tokenizer_pickle_path, 'rb') as f:
                    tokenizer = pickle.load(f)
                logger.info("Tokenizer loaded from pickle file")
                # Save as JSON for future use
                tokenizer.save(tokenizer_json_path)
                logger.info("Tokenizer saved as JSON for future use")
            except (AttributeError, pickle.UnpicklingError, ImportError) as e:
                logger.warning(f"Failed to load tokenizer from pickle: {e}")
                tokenizer = None

        # If both failed, recreate from training data
        if tokenizer is None:
            logger.info("Recreating tokenizer from training data...")
            try:
                df = pd.read_csv('../preprocessing_eda/data/processed/Preprocessed_Data.csv')
                tokenizer = SimpleTokenizer(max_features=10000)  # Match original: 10000
                tokenizer.fit_on_texts(df['clean_text'].dropna())
                # Save for future use
                tokenizer.save(tokenizer_json_path)
                logger.info("Tokenizer recreated and saved successfully")
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
        predicted_label = label_encoder.classes_[predicted_class]
        if predicted_label == -1.0:
            sentiment = "negative"
        elif predicted_label == 0.0:
            sentiment = "neutral"
        elif predicted_label == 1.0:
            sentiment = "positive"
        else:
            sentiment = "neutral"  # fallback
        confidence = float(probabilities[predicted_class])

        # Create probabilities dict
        label_to_sentiment = {-1.0: "negative", 0.0: "neutral", 1.0: "positive"}
        probs_dict = {label_to_sentiment.get(label, "neutral"): float(prob) for label, prob in zip(label_encoder.classes_, probabilities)}

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
    print(results)
    return {"results": results, "summary": summary}

# WebSocket endpoint for real-time dashboard updates
@app.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    await websocket.accept()
    logger.info("Dashboard WebSocket connected")
    try:
        while True:
            # Receive data from Chrome extension
            data = await websocket.receive_text()
            try:
                message_data = json.loads(data)
                
                # Process sentiment analysis data
                if "action" in message_data and message_data["action"] == "sentiment_update":
                    # Send back processed data for dashboard
                    response_data = {
                        "sentiment_distribution": message_data.get("sentiment_distribution", {}),
                        "total_messages": message_data.get("total_messages", 0),
                        "timestamp": datetime.now().isoformat(),
                        "recent_messages": message_data.get("recent_messages", [])
                    }
                    await websocket.send_text(json.dumps(response_data))
                else:
                    # Echo back for other messages
                    await websocket.send_text(json.dumps({
                        "status": "received",
                        "data": message_data
                    }))
                    
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "error": "Invalid JSON data"
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
