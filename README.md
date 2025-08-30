# YouTube Live Chat Sentiment Analyser

This project is a real-time sentiment analysis tool for YouTube Live Chats, built with a **Python FastAPI backend** and a **custom-trained machine learning model**, and a **TypeScript-based Chrome extension** for the front end. It is designed to provide content creators and moderators with instant insights into the sentiment of their live chat audience.

---

<img width="1897" height="931" alt="image" src="https://github.com/user-attachments/assets/daee023d-4197-4f7c-895b-4ab80b83d86b" />


## Features

- **Real-time Analysis**: Monitors and analyzes YouTube live chat messages as they are posted.  
- **Interactive Dashboard**: A web-based interface provides a dashboard with charts and key statistics for sentiment distribution.  
- **Confidence Scores**: Each sentiment prediction includes a confidence score, giving users a better understanding of the model's certainty.  
- **Auto-start**: The analysis automatically begins when a user navigates to a YouTube live stream page.  
- **Customizable**: The Chrome extension is configurable, allowing users to specify the API server URL and other analysis settings.  

---

## Project Structure

The project is divided into two main components: a **server** and a **chrome_extension**.

### Server
The `server` directory contains the backend of the application.  
- **Framework**: Built using FastAPI for high performance and easy API creation.  
- **Machine Learning**: Hosts a custom-trained sentiment analysis model. The model is trained and managed using MLflow for experiment tracking, versioning, and easy deployment.  
- **Tokenizer**: A custom tokenizer is used to preprocess text data for the model.  

### Chrome Extension
The `chrome_extension` directory holds the user-facing part of the application.  
- **Languages**: Developed using TypeScript to ensure a robust and maintainable codebase.  
- **Functionality**: It injects a content script into YouTube live stream pages to capture chat messages and send them to the FastAPI server for analysis. The results are then displayed on an interactive dashboard.  

---

## Getting Started

### Prerequisites
- Python **3.8+**  
- Node.js and npm  
- Google Chrome browser  
- MLflow (`pip install mlflow`)  

---

### Installation

#### 1. Backend Server Setup
Navigate to the `server` directory and install the required Python dependencies:

```bash
pip install -r requirements.txt
````

#### 2. Chrome Extension Setup

Navigate to the `chrome_extension` directory and install the Node.js packages:

```bash
npm install
```

After installation, build the extension:

```bash
npm run build
```

This will create a `dist` directory.

---

## Running the Application

### Start the Server

From the `server` directory, run the FastAPI application:

```bash
uvicorn main:app --reload
```

The server will start at **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.

### Load the Chrome Extension

1. Open **Google Chrome** and navigate to `chrome://extensions/`.
2. Enable **"Developer mode"** in the top-right corner.
3. Click **"Load unpacked"** and select the `dist` folder from your `chrome_extension` directory.
4. The extension icon will appear in your browser toolbar.

### Start Analyzing

* Go to a YouTube live stream with an active chat.
* Click the extension icon to open the popup.
* The analysis should begin automatically.

---

## MLflow

The **MLflow tracking server** is used to manage and serve the machine learning models.

To run the MLflow UI and view your experiments:

1. Navigate to the project's root directory.
2. Start the MLflow UI:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

You can then access the UI in your browser at:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

```
