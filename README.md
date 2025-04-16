# SignBridge

SignBridge is an interactive web application designed to help users learn American Sign Language (ASL) through a gamified approach. The application uses computer vision and machine learning to recognize sign language gestures in real-time through a webcam, providing immediate feedback to users.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Models](#models)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)

## Overview

SignBridge is a comprehensive learning platform that allows users to learn ASL through three progressive levels:

1. **Level 1: Alphabet Recognition** - Learn to sign individual letters (A, B, C, D, E, F)
2. **Level 2: Word Recognition** - Learn to sign common words ("busy", "hello", "help")
3. **Level 3: Sentence Recognition** - Learn to sign basic sentences ("how are you", "nice to meet you", "what is your name")

The application uses a webcam to capture the user's hand gestures, processes them using machine learning models, and provides real-time feedback on the accuracy of the signs.

## Features

- **User Authentication**: Register, login, and maintain user profiles
- **Progressive Learning**: Three levels of increasing difficulty
- **Real-time Feedback**: Immediate feedback on sign accuracy
- **Score Tracking**: Track progress and unlock new levels
- **Responsive Design**: Works on various screen sizes
- **Interactive UI**: Engaging user interface with clear instructions

## Project Structure

The project is organized into three main components:

### 1. Website (Frontend)

Located in the `website` directory, the frontend is built with React and includes:

- **Components**: UI components for different pages and features
  - `GamePage.jsx`: Main game interface with webcam integration
  - `Layout.jsx`: Common layout for all pages
  - `Login.jsx` & `Signup.jsx`: Authentication forms
  - And many more components for different sections of the app
  
- **Context**: React context for state management
  - `AuthContext.jsx`: Manages user authentication state
  - `ProgressContext.jsx`: Tracks and updates user progress
  
- **Assets**: Static resources like images, videos, and CSS files

### 2. Server (Backend)

Located in the `server` directory, the backend consists of:

- **Node.js Server**: Main server handling authentication, user management, and progress tracking
  - `server.js`: Entry point with Express routes for user authentication and progress tracking
  - MongoDB integration for data persistence
  
- **Python Server**: Handles ASL prediction using machine learning models
  - `server/python/server.py`: Flask server with endpoints for prediction
  - `server/python/asl_predictor.py`: Core logic for processing frames and making predictions

### 3. Models

Located in the `model` directory, this contains the trained machine learning models:

- **Alphabet Model**: For recognizing individual letters
  - `asl_alphabet_model.h5`: TensorFlow model for alphabet recognition
  - `label_encoder.pkl`: Label encoder for alphabet classes
  
- **Word Model**: For recognizing words
  - `Model_words/asl_word_lstm_model.h5`: LSTM model for word recognition
  - `Model_words/label_encoder.pkl`: Label encoder for word classes
  
- **Sentence Model**: For recognizing sentences
  - `model_sentences/asl_sentence_lstm_model.h5`: LSTM model for sentence recognition
  - `model_sentences/label_encoder.pkl`: Label encoder for sentence classes

## Technologies Used

### Frontend
- React.js
- React Router
- React Webcam
- Context API for state management
- CSS/Tailwind CSS for styling

### Backend
- Node.js with Express
- MongoDB with Mongoose
- JWT for authentication
- Flask (Python) for ML model serving

### Machine Learning
- TensorFlow/Keras
- MediaPipe for hand landmark detection
- OpenCV for image processing
- Scikit-learn for data preprocessing

## Models

### Alphabet Recognition Model
- **Architecture**: Sequential neural network
- **Input**: 63 hand landmark features from a single hand
- **Output**: Probability distribution over 6 alphabet classes (A, B, C, D, E, F)
- **Processing**: Uses MediaPipe to extract hand landmarks from a single frame

### Word Recognition Model
- **Architecture**: LSTM (Long Short-Term Memory) network
- **Input**: Sequence of hand landmarks from multiple frames
- **Output**: Probability distribution over 3 word classes ("busy", "hello", "help")
- **Processing**: Uses MediaPipe to extract hand landmarks from a sequence of frames
- **Fallback**: Uses a frame-count heuristic when model confidence is low

### Sentence Recognition Model
- **Architecture**: LSTM network
- **Input**: Sequence of hand landmarks from multiple frames
- **Output**: Probability distribution over 3 sentence classes ("how are you", "nice to meet you", "what is your name")
- **Processing**: Uses MediaPipe to extract hand landmarks from a sequence of frames
- **Fallback**: Uses a frame-count heuristic when model confidence is low

## Installation

### Prerequisites
- Node.js (v14 or higher)
- Python 3.8 or higher
- MongoDB
- Git

### Clone the Repository
```bash
git clone https://github.com/yourusername/SignBridge.git
cd SignBridge
```

### Install Frontend Dependencies
```bash
cd website
npm install
```

### Install Backend Dependencies
```bash
cd ../server
npm install
```

### Install Python Dependencies
```bash
cd python
pip install -r requirements.txt
```

The `requirements.txt` file should include:
```
flask
flask-cors
opencv-python
numpy
tensorflow
mediapipe
scikit-learn
joblib
h5py
```

## Running the Application

### Start MongoDB
Ensure MongoDB is running on your system. You can start it with:
```bash
mongod
```

### Start the Node.js Server
```bash
cd server
npm run dev
```
This will start the Node.js server on port 8000.

### Start the Python Server
```bash
cd server/python
python server.py
```
This will start the Python server on port 5001.

### Start the Frontend
```bash
cd website
npm run dev -- --port 3000 --host
```
This will start the React development server on port 3000.

### Access the Application
Open your browser and navigate to:
```
http://localhost:3000
```

## Usage

1. **Register/Login**: Create an account or log in to an existing account
2. **Level 1 (Alphabet)**: Practice signing individual letters
   - Position your hand in the webcam view
   - Make the sign for the displayed letter
   - Click "Check Sign" to verify your sign
   - Earn 10 points for each correct sign
   - Reach 30 points to unlock Level 2

3. **Level 2 (Words)**: Practice signing common words
   - Position your hand in the webcam view
   - Make the sign for the displayed word
   - Click "Start Recording" to begin capturing frames
   - Click "Stop Recording" when finished
   - Click "Check Sign" to verify your sign
   - Earn 10 points for each correct sign
   - Reach 60 points to unlock Level 3

4. **Level 3 (Sentences)**: Practice signing sentences
   - Position your hand in the webcam view
   - Make the sign for the displayed sentence
   - Click "Start Recording" to begin capturing frames
   - Click "Stop Recording" when finished
   - Click "Check Sign" to verify your sign
   - Earn 10 points for each correct sign

## API Endpoints

### Authentication
- `POST /api/auth/register`: Register a new user
- `POST /api/auth/login`: Login an existing user

### User Progress
- `GET /api/user/progress`: Get user progress
- `POST /api/user/progress`: Update user progress
- `POST /api/user/unlock-level2`: Unlock Level 2
- `POST /api/user/unlock-level3`: Unlock Level 3

### ASL Prediction
- `POST /api/asl/predict/alphabet`: Predict an alphabet from a single frame
- `POST /api/asl/predict/word`: Predict a word from a sequence of frames
- `POST /api/asl/predict/sentence`: Predict a sentence from a sequence of frames

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
