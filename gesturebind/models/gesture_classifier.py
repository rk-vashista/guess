"""
Gesture classifier model for GestureBind

This module handles the machine learning models for gesture classification.
"""

import os
import numpy as np
import logging
import json
import pickle
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)

class GestureClassifier:
    """
    Handles training and classification of custom gestures using machine learning.
    """
    
    def __init__(self, config):
        """
        Initialize the gesture classifier with configuration.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.user_data_path = Path(config["app"]["save_user_data_path"])
        self.model_type = "knn"  # Default model type
        self.model = None
        self.gesture_classes = []
        
        # Ensure user data directory exists
        self._ensure_data_dirs()
    
    def train_model(self, gesture_data):
        """
        Train a new model with user gesture data.
        
        Args:
            gesture_data (dict): Dictionary mapping gesture names to feature arrays
            
        Returns:
            bool: Success or failure
        """
        try:
            # Extract features and labels from gesture data
            features = []
            labels = []
            self.gesture_classes = list(gesture_data.keys())
            
            for class_idx, class_name in enumerate(self.gesture_classes):
                class_features = gesture_data[class_name]
                features.extend(class_features)
                labels.extend([class_idx] * len(class_features))
            
            # Convert to numpy arrays
            features = np.array(features)
            labels = np.array(labels)
            
            if len(features) == 0 or len(labels) == 0:
                logger.error("No training data provided")
                return False
            
            logger.info(f"Training model with {len(features)} samples and {len(self.gesture_classes)} classes")
            
            # Create and train the model
            if self.model_type == "knn":
                self.model = KNeighborsClassifier(n_neighbors=3, weights="distance")
                self.model.fit(features, labels)
            elif self.model_type == "nn":
                # Simple neural network model
                self.model = self._create_neural_network(features.shape[1], len(self.gesture_classes))
                self.model.fit(features, keras.utils.to_categorical(labels), 
                             epochs=50, batch_size=32, verbose=0)
            
            # Save the trained model and class mapping
            self._save_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Error training gesture model: {e}")
            return False
    
    def classify_gesture(self, feature_vector):
        """
        Classify a feature vector as a gesture.
        
        Args:
            feature_vector (numpy.ndarray): Feature vector to classify
            
        Returns:
            tuple: (gesture_name, confidence)
        """
        if self.model is None:
            # Try to load a saved model
            if not self._load_model():
                logger.warning("No model available for classification")
                return None, 0.0
        
        try:
            # Reshape feature vector if needed
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            if self.model_type == "knn":
                # Get prediction and distances
                prediction = self.model.predict(feature_vector)[0]
                distances = self.model.kneighbors(feature_vector, return_distance=True)[0][0]
                
                # Convert distance to confidence score (closer = more confident)
                max_distance = 2.0  # Arbitrary max distance for scaling
                confidence = max(0.0, min(1.0, 1.0 - (distances[0] / max_distance)))
                
                gesture_name = self.gesture_classes[prediction]
                return gesture_name, confidence
                
            elif self.model_type == "nn":
                # Get prediction probabilities
                probs = self.model.predict(feature_vector)[0]
                prediction = np.argmax(probs)
                confidence = float(probs[prediction])
                
                gesture_name = self.gesture_classes[prediction]
                return gesture_name, confidence
            
            return None, 0.0
            
        except Exception as e:
            logger.error(f"Error classifying gesture: {e}")
            return None, 0.0
    
    def add_gesture_sample(self, gesture_name, feature_vector):
        """
        Add a new sample for a gesture class.
        
        Args:
            gesture_name (str): Name of the gesture
            feature_vector (numpy.ndarray): Feature vector for the gesture
            
        Returns:
            bool: Success or failure
        """
        try:
            # Load existing gesture data
            gesture_data = self._load_gesture_data()
            
            # Add new sample
            if gesture_name not in gesture_data:
                gesture_data[gesture_name] = []
            
            gesture_data[gesture_name].append(feature_vector.tolist())
            
            # Save updated gesture data
            self._save_gesture_data(gesture_data)
            
            logger.info(f"Added new sample for gesture '{gesture_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error adding gesture sample: {e}")
            return False
    
    def delete_gesture(self, gesture_name):
        """
        Delete all samples for a gesture class.
        
        Args:
            gesture_name (str): Name of the gesture to delete
            
        Returns:
            bool: Success or failure
        """
        try:
            # Load existing gesture data
            gesture_data = self._load_gesture_data()
            
            # Remove gesture class if it exists
            if gesture_name in gesture_data:
                del gesture_data[gesture_name]
                
                # Save updated gesture data
                self._save_gesture_data(gesture_data)
                
                # Retrain the model with updated data
                self.train_model(gesture_data)
                
                logger.info(f"Deleted gesture class '{gesture_name}'")
                return True
            else:
                logger.warning(f"Gesture class '{gesture_name}' not found")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting gesture class: {e}")
            return False
    
    def get_all_gestures(self):
        """
        Get a list of all available gestures and sample counts.
        
        Returns:
            dict: Mapping of gesture names to sample counts
        """
        try:
            # Load existing gesture data
            gesture_data = self._load_gesture_data()
            
            # Count samples for each gesture
            gesture_counts = {name: len(samples) for name, samples in gesture_data.items()}
            
            return gesture_counts
            
        except Exception as e:
            logger.error(f"Error getting gesture counts: {e}")
            return {}
    
    def extract_features_from_landmarks(self, landmarks):
        """
        Extract features from MediaPipe hand landmarks.
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            numpy.ndarray: Feature vector
        """
        # Extract coordinates from landmarks
        points = []
        for landmark in landmarks.landmark:
            points.append([landmark.x, landmark.y, landmark.z])
        
        # Normalize coordinates relative to wrist position
        points = np.array(points)
        wrist = points[0]
        points = points - wrist
        
        # Convert to feature vector (flatten the array)
        features = points.flatten()
        
        return features
    
    def _create_neural_network(self, input_size, num_classes):
        """Create a simple neural network model for gesture classification"""
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(input_size,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    def _ensure_data_dirs(self):
        """Ensure data directories exist"""
        self.user_data_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured data directory exists: {self.user_data_path}")
    
    def _save_model(self):
        """Save the trained model and class mapping"""
        try:
            model_dir = self.user_data_path / "models"
            model_dir.mkdir(exist_ok=True)
            
            if self.model_type == "knn":
                # Save the KNN model
                model_path = model_dir / "gesture_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(self.model, f)
            elif self.model_type == "nn":
                # Save the neural network model
                model_path = model_dir / "gesture_model.h5"
                self.model.save(model_path)
            
            # Save class mapping
            classes_path = model_dir / "gesture_classes.json"
            with open(classes_path, 'w') as f:
                json.dump(self.gesture_classes, f)
                
            logger.info(f"Saved model to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def _load_model(self):
        """Load the trained model and class mapping"""
        try:
            model_dir = self.user_data_path / "models"
            
            # Check if model files exist
            knn_path = model_dir / "gesture_model.pkl"
            nn_path = model_dir / "gesture_model.h5"
            classes_path = model_dir / "gesture_classes.json"
            
            if not classes_path.exists():
                logger.warning("No saved gesture classes found")
                return False
            
            # Load class mapping
            with open(classes_path, 'r') as f:
                self.gesture_classes = json.load(f)
            
            # Load appropriate model
            if knn_path.exists():
                self.model_type = "knn"
                with open(knn_path, 'rb') as f:
                    self.model = pickle.load(f)
            elif nn_path.exists():
                self.model_type = "nn"
                self.model = keras.models.load_model(nn_path)
            else:
                logger.warning("No saved model found")
                return False
            
            logger.info(f"Loaded {self.model_type} model with {len(self.gesture_classes)} gesture classes")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _load_gesture_data(self):
        """Load user gesture training data"""
        try:
            data_path = self.user_data_path / "gesture_data.json"
            
            if not data_path.exists():
                return {}
            
            with open(data_path, 'r') as f:
                gesture_data = json.load(f)
                
            return gesture_data
            
        except Exception as e:
            logger.error(f"Error loading gesture data: {e}")
            return {}
    
    def _save_gesture_data(self, gesture_data):
        """Save user gesture training data"""
        try:
            data_path = self.user_data_path / "gesture_data.json"
            
            with open(data_path, 'w') as f:
                json.dump(gesture_data, f)
                
            logger.info(f"Saved gesture data to {data_path}")
            
        except Exception as e:
            logger.error(f"Error saving gesture data: {e}")