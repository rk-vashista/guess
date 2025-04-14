"""
Gesture Classifier module for GestureBind

This module provides machine learning capabilities for classifying hand gestures.
"""

import os
import json
import pickle
import logging
import numpy as np
from pathlib import Path

# Try to import ML libraries with graceful fallbacks
try:
    from sklearn.neighbors import KNeighborsClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

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
        self.user_data_path = Path(os.path.expanduser(config.get("app", {}).get("save_user_data_path", "~/gesturebind/data")))
        self.model_type = "knn"  # Default model type
        self.model = None
        self.gesture_classes = []
        self.current_profile = config.get("actions", {}).get("default_profile", "default")
        
        # Ensure user data directory exists
        self._ensure_data_dirs()
        
        # Try to load existing model
        self._load_model()
    
    def train_model(self, custom_data=None):
        """
        Train a new model with user gesture data.
        
        Args:
            custom_data (dict, optional): Optional dictionary mapping gesture names to feature arrays
            
        Returns:
            bool: Success or failure
        """
        try:
            # If custom data not provided, load from storage
            gesture_data = custom_data or self._load_gesture_data()
            
            if not gesture_data:
                logger.error("No gesture data available for training")
                return False
            
            # Extract features and labels from gesture data
            features = []
            labels = []
            self.gesture_classes = list(gesture_data.keys())
            
            for class_idx, class_name in enumerate(self.gesture_classes):
                class_samples = gesture_data[class_name].get("samples", [])
                if len(class_samples) == 0:
                    continue
                
                for feature_vector in class_samples:
                    features.append(feature_vector)
                    labels.append(class_idx)
            
            # Convert to numpy arrays for ML processing
            features = np.array(features)
            labels = np.array(labels)
            
            if len(features) == 0 or len(labels) == 0:
                logger.error("No training data provided")
                return False
            
            logger.info(f"Training model with {len(features)} samples and {len(self.gesture_classes)} classes")
            
            # Create and train the model based on the selected type
            if self.model_type == "knn" and SKLEARN_AVAILABLE:
                self.model = KNeighborsClassifier(n_neighbors=3, weights="distance")
                self.model.fit(features, labels)
                logger.info("Trained KNN classifier successfully")
            elif self.model_type == "nn" and TF_AVAILABLE:
                # Create a neural network model
                self.model = self._create_neural_network(features.shape[1], len(self.gesture_classes))
                
                # Convert labels to one-hot encoded format
                one_hot_labels = keras.utils.to_categorical(labels, num_classes=len(self.gesture_classes))
                
                # Train the model
                self.model.fit(
                    features, 
                    one_hot_labels,
                    epochs=100,  # Can be configured
                    batch_size=16,  # Can be configured 
                    verbose=0  # Quiet mode
                )
                logger.info("Trained neural network classifier successfully")
            else:
                if not SKLEARN_AVAILABLE and not TF_AVAILABLE:
                    logger.error("Neither scikit-learn nor TensorFlow is available")
                    return False
                elif not SKLEARN_AVAILABLE and self.model_type == "knn":
                    logger.error("scikit-learn is not available, cannot use KNN classifier")
                    return False
                elif not TF_AVAILABLE and self.model_type == "nn":
                    logger.error("TensorFlow is not available, cannot use neural network classifier")
                    return False
            
            # Mark all gestures as trained
            for gesture_name in gesture_data:
                if "samples" in gesture_data[gesture_name] and len(gesture_data[gesture_name]["samples"]) > 0:
                    gesture_data[gesture_name]["trained"] = True
            
            # Save the trained model and updated gesture data
            self._save_model()
            self._save_gesture_data(gesture_data)
            
            return True
        except Exception as e:
            logger.error(f"Error training gesture model: {e}")
            return False
    
    def classify_gesture(self, feature_vector):
        """
        Classify a feature vector as a gesture.
        
        Args:
            feature_vector (numpy.ndarray or list): Feature vector to classify
            
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
            if isinstance(feature_vector, list):
                feature_vector = np.array(feature_vector)
            
            feature_vector = feature_vector.reshape(1, -1)
            
            # Check dimensions
            if not self.gesture_classes:
                logger.error("No gesture classes defined in the model")
                return None, 0.0
            
            # Classify based on model type
            if self.model_type == "knn":
                # Get prediction and distances
                prediction = self.model.predict(feature_vector)[0]
                distances = self.model.kneighbors(feature_vector, return_distance=True)[0][0]
                
                # Convert distance to confidence score (closer = more confident)
                max_distance = 2.0  # Arbitrary max distance for scaling
                confidence = max(0.0, min(1.0, 1.0 - (distances[0] / max_distance)))
                
                if prediction < len(self.gesture_classes):
                    gesture_name = self.gesture_classes[prediction]
                    return gesture_name, confidence
                
            elif self.model_type == "nn":
                # Get prediction probabilities
                probs = self.model.predict(feature_vector)[0]
                prediction = np.argmax(probs)
                confidence = float(probs[prediction])
                
                if prediction < len(self.gesture_classes):
                    gesture_name = self.gesture_classes[prediction]
                    return gesture_name, confidence
            
            return None, 0.0
            
        except Exception as e:
            logger.error(f"Error classifying gesture: {e}")
            return None, 0.0
    
    def add_gesture_sample(self, gesture_name, landmarks):
        """
        Add a new sample for a gesture class.
        
        Args:
            gesture_name (str): Name of the gesture
            landmarks: MediaPipe landmarks or feature vector
            
        Returns:
            bool: Success or failure
        """
        try:
            # Extract features if landmarks provided
            if not isinstance(landmarks, (list, np.ndarray)) or (isinstance(landmarks, list) and isinstance(landmarks[0], dict)):
                feature_vector = self.extract_features_from_landmarks(landmarks)
            else:
                feature_vector = landmarks
            
            # Load existing gesture data
            gesture_data = self._load_gesture_data()
            
            # Create new gesture entry if not exists
            if gesture_name not in gesture_data:
                gesture_data[gesture_name] = {
                    "samples": [],
                    "trained": False
                }
            
            # Add new sample
            if isinstance(feature_vector, np.ndarray):
                feature_vector = feature_vector.tolist()
                
            gesture_data[gesture_name]["samples"].append(feature_vector)
            gesture_data[gesture_name]["trained"] = False
            
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
            dict: Mapping of gesture names to gesture info
        """
        try:
            # Load existing gesture data
            gesture_data = self._load_gesture_data()
            
            # Build gesture info dictionary
            gesture_info = {}
            for name, data in gesture_data.items():
                samples = data.get("samples", [])
                gesture_info[name] = {
                    "sample_count": len(samples),
                    "trained": data.get("trained", False)
                }
            
            return gesture_info
            
        except Exception as e:
            logger.error(f"Error getting gesture info: {e}")
            return {}
    
    def extract_features_from_landmarks(self, landmarks):
        """
        Extract features from MediaPipe hand landmarks.
        
        Args:
            landmarks: MediaPipe hand landmarks or list of landmark dictionaries
            
        Returns:
            numpy.ndarray: Feature vector
        """
        # Extract coordinates from landmarks
        if hasattr(landmarks, 'landmark'):
            # MediaPipe landmarks object
            points = []
            for landmark in landmarks.landmark:
                points.append([landmark.x, landmark.y, landmark.z])
        else:
            # List of dictionaries with x, y, z keys
            points = []
            for landmark in landmarks:
                points.append([landmark['x'], landmark['y'], landmark['z']])
        
        # Convert to numpy array
        points = np.array(points)
        
        # Normalize coordinates relative to wrist position
        wrist = points[0]
        points = points - wrist
        
        # Convert to feature vector (flatten the array)
        features = points.flatten()
        
        return features
    
    def _create_neural_network(self, input_size, num_classes):
        """Create a simple neural network model for gesture classification"""
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available, cannot create neural network")
            return None
            
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
        try:
            profile_dir = self.user_data_path / self.current_profile
            profile_dir.mkdir(parents=True, exist_ok=True)
            
            model_dir = self.user_data_path / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            logger.debug(f"Ensured data directories exist")
        except Exception as e:
            logger.error(f"Error creating data directories: {e}")
    
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
                    
                logger.info(f"Saved KNN model to {model_path}")
                
            elif self.model_type == "nn" and TF_AVAILABLE:
                # Save the neural network model
                model_path = model_dir / "gesture_model.h5"
                self.model.save(model_path)
                
                logger.info(f"Saved neural network model to {model_path}")
            
            # Save class mapping
            classes_path = model_dir / "gesture_classes.json"
            with open(classes_path, 'w') as f:
                json.dump(self.gesture_classes, f, indent=2)
                
            logger.info(f"Saved gesture classes to {classes_path}")
            
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
                logger.info("No saved gesture classes found")
                return False
            
            # Load class mapping
            with open(classes_path, 'r') as f:
                self.gesture_classes = json.load(f)
            
            # Load appropriate model
            if knn_path.exists() and SKLEARN_AVAILABLE:
                self.model_type = "knn"
                with open(knn_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded KNN model with {len(self.gesture_classes)} gesture classes")
                return True
            elif nn_path.exists() and TF_AVAILABLE:
                self.model_type = "nn"
                self.model = keras.models.load_model(nn_path)
                logger.info(f"Loaded neural network model with {len(self.gesture_classes)} gesture classes")
                return True
            else:
                if not SKLEARN_AVAILABLE and not TF_AVAILABLE:
                    logger.error("Neither scikit-learn nor TensorFlow is available")
                elif knn_path.exists() and not SKLEARN_AVAILABLE:
                    logger.error("scikit-learn is not available but KNN model exists")
                elif nn_path.exists() and not TF_AVAILABLE:
                    logger.error("TensorFlow is not available but neural network model exists")
                else:
                    logger.warning("No saved model found")
                return False
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _load_gesture_data(self):
        """Load user gesture training data"""
        try:
            profile_dir = self.user_data_path / self.current_profile
            profile_dir.mkdir(parents=True, exist_ok=True)
            
            data_path = profile_dir / "gesture_data.json"
            
            if not data_path.exists():
                logger.info(f"No gesture data found at {data_path}, returning empty data")
                return {}
            
            with open(data_path, 'r') as f:
                gesture_data = json.load(f)
                
            logger.debug(f"Loaded gesture data from {data_path}")
            return gesture_data
            
        except Exception as e:
            logger.error(f"Error loading gesture data: {e}")
            return {}
    
    def _save_gesture_data(self, gesture_data):
        """Save user gesture training data"""
        try:
            profile_dir = self.user_data_path / self.current_profile
            profile_dir.mkdir(parents=True, exist_ok=True)
            
            data_path = profile_dir / "gesture_data.json"
            
            with open(data_path, 'w') as f:
                json.dump(gesture_data, f, indent=2)
                
            logger.info(f"Saved gesture data to {data_path}")
            
        except Exception as e:
            logger.error(f"Error saving gesture data: {e}")
    
    def set_current_profile(self, profile_name):
        """
        Change the current gesture profile.
        
        Args:
            profile_name (str): Name of the profile to use
        """
        if profile_name == self.current_profile:
            return
            
        logger.info(f"Switching to gesture profile: {profile_name}")
        self.current_profile = profile_name
        
        # Create profile directory if it doesn't exist
        profile_dir = self.user_data_path / profile_name
        profile_dir.mkdir(parents=True, exist_ok=True)
        
        # Reload the model for this profile
        self._load_model()