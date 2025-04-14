"""
Gesture Classifier module for GestureBind

This module provides machine learning capabilities for classifying hand gestures.
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
from pathlib import Path

# Fix imports by adding the project root to sys.path if necessary
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use direct imports instead of namespace package imports
try:
    # Direct import using path
    from utils.path_manager import (
        get_workspace_root, resolve_path, get_data_dir, 
        get_gesture_data_file, get_gesture_data_path, get_models_dir
    )
    logger = logging.getLogger(__name__)
    logger.info("Successfully imported path_manager using direct import")
except ImportError as e:
    # Fallback to relative imports if direct imports fail
    try:
        from ..utils.path_manager import (
            get_workspace_root, resolve_path, get_data_dir,
            get_gesture_data_file, get_gesture_data_path, get_models_dir
        )
        logger = logging.getLogger(__name__)
        logger.info("Successfully imported path_manager using relative import")
    except ImportError:
        # Last resort: try to find the module manually
        import logging
        logger = logging.getLogger(__name__)
        
        # Get the absolute path to the project root (parent of models directory)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            logger.info(f"Added project root to sys.path: {project_root}")
        
        # Define the path functions directly if imports fail
        logger.warning("Couldn't import path_manager, using fallback implementations")
        
        # Get the absolute path to the workspace root directory
        WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        
        # Make sure we use the correct data path in the current workspace, not another location
        DATA_ROOT = os.path.join(WORKSPACE_ROOT, "gesturebind", "data")
        
        logger.info(f"Fallback workspace root: {WORKSPACE_ROOT}")
        logger.info(f"Fallback data root: {DATA_ROOT}")
        
        def get_workspace_root():
            """Get the absolute path to the workspace root directory"""
            return WORKSPACE_ROOT
        
        def resolve_path(path, create_if_missing=False):
            """Resolve a path to be relative to the workspace root"""
            # If path starts with ./, it's already workspace-relative
            if path.startswith("./"):
                abs_path = os.path.join(WORKSPACE_ROOT, path[2:])
            # If path is absolute but outside workspace, convert to workspace-relative
            elif os.path.isabs(path):
                # Check if this is a home-directory path
                if path.startswith("~"):
                    path = os.path.expanduser(path)
                
                # Extract the last components of the path
                # Example: /home/user/gesturebind/data -> gesturebind/data
                path_components = path.split(os.sep)
                for i in range(len(path_components) - 1, 0, -1):
                    if path_components[i] == "gesturebind":
                        rel_path = os.path.join(*path_components[i:])
                        abs_path = os.path.join(WORKSPACE_ROOT, rel_path)
                        break
                else:
                    # If we can't find a good relative path, use just the last component
                    rel_path = path_components[-1]
                    abs_path = os.path.join(WORKSPACE_ROOT, "data", rel_path)
                    logger.warning(f"Converting external path '{path}' to workspace path '{abs_path}'")
            else:
                # Relative path, but not starting with ./
                abs_path = os.path.join(WORKSPACE_ROOT, path)
            
            # Create directory if requested
            if create_if_missing:
                directory = abs_path
                if os.path.splitext(abs_path)[1]:  # If it has an extension, it's probably a file
                    directory = os.path.dirname(abs_path)
                
                if directory and not os.path.exists(directory):
                    try:
                        os.makedirs(directory, exist_ok=True)
                        logger.info(f"Created directory: {directory}")
                    except Exception as e:
                        logger.error(f"Failed to create directory {directory}: {e}")
            
            return abs_path
        
        def get_data_dir(subdir=None, create=True):
            """Get data directory within workspace"""
            if subdir:
                path = os.path.join(DATA_ROOT, subdir)
            else:
                path = DATA_ROOT
            
            if create and not os.path.exists(path):
                try:
                    os.makedirs(path, exist_ok=True)
                    logger.info(f"Created data directory: {path}")
                except Exception as e:
                    logger.error(f"Failed to create data directory {path}: {e}")
            
            return path
        
        def get_gesture_data_path():
            """Get gesture data directory"""
            return get_data_dir("gestures", create=True)
        
        def get_gesture_data_file():
            """Get main gesture data file"""
            # Use the data_dir function to ensure consistent path resolution
            return os.path.join(get_data_dir("default", create=True), "gesture_data.json")
        
        def get_models_dir():
            """Fallback models dir"""
            return get_data_dir("models", create=True)

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
        # Use path_manager to get the base data path within the workspace
        self.user_data_path = Path(get_data_dir()) # Use path_manager's data dir
        self.model_type = "knn"  # Default model type
        self.model = None
        self.gesture_classes = []
        self.current_profile = config.get("actions", {}).get("default_profile", "default")
        
        # Ensure user data directory exists using path_manager
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
            features = np.array(features, dtype=np.float32)
            labels = np.array(labels, dtype=np.int32)
            
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
                # Force CPU usage to avoid CUDA errors
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                
                # Verify we're using CPU
                import tensorflow as tf
                logger.info(f"Using TensorFlow devices: {tf.config.list_physical_devices()}")
                
                # Create a neural network model
                input_size = features.shape[1]
                num_classes = len(self.gesture_classes)
                
                # Handle the case where we have only one sample per class
                if len(features) < num_classes * 2:
                    logger.error("Not enough samples for neural network training. Need at least 2 samples per gesture class.")
                    return False
                
                self.model = self._create_neural_network(input_size, num_classes)
                
                # Convert labels to one-hot encoded format
                one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
                
                # Train the model with validation split to prevent overfitting
                self.model.fit(
                    features, 
                    one_hot_labels,
                    epochs=100,
                    batch_size=min(16, len(features)//2 or 1),  # Ensure batch size is appropriate
                    verbose=1,  # Show progress
                    validation_split=0.2  # Use 20% of data for validation
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
            import traceback
            logger.error(traceback.format_exc())
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
            # Check if landmarks is a list/tuple of numbers (already features) or needs extraction
            is_already_features = isinstance(landmarks, (list, np.ndarray)) and \
                                  all(isinstance(item, (int, float, np.number)) for item in landmarks)

            if not is_already_features:
                 feature_vector = self.extract_features_from_landmarks(landmarks)
                 if feature_vector is None:
                      logger.error(f"Failed to extract features for gesture '{gesture_name}'")
                      return False # Stop if feature extraction failed
            else:
                 feature_vector = np.array(landmarks, dtype=np.float32) # Ensure it's a numpy array

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
                feature_vector = feature_vector.tolist() # Convert numpy array to list for JSON serialization
                
            gesture_data[gesture_name]["samples"].append(feature_vector)
            gesture_data[gesture_name]["trained"] = False
            
            # Save updated gesture data
            self._save_gesture_data(gesture_data)
            
            logger.info(f"Added new sample for gesture '{gesture_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error adding gesture sample: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
            numpy.ndarray: Feature vector or None if extraction fails
        """
        points = []
        try:
            # Extract coordinates from landmarks
            if hasattr(landmarks, 'landmark'):
                # MediaPipe landmarks object
                for landmark in landmarks.landmark:
                    # Ensure coordinates are floats
                    points.append([float(landmark.x), float(landmark.y), float(landmark.z)])
            elif isinstance(landmarks, list) and landmarks:
                # List of dictionaries with x, y, z keys
                for landmark in landmarks:
                    # Ensure coordinates are floats and handle potential string values gracefully
                    x = float(landmark.get('x', 0.0))
                    y = float(landmark.get('y', 0.0))
                    z = float(landmark.get('z', 0.0))
                    points.append([x, y, z])
            else:
                logger.error(f"Invalid landmarks format received: {type(landmarks)}")
                return None

            if not points:
                logger.error("No valid points extracted from landmarks")
                return None

            # Convert to numpy array with explicit float type
            points = np.array(points, dtype=np.float32)
            
            # Check if points array is valid
            if points.shape[0] == 0:
                 logger.error("Extracted points array is empty")
                 return None

            # Normalize coordinates relative to wrist position (landmark 0)
            wrist = points[0]
            # Perform subtraction - this is where the error occurred previously
            normalized_points = points - wrist
            
            # Convert to feature vector (flatten the array)
            features = normalized_points.flatten()
            
            return features

        except (ValueError, TypeError) as e:
             logger.error(f"Error converting landmark data to float: {e}. Landmarks: {landmarks}")
             return None
        except Exception as e:
            logger.error(f"Unexpected error during feature extraction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
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
        """Ensure data directories exist using path_manager"""
        try:
            # Ensure base data, gestures, default, and models directories exist
            get_data_dir(create=True)
            get_gesture_data_path() # Ensures ./data/gestures
            get_data_dir("default", create=True) # Ensures ./data/default
            get_models_dir() # Ensures ./data/models

            # Ensure profile-specific directory exists within the main data dir
            # Note: Profile data itself (like gesture_data.json) is handled by get_gesture_data_file
            # We might not need separate profile directories here unless models are profile-specific
            # profile_dir = self.user_data_path / self.current_profile # Old way
            # profile_dir.mkdir(parents=True, exist_ok=True)

            logger.debug(f"Ensured data directories exist via path_manager")
        except Exception as e:
            logger.error(f"Error creating data directories: {e}")
    
    def _save_model(self):
        """Save the trained model and class mapping to the workspace models directory"""
        try:
            # Use path_manager to get the correct models directory
            model_dir = Path(get_models_dir())
            model_dir.mkdir(exist_ok=True) # Should already exist due to ensure_data_dirs

            model_base_name = f"gesture_model_{self.current_profile}" # Profile-specific model name
            classes_base_name = f"gesture_classes_{self.current_profile}.json" # Profile-specific classes

            if self.model_type == "knn":
                # Save the KNN model
                model_path = model_dir / f"{model_base_name}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(self.model, f)
                logger.info(f"Saved KNN model to {model_path}")

            elif self.model_type == "nn" and TF_AVAILABLE:
                # Save the neural network model
                model_path = model_dir / f"{model_base_name}.h5"
                self.model.save(model_path)
                logger.info(f"Saved neural network model to {model_path}")

            # Save class mapping
            classes_path = model_dir / classes_base_name
            with open(classes_path, 'w') as f:
                json.dump(self.gesture_classes, f, indent=2)
            logger.info(f"Saved gesture classes to {classes_path}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def _load_model(self):
        """Load the trained model and class mapping from the workspace models directory"""
        try:
            # Use path_manager to get the correct models directory
            model_dir = Path(get_models_dir())

            model_base_name = f"gesture_model_{self.current_profile}" # Profile-specific model name
            classes_base_name = f"gesture_classes_{self.current_profile}.json" # Profile-specific classes

            # Check if model files exist
            knn_path = model_dir / f"{model_base_name}.pkl"
            nn_path = model_dir / f"{model_base_name}.h5"
            classes_path = model_dir / classes_base_name

            if not classes_path.exists():
                logger.info(f"No saved gesture classes found for profile '{self.current_profile}' at {classes_path}")
                self.model = None # Ensure model is None if classes aren't found
                self.gesture_classes = []
                return False

            # Load class mapping
            with open(classes_path, 'r') as f:
                self.gesture_classes = json.load(f)
            logger.info(f"Loaded {len(self.gesture_classes)} gesture classes from {classes_path}")

            # Load appropriate model
            model_loaded = False
            if knn_path.exists() and SKLEARN_AVAILABLE:
                self.model_type = "knn"
                with open(knn_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded KNN model from {knn_path}")
                model_loaded = True
            elif nn_path.exists() and TF_AVAILABLE:
                self.model_type = "nn"
                # Ensure TF uses CPU if needed
                if 'CUDA_VISIBLE_DEVICES' not in os.environ:
                     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                import tensorflow as tf # Import here to respect env var
                self.model = tf.keras.models.load_model(nn_path)
                logger.info(f"Loaded neural network model from {nn_path}")
                model_loaded = True

            if not model_loaded:
                 self.model = None # Ensure model is None if no file was loaded
                 if not SKLEARN_AVAILABLE and not TF_AVAILABLE:
                     logger.error("Neither scikit-learn nor TensorFlow is available")
                 elif knn_path.exists() and not SKLEARN_AVAILABLE:
                     logger.error(f"KNN model exists ({knn_path}) but scikit-learn is not available")
                 elif nn_path.exists() and not TF_AVAILABLE:
                     logger.error(f"NN model exists ({nn_path}) but TensorFlow is not available")
                 else:
                     logger.warning(f"No compatible saved model file found for profile '{self.current_profile}' in {model_dir}")
                 return False

            return True

        except Exception as e:
            logger.error(f"Error loading model for profile '{self.current_profile}': {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.model = None # Ensure model is None on error
            self.gesture_classes = []
            return False
    
    def _load_gesture_data(self):
        """Load user gesture training data using path_manager"""
        try:
            # Use path_manager to get the correct gesture data file path
            # This now points to ./data/default/gesture_data.json
            data_path = get_gesture_data_file()

            if not os.path.exists(data_path):
                logger.info(f"No gesture data found at {data_path}, returning empty data")
                return {}

            with open(data_path, 'r') as f:
                # Handle potential empty file
                content = f.read()
                if not content:
                     logger.warning(f"Gesture data file is empty: {data_path}")
                     return {}
                gesture_data = json.loads(content)

            # Check format - should be {"gesture_name": {"samples": [...], "trained": bool}}
            if not isinstance(gesture_data, dict):
                 logger.error(f"Invalid format in {data_path}, expected a dictionary.")
                 return {}
            # Handle old format {"gestures": {...}} if necessary
            if "gestures" in gesture_data and isinstance(gesture_data["gestures"], dict) and len(gesture_data) == 1:
                 logger.warning(f"Detected old gesture data format in {data_path}. Using 'gestures' content.")
                 gesture_data = gesture_data["gestures"]


            logger.debug(f"Loaded gesture data from {data_path}")
            return gesture_data

        except json.JSONDecodeError:
             logger.error(f"Error decoding JSON from gesture data file: {data_path}")
             return {}
        except Exception as e:
            logger.error(f"Error loading gesture data: {e}")
            return {}
    
    def _save_gesture_data(self, gesture_data):
        """Save user gesture training data using path_manager"""
        try:
            # Use path_manager to get the correct gesture data file path
            data_path = get_gesture_data_file()

            # Ensure the directory exists (path_manager's get_data_dir should handle this)
            os.makedirs(os.path.dirname(data_path), exist_ok=True)

            # Ensure gesture_data is a dictionary
            if not isinstance(gesture_data, dict):
                 logger.error(f"Attempted to save invalid gesture data (not a dict): {type(gesture_data)}")
                 return

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
        
        # No need to create profile-specific directories here anymore
        # as models and data are stored centrally or per-profile named files

        # Reload the model for this profile
        self._load_model()