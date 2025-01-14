"""
Wardrobing Detection Neural Network Model
A deep learning model to detect fraudulent returns in e-commerce
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WardrobingDetector:
    def __init__(self, input_dim):
        """
        Initialize the neural network model
        
        Args:
            input_dim (int): Number of input features
            
        Neural Network Concepts:
        - Layers: Building blocks that transform data
        - Neurons: Individual units that process information
        - Activation Functions: Add non-linearity to learn complex patterns
        """
        self.input_dim = input_dim
        
        # Create the neural network architecture
        self.model = tf.keras.Sequential([
            # Input Layer: Where data enters the network
            # Dense means fully connected (each neuron connects to all neurons in next layer)
            tf.keras.layers.Dense(
                units=128,               # Number of neurons
                input_shape=(input_dim,), # Shape of input data
                activation='relu',        # ReLU activation: max(0, x)
                kernel_regularizer=tf.keras.regularizers.l2(0.01)  # Prevent overfitting
            ),
            
            # Dropout Layer: Randomly turns off neurons during training
            # This prevents the network from becoming too dependent on any one feature
            tf.keras.layers.Dropout(0.3),  # 30% of neurons are randomly disabled
            
            # Hidden Layer 1: Processes features from previous layer
            tf.keras.layers.Dense(
                units=64,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
            ),
            
            # Dropout Layer 2
            tf.keras.layers.Dropout(0.2),
            
            # Hidden Layer 2: Further feature processing
            tf.keras.layers.Dense(
                units=32,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
            ),
            
            # Output Layer: Produces fraud probability
            tf.keras.layers.Dense(
                units=1,                 # Single output (fraud probability)
                activation='sigmoid'      # Sigmoid squashes output to 0-1 range
            )
        ])
        
        # Initialize metrics to track model performance
        self.metrics = {
            'accuracy': tf.keras.metrics.BinaryAccuracy(),
            'precision': tf.keras.metrics.Precision(),
            'recall': tf.keras.metrics.Recall(),
            'auc': tf.keras.metrics.AUC()
        }
        
        # Track training history
        self.history = None
        
        # Initialize feature importance tracker
        self.feature_importance = None

    def compile_model(self, learning_rate=0.001):
        """
        Configure the model for training
        
        Args:
            learning_rate (float): How fast the model learns
            
        Machine Learning Concepts:
        - Optimizer: Adjusts weights to minimize error
        - Loss Function: Measures how wrong the predictions are
        - Learning Rate: Step size for weight updates
        """
        # Adam optimizer: Adaptive learning rate optimization
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,  # Initial learning rate
            beta_1=0.9,                  # Momentum parameter
            beta_2=0.999                 # RMSprop parameter
        )
        
        # Compile model with binary crossentropy loss (good for binary classification)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',  # Loss function for binary problems
            metrics=[                    # Metrics to track during training
                'accuracy',              # Percentage of correct predictions
                tf.keras.metrics.Precision(),  # True positives / (True + False positives)
                tf.keras.metrics.Recall(),     # True positives / (True + False negatives)
                tf.keras.metrics.AUC()         # Area under ROC curve
            ]
        )

    def create_callbacks(self):
        """
        Create training callbacks for monitoring and improvement
        
        Callbacks Explained:
        - EarlyStopping: Prevents overfitting by stopping when model stops improving
        - ModelCheckpoint: Saves best model during training
        - ReduceLROnPlateau: Reduces learning rate when progress stalls
        """
        callbacks = [
            # Stop training when validation loss stops improving
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',      # Watch validation loss
                patience=10,             # Number of epochs to wait
                restore_best_weights=True # Keep best weights
            ),
            
            # Save model when validation loss improves
            tf.keras.callbacks.ModelCheckpoint(
                filepath='../models/best_model.h5',
                monitor='val_loss',
                save_best_only=True
            ),
            
            # Reduce learning rate when progress plateaus
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,              # Multiply learning rate by 0.2
                patience=5,              # Wait 5 epochs before reducing
                min_lr=0.00001           # Don't go below this learning rate
            )
        ]
        return callbacks