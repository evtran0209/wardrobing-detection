"""
Wardrobing Detection System - Model Training Module

This module implements and trains models for wardrobing detection.
It demonstrates both traditional ML (scikit-learn) and neural network (TensorFlow) approaches.

Key ML Concepts Demonstrated:
1. Model Training and Evaluation
2. Handling Class Imbalance
3. Model Selection
4. Performance Metrics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns

class WardrobingDetector:
    """
    A class that implements both traditional ML and neural network models
    for wardrobing detection.
    """
    
    def __init__(self, use_neural_net=False):
        """
        Initialize the detector with chosen model type.
        
        Parameters:
        -----------
        use_neural_net : bool
            If True, uses neural network, otherwise uses logistic regression
        """
        self.use_neural_net = use_neural_net
        self.model = None
        self.scaler = StandardScaler()
        
    def build_neural_net(self, input_dim):
        """
        Build a simple neural network for binary classification.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        """
        model = models.Sequential([
            # Input layer
            layers.Dense(64, activation='relu', input_dim=input_dim),
            # Hidden layer
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),  # Prevent overfitting
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
    
    def prepare_data(self, df, features, target):
        """
        Prepare data for model training.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        features : list
            List of feature column names
        target : str
            Name of target column
            
        Returns:
        --------
        tuple
            Training and testing data splits
        """
        # Split data
        X = df[features]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train, y_train):
        """
        Train the selected model.
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training labels
        """
        if self.use_neural_net:
            # Neural Network approach
            self.model = self.build_neural_net(X_train.shape[1])
            
            # Calculate class weights for imbalanced data
            n_samples = len(y_train)
            n_returns = sum(y_train)
            n_non_returns = n_samples - n_returns
            
            class_weight = {
                0: n_samples / (2 * n_non_returns),
                1: n_samples / (2 * n_returns)
            }
            
            # Train the model
            history = self.model.fit(
                X_train, y_train,
                epochs=10,
                batch_size=32,
                validation_split=0.2,
                class_weight=class_weight,
                verbose=1
            )
            
            return history
            
        else:
            # Logistic Regression approach
            self.model = LogisticRegression(
                class_weight='balanced',
                random_state=42
            )
            self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model's performance.
        
        Parameters:
        -----------
        X_test : numpy.ndarray
            Test features
        y_test : numpy.ndarray
            Test labels
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        if self.use_neural_net:
            y_pred_proba = self.model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        return {
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def plot_results(self, evaluation_results, history=None):
        """
        Plot model performance metrics.
        
        Parameters:
        -----------
        evaluation_results : dict
            Results from evaluate method
        history : tensorflow.keras.callbacks.History, optional
            Training history for neural network
        """
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            evaluation_results['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues'
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # For neural network, plot training history
        if history is not None:
            plt.figure(figsize=(10, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    # Load processed data
    print("Loading processed data...")
    df = pd.read_csv('../data/processed_data.csv')
    
    # Define features and target
    features = [
        'product_price',
        'days_to_return',
        'purchase_month',
        'is_weekend',
        'customer_return_rate',
        'customer_avg_order_value',
        'customer_order_value_std',
        'customer_total_orders',
        'category_return_rate',
        'product_category_encoded',
        'account_age_days'
    ]
    target = 'is_return'
    
    # Train and evaluate both models
    for use_neural_net in [False, True]:
        print(f"\nTraining {'Neural Network' if use_neural_net else 'Logistic Regression'}...")
        
        # Initialize detector
        detector = WardrobingDetector(use_neural_net=use_neural_net)
        
        # Prepare data
        X_train, X_test, y_train, y_test = detector.prepare_data(df, features, target)
        
        # Train model
        history = detector.train(X_train, y_train)
        
        # Evaluate model
        results = detector.evaluate(X_test, y_test)
        
        # Print results
        print("\nClassification Report:")
        print(results['classification_report'])
        
        # Plot results
        detector.plot_results(results, history if use_neural_net else None)