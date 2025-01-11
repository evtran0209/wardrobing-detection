"""
Feature Engineering for Wardrobing Detection
Transforms raw order data into machine learning-ready features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime
import logging

# Set up logging to track feature engineering process
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        """
        Initialize the feature engineering pipeline
        Sets up scalers, encoders, and defines which features to process
        """
        # Initialize scalers for converting numbers to same scale
        self.numerical_scaler = StandardScaler()
        
        # Initialize encoder for converting categories to numbers
        self.categorical_encoder = OneHotEncoder(
            sparse=False,  # Return dense array instead of sparse matrix
            handle_unknown='ignore'  # Don't fail on new categories
        )
        
        # Features that need numerical scaling (converting to similar ranges)
        self.numerical_features = [
            'product_price',          # Raw price needs scaling (e.g., $20 to $2000)
            'account_age_days',       # Account age varies widely
            'customer_total_orders',  # Order count varies by customer
            'days_to_return',         # Return timing varies
            'customer_return_rate',   # Return rate is 0-1
            'order_frequency',        # Orders per day varies
            'price_category_zscore'   # Price deviation from category average
        ]
        
        # Features that need category encoding (converting text to numbers)
        self.categorical_features = [
            'product_category',      # Type of product
            'product_subcategory',   # Specific product type
            'return_reason',         # Why item was returned
            'customer_type'          # Type of customer behavior
        ]
        
        # Track feature statistics for monitoring
        self.feature_stats = {}

    def create_time_features(self, df):
        """
        Create features from date and time information
        Args:
            df (pandas.DataFrame): Input dataframe with date columns
        Returns:
            pandas.DataFrame: Time-based features
        """
        logger.info("Creating time-based features...")
        time_features = pd.DataFrame()
        
        # Convert string dates to datetime if they're not already
        for col in ['purchase_date', 'return_date']:
            if df[col].dtype == 'object':
                df[col] = pd.to_datetime(df[col])
        
        # Extract month (1-12) to capture seasonal patterns
        time_features['purchase_month'] = df['purchase_date'].dt.month
        
        # Extract day of week (0-6) where 0 is Monday
        time_features['purchase_day'] = df['purchase_date'].dt.dayofweek
        
        # Create weekend indicator (1 if weekend, 0 if weekday)
        time_features['is_weekend'] = (time_features['purchase_day'] >= 5).astype(int)
        
        # Calculate days between purchase and return for returned items
        mask = df['return_date'].notna()
        time_features.loc[mask, 'days_to_return'] = (
            (df.loc[mask, 'return_date'] - df.loc[mask, 'purchase_date']).dt.days
        )
        
        # Fill missing return days with -1 for non-returned items
        time_features['days_to_return'].fillna(-1, inplace=True)
        
        # Create seasonal flags based on known high-risk periods
        time_features['is_holiday_season'] = time_features['purchase_month'].isin([11, 12])
        time_features['is_wedding_season'] = time_features['purchase_month'].isin([5, 6, 7, 8])
        time_features['is_prom_season'] = time_features['purchase_month'].isin([4, 5])
        
        # Track statistics for monitoring
        self.feature_stats['time_features'] = {
            'avg_days_to_return': time_features['days_to_return'][mask].mean(),
            'weekend_purchase_rate': time_features['is_weekend'].mean()
        }
        
        return time_features

    def create_price_features(self, df):
        """
        Create features from price information
        Args:
            df (pandas.DataFrame): Input dataframe with price information
        Returns:
            pandas.DataFrame: Price-based features
        """
        logger.info("Creating price-based features...")
        price_features = pd.DataFrame()
        
        # Basic price feature (will be scaled later)
        price_features['price'] = df['product_price']
        
        # Calculate price percentile within each product category (0-1 scale)
        price_features['category_price_percentile'] = df.groupby('product_category')[
            'product_price'
        ].transform(
            lambda x: pd.qcut(x, q=10, labels=False, duplicates='drop')
        ) / 10.0
        
        # Flag for high-value items (top 25% of prices)
        price_features['is_high_value'] = (
            df['product_price'] > df['product_price'].quantile(0.75)
        ).astype(int)
        
        # Calculate how much item price deviates from category average
        category_means = df.groupby('product_category')['product_price'].transform('mean')
        category_stds = df.groupby('product_category')['product_price'].transform('std')
        price_features['price_category_zscore'] = (
            (df['product_price'] - category_means) / category_stds
        )
        
        # Track statistics
        self.feature_stats['price_features'] = {
            'avg_price': price_features['price'].mean(),
            'high_value_rate': price_features['is_high_value'].mean()
        }
        
        return price_features

    def create_customer_features(self, df):
        """
        Create features from customer behavior patterns
        Args:
            df (pandas.DataFrame): Input dataframe with customer information
        Returns:
            pandas.DataFrame: Customer behavior features
        """
        logger.info("Creating customer behavior features...")
        customer_features = pd.DataFrame()
        
        # Calculate return rate for each customer
        customer_features['customer_return_rate'] = df.groupby('customer_id')[
            'is_returned'
        ].transform('mean')
        
        # Calculate average order value per customer
        customer_features['customer_avg_order'] = df.groupby('customer_id')[
            'product_price'
        ].transform('mean')
        
        # Calculate order frequency (orders per day of account age)
        customer_features['order_frequency'] = (
            df.groupby('customer_id')['order_id'].transform('count') / 
            df['account_age_days']
        )
        
        # Calculate average days to return for each customer
        return_days = df[df['is_returned']].groupby('customer_id')['days_to_return'].mean()
        customer_features['avg_return_days'] = df['customer_id'].map(return_days)
        customer_features['avg_return_days'].fillna(-1, inplace=True)
        
        # Calculate return rate for high-value items
        high_value_mask = df['product_price'] > df['product_price'].quantile(0.75)
        customer_features['high_value_return_rate'] = df[high_value_mask].groupby(
            'customer_id'
        )['is_returned'].transform('mean')
        customer_features['high_value_return_rate'].fillna(0, inplace=True)
        
        # Track statistics
        self.feature_stats['customer_features'] = {
            'avg_return_rate': customer_features['customer_return_rate'].mean(),
            'avg_order_frequency': customer_features['order_frequency'].mean()
        }
        
        return customer_features

    def scale_numerical_features(self, features_df):
        """
        Scale numerical features to standard normal distribution
        Args:
            features_df (pandas.DataFrame): Features to scale
        Returns:
            pandas.DataFrame: Scaled features
        """
        logger.info("Scaling numerical features...")
        scaled_features = features_df.copy()
        
        # Fit scaler and transform features to standard normal distribution
        scaled_features[self.numerical_features] = self.numerical_scaler.fit_transform(
            features_df[self.numerical_features]
        )
        
        # Track scaling parameters
        self.feature_stats['scaling'] = {
            feature: {
                'mean': self.numerical_scaler.mean_[i],
                'scale': self.numerical_scaler.scale_[i]
            }
            for i, feature in enumerate(self.numerical_features)
        }
        
        return scaled_features

    def encode_categorical_features(self, features_df):
        """
        Convert categorical features to numerical using one-hot encoding
        Args:
            features_df (pandas.DataFrame): Features to encode
        Returns:
            pandas.DataFrame: Encoded features
        """
        logger.info("Encoding categorical features...")
        
        # Fit encoder and transform categories to binary columns
        encoded_features = self.categorical_encoder.fit_transform(
            features_df[self.categorical_features]
        )
        
        # Get feature names for encoded columns
        feature_names = self.categorical_encoder.get_feature_names_out(
            self.categorical_features
        )
        
        # Create DataFrame with proper column names
        encoded_df = pd.DataFrame(
            encoded_features,
            columns=feature_names,
            index=features_df.index
        )
        
        # Track encoding information
        self.feature_stats['encoding'] = {
            feature: list(categories)
            for feature, categories in zip(
                self.categorical_features,
                self.categorical_encoder.categories_
            )
        }
        
        return encoded_df

    def transform(self, df):
        """
        Transform raw data into ML-ready features
        Args:
            df (pandas.DataFrame): Raw input data
        Returns:
            pandas.DataFrame: Processed features ready for ML
        """
        logger.info("Starting feature transformation pipeline...")
        
        # Create different types of features
        time_features = self.create_time_features(df)
        price_features = self.create_price_features(df)
        customer_features = self.create_customer_features(df)
        
        # Combine all numerical features
        numerical_df = pd.concat(
            [time_features, price_features, customer_features],
            axis=1
        )
        
        # Scale numerical features
        scaled_features = self.scale_numerical_features(numerical_df)
        
        # Encode categorical features
        encoded_features = self.encode_categorical_features(df)
        
        # Combine all features
        final_features = pd.concat([scaled_features, encoded_features], axis=1)
        
        logger.info(f"Feature engineering complete. Created {final_features.shape[1]} features.")
        return final_features

    def get_feature_stats(self):
        """
        Return statistics about the engineered features
        Returns:
            dict: Feature statistics and parameters
        """
        return self.feature_stats

if __name__ == "__main__":
    # Load raw data
    logger.info("Loading raw data...")
    df = pd.read_csv('../data/synthetic_data.csv')
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Transform data
    logger.info("Starting feature engineering process...")
    features = engineer.transform(df)
    
    # Save engineered features
    logger.info("Saving engineered features...")
    features.to_csv('../data/engineered_features.csv', index=False)
    
    # Print feature statistics
    stats = engineer.get_feature_stats()
    print("\nFeature Engineering Summary:")
    print(f"Total features created: {features.shape[1]}")
    print("\nNumerical Feature Statistics:")
    for feature, params in stats['scaling'].items():
        print(f"{feature}:")
        print(f"  Mean: {params['mean']:.2f}")
        print(f"  Scale: {params['scale']:.2f}")
    
    print("\nCategorical Feature Counts:")
    for feature, categories in stats['encoding'].items():
        print(f"{feature}: {len(categories)} unique values")