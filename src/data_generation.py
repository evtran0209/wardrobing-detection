"""
Wardrobing Fraud Detection System - Data Generation
Generates synthetic e-commerce data with realistic fraud patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

class ProductCategory:
    def __init__(self):
        self.categories = {
            'Formal Wear': {
                'subcategories': ['Evening Gowns', 'Tuxedos', 'Cocktail Dresses'],
                'price_range': (200, 1000),
                'fraud_risk': 0.25,
                'seasonal_factors': {
                    'wedding_season': 1.5,
                    'prom_season': 2.0,
                    'holiday_season': 1.3
                },
                'return_window': 30,
                'typical_wear_time': 1
            },
            'Designer Accessories': {
                'subcategories': ['Handbags', 'Shoes', 'Jewelry'],
                'price_range': (300, 3000),
                'fraud_risk': 0.20,
                'seasonal_factors': {
                    'holiday_season': 1.8
                },
                'return_window': 30,
                'typical_wear_time': 1
            },
            'Everyday Clothing': {
                'subcategories': ['T-shirts', 'Jeans', 'Casual Wear'],
                'price_range': (20, 150),
                'fraud_risk': 0.05,
                'seasonal_factors': {
                    'back_to_school': 1.2
                },
                'return_window': 30,
                'typical_wear_time': 7
            },
            'Luxury Outerwear': {
                'subcategories': ['Winter Coats', 'Designer Jackets'],
                'price_range': (500, 2000),
                'fraud_risk': 0.15,
                'seasonal_factors': {
                    'winter_season': 1.6
                },
                'return_window': 30,
                'typical_wear_time': 3
            },
            'Special Occasion': {
                'subcategories': ['Wedding Guest Dresses', 'Interview Suits'],
                'price_range': (150, 800),
                'fraud_risk': 0.30,
                'seasonal_factors': {
                    'wedding_season': 2.0,
                    'graduation_season': 1.7
                },
                'return_window': 30,
                'typical_wear_time': 1
            }
        }

class CustomerProfile:
    def __init__(self, customer_id):
        # Determine customer type with weighted probability
        self.customer_type = random.choices(
            ['genuine', 'occasional_fraudster', 'frequent_fraudster'],
            weights=[0.80, 0.15, 0.05]
        )[0]
        
        self.customer_id = customer_id
        self.profile = {
            'genuine': {
                'return_rate': random.uniform(0.05, 0.20),
                'fraud_probability': random.uniform(0.01, 0.05),
                'account_age': random.randint(1, 1095),  # Up to 3 years
                'total_orders': random.randint(1, 50)
            },
            'occasional_fraudster': {
                'return_rate': random.uniform(0.20, 0.40),
                'fraud_probability': random.uniform(0.30, 0.50),
                'account_age': random.randint(1, 365),   # Newer accounts
                'total_orders': random.randint(1, 20)
            },
            'frequent_fraudster': {
                'return_rate': random.uniform(0.40, 0.70),
                'fraud_probability': random.uniform(0.70, 0.90),
                'account_age': random.randint(1, 180),   # Very new accounts
                'total_orders': random.randint(1, 10)
            }
        }[self.customer_type]

def calculate_return_probability(customer, product_info, purchase_date):
    """Calculate return probability based on multiple risk factors"""
    
    # Base probability from customer profile
    base_prob = customer.profile['return_rate']
    
    # Product category risk
    category_risk = product_info['base_fraud_risk']
    
    # Seasonal factor
    month = purchase_date.month
    seasonal_multiplier = 1.0
    if month in [5,6,7,8]:  # Wedding season
        seasonal_multiplier = product_info.get('seasonal_factors', {}).get('wedding_season', 1.0)
    elif month in [11,12]:  # Holiday season
        seasonal_multiplier = product_info.get('seasonal_factors', {}).get('holiday_season', 1.0)
    
    # Price factor (higher priced items more likely to be returned)
    price_factor = min(product_info['price'] / 1000, 1.5)
    
    return min(base_prob * category_risk * seasonal_multiplier * price_factor, 1.0)

def generate_synthetic_data(num_customers=1000, num_orders=5000):
    """Generate synthetic e-commerce data with realistic fraud patterns"""
    
    # Initialize random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Initialize product categories and customers
    products = ProductCategory()
    customers = {i: CustomerProfile(i) for i in range(num_customers)}
    
    # Generate orders
    data = []
    for order_id in range(num_orders):
        # Select customer (weighted by return rate)
        customer = random.choices(
            list(customers.values()),
            weights=[c.profile['return_rate'] for c in customers.values()]
        )[0]
        
        # Select product category
        category_weights = {
            'Formal Wear': 0.2,
            'Designer Accessories': 0.15,
            'Everyday Clothing': 0.4,
            'Luxury Outerwear': 0.1,
            'Special Occasion': 0.15
        }
        
        category_name = random.choices(
            list(category_weights.keys()),
            weights=list(category_weights.values())
        )[0]
        
        # Generate product info
        category = products.categories[category_name]
        subcategory = random.choice(category['subcategories'])
        price = round(random.uniform(*category['price_range']), 2)
        
        product_info = {
            'category_name': category_name,
            'subcategory': subcategory,
            'base_fraud_risk': category['fraud_risk'],
            'price': price,
            'seasonal_factors': category['seasonal_factors']
        }
        
        # Generate purchase date
        purchase_date = datetime(2023, 1, 1) + timedelta(
            days=random.randint(0, 365)
        )
        
        # Calculate return probability
        return_probability = calculate_return_probability(
            customer, product_info, purchase_date
        )
        
        # Determine if order will be returned
        is_returned = random.random() < return_probability
        
        # Generate return information if applicable
        return_date = None
        return_reason = None
        is_fraudulent = False
        wear_signs = False
        packaging_damage = False
        
        if is_returned:
            # Calculate fraud probability
            fraud_probability = (
                customer.profile['fraud_probability'] * 
                product_info['base_fraud_risk'] * 
                (1.5 if product_info['price'] > 500 else 1.0)
            )
            is_fraudulent = random.random() < fraud_probability
            
            # Generate return details based on fraud status
            if is_fraudulent:
                days_to_return = random.randint(25, category['return_window'])
                wear_signs = random.random() < 0.8
                packaging_damage = random.random() < 0.7
                reasons = {
                    "Doesn't fit": 0.45,
                    "Changed mind": 0.35,
                    "Not as expected": 0.20
                }
            else:
                days_to_return = random.randint(1, 15)
                wear_signs = random.random() < 0.1
                packaging_damage = random.random() < 0.2
                reasons = {
                    "Doesn't fit": 0.30,
                    "Damaged/defective": 0.25,
                    "Not as expected": 0.25,
                    "Changed mind": 0.20
                }
            
            return_date = purchase_date + timedelta(days=days_to_return)
            return_reason = random.choices(
                list(reasons.keys()),
                weights=list(reasons.values())
            )[0]
        
        # Create order record
        data.append({
            'order_id': order_id,
            'customer_id': customer.customer_id,
            'customer_type': customer.customer_type,
            'product_category': category_name,
            'product_subcategory': subcategory,
            'purchase_date': purchase_date,
            'return_date': return_date,
            'days_to_return': (return_date - purchase_date).days if return_date else None,
            'return_reason': return_reason,
            'product_price': price,
            'account_age_days': customer.profile['account_age'],
            'customer_total_orders': customer.profile['total_orders'],
            'is_returned': is_returned,
            'is_fraudulent': is_fraudulent,
            'wear_signs': wear_signs if is_returned else None,
            'packaging_damage': packaging_damage if is_returned else None
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate the data
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    
    # Save to CSV
    print("Saving data to CSV...")
    df.to_csv('../data/synthetic_data.csv', index=False)
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total Orders: {len(df)}")
    print(f"Return Rate: {(df['is_returned'].mean() * 100):.1f}%")
    print(f"Fraud Rate: {(df['is_fraudulent'].mean() * 100):.1f}%")
    print(f"Average Order Value: ${df['product_price'].mean():.2f}")
    
    # Print fraud patterns
    print("\nFraud by Category:")
    print(df.groupby('product_category')['is_fraudulent'].mean().sort_values(ascending=False))
    
    print("\nFraud by Customer Type:")
    print(df.groupby('customer_type')['is_fraudulent'].mean())
    
    print("\nData generated and saved successfully!")