"""
Configuration file for Inoreader News Classifier
Contains API credentials and application settings
"""

import os
from typing import Dict, List

class Config:
    """Configuration settings for the Inoreader News Classifier"""
    
    # Inoreader API Configuration
    INOREADER_BASE_URL = "https://www.inoreader.com"
    INOREADER_API_URL = f"{INOREADER_BASE_URL}/reader/api/0"
    
    # API Credentials (set these in environment variables or config file)
    INOREADER_EMAIL = os.getenv('INOREADER_EMAIL', '')
    INOREADER_PASSWORD = os.getenv('INOREADER_PASSWORD', '')
    INOREADER_APP_ID = os.getenv('INOREADER_APP_ID', 'InoreaderNewsClassifier')
    INOREADER_APP_KEY = os.getenv('INOREADER_APP_KEY', '')
    
    # Classification Categories
    NEWS_CATEGORIES = [
        'business',
        'technology', 
        'politics',
        'sports',
        'entertainment',
        'science',
        'health',
        'world',
        'local',
        'opinion'
    ]
    
    # Model Configuration
    MODEL_PATH = 'models/'
    CLASSIFIER_MODEL_FILE = 'news_classifier.pkl'
    VECTORIZER_FILE = 'text_vectorizer.pkl'
    
    # Data Storage
    DATABASE_PATH = 'data/news_classifier.db'
    CACHE_DURATION_HOURS = 24
    
    # Text Processing
    MAX_FEATURES = 10000
    MIN_TEXT_LENGTH = 50
    STOPWORDS_LANGUAGE = 'english'
    
    # API Rate Limiting
    API_RATE_LIMIT = 100  # requests per day for free tier
    REQUEST_DELAY = 0.5   # seconds between requests
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'logs/news_classifier.log'
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration is present"""
        if not cls.INOREADER_EMAIL or not cls.INOREADER_PASSWORD:
            return False
        return True
    
    @classmethod
    def get_headers(cls) -> Dict[str, str]:
        """Get common headers for API requests"""
        return {
            'User-Agent': f'{cls.INOREADER_APP_ID}/1.0',
            'Content-Type': 'application/x-www-form-urlencoded'
        }