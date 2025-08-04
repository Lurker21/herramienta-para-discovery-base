"""
News Classification Module
Implements machine learning models to classify news articles into categories
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from bs4 import BeautifulSoup
from config import Config

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class NewsClassifier:
    """Machine Learning classifier for news articles"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.categories = Config.NEWS_CATEGORIES
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(Config.STOPWORDS_LANGUAGE))
        self.model_path = Config.MODEL_PATH
        self.is_trained = False
        
        # Ensure model directory exists
        os.makedirs(self.model_path, exist_ok=True)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text for classification
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove special characters and digits, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def extract_features_from_article(self, article: Dict) -> str:
        """
        Extract text features from an article dictionary
        """
        features = []
        
        # Title (weighted more heavily)
        title = article.get('title', '')
        if title:
            # Add title multiple times to give it more weight
            clean_title = self.clean_text(title)
            features.extend([clean_title] * 3)
        
        # Summary/content
        summary = ''
        if 'summary' in article and article['summary']:
            if isinstance(article['summary'], dict):
                summary = article['summary'].get('content', '')
            else:
                summary = str(article['summary'])
        
        if summary:
            features.append(self.clean_text(summary))
        
        # Content if available
        content = article.get('content', '')
        if content:
            if isinstance(content, dict):
                content = content.get('content', '')
            features.append(self.clean_text(content))
        
        # Categories/tags if available
        categories = article.get('categories', [])
        if categories:
            for cat in categories:
                if isinstance(cat, str):
                    features.append(self.clean_text(cat))
                elif isinstance(cat, dict):
                    features.append(self.clean_text(cat.get('label', '')))
        
        return ' '.join(features)
    
    def create_training_data(self, articles: List[Dict], labels: List[str]) -> Tuple[List[str], List[str]]:
        """
        Create training data from articles and their labels
        """
        texts = []
        clean_labels = []
        
        for article, label in zip(articles, labels):
            text = self.extract_features_from_article(article)
            if len(text) >= Config.MIN_TEXT_LENGTH:
                texts.append(text)
                clean_labels.append(label.lower())
        
        return texts, clean_labels
    
    def train_model(self, training_articles: List[Dict], labels: List[str], 
                   algorithm: str = 'random_forest') -> Dict[str, Any]:
        """
        Train the classification model
        
        Args:
            training_articles: List of article dictionaries
            labels: List of corresponding category labels
            algorithm: Algorithm to use ('random_forest', 'logistic_regression', 'naive_bayes', 'svm')
        """
        logger.info(f"Training model with {len(training_articles)} articles...")
        
        # Prepare training data
        texts, clean_labels = self.create_training_data(training_articles, labels)
        
        if len(texts) < 10:
            raise ValueError("Insufficient training data. Need at least 10 articles.")
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=Config.MAX_FEATURES,
            stop_words='english',
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8  # Ignore terms that appear in more than 80% of documents
        )
        
        # Vectorize texts
        X = self.vectorizer.fit_transform(texts)
        y = clean_labels
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Choose and train model
        if algorithm == 'random_forest':
            base_model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                class_weight='balanced'
            )
        elif algorithm == 'logistic_regression':
            base_model = LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'
            )
        elif algorithm == 'naive_bayes':
            base_model = MultinomialNB(alpha=0.1)
        elif algorithm == 'svm':
            base_model = SVC(
                kernel='linear', 
                random_state=42, 
                class_weight='balanced',
                probability=True
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Use OneVsRest for multi-class classification
        self.model = OneVsRestClassifier(base_model)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get detailed classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        self.is_trained = True
        
        training_results = {
            'algorithm': algorithm,
            'accuracy': accuracy,
            'training_samples': len(texts),
            'test_samples': len(y_test),
            'classification_report': report,
            'feature_count': X.shape[1]
        }
        
        logger.info(f"Model training completed. Accuracy: {accuracy:.3f}")
        return training_results
    
    def classify_article(self, article: Dict, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Classify a single article and return top-k predictions with confidence scores
        """
        if not self.is_trained or not self.model or not self.vectorizer:
            raise ValueError("Model not trained. Call train_model() first or load_model().")
        
        # Extract and clean text
        text = self.extract_features_from_article(article)
        
        if len(text) < Config.MIN_TEXT_LENGTH:
            logger.warning("Article text too short for reliable classification")
            return [('unknown', 0.0)]
        
        # Vectorize text
        X = self.vectorizer.transform([text])
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X)[0]
        
        # Get class labels
        classes = self.model.classes_
        
        # Create list of (class, probability) tuples
        predictions = list(zip(classes, probabilities))
        
        # Sort by probability (descending) and return top-k
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:top_k]
    
    def classify_articles(self, articles: List[Dict], top_k: int = 3) -> List[List[Tuple[str, float]]]:
        """
        Classify multiple articles efficiently
        """
        if not self.is_trained or not self.model or not self.vectorizer:
            raise ValueError("Model not trained. Call train_model() first or load_model().")
        
        # Extract texts
        texts = [self.extract_features_from_article(article) for article in articles]
        
        # Filter out short texts
        valid_indices = [i for i, text in enumerate(texts) if len(text) >= Config.MIN_TEXT_LENGTH]
        valid_texts = [texts[i] for i in valid_indices]
        
        if not valid_texts:
            return [[('unknown', 0.0)] for _ in articles]
        
        # Vectorize all texts at once
        X = self.vectorizer.transform(valid_texts)
        
        # Get predictions
        probabilities = self.model.predict_proba(X)
        classes = self.model.classes_
        
        # Process results
        results = []
        valid_idx = 0
        
        for i, article in enumerate(articles):
            if i in valid_indices:
                probs = probabilities[valid_idx]
                predictions = list(zip(classes, probs))
                predictions.sort(key=lambda x: x[1], reverse=True)
                results.append(predictions[:top_k])
                valid_idx += 1
            else:
                results.append([('unknown', 0.0)])
        
        return results
    
    def save_model(self, filename_suffix: str = "") -> bool:
        """
        Save the trained model and vectorizer to disk
        """
        if not self.is_trained or not self.model or not self.vectorizer:
            logger.error("No trained model to save")
            return False
        
        try:
            # Create filenames
            model_file = os.path.join(
                self.model_path, 
                f"{filename_suffix}_{Config.CLASSIFIER_MODEL_FILE}" if filename_suffix else Config.CLASSIFIER_MODEL_FILE
            )
            vectorizer_file = os.path.join(
                self.model_path,
                f"{filename_suffix}_{Config.VECTORIZER_FILE}" if filename_suffix else Config.VECTORIZER_FILE
            )
            
            # Save model and vectorizer
            with open(model_file, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(vectorizer_file, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            logger.info(f"Model saved to {model_file}")
            logger.info(f"Vectorizer saved to {vectorizer_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filename_suffix: str = "") -> bool:
        """
        Load a previously trained model and vectorizer from disk
        """
        try:
            # Create filenames
            model_file = os.path.join(
                self.model_path, 
                f"{filename_suffix}_{Config.CLASSIFIER_MODEL_FILE}" if filename_suffix else Config.CLASSIFIER_MODEL_FILE
            )
            vectorizer_file = os.path.join(
                self.model_path,
                f"{filename_suffix}_{Config.VECTORIZER_FILE}" if filename_suffix else Config.VECTORIZER_FILE
            )
            
            # Check if files exist
            if not os.path.exists(model_file) or not os.path.exists(vectorizer_file):
                logger.error(f"Model files not found: {model_file}, {vectorizer_file}")
                return False
            
            # Load model and vectorizer
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(vectorizer_file, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            self.is_trained = True
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get most important features for each class (works with tree-based models)
        """
        if not self.is_trained or not self.model or not self.vectorizer:
            return {}
        
        try:
            feature_names = self.vectorizer.get_feature_names_out()
            classes = self.model.classes_
            
            # Get feature importance for each classifier
            importance_by_class = {}
            
            for i, class_name in enumerate(classes):
                if hasattr(self.model.estimators_[i], 'feature_importances_'):
                    # Tree-based models
                    importances = self.model.estimators_[i].feature_importances_
                elif hasattr(self.model.estimators_[i], 'coef_'):
                    # Linear models
                    importances = abs(self.model.estimators_[i].coef_[0])
                else:
                    continue
                
                # Get top features
                top_indices = np.argsort(importances)[-top_n:][::-1]
                top_features = [(feature_names[idx], importances[idx]) for idx in top_indices]
                importance_by_class[class_name] = top_features
            
            return importance_by_class
            
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return {}
    
    def evaluate_model(self, test_articles: List[Dict], test_labels: List[str]) -> Dict[str, Any]:
        """
        Evaluate model performance on test data
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Prepare test data
        texts, labels = self.create_training_data(test_articles, test_labels)
        
        if not texts:
            raise ValueError("No valid test data")
        
        # Vectorize
        X_test = self.vectorizer.transform(texts)
        y_test = labels
        
        # Predict
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'test_samples': len(texts)
        }