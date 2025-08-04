"""
Main News Classifier Application
Orchestrates the entire workflow of fetching, classifying, and storing news articles
"""

import logging
import os
import sys
import argparse
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from inoreader_client import InoreaderClient
from news_classifier import NewsClassifier
from data_storage import DataStorage

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(Config.LOG_FILE)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )

logger = logging.getLogger(__name__)

class NewsClassifierApp:
    """Main application class for the Inoreader News Classifier"""
    
    def __init__(self):
        self.inoreader_client = InoreaderClient()
        self.classifier = NewsClassifier()
        self.storage = DataStorage()
        self.model_version = "1.0"
        
    def authenticate(self) -> bool:
        """Authenticate with Inoreader API"""
        logger.info("Authenticating with Inoreader...")
        success = self.inoreader_client.authenticate()
        if success:
            user_info = self.inoreader_client.get_user_info()
            if user_info:
                logger.info(f"Authenticated as {user_info.get('userName', 'Unknown User')}")
        return success
    
    def fetch_articles(self, max_articles: int = 100, unread_only: bool = True) -> List[Dict]:
        """
        Fetch articles from Inoreader
        
        Args:
            max_articles: Maximum number of articles to fetch
            unread_only: Whether to fetch only unread articles
        """
        logger.info(f"Fetching {'unread' if unread_only else 'all'} articles...")
        
        if unread_only:
            articles = self.inoreader_client.get_all_unread_articles(max_articles)
        else:
            articles = self.inoreader_client.get_stream_contents(count=max_articles, exclude_read=False)
        
        logger.info(f"Fetched {len(articles)} articles")
        return articles
    
    def classify_and_store_articles(self, articles: List[Dict], 
                                   store_articles: bool = True, 
                                   store_classifications: bool = True) -> Dict[str, Any]:
        """
        Classify articles and optionally store them
        
        Args:
            articles: List of articles to classify
            store_articles: Whether to store articles in database
            store_classifications: Whether to store classification results
        """
        logger.info(f"Classifying {len(articles)} articles...")
        
        results = {
            'total_articles': len(articles),
            'stored_articles': 0,
            'classified_articles': 0,
            'classification_results': {},
            'errors': []
        }
        
        # Store articles if requested
        if store_articles and articles:
            results['stored_articles'] = self.storage.save_articles(articles)
        
        # Classify articles
        if not self.classifier.is_trained:
            logger.warning("Classifier not trained. Attempting to load existing model...")
            if not self.classifier.load_model():
                logger.error("No trained model found. Please train the model first.")
                return results
        
        try:
            # Classify all articles at once for efficiency
            classifications = self.classifier.classify_articles(articles, top_k=3)
            
            for i, (article, predictions) in enumerate(zip(articles, classifications)):
                article_id = article.get('id', f'unknown_{i}')
                
                try:
                    # Store classification results
                    if store_classifications and predictions and predictions[0][0] != 'unknown':
                        self.storage.save_classification(article_id, predictions, self.model_version)
                        results['classified_articles'] += 1
                    
                    # Aggregate results by category
                    if predictions and predictions[0][0] != 'unknown':
                        category = predictions[0][0]
                        confidence = predictions[0][1]
                        
                        if category not in results['classification_results']:
                            results['classification_results'][category] = {
                                'count': 0,
                                'avg_confidence': 0,
                                'articles': []
                            }
                        
                        cat_results = results['classification_results'][category]
                        cat_results['count'] += 1
                        cat_results['avg_confidence'] = (
                            (cat_results['avg_confidence'] * (cat_results['count'] - 1) + confidence) / 
                            cat_results['count']
                        )
                        cat_results['articles'].append({
                            'id': article_id,
                            'title': article.get('title', ''),
                            'confidence': confidence
                        })
                
                except Exception as e:
                    error_msg = f"Error processing article {article_id}: {e}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
        
        except Exception as e:
            error_msg = f"Error during classification: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        logger.info(f"Classification complete. {results['classified_articles']} articles classified.")
        return results
    
    def train_model(self, use_existing_data: bool = True, 
                   external_training_file: str = None,
                   algorithm: str = 'random_forest') -> Dict[str, Any]:
        """
        Train the classification model
        
        Args:
            use_existing_data: Whether to use articles stored in database for training
            external_training_file: Path to external training data file (JSON format)
            algorithm: Algorithm to use for training
        """
        logger.info("Training classification model...")
        
        training_articles = []
        training_labels = []
        
        # Load training data from database
        if use_existing_data:
            training_data = self.storage.get_training_data()
            if training_data:
                training_articles.extend([article for article, label in training_data])
                training_labels.extend([label for article, label in training_data])
                logger.info(f"Loaded {len(training_data)} training samples from database")
        
        # Load external training data if provided
        if external_training_file and os.path.exists(external_training_file):
            try:
                with open(external_training_file, 'r', encoding='utf-8') as f:
                    external_data = json.load(f)
                
                for item in external_data:
                    if 'article' in item and 'category' in item:
                        training_articles.append(item['article'])
                        training_labels.append(item['category'])
                
                logger.info(f"Loaded {len(external_data)} training samples from file")
            
            except Exception as e:
                logger.error(f"Failed to load external training data: {e}")
        
        # Generate synthetic training data if insufficient data
        if len(training_articles) < 50:
            logger.warning("Insufficient training data. Generating synthetic samples...")
            synthetic_data = self._generate_synthetic_training_data()
            training_articles.extend([article for article, label in synthetic_data])
            training_labels.extend([label for article, label in synthetic_data])
        
        if len(training_articles) < 10:
            raise ValueError("Insufficient training data. Need at least 10 labeled articles.")
        
        # Train the model
        try:
            training_results = self.classifier.train_model(
                training_articles, 
                training_labels, 
                algorithm=algorithm
            )
            
            # Save the trained model
            if self.classifier.save_model():
                logger.info("Model saved successfully")
            
            # Save model metadata
            self.storage.save_model_metadata(
                self.model_version,
                training_results['algorithm'],
                training_results['accuracy'],
                training_results['training_samples'],
                training_results['feature_count']
            )
            
            return training_results
        
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    def run_classification_pipeline(self, max_articles: int = 100, 
                                  unread_only: bool = True) -> Dict[str, Any]:
        """
        Run the complete pipeline: authenticate, fetch, classify, and store
        """
        logger.info("Starting news classification pipeline...")
        
        pipeline_results = {
            'authenticated': False,
            'articles_fetched': 0,
            'classification_results': {},
            'errors': [],
            'start_time': datetime.now().isoformat()
        }
        
        try:
            # Step 1: Authenticate
            if not self.authenticate():
                pipeline_results['errors'].append("Authentication failed")
                return pipeline_results
            
            pipeline_results['authenticated'] = True
            
            # Step 2: Fetch articles
            articles = self.fetch_articles(max_articles, unread_only)
            pipeline_results['articles_fetched'] = len(articles)
            
            if not articles:
                logger.info("No articles to process")
                return pipeline_results
            
            # Step 3: Classify and store articles
            classification_results = self.classify_and_store_articles(articles)
            pipeline_results['classification_results'] = classification_results
            
            # Step 4: Mark articles as read (optional)
            if unread_only:
                self._mark_articles_as_read(articles)
            
        except Exception as e:
            error_msg = f"Pipeline error: {e}"
            logger.error(error_msg)
            pipeline_results['errors'].append(error_msg)
        
        pipeline_results['end_time'] = datetime.now().isoformat()
        logger.info("Pipeline completed")
        return pipeline_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get application statistics"""
        stats = self.storage.get_statistics()
        
        # Add model information
        if self.classifier.is_trained:
            feature_importance = self.classifier.get_feature_importance()
            stats['feature_importance'] = feature_importance
        
        # Add rate limit information
        rate_limit_info = self.inoreader_client.get_rate_limit_info()
        stats['rate_limit_info'] = rate_limit_info
        
        return stats
    
    def search_articles(self, query: str, category: str = None, 
                       limit: int = 50) -> List[Dict]:
        """
        Search for articles by query and/or category
        """
        # Search in Inoreader
        inoreader_results = self.inoreader_client.search_articles(query, limit)
        
        # Search in local database
        stored_articles = self.storage.get_articles(limit=limit, category=category)
        
        # Filter stored articles by query
        filtered_stored = []
        if query:
            query_lower = query.lower()
            for article in stored_articles:
                title = article.get('title', '').lower()
                summary = article.get('summary', '').lower()
                if query_lower in title or query_lower in summary:
                    filtered_stored.append(article)
        else:
            filtered_stored = stored_articles
        
        # Combine and deduplicate results
        all_results = []
        seen_ids = set()
        
        for article in inoreader_results + filtered_stored:
            article_id = article.get('id')
            if article_id and article_id not in seen_ids:
                seen_ids.add(article_id)
                all_results.append(article)
        
        return all_results[:limit]
    
    def provide_feedback(self, article_id: str, predicted_category: str, 
                        actual_category: str) -> bool:
        """
        Provide feedback on classification results
        """
        return self.storage.save_user_feedback(
            article_id, predicted_category, actual_category
        )
    
    def _generate_synthetic_training_data(self) -> List[tuple]:
        """
        Generate synthetic training data based on category keywords
        """
        synthetic_data = []
        
        category_keywords = {
            'technology': [
                'AI', 'artificial intelligence', 'machine learning', 'software', 'tech',
                'programming', 'computer', 'digital', 'innovation', 'startup'
            ],
            'business': [
                'market', 'economy', 'finance', 'investment', 'company', 'corporate',
                'industry', 'revenue', 'profit', 'earnings'
            ],
            'sports': [
                'football', 'basketball', 'soccer', 'tennis', 'championship', 'team',
                'player', 'game', 'match', 'tournament'
            ],
            'politics': [
                'government', 'election', 'president', 'congress', 'policy', 'law',
                'political', 'democrat', 'republican', 'vote'
            ],
            'health': [
                'medical', 'healthcare', 'hospital', 'doctor', 'disease', 'treatment',
                'medicine', 'health', 'wellness', 'fitness'
            ]
        }
        
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                synthetic_article = {
                    'title': f"News about {keyword}",
                    'summary': {
                        'content': f"This is a news article about {keyword} and related topics in {category}."
                    },
                    'id': f"synthetic_{category}_{keyword}"
                }
                synthetic_data.append((synthetic_article, category))
        
        logger.info(f"Generated {len(synthetic_data)} synthetic training samples")
        return synthetic_data
    
    def _mark_articles_as_read(self, articles: List[Dict]) -> int:
        """Mark articles as read in Inoreader"""
        marked_count = 0
        for article in articles:
            article_id = article.get('id')
            if article_id and self.inoreader_client.mark_article_as_read(article_id):
                marked_count += 1
        
        logger.info(f"Marked {marked_count}/{len(articles)} articles as read")
        return marked_count

def main():
    """Main entry point"""
    setup_logging()
    
    parser = argparse.ArgumentParser(description='Inoreader News Classifier')
    parser.add_argument('--action', choices=['classify', 'train', 'stats', 'search'], 
                       default='classify', help='Action to perform')
    parser.add_argument('--max-articles', type=int, default=100, 
                       help='Maximum number of articles to process')
    parser.add_argument('--unread-only', action='store_true', default=True,
                       help='Process only unread articles')
    parser.add_argument('--algorithm', choices=['random_forest', 'logistic_regression', 'naive_bayes', 'svm'],
                       default='random_forest', help='Algorithm for training')
    parser.add_argument('--training-file', type=str, 
                       help='Path to external training data file')
    parser.add_argument('--query', type=str, help='Search query')
    parser.add_argument('--category', type=str, help='Filter by category')
    
    args = parser.parse_args()
    
    try:
        app = NewsClassifierApp()
        
        if args.action == 'classify':
            results = app.run_classification_pipeline(args.max_articles, args.unread_only)
            print("\n=== CLASSIFICATION RESULTS ===")
            print(json.dumps(results, indent=2))
            
        elif args.action == 'train':
            results = app.train_model(
                use_existing_data=True,
                external_training_file=args.training_file,
                algorithm=args.algorithm
            )
            print("\n=== TRAINING RESULTS ===")
            print(f"Algorithm: {results['algorithm']}")
            print(f"Accuracy: {results['accuracy']:.3f}")
            print(f"Training samples: {results['training_samples']}")
            print(f"Features: {results['feature_count']}")
            
        elif args.action == 'stats':
            stats = app.get_statistics()
            print("\n=== STATISTICS ===")
            print(json.dumps(stats, indent=2, default=str))
            
        elif args.action == 'search':
            if not args.query and not args.category:
                print("Please provide either --query or --category for search")
                return
            
            results = app.search_articles(
                query=args.query or '',
                category=args.category,
                limit=args.max_articles
            )
            print(f"\n=== SEARCH RESULTS ({len(results)} articles) ===")
            for article in results:
                print(f"- {article.get('title', 'No title')}")
                print(f"  {article.get('summary', 'No summary')[:100]}...")
                print()
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()