"""
Data Storage Module
Handles storing and retrieving articles and classifications using SQLite
"""

import sqlite3
import json
import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from config import Config

logger = logging.getLogger(__name__)

class DataStorage:
    """Handles data persistence for articles and classifications"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or Config.DATABASE_PATH
        self._ensure_db_directory()
        self._init_database()
    
    def _ensure_db_directory(self):
        """Ensure the database directory exists"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
    
    def _init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Articles table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS articles (
                        id TEXT PRIMARY KEY,
                        title TEXT,
                        summary TEXT,
                        content TEXT,
                        url TEXT,
                        author TEXT,
                        published_date TEXT,
                        feed_url TEXT,
                        feed_title TEXT,
                        raw_data TEXT,
                        created_at TEXT,
                        updated_at TEXT
                    )
                ''')
                
                # Classifications table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS classifications (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        article_id TEXT,
                        category TEXT,
                        confidence REAL,
                        rank INTEGER,
                        model_version TEXT,
                        classified_at TEXT,
                        FOREIGN KEY (article_id) REFERENCES articles (id)
                    )
                ''')
                
                # User feedback table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        article_id TEXT,
                        predicted_category TEXT,
                        actual_category TEXT,
                        feedback_type TEXT,
                        created_at TEXT,
                        FOREIGN KEY (article_id) REFERENCES articles (id)
                    )
                ''')
                
                # Training data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS training_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        article_id TEXT,
                        category TEXT,
                        source TEXT,
                        quality_score REAL,
                        created_at TEXT,
                        FOREIGN KEY (article_id) REFERENCES articles (id)
                    )
                ''')
                
                # Model metadata table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS model_metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_version TEXT,
                        algorithm TEXT,
                        accuracy REAL,
                        training_samples INTEGER,
                        feature_count INTEGER,
                        created_at TEXT,
                        is_active BOOLEAN
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_articles_created_at ON articles (created_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_classifications_article_id ON classifications (article_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_classifications_category ON classifications (category)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_feedback_article_id ON user_feedback (article_id)')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except sqlite3.Error as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def save_article(self, article: Dict) -> bool:
        """
        Save an article to the database
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Extract article data
                article_id = article.get('id', '')
                title = article.get('title', '')
                summary = self._extract_summary(article)
                content = self._extract_content(article)
                url = article.get('alternate', [{}])[0].get('href', '') if article.get('alternate') else ''
                author = article.get('author', '')
                published_date = self._extract_published_date(article)
                feed_url = self._extract_feed_url(article)
                feed_title = self._extract_feed_title(article)
                raw_data = json.dumps(article)
                created_at = datetime.now().isoformat()
                
                # Insert or update article
                cursor.execute('''
                    INSERT OR REPLACE INTO articles 
                    (id, title, summary, content, url, author, published_date, 
                     feed_url, feed_title, raw_data, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (article_id, title, summary, content, url, author, 
                      published_date, feed_url, feed_title, raw_data, 
                      created_at, created_at))
                
                conn.commit()
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Failed to save article {article.get('id', 'unknown')}: {e}")
            return False
    
    def save_articles(self, articles: List[Dict]) -> int:
        """
        Save multiple articles to the database
        Returns the number of successfully saved articles
        """
        saved_count = 0
        for article in articles:
            if self.save_article(article):
                saved_count += 1
        
        logger.info(f"Saved {saved_count}/{len(articles)} articles")
        return saved_count
    
    def save_classification(self, article_id: str, predictions: List[Tuple[str, float]], 
                          model_version: str = "1.0") -> bool:
        """
        Save classification results for an article
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                classified_at = datetime.now().isoformat()
                
                # Delete existing classifications for this article and model
                cursor.execute('''
                    DELETE FROM classifications 
                    WHERE article_id = ? AND model_version = ?
                ''', (article_id, model_version))
                
                # Insert new classifications
                for rank, (category, confidence) in enumerate(predictions, 1):
                    cursor.execute('''
                        INSERT INTO classifications 
                        (article_id, category, confidence, rank, model_version, classified_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (article_id, category, confidence, rank, model_version, classified_at))
                
                conn.commit()
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Failed to save classification for article {article_id}: {e}")
            return False
    
    def get_article(self, article_id: str) -> Optional[Dict]:
        """
        Get an article by ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM articles WHERE id = ?', (article_id,))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_article_dict(cursor, row)
                return None
                
        except sqlite3.Error as e:
            logger.error(f"Failed to get article {article_id}: {e}")
            return None
    
    def get_articles(self, limit: int = 100, offset: int = 0, 
                    category: str = None, since: datetime = None) -> List[Dict]:
        """
        Get articles with optional filtering
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = 'SELECT * FROM articles'
                params = []
                where_clauses = []
                
                if since:
                    where_clauses.append('created_at >= ?')
                    params.append(since.isoformat())
                
                if category:
                    where_clauses.append('''
                        id IN (SELECT article_id FROM classifications 
                               WHERE category = ? AND rank = 1)
                    ''')
                    params.append(category)
                
                if where_clauses:
                    query += ' WHERE ' + ' AND '.join(where_clauses)
                
                query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
                params.extend([limit, offset])
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_article_dict(cursor, row) for row in rows]
                
        except sqlite3.Error as e:
            logger.error(f"Failed to get articles: {e}")
            return []
    
    def get_unclassified_articles(self, model_version: str = "1.0", limit: int = 100) -> List[Dict]:
        """
        Get articles that haven't been classified yet
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM articles 
                    WHERE id NOT IN (
                        SELECT article_id FROM classifications 
                        WHERE model_version = ?
                    )
                    ORDER BY created_at DESC 
                    LIMIT ?
                ''', (model_version, limit))
                
                rows = cursor.fetchall()
                return [self._row_to_article_dict(cursor, row) for row in rows]
                
        except sqlite3.Error as e:
            logger.error(f"Failed to get unclassified articles: {e}")
            return []
    
    def get_classifications(self, article_id: str, model_version: str = "1.0") -> List[Tuple[str, float]]:
        """
        Get classifications for an article
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT category, confidence FROM classifications 
                    WHERE article_id = ? AND model_version = ?
                    ORDER BY rank
                ''', (article_id, model_version))
                
                return cursor.fetchall()
                
        except sqlite3.Error as e:
            logger.error(f"Failed to get classifications for article {article_id}: {e}")
            return []
    
    def save_user_feedback(self, article_id: str, predicted_category: str, 
                          actual_category: str, feedback_type: str = "correction") -> bool:
        """
        Save user feedback for improving the model
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO user_feedback 
                    (article_id, predicted_category, actual_category, feedback_type, created_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (article_id, predicted_category, actual_category, 
                      feedback_type, datetime.now().isoformat()))
                
                conn.commit()
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Failed to save user feedback: {e}")
            return False
    
    def get_training_data(self, min_quality_score: float = 0.0) -> List[Tuple[Dict, str]]:
        """
        Get training data (articles and their labels)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT a.*, t.category FROM articles a
                    JOIN training_data t ON a.id = t.article_id
                    WHERE t.quality_score >= ?
                    ORDER BY t.quality_score DESC
                ''', (min_quality_score,))
                
                rows = cursor.fetchall()
                training_data = []
                
                for row in rows:
                    article = self._row_to_article_dict(cursor, row[:-1])  # Exclude category
                    category = row[-1]  # Last column is category
                    training_data.append((article, category))
                
                return training_data
                
        except sqlite3.Error as e:
            logger.error(f"Failed to get training data: {e}")
            return []
    
    def save_model_metadata(self, model_version: str, algorithm: str, 
                           accuracy: float, training_samples: int, 
                           feature_count: int) -> bool:
        """
        Save model training metadata
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Deactivate previous models
                cursor.execute('UPDATE model_metadata SET is_active = 0')
                
                # Insert new model metadata
                cursor.execute('''
                    INSERT INTO model_metadata 
                    (model_version, algorithm, accuracy, training_samples, 
                     feature_count, created_at, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                ''', (model_version, algorithm, accuracy, training_samples, 
                      feature_count, datetime.now().isoformat()))
                
                conn.commit()
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Failed to save model metadata: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Total articles
                cursor.execute('SELECT COUNT(*) FROM articles')
                stats['total_articles'] = cursor.fetchone()[0]
                
                # Articles by day (last 7 days)
                cursor.execute('''
                    SELECT DATE(created_at) as date, COUNT(*) as count
                    FROM articles 
                    WHERE created_at >= date('now', '-7 days')
                    GROUP BY DATE(created_at)
                    ORDER BY date DESC
                ''')
                stats['articles_by_day'] = cursor.fetchall()
                
                # Classifications by category
                cursor.execute('''
                    SELECT category, COUNT(*) as count
                    FROM classifications 
                    WHERE rank = 1
                    GROUP BY category
                    ORDER BY count DESC
                ''')
                stats['classifications_by_category'] = cursor.fetchall()
                
                # User feedback count
                cursor.execute('SELECT COUNT(*) FROM user_feedback')
                stats['user_feedback_count'] = cursor.fetchone()[0]
                
                # Model accuracy (latest)
                cursor.execute('''
                    SELECT accuracy FROM model_metadata 
                    WHERE is_active = 1
                    ORDER BY created_at DESC 
                    LIMIT 1
                ''')
                result = cursor.fetchone()
                stats['current_model_accuracy'] = result[0] if result else None
                
                return stats
                
        except sqlite3.Error as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """
        Clean up old data to save space
        Returns number of deleted records
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
            deleted_count = 0
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old classifications first (foreign key constraint)
                cursor.execute('''
                    DELETE FROM classifications 
                    WHERE article_id IN (
                        SELECT id FROM articles 
                        WHERE created_at < ?
                    )
                ''', (cutoff_date,))
                deleted_count += cursor.rowcount
                
                # Delete old articles
                cursor.execute('DELETE FROM articles WHERE created_at < ?', (cutoff_date,))
                deleted_count += cursor.rowcount
                
                conn.commit()
                logger.info(f"Cleaned up {deleted_count} old records")
                return deleted_count
                
        except sqlite3.Error as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0
    
    def _extract_summary(self, article: Dict) -> str:
        """Extract summary from article data"""
        summary = article.get('summary', {})
        if isinstance(summary, dict):
            return summary.get('content', '')
        return str(summary) if summary else ''
    
    def _extract_content(self, article: Dict) -> str:
        """Extract content from article data"""
        content = article.get('content', {})
        if isinstance(content, dict):
            return content.get('content', '')
        elif isinstance(content, list) and content:
            return content[0].get('content', '') if isinstance(content[0], dict) else str(content[0])
        return str(content) if content else ''
    
    def _extract_published_date(self, article: Dict) -> str:
        """Extract published date from article data"""
        published = article.get('published', 0)
        if isinstance(published, (int, float)):
            return datetime.fromtimestamp(published).isoformat()
        return str(published) if published else ''
    
    def _extract_feed_url(self, article: Dict) -> str:
        """Extract feed URL from article data"""
        origin = article.get('origin', {})
        return origin.get('streamId', '') if isinstance(origin, dict) else ''
    
    def _extract_feed_title(self, article: Dict) -> str:
        """Extract feed title from article data"""
        origin = article.get('origin', {})
        return origin.get('title', '') if isinstance(origin, dict) else ''
    
    def _row_to_article_dict(self, cursor, row) -> Dict:
        """Convert database row to article dictionary"""
        columns = [desc[0] for desc in cursor.description]
        article_dict = dict(zip(columns, row))
        
        # Parse raw_data if available
        if article_dict.get('raw_data'):
            try:
                raw_data = json.loads(article_dict['raw_data'])
                article_dict.update(raw_data)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse raw_data for article {article_dict.get('id')}")
        
        return article_dict