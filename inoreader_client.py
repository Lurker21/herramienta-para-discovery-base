"""
Inoreader API Client
Handles authentication and data fetching from Inoreader API
"""

import requests
import time
import json
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode
from config import Config

logger = logging.getLogger(__name__)

class InoreaderClient:
    """Client for interacting with Inoreader API"""
    
    def __init__(self):
        self.base_url = Config.INOREADER_API_URL
        self.session = requests.Session()
        self.auth_token = None
        self._setup_session()
    
    def _setup_session(self):
        """Setup session with default headers"""
        self.session.headers.update(Config.get_headers())
    
    def authenticate(self) -> bool:
        """
        Authenticate with Inoreader API using email/password
        Returns True if successful, False otherwise
        """
        if not Config.validate_config():
            logger.error("Missing required configuration: email and password")
            return False
        
        auth_url = f"{self.base_url}/accounts/ClientLogin"
        auth_data = {
            'Email': Config.INOREADER_EMAIL,
            'Passwd': Config.INOREADER_PASSWORD,
            'service': 'reader',
            'accountType': 'HOSTED_OR_GOOGLE',
            'client': Config.INOREADER_APP_ID,
            'output': 'json'
        }
        
        try:
            logger.info("Attempting to authenticate with Inoreader...")
            response = self.session.post(auth_url, data=auth_data)
            response.raise_for_status()
            
            # Parse authentication response
            if response.headers.get('content-type', '').startswith('application/json'):
                auth_result = response.json()
                if 'Auth' in auth_result:
                    self.auth_token = auth_result['Auth']
                else:
                    logger.error(f"Authentication failed: {auth_result}")
                    return False
            else:
                # Parse text response format
                auth_text = response.text
                for line in auth_text.split('\n'):
                    if line.startswith('Auth='):
                        self.auth_token = line.split('=', 1)[1]
                        break
                
                if not self.auth_token:
                    logger.error("Could not extract auth token from response")
                    return False
            
            # Set authorization header for future requests
            self.session.headers['Authorization'] = f'GoogleLogin auth={self.auth_token}'
            logger.info("Authentication successful")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def _make_request(self, endpoint: str, params: Dict = None, method: str = 'GET') -> Optional[Dict]:
        """
        Make authenticated request to Inoreader API
        """
        if not self.auth_token:
            logger.error("Not authenticated. Call authenticate() first.")
            return None
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            # Add rate limiting
            time.sleep(Config.REQUEST_DELAY)
            
            if method.upper() == 'GET':
                response = self.session.get(url, params=params)
            else:
                response = self.session.post(url, data=params)
            
            response.raise_for_status()
            
            if response.headers.get('content-type', '').startswith('application/json'):
                return response.json()
            else:
                return {'content': response.text}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def get_user_info(self) -> Optional[Dict]:
        """Get user information"""
        return self._make_request('user-info', {'output': 'json'})
    
    def get_subscriptions(self) -> Optional[List[Dict]]:
        """Get list of subscribed feeds"""
        result = self._make_request('subscription/list', {'output': 'json'})
        if result and 'subscriptions' in result:
            return result['subscriptions']
        return []
    
    def get_unread_count(self) -> Optional[Dict]:
        """Get unread count for all feeds"""
        return self._make_request('unread-count', {'output': 'json'})
    
    def get_stream_contents(self, stream_id: str = None, count: int = 50, 
                          exclude_read: bool = True) -> Optional[List[Dict]]:
        """
        Get articles from a stream (feed or reading list)
        
        Args:
            stream_id: Stream ID (default: reading list)
            count: Number of articles to fetch
            exclude_read: Whether to exclude already read articles
        """
        if stream_id is None:
            stream_id = 'user/-/state/com.google/reading-list'
        
        params = {
            's': stream_id,
            'n': min(count, 1000),  # API limit
            'output': 'json'
        }
        
        if exclude_read:
            params['xt'] = 'user/-/state/com.google/read'
        
        result = self._make_request('stream/contents', params)
        if result and 'items' in result:
            return result['items']
        return []
    
    def get_all_unread_articles(self, max_articles: int = 500) -> List[Dict]:
        """
        Get all unread articles from all subscriptions
        """
        articles = []
        
        # Get reading list articles
        reading_list_articles = self.get_stream_contents(
            stream_id='user/-/state/com.google/reading-list',
            count=max_articles,
            exclude_read=True
        )
        
        if reading_list_articles:
            articles.extend(reading_list_articles)
            logger.info(f"Retrieved {len(reading_list_articles)} unread articles")
        
        return articles[:max_articles]
    
    def mark_article_as_read(self, article_id: str) -> bool:
        """Mark an article as read"""
        params = {
            'i': article_id,
            'a': 'user/-/state/com.google/read'
        }
        
        result = self._make_request('edit-tag', params, method='POST')
        return result is not None
    
    def get_article_content(self, article_ids: List[str]) -> Optional[List[Dict]]:
        """
        Get full content for specific articles
        """
        if not article_ids:
            return []
        
        # API allows multiple article IDs
        params = {'output': 'json'}
        for article_id in article_ids[:20]:  # Limit to 20 articles per request
            params[f'i'] = article_id
        
        result = self._make_request('stream/items/contents', params, method='POST')
        if result and 'items' in result:
            return result['items']
        return []
    
    def search_articles(self, query: str, count: int = 50) -> List[Dict]:
        """
        Search for articles containing specific terms
        Note: This uses the stream contents with a text filter approach
        """
        # Get recent articles and filter by query
        all_articles = self.get_stream_contents(count=min(count * 2, 1000))
        
        if not all_articles:
            return []
        
        # Simple text search in title and summary
        matching_articles = []
        query_lower = query.lower()
        
        for article in all_articles:
            title = article.get('title', '').lower()
            summary = article.get('summary', {}).get('content', '').lower()
            
            if query_lower in title or query_lower in summary:
                matching_articles.append(article)
                
                if len(matching_articles) >= count:
                    break
        
        return matching_articles
    
    def get_feed_articles(self, feed_url: str, count: int = 50) -> List[Dict]:
        """
        Get articles from a specific feed URL
        """
        # First, try to find the feed in subscriptions
        subscriptions = self.get_subscriptions()
        feed_stream_id = None
        
        for sub in subscriptions or []:
            if sub.get('url') == feed_url or sub.get('htmlUrl') == feed_url:
                feed_stream_id = sub.get('id')
                break
        
        if not feed_stream_id:
            logger.warning(f"Feed {feed_url} not found in subscriptions")
            return []
        
        return self.get_stream_contents(stream_id=feed_stream_id, count=count)
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """
        Get current rate limit information from response headers
        """
        # Make a simple request to get rate limit headers
        self._make_request('user-info', {'output': 'json'})
        
        rate_limit_info = {}
        last_response = self.session.cookies
        
        # Extract rate limit info from headers if available
        if hasattr(self.session, 'last_response'):
            headers = self.session.last_response.headers
            rate_limit_info = {
                'zone1_limit': headers.get('X-Reader-Zone1-Limit'),
                'zone1_usage': headers.get('X-Reader-Zone1-Usage'),
                'zone2_limit': headers.get('X-Reader-Zone2-Limit'),
                'zone2_usage': headers.get('X-Reader-Zone2-Usage'),
                'reset_after': headers.get('X-Reader-Limits-Reset-After')
            }
        
        return rate_limit_info