# Inoreader News Classifier

A comprehensive tool to automatically read, fetch, classify, and organize news articles from your Inoreader account using machine learning. This tool connects to the Inoreader API to retrieve your news feeds and uses natural language processing to automatically categorize articles into topics like technology, business, sports, politics, health, and more.

## Features

- **Inoreader Integration**: Authenticate and fetch articles from your Inoreader account
- **Machine Learning Classification**: Automatically categorize news articles using multiple ML algorithms
- **Multiple Algorithms**: Support for Random Forest, Logistic Regression, Naive Bayes, and SVM
- **Data Storage**: Local SQLite database for storing articles and classifications
- **Search & Filter**: Search articles by keywords and filter by categories
- **Training Data Management**: Use existing articles or external datasets for model training
- **Rate Limiting**: Respects Inoreader API rate limits
- **User Feedback**: Provide feedback to improve classification accuracy
- **Statistics & Analytics**: View classification statistics and model performance

## Project Structure

```
inoreader-news-classifier/
├── config.py                  # Configuration settings
├── inoreader_client.py        # Inoreader API client
├── news_classifier.py         # Machine learning classification logic
├── data_storage.py           # SQLite database management
├── news_classifier_app.py    # Main application
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── data/                    # Database storage
├── models/                  # Trained ML models
└── logs/                   # Application logs
```

## Installation

### Prerequisites

- Python 3.7 or higher
- An active Inoreader account
- Inoreader account credentials (email and password)

### Setup

1. **Clone or download the project files**
   ```bash
   # If using git
   git clone <repository-url>
   cd inoreader-news-classifier
   
   # Or download and extract the files
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   Create a `.env` file in the project root or set environment variables:
   ```bash
   export INOREADER_EMAIL="your_email@example.com"
   export INOREADER_PASSWORD="your_password"
   ```

   Or create a `.env` file:
   ```
   INOREADER_EMAIL=your_email@example.com
   INOREADER_PASSWORD=your_password
   ```

4. **Create necessary directories**
   ```bash
   mkdir -p data models logs
   ```

## Configuration

The application uses `config.py` for configuration. Key settings include:

### API Settings
- `INOREADER_EMAIL`: Your Inoreader email
- `INOREADER_PASSWORD`: Your Inoreader password
- `API_RATE_LIMIT`: API request limits (default: 100/day for free tier)

### Classification Categories
```python
NEWS_CATEGORIES = [
    'business', 'technology', 'politics', 'sports',
    'entertainment', 'science', 'health', 'world',
    'local', 'opinion'
]
```

### Model Settings
- `MAX_FEATURES`: Maximum number of features for text vectorization (default: 10000)
- `MIN_TEXT_LENGTH`: Minimum text length for classification (default: 50)

## Usage

### Command Line Interface

The main application provides several commands:

#### 1. Classify Articles (Default)
Fetch and classify new articles from your Inoreader account:

```bash
python news_classifier_app.py --action classify --max-articles 50
```

Options:
- `--max-articles`: Number of articles to process (default: 100)
- `--unread-only`: Process only unread articles (default: True)

#### 2. Train the Model
Train a new classification model:

```bash
python news_classifier_app.py --action train --algorithm random_forest
```

Options:
- `--algorithm`: Choose from `random_forest`, `logistic_regression`, `naive_bayes`, `svm`
- `--training-file`: Path to external training data (JSON format)

#### 3. View Statistics
Get comprehensive statistics about your articles and classifications:

```bash
python news_classifier_app.py --action stats
```

#### 4. Search Articles
Search for articles by keywords or category:

```bash
python news_classifier_app.py --action search --query "artificial intelligence"
python news_classifier_app.py --action search --category technology
```

### Python API Usage

You can also use the classes directly in your Python code:

```python
from news_classifier_app import NewsClassifierApp

# Initialize the application
app = NewsClassifierApp()

# Authenticate with Inoreader
if app.authenticate():
    # Fetch articles
    articles = app.fetch_articles(max_articles=50)
    
    # Classify articles
    results = app.classify_and_store_articles(articles)
    print(f"Classified {results['classified_articles']} articles")
    
    # Get statistics
    stats = app.get_statistics()
    print(f"Total articles in database: {stats['total_articles']}")
```

## Training Data Format

If you want to provide your own training data, create a JSON file with this structure:

```json
[
    {
        "article": {
            "title": "Apple Releases New iPhone",
            "summary": {"content": "Apple announced a new iPhone with advanced features..."},
            "id": "article_1"
        },
        "category": "technology"
    },
    {
        "article": {
            "title": "Stock Market Hits Record High",
            "summary": {"content": "The stock market reached new heights..."},
            "id": "article_2"
        },
        "category": "business"
    }
]
```

## API Rate Limits

The Inoreader API has rate limits:
- **Free tier**: 100 requests per day
- **Zone 1** (reading): User info, subscriptions, stream contents
- **Zone 2** (writing): Marking as read, editing subscriptions

The application automatically handles rate limiting and includes delays between requests.

## Database Schema

The application uses SQLite with the following tables:

- `articles`: Store article content and metadata
- `classifications`: Store classification results with confidence scores
- `user_feedback`: Store user corrections for model improvement
- `training_data`: Store labeled training data
- `model_metadata`: Store model training information

## Machine Learning Models

### Supported Algorithms

1. **Random Forest** (default)
   - Good for handling mixed data types
   - Provides feature importance
   - Robust against overfitting

2. **Logistic Regression**
   - Fast training and prediction
   - Good baseline performance
   - Interpretable results

3. **Naive Bayes**
   - Excellent for text classification
   - Fast and memory efficient
   - Works well with small datasets

4. **Support Vector Machine (SVM)**
   - High accuracy potential
   - Good for high-dimensional data
   - Memory efficient

### Text Processing Pipeline

1. **HTML Cleaning**: Remove HTML tags and entities
2. **Text Normalization**: Convert to lowercase, remove special characters
3. **Tokenization**: Split text into individual words
4. **Stop Word Removal**: Remove common words (the, and, etc.)
5. **Lemmatization**: Reduce words to their root form
6. **TF-IDF Vectorization**: Convert text to numerical features

## Troubleshooting

### Common Issues

#### Authentication Failed
- Verify your email and password are correct
- Check if you have a regular Inoreader account (not just OAuth login)
- Ensure you're not hitting rate limits

#### No Articles Found
- Check if you have unread articles in your Inoreader account
- Try with `--unread-only false` to include read articles
- Verify your subscriptions are active

#### Classification Accuracy Low
- Train with more diverse training data
- Try different algorithms
- Increase the number of training samples
- Provide user feedback to improve the model

#### Import Errors
```bash
# Install missing dependencies
pip install -r requirements.txt

# For NLTK data issues
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Logging

The application logs to both console and file (`logs/news_classifier.log`). Set log level in `config.py`:

```python
LOG_LEVEL = 'DEBUG'  # DEBUG, INFO, WARNING, ERROR
```

## Performance Tips

1. **Batch Processing**: Process articles in batches of 50-100 for optimal performance
2. **Model Caching**: The trained model is saved and loaded automatically
3. **Database Indexing**: The database includes indexes for better query performance
4. **Rate Limiting**: Respect API limits to avoid being blocked

## Contributing

To contribute to this project:

1. Add new classification categories in `config.py`
2. Implement additional ML algorithms in `news_classifier.py`
3. Add new features to the main application
4. Improve text processing pipeline
5. Add web interface or GUI

## Examples

### Example 1: Daily News Classification

Set up a daily cron job to classify new articles:

```bash
# Add to crontab (crontab -e)
0 9 * * * cd /path/to/project && python news_classifier_app.py --action classify --max-articles 100
```

### Example 2: Training with Custom Data

```python
# Prepare training data
training_data = [
    {
        "article": {"title": "AI breakthrough", "summary": {"content": "New AI technology..."}},
        "category": "technology"
    }
]

# Save to file
import json
with open('training_data.json', 'w') as f:
    json.dump(training_data, f)

# Train model
python news_classifier_app.py --action train --training-file training_data.json
```

### Example 3: Search and Filter

```python
app = NewsClassifierApp()
app.authenticate()

# Search for AI articles
ai_articles = app.search_articles("artificial intelligence", limit=20)

# Get all technology articles
tech_articles = app.search_articles("", category="technology", limit=50)
```

## License

This project is open source. Please check the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the logs for error messages
3. Ensure all dependencies are installed correctly
4. Verify your Inoreader credentials are valid

## Acknowledgments

- Inoreader for providing the RSS reader API
- scikit-learn for machine learning algorithms
- NLTK for natural language processing tools
- SQLite for lightweight database storage