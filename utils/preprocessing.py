import re
import emoji
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download(['punkt', 'stopwords', 'wordnet'])

def clean_tweet(tweet):
    """Full text cleaning pipeline"""
    tweet = str(tweet).lower()
    tweet = re.sub(r'http\S+', '', tweet)  # Remove URLs
    tweet = re.sub(r'@\S+', '', tweet)     # Remove mentions
    tweet = re.sub(r'#(\S+)', r'\1', tweet) # Keep hashtag text
    tweet = emoji.demojize(tweet, delimiters=(" ", " "))
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(tweet)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return ' '.join(lemmatized_words)