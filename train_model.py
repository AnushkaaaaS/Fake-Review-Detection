import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import requests
from bs4 import BeautifulSoup as bs
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def scrape_amazon_reviews(url, maxpages=20):
    try:
        reviews = []
        pagenumber = 1
        while pagenumber <= maxpages:
            page = requests.get(f"{url}&pageNumber={pagenumber}")
            soup = bs(page.content, 'html.parser')
            review = soup.find_all('span', {'data-hook': 'review-body'})
            for i in range(0, len(review)):
                reviews.append(review[i].get_text())
            pagenumber += 1
        return reviews
    except Exception as e:
        print(f"Error scraping reviews: {e}")
        return None

dataset = pd.read_csv('./data/fake_reviews_dataset.csv')
X = dataset['review'].values
y = dataset['label'].values
le = LabelEncoder()
y = le.fit_transform(y)
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = tfidf.fit_transform(X)
classifier = LogisticRegression()
classifier.fit(X_tfidf, y)

analyzer = SentimentIntensityAnalyzer()

def predict_reviews(url):
    reviews = scrape_amazon_reviews(url)
    if reviews:
        new_reviews_df = pd.DataFrame({'reviews': reviews})
        
        # Sentiment analysis
        new_reviews_df['sentiment'] = new_reviews_df['reviews'].apply(lambda review: analyze_sentiment(review))
        
        new_reviews = new_reviews_df['reviews'].values
        X_new_tfidf = tfidf.transform(new_reviews)
        predictions = classifier.predict(X_new_tfidf)
        predicted_labels = le.inverse_transform(predictions)
        new_reviews_df['prediction'] = predicted_labels

        # Scrape product name and image URL
        page = requests.get(url)
        soup = bs(page.content, 'html.parser')
        name = soup.find('span', {'id': 'productTitle'}).get_text().strip()
        image_url = soup.find('img', {'id': 'landingImage'})['src']

        return new_reviews_df, name, image_url
    else:
        return pd.DataFrame(), '', ''

def analyze_sentiment(text):
    sentiment_score = analyzer.polarity_scores(text)
    compound_score = sentiment_score['compound']
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def get_recommendation(reviews_df):
    positive_reviews = reviews_df[reviews_df['sentiment'] == 'Positive']
    negative_reviews = reviews_df[reviews_df['sentiment'] == 'Negative']
    fake_reviews_percentage = (reviews_df['prediction'] == 'Fake').mean() * 100

    if len(positive_reviews) > len(negative_reviews) and fake_reviews_percentage < 20:
        return "Recommendation: Good product. Go for it!"
    else:
        return "Recommendation: Reviews don't seem that great. Be cautious!"
