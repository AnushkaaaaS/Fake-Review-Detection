import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import requests
from bs4 import BeautifulSoup as bs
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from train_model import predict_reviews, get_recommendation, scrape_amazon_reviews,analyze_sentiment



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






st.title('Fake Reviews Detection')

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Check Review", "Product Analysis"])

if page == "Check Review":
    st.write("Welcome to the Amazon Review Fake Detection app! This tool helps you identify fake reviews on Amazon products using advanced machine learning techniques. Simply enter a product URL, and the app will classify reviews as genuine or fake, providing insights into their authenticity and overall sentiment. Enhance your shopping experience with more reliable information!")
    review_input = st.text_area("Enter a review to check if it is fake or genuine:")
    if st.button("Check Review"):
        if review_input:
            review_transformed = tfidf.transform([review_input])
            prediction = classifier.predict(review_transformed)
            label = le.inverse_transform(prediction)[0]
            if(label=='CG'):
             st.write("The review is predicted to be original")
            else:
                st.write("The review is predicted to be fake")
        else:
            st.warning("Please enter a review to check.")




elif page == "Product Analysis":
    url = st.text_input('Enter the Amazon product URL:')
    if url:
        with st.spinner('Scraping reviews...'):
            reviews_df, name, image_url = predict_reviews(url)

        if not reviews_df.empty:
            st.subheader("Product details:")
            col1, col2 = st.columns([7, 8])  

            with col1:
                st.image(image_url, width=300)  

            with col2:
                st.markdown(f"""
                    <h1 style='text-align: left; font-size: 30px; margin-left: 20px;'>{name}</h1>
                """, unsafe_allow_html=True)

            st.subheader('Reviews Analysis')
            label_counts = reviews_df['prediction'].value_counts()
            labels = label_counts.index
            sizes = label_counts.values
            colors = ['lightcoral', 'lightskyblue'][:len(labels)]
            explode = [0.1] * len(labels)

            fig, ax = plt.subplots(figsize=(10, 6))  
            ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
            ax.set_title('Percentage of Fake and Real Reviews')
            ax.axis('equal')

            st.pyplot(fig)

            st.subheader('Sentiment Analysis')
            sentiment_counts = reviews_df['sentiment'].value_counts()
            sentiment_labels = sentiment_counts.index
            sentiment_sizes = sentiment_counts.values
            sentiment_colors = ['lightgreen', 'lightcoral', 'lightskyblue'][:len(sentiment_labels)]
            sentiment_explode = [0.1] * len(sentiment_labels)

            fig, ax = plt.subplots(figsize=(10, 6))  
            ax.pie(sentiment_sizes, explode=sentiment_explode, labels=sentiment_labels, colors=sentiment_colors, autopct='%1.1f%%', shadow=True, startangle=140)
            ax.set_title('Sentiment Distribution of Reviews')
            ax.axis('equal')

            st.pyplot(fig)

            st.write(reviews_df)
            recommendation = get_recommendation(reviews_df)
            st.subheader(recommendation)
        else:
            st.info('No reviews were retrieved. Please check the URL or try a different product.')
