import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import streamlit as st

# Set page configuration
st.set_page_config(page_title="Customer Feedback Sentiment Analysis", layout="wide")

# Inject background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f9f9f9;
        font-family: 'Arial';
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("Customer Feedback Sentiment Analysis")

# Sidebar with checkboxes
st.sidebar.header("Select Features to Display")
show_sentiment = st.sidebar.checkbox("Customer Sentiment Distribution")
show_campaign = st.sidebar.checkbox("Recommended Campaign Actions")
show_table = st.sidebar.checkbox("Strategy Table")
show_wordcloud = st.sidebar.checkbox("Sentiment-Weighted Word Cloud")
show_keywords = st.sidebar.checkbox("Important Keywords (TF-IDF)")
show_clustering = st.sidebar.checkbox("Feedback Clustering (KMeans)")
show_summary = st.sidebar.checkbox("Summary Dashboard")

submit = st.sidebar.button("Submit")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None and submit:
    df = pd.read_csv(uploaded_file)

    if 'customer_feedback' not in df.columns:
        st.error("The 'customer_feedback' column is missing in the uploaded file.")
    else:
        st.subheader("Data Preview")
        st.write(df.head())

        # Sentiment Analysis
        df['SentimentScore'] = df['customer_feedback'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

        def classify_sentiment(score):
            if score > 0.2:
                return 'Positive'
            elif score < -0.2:
                return 'Negative'
            else:
                return 'Neutral'

        df['Sentiment'] = df['SentimentScore'].apply(classify_sentiment)

        def suggest_campaign(sentiment):
            if sentiment == 'Positive':
                return 'Upsell or Loyalty Reward'
            elif sentiment == 'Neutral':
                return 'Follow-up Survey'
            else:
                return 'Apology with Discount'

        df['CampaignAction'] = df['Sentiment'].apply(suggest_campaign)

        # Sentiment Distribution
        if show_sentiment:
            st.subheader("Customer Sentiment Distribution")
            fig1, ax1 = plt.subplots()
            sns.countplot(data=df, x='Sentiment', hue='Sentiment', palette='Set2', legend=False, ax=ax1)
            st.pyplot(fig1)

        # Campaign Action Count
        if show_campaign:
            st.subheader("Recommended Campaign Actions")
            fig2, ax2 = plt.subplots()
            sns.countplot(data=df, x='CampaignAction', hue='CampaignAction', palette='Set3', legend=False, ax=ax2)
            plt.xticks(rotation=15)
            st.pyplot(fig2)

        # Strategy Table
        if show_table:
            st.subheader("Strategy Table")
            cols_to_show = ['campaign_id', 'customer_feedback', 'Sentiment', 'CampaignAction'] if 'campaign_id' in df.columns else ['customer_feedback', 'Sentiment', 'CampaignAction']
            st.dataframe(df[cols_to_show])

        # WordCloud
        if show_wordcloud:
            word_freq = Counter()
            for index, row in df.iterrows():
                feedback = str(row['customer_feedback'])
                sentiment = row['Sentiment']
                words = feedback.split()
                weight = 3 if sentiment == 'Positive' else 2 if sentiment == 'Neutral' else 1
                for word in words:
                    word_freq[word.lower()] += weight

            wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(word_freq)
            st.subheader("Sentiment-Weighted Word Cloud")
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            ax3.imshow(wordcloud, interpolation='bilinear')
            ax3.axis('off')
            st.pyplot(fig3)

        # TF-IDF Keywords
        if show_keywords:
            st.subheader("Important Keywords (TF-IDF)")
            tfidf = TfidfVectorizer(max_features=15, stop_words='english')
            tfidf_matrix = tfidf.fit_transform(df['customer_feedback'].astype(str))
            tfidf_scores = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
            sorted_keywords = sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)
            st.write("Top Keywords:")
            st.write(pd.DataFrame(sorted_keywords, columns=['Keyword', 'Score']))

        # KMeans Clustering
        if show_clustering:
            st.subheader("Feedback Clustering (KMeans)")
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(tfidf_matrix)
            st.write(df[['customer_feedback', 'Cluster']].head(10))

        # Summary Dashboard
        if show_summary:
            st.subheader("Summary Dashboard")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Feedbacks", len(df))
            col2.metric("Positive", (df['Sentiment'] == 'Positive').sum())
            col3.metric("Negative", (df['Sentiment'] == 'Negative').sum())
            st.success("Use these insights to tailor customer engagement strategies.")

elif uploaded_file is None:
    st.info("Please upload a CSV file to get started.")
elif not submit:
    st.warning("Please select options and click Submit from the sidebar.")
