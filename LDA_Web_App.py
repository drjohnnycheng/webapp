# LDA Web App #

# Dependencies
import streamlit as st
from streamlit import components
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
from sklearn.datasets import fetch_20newsgroups
# from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.gensim_models


# Load the 20 Newsgroups dataset
#@st.cache
@st.cache_data
def load_data():
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    return newsgroups.data, newsgroups.target

# Preprocess text
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(docs):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    cleaned_texts = []
    for doc in docs:
        # Remove punctuation and numbers
        doc = re.sub(r'[^a-zA-Z\s]', '', doc)
        # Lowercase
        doc = doc.lower()
        # Tokenization, remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(word) for word in doc.split() if word not in stop_words and len(word) > 2]
        cleaned_texts.append(' '.join(tokens))

    return cleaned_texts

# Prepare data for LDA
def prepare_data(docs):
    # Tokenize and preprocess
    texts = [[word for word in doc.lower().split() if len(word) > 3] for doc in docs]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return dictionary, corpus

# Train LDA model
def train_lda(corpus, dictionary, num_topics):
    # lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    # lda_model.fit(corpus)
    lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15, random_state=42)
    return lda_model


# Streamlit app
def main():
    # st.set_option('client.showErrorDetails', False)

    st.title("Topic Modeling with LDA on 20 Newsgroups")
    
    # Load data
    docs, target = load_data()
    
    st.sidebar.header("User Input")
    num_topics = st.sidebar.slider("Number of Topics", 2, 20, 5)

    if st.sidebar.button("Run LDA"):
        # Preprocess text
        st.text("Preprocessing text ...")
        clean_docs = preprocess_text(docs)

        # Prepare data
        st.text("Preparing data ...")
        dictionary, corpus = prepare_data(clean_docs)
      
        # Train LDA model
        st.text("Training LDA model ...")
        lda_model = train_lda(corpus, dictionary, num_topics)

        # Compute coherence score
        st.text("Computing coherence score ...")
        clean_docs = [doc.split() for doc in clean_docs]
        coherence_model = CoherenceModel(model=lda_model, texts=clean_docs, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        st.text("Coherence Score: " + str(coherence_score))

        # Visualize with pyLDAvis
        st.text("Visualizing trained LDA model ...")
        vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
        # pyLDAvis.save_html(vis, 'lda.html')

        # Display visualization
        st.text("Displaying visualization ...")
        # st.write(pyLDAvis.prepared_data_to_html(vis), unsafe_allow_html=True)
        html_string = pyLDAvis.prepared_data_to_html(vis)
        components.v1.html(html_string, width=1200, height=800)

if __name__ == "__main__":
    main()
