import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

def extract_text_features(df: pd.DataFrame, vectorizer=None, train_mode: bool = True) -> tuple:
    df = df.copy()
    
    # Handle missing descriptions
    descriptions = df['description'].fillna('').astype(str).str.lower().str.strip()
    
    if train_mode:
        vectorizer = TfidfVectorizer(
            max_features=300,
            ngram_range=(1, 2),
            min_df=5,
            stop_words='english' # can remove stop words to help clean
        )
        tfidf_matrix = vectorizer.fit_transform(descriptions)
    else:
        tfidf_matrix = vectorizer.transform(descriptions)
        
    # Convert TF-IDF sparse matrix to DataFrame features
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
        index=df.index
    )
    
    # Concatenate with df
    df = pd.concat([df, tfidf_df], axis=1)
    
    # Compute basic text stats
    df['desc_length'] = df['description'].fillna('').apply(len)
    df['desc_word_count'] = df['description'].fillna('').apply(lambda x: len(x.split()))
    
    return df, vectorizer
