import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from utils import *


# Basic Overview
def basic_overview(df):
    print("Dataset Shape:", df.shape)
    print("\nColumn Data Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())

# Text Length Analysis
def text_length_analysis(df, text_col):
    df['text_length'] = df[text_col].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(10,6))
    sns.histplot(df['text_length'], bins=30, kde=True)
    plt.title('Text Length Distribution')
    plt.xticks(rotation=90)
    plt.show()

# Target Distribution
def target_distribution(df, target_col):
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x=target_col)
    plt.title('Target Variable Distribution')
    plt.xticks(rotation=90)
    plt.show()
    print("\nTarget Value Counts:\n", df[target_col].value_counts())

# Word Frequency Analysis
def word_frequency(df, text_col):
    all_words = ' '.join(df[text_col].dropna())
    word_counts = Counter(all_words.split())
    most_common_words = word_counts.most_common(20)
    print("\nMost Common Words:\n", most_common_words)

    wordcloud = WordCloud(width=800, height=400, max_words=100).generate(all_words)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

# Running EDA
if __name__=="__main__":
    df = pd.read_excel(r'C:\Users\ASUS\Documents\Projects\wysa\main-2\NLP Engineer Assignment Dataset (1) (1) (1) (1).xlsx',sheet_name="Train")
    df = preprocess_data(df)
    basic_overview(df)
    text_length_analysis(df, 'text')  
    target_distribution(df, 'target')  
    word_frequency(df, 'text')  
