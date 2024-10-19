import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import joblib
from sklearn.metrics import classification_report
import numpy as np
import re
from imblearn.over_sampling import RandomOverSampler
import random
from nltk.corpus import wordnet
import nltk
from warnings import filterwarnings
filterwarnings('ignore')

nltk.download('wordnet', quiet=True)

def clean_text(text):
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'{links}', '', text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(df):
    df.columns = ['text', 'target', 'emotion']
    df = df[df['emotion'] != 'I can\'t tell']
    df = df[~df['text'].isna()].reset_index(drop=True)

    target_mapping = {
        'Google': 'Google',
        'Apple': 'Apple',
        'iPad': 'iPad',
        'iPhone': 'iPhone',
        'Other Google product or service': 'Google',
        'Other Apple product or service': 'Apple',
        'Android': 'Android',
        'Android App': 'Android App',
        'iPad or iPhone App': 'iPad or iPhone App',
    }

    df['new_text'] = df['text'].apply(clean_text)
    df['new_target'] = df['target'].apply(lambda x: target_mapping.get(x, 'No Product'))
    df['target'] = df['target'].fillna('No Product')

    return df

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': (predictions == labels).mean(),
    }

def random_word_drop(text, max_words=3):
    words = text.split()
    if len(words) <= max_words:
        return text
    num_to_drop = random.randint(1, min(max_words, len(words) - 1))
    drop_indices = random.sample(range(len(words)), num_to_drop)
    return ' '.join([word for i, word in enumerate(words) if i not in drop_indices])

def synonym_replacement(text, n=1):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word.isalnum()]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = []
        for syn in wordnet.synsets(random_word):
            for l in syn.lemmas():
                synonyms.append(l.name())
        if len(synonyms) >= 1:
            synonym = random.choice(list(set(synonyms)))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)

def print_class_distribution(df, stage):
    class_dist = df['sentiment_label'].value_counts().sort_index()
    total = len(df)
    print(f"\nClass distribution - {stage}:")
    for label, count in class_dist.items():
        percentage = (count / total) * 100
        print(f"Class {label}: {count} ({percentage:.2f}%)")

def augment_data(df):
    class_counts = df['sentiment_label'].value_counts()
    max_class = class_counts.idxmax()
    min_class = class_counts.idxmin()

    augmented_data = []
    for _, row in df.iterrows():
        # Original data
        augmented_data.append({
            'new_text': row['new_text'],
            'new_target': row['new_target'],
            'sentiment_label': row['sentiment_label']
        })

        # Augment less for the highest class
        if row['sentiment_label'] == max_class:
            if random.random() < 0.4:  # 50% chance to augment
                new_text = random_word_drop(row['new_text'])
                augmented_data.append({
                    'new_text': new_text,
                    'new_target': row['new_target'],
                    'sentiment_label': row['sentiment_label']
                })
        else:
            # Random word drop
            new_text = random_word_drop(row['new_text'])
            augmented_data.append({
                'new_text': new_text,
                'new_target': row['new_target'],
                'sentiment_label': row['sentiment_label']
            })

            # Synonym replacement
            new_text = synonym_replacement(row['new_text'])
            augmented_data.append({
                'new_text': new_text,
                'new_target': row['new_target'],
                'sentiment_label': row['sentiment_label']
            })

        # Extra augmentation for the lowest class
        if row['sentiment_label'] == min_class:
            new_text = random_word_drop(synonym_replacement(row['new_text']))
            augmented_data.append({
                'new_text': new_text,
                'new_target': row['new_target'],
                'sentiment_label': row['sentiment_label']
            })

    augmented_df = pd.DataFrame(augmented_data)
    augmented_df['combined_input'] = augmented_df['new_text'] + " [SEP] " + augmented_df['new_target']
    return augmented_df

def balance_data(df):
    X = df[['combined_input']]
    y = df['sentiment_label']

    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    balanced_df = pd.DataFrame({
        'combined_input': X_resampled['combined_input'],
        'sentiment_label': y_resampled
    })

    return balanced_df