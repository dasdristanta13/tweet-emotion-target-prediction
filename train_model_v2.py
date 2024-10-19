import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import joblib
from sklearn.metrics import classification_report
import numpy as np
import re
import wandb
from imblearn.over_sampling import RandomOverSampler
import random
from nltk.corpus import wordnet
import nltk
from warnings import filterwarnings
filterwarnings('ignore')

nltk.download('wordnet', quiet=True)
wandb.init(mode="disabled")

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

def train_model_v2(df):
    emo_dict = {'Negative emotion': 0, 'Positive emotion': 1, 'No emotion toward brand or product': 2}
    df['sentiment_label'] = df['emotion'].apply(lambda x: emo_dict[x])

    train_data, val_data = train_test_split(df, test_size=0.2, stratify=df['sentiment_label'], random_state=42)

    print_class_distribution(train_data, "Before augmentation and balancing")

    # Augment and balance only the training data
    train_data = augment_data(train_data)
    print_class_distribution(train_data, "After augmentation")

    train_data = balance_data(train_data)
    print_class_distribution(train_data, "After balancing")

    val_data['combined_input'] = val_data['new_text'] + " [SEP] " + val_data['new_target']

    train_data = train_data[['combined_input', 'sentiment_label']]
    val_data = val_data[['combined_input', 'sentiment_label']]

    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

    def preprocess_function(examples):
        tokenized_inputs = tokenizer(examples['combined_input'], padding='max_length', truncation=True, max_length=128)
        tokenized_inputs['labels'] = examples['sentiment_label']
        return tokenized_inputs

    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data[['combined_input', 'sentiment_label']])

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment", num_labels=3)

    for name, param in model.named_parameters():
        if not any(layer in name for layer in ['encoder.layer.11', 'encoder.layer.10','encoder.layer.9','encoder.layer.8', 'pooler', 'classifier']):
            param.requires_grad = False

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=1e-5,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        num_train_epochs=8,
        weight_decay=0.01,
        logging_strategy="steps",
        load_best_model_at_end=True,
        report_to="none",
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Save the model and tokenizer
    output_dir = "./emotion_model"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model and tokenizer saved to {output_dir}")

    # Generate and print classification report
    predictions = trainer.predict(val_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=list(emo_dict.keys())))

if __name__ == "__main__":
    df = pd.read_excel('NLP Engineer Assignment Dataset (1) (1) (1) (1).xlsx', sheet_name='Train')
    processed_df = preprocess_data(df)
    train_model_v2(processed_df)
