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
from utils import *

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
