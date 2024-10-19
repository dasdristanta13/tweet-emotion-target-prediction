
import os
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Load the model and tokenizer
model_path = "./emotion_model"
hub_path = "dasdristanta13/twitter-emotion-model"
if os.path.isdir(model_path):
    emotion_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    emotion_tokenizer = AutoTokenizer.from_pretrained(model_path)
else:
    emotion_model = AutoModelForSequenceClassification.from_pretrained(hub_path)
    emotion_tokenizer = AutoTokenizer.from_pretrained(hub_path)

# Move the model to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
emotion_model = emotion_model.to(device)

# Load a pre-trained sentence transformer model for semantic similarity
semantic_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
semantic_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
semantic_model = semantic_model.to(device)

target_mapping = {
    'Google': 'Google',
    'Apple': 'Apple',
    'iPad': 'Apple',
    'iPhone': 'Apple',
    'Other Google product or service': 'Google',
    'Other Apple product or service': 'Apple',
    'Android': 'Google',
    'Android App': 'Google',
    'iPad or iPhone App': 'Apple',
}

def get_embedding(text):
    inputs = semantic_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = semantic_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def predict_target(text):
    text_embedding = get_embedding(text)
    target_embeddings = {target: get_embedding(target) for target in target_mapping.keys()}

    similarities = {target: cosine_similarity(text_embedding, emb)[0][0] for target, emb in target_embeddings.items()}
    predicted_target = max(similarities, key=similarities.get)

    return predicted_target

def predict_emotion(text, target):
    combined_input = f"{text} [SEP] {target}"
    inputs = emotion_tokenizer(combined_input, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = emotion_model(**inputs)

    probabilities = outputs.logits.softmax(dim=-1).squeeze().cpu().numpy()

    emotion_labels = ['Negative emotion', 'Positive emotion','No emotion']
    predicted_emotion = emotion_labels[np.argmax(probabilities)]

    return predicted_emotion, {label: float(prob) for label, prob in zip(emotion_labels, probabilities)}

def process_test_data(test_df):
    results = []
    for _, row in test_df.iterrows():
        text = row['Tweet']
        predicted_target = predict_target(text)
        predicted_emotion, emotion_probs = predict_emotion(text, predicted_target)

        results.append({
            'Tweet': text,
            'Predicted Target': predicted_target,
            'Predicted Emotion': predicted_emotion,
            'Emotion Probabilities': emotion_probs
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    test_df = pd.read_excel('NLP Engineer Assignment Dataset (1) (1) (1) (1).xlsx', sheet_name='Test')
    results_df = process_test_data(test_df)
    results_df.to_csv('test_results.csv', index=False)
    print("Results saved to test_results.csv")
