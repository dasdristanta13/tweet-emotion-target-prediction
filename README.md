---
title: Twitter_Emotion_and_Target_Prediction
app_file: run_gradio_v3.py
sdk: gradio
sdk_version: 5.1.0
---

Hosted_link: [Twitter_Emotion_and_Target_Prediction](https://huggingface.co/spaces/dasdristanta13/Twitter_Emotion_and_Target_Prediction)
# Tweet Emotion and Target Prediction

This project implements a machine learning pipeline for predicting the emotion and target of tweets. It includes model training, data preprocessing, data augmentation, inference, and a Gradio-based web interface for easy interaction.

## Project Structure

- `train_model_v2.py`: Trains the emotion classification model using a fine-tuned RoBERTa model.
- `inference.py`: Implements the prediction pipeline using the trained models.
- `run_gradio_v3.py`: Creates a Gradio web interface for interactive predictions.

## Setup and Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/dasdristanta13/tweet-emotion-target-prediction.git
   cd tweet-emotion-target-prediction
   ```

2. Install the required packages:
   ```bash
   pip install pandas numpy scikit-learn transformers datasets torch gradio joblib imbalanced-learn xgboost
   ```

3. Download the dataset file `NLP Engineer Assignment Dataset (1) (1) (1) (1).xlsx` and place it in the project root directory.

## Training the Models

### Emotion Classification Model

Run the following command to train the emotion classification model:

```bash
python train_model_v2.py
```

This script will:
- **Preprocess the data**: Apply basic cleaning techniques and tokenization.
- **Data augmentation**: Use oversampling techniques to handle class imbalance, ensuring the model learns well from underrepresented emotions.
- **Fine-tune a RoBERTa model**: Use the `cardiffnlp/twitter-roberta-base-sentiment` for transfer learning, fine-tuning it on the tweet emotion dataset.
- **Save artifacts**: The fine-tuned RoBERTa model and tokenizer will be saved for inference.

## Data Augmentation and Handling Imbalance

- **Random Word Drop**: A function that removes a random subset of words from the input text.
   - This operation is probabilistically applied to reduce the highest class' dominance and augment the lower classes.

- **Synonym Replacement**: A function leveraging the WordNet corpus to replace random words with their synonyms, generating alternative versions of the input text.
   - Synonym replacement is more heavily applied to the minority classes to balance the dataset.

- **Augmentation Strategy**: 
   - The largest class undergoes minimal augmentation, while the smaller classes receive extra augmentation (both word drop and synonym replacement). The smallest class gets further augmentation by combining both techniques (word drop after synonym replacement).

## Running Inference

To process the test data and generate predictions, run:

```bash
python inference.py
```

This script will:
- **Load the trained models**: Load both the target classification and emotion classification models.
- **Process the test data**: The test dataset is preprocessed similarly to the training dataset.
- **Generate predictions**: Predictions for both target and emotion are produced for each tweet.
- **Save the results**: The predictions are saved to `test_results.csv` for analysis.

## Launching the Gradio Interface

To launch the Gradio web interface for interactive predictions, run:

```bash
python run_gradio_v2.py
```

This will start a local server and provide a URL to access the web interface.

## Model Details

### Emotion Classification Model
- **Model Architecture**: Fine-tuned `RoBERTa` model ("cardiffnlp/twitter-roberta-base-sentiment").
- **Data Augmentation**: Uses SMOTE and other oversampling methods to handle imbalanced class distribution.
- **Transfer Learning**: Leverages pre-trained `RoBERTa` for sentiment analysis, fine-tuning it on emotion-labeled tweet data.

## Gradio Interface

The Gradio interface provides:
- **Input field** for tweet text.
- **Text analysis**: Displays word count, character count, hashtags, mentions, URLs, emojis in the tweet.
- **Predicted target and emotion**: Real-time display of predictions based on user input.
- **Emotion probabilities**: Displays the probability distribution of the predicted emotions.
- **Summary table of predictions**: A table summarizing the tweet text, predicted target, emotion, and associated probabilities.
