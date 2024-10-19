import gradio as gr
from inference import predict_target, predict_emotion
import pandas as pd
import re
from PIL import Image

def predict(text):
    predicted_target = predict_target(text)
    predicted_emotion, emotion_probs = predict_emotion(text, predicted_target)
    
    summary_df = pd.DataFrame({
        "Aspect": ["Target", "Emotion"],
        "Prediction": [predicted_target, predicted_emotion]
    })
    
    return predicted_target, predicted_emotion, emotion_probs, summary_df

def analyze_text(text):
    word_count = len(text.split())
    char_count = len(text)
    
    hashtags = re.findall(r'#\w+', text)
    mentions = re.findall(r'@\w+', text)
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    emojis = re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', text)
    
    analysis = f"""Word count: {word_count}
Character count: {char_count}
Hashtags: {len(hashtags)} {', '.join(hashtags)}
Mentions: {len(mentions)} {', '.join(mentions)}
URLs: {len(urls)}
Emojis: {len(emojis)} {''.join(emojis)}"""
    
    return analysis

def load_readme():
    with open("README.md", "r") as file:
        return file.read()

logo = Image.open("freepik__pixel-art-8bits-create-an-icon-for-tweet-sentiment__72933.jpeg")
logo.thumbnail((100, 100))

with gr.Blocks(title="Tweet Analysis Dashboard") as iface:
    page = gr.State("inference")
    
    with gr.Row():
        gr.Markdown("# Tweet Analysis Dashboard")
        gr.Image(logo, scale=1, min_width=100)
    
    with gr.Row():
        inference_btn = gr.Button("Inference")
        readme_btn = gr.Button("README")
    
    with gr.Column() as inference_page:
        gr.Markdown("## Tweet Emotion and Target Prediction")
        gr.Markdown("Enter a tweet to predict its target and emotion, and get additional text analysis.")
        
        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(label="Tweet Text", lines=5)
                submit_btn = gr.Button("Analyze")
            with gr.Column(scale=1):
                text_analysis = gr.Textbox(label="Text Analysis", interactive=False)
        
        with gr.Row():
            target_output = gr.Textbox(label="Predicted Target")
            emotion_output = gr.Textbox(label="Predicted Emotion")
        
        emotion_probs_output = gr.Label(label="Emotion Probabilities")
        summary_output = gr.Dataframe(label="Prediction Summary", headers=["Aspect", "Prediction"])
    
    with gr.Column(visible=False) as readme_page:
        readme_content = gr.Markdown(load_readme())
    
    def show_inference():
        return {
            inference_page: gr.update(visible=True),
            readme_page: gr.update(visible=False),
            page: "inference"
        }
    
    def show_readme():
        return {
            inference_page: gr.update(visible=False),
            readme_page: gr.update(visible=True),
            page: "readme"
        }
    
    inference_btn.click(show_inference, outputs=[inference_page, readme_page, page])
    readme_btn.click(show_readme, outputs=[inference_page, readme_page, page])
    
    submit_btn.click(
        fn=predict,
        inputs=input_text,
        outputs=[target_output, emotion_output, emotion_probs_output, summary_output]
    )
    
    submit_btn.click(
        fn=analyze_text,
        inputs=input_text,
        outputs=text_analysis
    )

if __name__ == "__main__":
    iface.launch(share=True,debug=True)