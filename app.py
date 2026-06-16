import gradio as gr
import pickle

# Load detector
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def detect_fake_news(news_text):
    news_vec = vectorizer.transform([news_text])

    prediction = model.predict(news_vec)[0]
    confidence = max(model.predict_proba(news_vec)[0]) * 100

    if prediction == 0:
        return f"FAKE NEWS ({confidence:.2f}%)"

    return f"REAL NEWS ({confidence:.2f}%)"

# Gradio Interface (Single App)
app = gr.Interface(
    fn=detect_fake_news,
    inputs=gr.Textbox(lines=8, label="Enter News"),
    outputs="text",
    title="Fake News Detector",
    description="Paste a news article and the model will classify it as REAL or FAKE."
)

app.launch(server_name="0.0.0.0")
