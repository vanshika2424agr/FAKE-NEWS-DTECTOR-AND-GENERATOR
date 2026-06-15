import gradio as gr
import pickle

model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def detect_fake_news(news_text):
    news_vec = vectorizer.transform([news_text])
    prediction = model.predict(news_vec)[0]

    if prediction == 0:
        return "FAKE NEWS"
    return "REAL NEWS"

demo = gr.Interface(
    fn=detect_fake_news,
    inputs=gr.Textbox(lines=8),
    outputs="text",
    title="Fake News Detector"
)

demo.launch(server_name="0.0.0.0", server_port=10000)
