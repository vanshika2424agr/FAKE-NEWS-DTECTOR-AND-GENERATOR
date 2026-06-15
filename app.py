import gradio as gr
import pickle
from transformers import pipeline

# Load detector
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load generator
generator = pipeline(
    "text-generation",
    model="gpt2"
)

def detect_fake_news(news_text):
    news_vec = vectorizer.transform([news_text])

    prediction = model.predict(news_vec)[0]
    confidence = max(model.predict_proba(news_vec)[0]) * 100

    if prediction == 0:
        return f"FAKE NEWS ({confidence:.2f}%)"

    return f"REAL NEWS ({confidence:.2f}%)"

def generate_news(topic):

    prompt = f"""
FICTIONAL NEWS ARTICLE FOR RESEARCH PURPOSES ONLY

Topic: {topic}

Article:
"""

    result = generator(
        prompt,
        max_length=200,
        do_sample=True,
        temperature=0.8
    )

    return result[0]["generated_text"]

detector_tab = gr.Interface(
    fn=detect_fake_news,
    inputs=gr.Textbox(lines=8, label="Enter News"),
    outputs="text",
    title="Fake News Detector"
)

generator_tab = gr.Interface(
    fn=generate_news,
    inputs=gr.Textbox(label="Enter Topic"),
    outputs="text",
    title="Fictional News Generator"
)

app = gr.TabbedInterface(
    [detector_tab, generator_tab],
    ["Detector", "Generator"]
)

app.launch(server_name="0.0.0.0")
