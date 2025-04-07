import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig
import streamlit as st
import numpy as np

# Label mapping dan emoji
label_dict = {
    0: ("Negatif", "😠"),
    1: ("Netral", "😐"),
    2: ("Positif", "😊")
}

st.set_page_config(page_title="Sentimen Analisis", page_icon="💬", layout="centered")

@st.cache_resource
def load_tokenizer_model():
    tokenizer = DistilBertTokenizer.from_pretrained("model/distilbert/tokenizer")

    config = DistilBertConfig(
        num_labels=3,  # sesuaikan dengan saat training
        vocab_size=30522,  # default distilBERT
        hidden_size=768,
        n_heads=12,
        n_layers=6
    )

    model = DistilBertForSequenceClassification(config)
    model.load_state_dict(torch.load("model/distilbert/distilbert_sentiment_state_dict.pt", map_location=torch.device("cpu")))
    model.eval()
    return tokenizer, model
    
tokenizer, model = load_tokenizer_model()

# Header
st.markdown("""
    <h1 style='text-align: center;'>💬 Analisis Sentimen Komentar</h1>
    <p style='text-align: center;'>Masukkan komentar atau review. Model akan memprediksi apakah sentimennya positif, netral, atau negatif.</p>
    <p style='text-align: center; color: red; font-size: 15px;'>⚠️ Saat ini hanya mendukung input berbahasa <b>Inggris</b>.</p>
    <hr style='margin-top: 10px; margin-bottom: 20px;'>
""", unsafe_allow_html=True)

# Input
user_input = st.text_area("✍️ Tulis komentarmu di sini:", height=150, placeholder="Contoh: This app is really helpful for learning...")

if st.button("🔍 Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")
    else:
        with st.spinner("⏳ Menganalisis..."):
            inputs = tokenizer.encode_plus(
                user_input,
                add_special_tokens=True,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding="max_length"
            )

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()

        # Hasil
        label, emoji = label_dict[pred]
        st.markdown(f"<h2 style='text-align: center;'>Hasil: <span style='color:#4CAF50'>{label} {emoji}</span></h2>", unsafe_allow_html=True)

        st.subheader("📊 Probabilitas Sentimen:")
        col1, col2, col3 = st.columns(3)
        for i, col in zip(range(3), [col1, col2, col3]):
            percent = probs[0][i].item() * 100
            sent_label, emo = label_dict[i]
            col.metric(f"{sent_label} {emo}", f"{percent:.2f} %")

        # Optional bar chart
        st.bar_chart({
            "Sentimen": {
                "Negatif": probs[0][0].item(),
                "Netral": probs[0][1].item(),
                "Positif": probs[0][2].item()
            }
        })

# Footer
st.markdown("""
<hr style='margin-top: 30px;'>
<p style='text-align: center; font-size: 13px;'>🚀 Dibuat oleh Maza menggunakan DistilBERT & Streamlit</p>
""", unsafe_allow_html=True)
