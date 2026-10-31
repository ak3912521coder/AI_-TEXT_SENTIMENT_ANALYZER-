import streamlit as st
import tensorflow as tf
import pandas as pd
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("sentiment_model_imdb.keras")

    word_index = imdb.get_word_index()

    word_index = {k: (v + 3) for k, v in word_index.items()}

    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3

    return model, word_index


st.title("💬 AI Sentiment Analyzer")
st.subheader("IMDB Movie Review Analysis")

st.write(
    "Enter your movie review below "
    "and let AI predict its sentiment."
)

try:
    model, word_index = load_resources()

    review = st.text_area(
        "Enter your review",
        placeholder="This movie was amazing and I loved it!",
        height=150
    )

    if st.button("Analyze Sentiment", use_container_width=True):

        if review.strip():

            words = review.lower().split()
            sequence = []

            for word in words:
                index = word_index.get(word, 2)

                if index < 10000:
                    sequence.append(index)
                else:
                    sequence.append(2)

            padded = pad_sequences(
                [sequence],
                maxlen=100,
                padding="post",
                truncating="post"
            )

            prediction = float(
                model.predict(padded, verbose=0)[0][0]
            )

            if prediction >= 0.5:
                sentiment = "Positive"
                confidence = prediction * 100
                st.success("😊 Sentiment: Positive")
            else:
                sentiment = "Negative"
                confidence = (1 - prediction) * 100
                st.error("😞 Sentiment: Negative")

            st.metric(
                label="Prediction Confidence",
                value=f"{confidence:.2f}%"
            )

            st.subheader("📊 Sentiment Confidence Graph")

            graph_data = pd.DataFrame(
                {
                    "Percentage": [
                        confidence,
                        100 - confidence
                    ]
                },
                index=[
                    "Prediction Confidence",
                    "Remaining Probability"
                ]
            )

            st.bar_chart(graph_data)

            st.subheader("📝 Review Summary")

            st.write(f"**Sentiment:** {sentiment}")
            st.write(f"**Model Score:** {prediction:.4f}")

        else:
            st.warning("Please enter a movie review first.")

except Exception as e:
    st.error(f"Error loading model or files: {e}")
    st.info(
        "Check that sentiment_model_imdb.keras "
        "is present in your project folder."
    )