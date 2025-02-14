import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

def create_navigation_menu(logo_url:
    """Creates a navigation menu for the Streamlit app with a logo."""

    menu_options = {
        "Home": "https://theseoconsultant.ai/",
        "About": "https://theseoconsultant.ai/about/",
        "Services": https://theseoconsultant.ai/seo-services/,
        "Blog": "https://theseoconsultant.ai/blog/",
        "Contact": "https://theseoconsultant.ai/contact/"
    }

    st.sidebar.image(logo_url, width=200)  # Adjust width as needed
    st.sidebar.header("Navigation")

    for key, value in menu_options.items():
        if isinstance(value, str):  # Simple link
            if st.sidebar.button(key):
                st.components.v1.iframe(value, height=200)
        else:  # Submenu
            with st.sidebar.expander(key):
                for sub_key, sub_value in value.items():
                    if st.button(sub_key):
                        st.components.v1.iframe(sub_value, height=200)

@st.cache_resource
def load_model():
    """Loads the BERT model and tokenizer."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    return tokenizer, model

def get_embedding(text, model, tokenizer):
    """
    Generates a BERT embedding for the given text.

    Args:
        text: The input text string.
        model: The pre-trained BERT model.
        tokenizer: The BERT tokenizer.

    Returns:
        A NumPy array representing the BERT embedding.
    """
    tokenizer.pad_token = tokenizer.unk_token  # Set pad token to be the same as unknown token for batched inference.
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512) # added padding and truncation
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def calculate_similarity(text, search_term, tokenizer, model):
    """Calculates similarity scores for each sentence in the text against the search term."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    sentence_embeddings = [get_embedding(sentence, model, tokenizer) for sentence in sentences]
    search_term_embedding = get_embedding(search_term, model, tokenizer)

    similarities = []
    for sentence_embedding in sentence_embeddings:
        similarity = cosine_similarity(sentence_embedding, search_term_embedding)[0][0]
        similarities.append(similarity)

    return sentences, similarities

def main():
    st.title("Cosine Similarity Score - Every Embedding")
    st.markdown("By: [The SEO Consultant.ai](https://theseoconsultant.ai)") # Credit and link

    # Input text area
    text = st.text_area("Enter Text:",
                         """""")

    # Search term input
    search_term = st.text_input("Enter Search Term:", "")

    if st.button("Calculate Similarity"):
        # Load the model
        tokenizer, model = load_model()

        with st.spinner("Calculating Similarities..."):
            # Calculate similarities
            sentences, similarities = calculate_similarity(text, search_term, tokenizer, model)

        st.subheader("Similarity Scores:")
        for i, (sentence, score) in enumerate(zip(sentences, similarities), 1):
            st.write(f"{i}. {sentence} (Similarity: {score:.4f})")

if __name__ == "__main__":
    main()
