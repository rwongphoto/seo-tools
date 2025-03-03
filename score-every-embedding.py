import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

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
    
def create_navigation_menu(logo_url):
    """Creates a top navigation menu for the Streamlit app with a logo above and centered."""

    menu_options = {
        "Home": "https://theseoconsultant.ai/",
        "About": "https://theseoconsultant.ai/about/",
        "Services": "https://theseoconsultant.ai/seo-consulting/",  # Flattened for top nav
        "Blog": "https://theseoconsultant.ai/blog/",
        "Contact": "https://theseoconsultant.ai/contact/"
    }

    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="{logo_url}" width="350">
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
        .topnav {
          overflow: hidden;
          background-color: #f1f1f1; /* Adjust color as needed */
          display: flex;              /* Use flexbox */
          justify-content: center;    /* Horizontally center items */
          margin-bottom: 35px;        /* Add space below the menu */
        }

        .topnav a {
          float: left;
          display: block;
          color: black; /* Adjust color as needed */
          text-align: center;
          padding: 14px 16px;
          text-decoration: none;
        }

        .topnav a:hover {
          background-color: #ddd; /* Adjust color as needed */
          color: black; /* Adjust color as needed */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create the top navigation menu
    menu_html = "<div class='topnav'>"
    for key, value in menu_options.items():
        menu_html += f"<a href='{value}' target='_blank'>{key}</a>"
    menu_html += "</div>"

    st.markdown(menu_html, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Cosine Similarity Score - Every Embedding | The SEO Consultant.ai") # set page title here
    logo_url = "https://theseoconsultant.ai/wp-content/uploads/2024/12/cropped-theseoconsultant-logo-2.jpg"
    create_navigation_menu(logo_url)
    st.title("Cosine Similarity Score - Every Embedding")
    
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

    st.markdown("---") # Add a horizontal line for visual separation
    st.markdown(
        "Powered by [The SEO Consultant.ai](https://theseoconsultant.ai)",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()



