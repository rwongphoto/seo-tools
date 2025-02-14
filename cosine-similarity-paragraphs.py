import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

@st.cache_resource
def load_models():
    """Loads the BERT tokenizer and model only once."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()  # Set the model to evaluation mode
    return tokenizer, model

def get_embedding(text, model, tokenizer):
    """Generates a BERT embedding for the given text."""
    tokenizer.pad_token = tokenizer.unk_token
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def rank_sentences_by_similarity(text, search_term, tokenizer, model):
    """
    Calculates cosine similarity between sentences and a search term using BERT.

    Returns:
        A list of tuples: (sentence, similarity score)
    """

    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    sentence_embeddings = [get_embedding(sentence, model, tokenizer) for sentence in sentences]
    search_term_embedding = get_embedding(search_term, model, tokenizer)

    similarities = [cosine_similarity(sentence_embedding, search_term_embedding)[0][0]
                    for sentence_embedding in sentence_embeddings]

    # Normalize the similarity scores to be between 0 and 1
    min_similarity = min(similarities)
    max_similarity = max(similarities)
    if max_similarity == min_similarity:
        normalized_similarities = [0.0] * len(similarities)  # Avoid division by zero
    else:
        normalized_similarities = [(s - min_similarity) / (max_similarity - min_similarity) for s in similarities]

    return list(zip(sentences, normalized_similarities))

def extract_text_from_url(url):
    """
    Extracts text from a URL using Selenium and BeautifulSoup, rendering JavaScript.
    Returns a single string of all extracted text.
    """
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        # Add User-Agent for iPhone
        user_agent = "Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.7.1 Mobile/15E148 Safari/604.1"
        chrome_options.add_argument(f"user-agent={user_agent}")

        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)

        # Explicit wait for the presence of at least one of the desired elements
        wait = WebDriverWait(driver, 10)
        wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "p, h1, h2, h3, li"))
        )

        page_source = driver.page_source
        driver.quit()
        soup = BeautifulSoup(page_source, "html.parser")

        # Exclude header and footer divs
        for tag in soup.find_all(['header', 'footer']):
            tag.decompose()

        # Extract all relevant tags and their text content
        all_relevant_tags = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'p'])
        text = ""
        for tag in all_relevant_tags:
            text += tag.get_text(separator=" ", strip=True) + "\n"
        return text.strip() # Returns a single string

    except Exception as e:
        print(f"Error fetching or processing URL {url}: {e}")
        return None

def highlight_text(text, search_term, tokenizer, model):
    """Highlights text based on similarity to the search term using HTML/CSS, adding paragraph breaks."""
    sentences_with_similarity = rank_sentences_by_similarity(text, search_term, tokenizer, model)

    highlighted_text = ""
    for sentence, similarity in sentences_with_similarity:
        if similarity < 0.35:
            color = "red"
        elif similarity < 0.65:
            color = "black"
        else:
            color = "green"

        # Enclose each sentence in a <p> tag for paragraph breaks
        highlighted_text += f'<p style="color:{color};">{sentence}</p>'
    return highlighted_text

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
    st.set_page_config(
        page_title="Cosine Similarity Heat Map - Paragraphs | The SEO Consultant.ai",
        page_icon=":art:"
    )
    logo_url = "https://theseoconsultant.ai/wp-content/uploads/2024/12/cropped-theseoconsultant-logo-2.jpg"
    create_navigation_menu(logo_url)
    st.title("Cosine Similarity Heat Map - Paragraphs")
    st.markdown("Green text is the most relevant to the search query. Red is the least relevant content to search query.")

    st.markdown("---")

    # Input fields
    source = st.radio("Select Input Source:", ("Text", "URL"))

    if source == "Text":
        input_text = st.text_area("Enter your text:", height=300, value="Paste your text here.")
    else:
        url = st.text_input("Enter URL:", "")
        input_text = extract_text_from_url(url)
        if input_text:
            st.success("Content extracted from URL.")
        elif url: # Show warning only if something was entered
            st.warning("Could not extract content from the URL. Check the URL or try a different one.")


    search_term = st.text_input("Enter your search term:", value="Enter Your SEO Keyword Here")

    if st.button("Highlight"):
        tokenizer, model = load_models()
        if not input_text or not search_term:
            st.error("Please enter both text and a search term.")
        else:
            with st.spinner("Generating highlighted text..."):
                highlighted_text = highlight_text(input_text, search_term, tokenizer, model)

            st.markdown(highlighted_text, unsafe_allow_html=True)  # Display highlighted text

    st.markdown("---")
    st.markdown(
        "Powered by [The SEO Consultant.ai](https://theseoconsultant.ai)",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
