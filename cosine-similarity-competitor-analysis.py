import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import matplotlib.pyplot as plt
import pandas as pd

@st.cache_resource
def initialize_model():
    """Initializes the BERT tokenizer and model."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    return tokenizer, model

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
        user_agent = "Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.7.1 Mobile/15E148 Safari/604.1"
        chrome_options.add_argument(f"user-agent={user_agent}")

        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)

        wait = WebDriverWait(driver, 10)
        wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "p, h1, h2, h3, li"))
        )

        page_source = driver.page_source
        driver.quit()
        soup = BeautifulSoup(page_source, "html.parser")

        for tag in soup.find_all(['header', 'footer']):
            tag.decompose()

        all_relevant_tags = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'p'])
        text = ""
        for tag in all_relevant_tags:
            text += tag.get_text(separator=" ", strip=True) + "\n"
        return text.strip()

    except Exception as e:
        st.error(f"Error fetching or processing URL {url}: {e}")
        return None

def get_embedding(text, model, tokenizer):
    """Generates a BERT embedding for the given text."""
    tokenizer.pad_token = tokenizer.unk_token
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def calculate_overall_similarity(urls, search_term, model, tokenizer):
    """
    Calculates the overall cosine similarity score for a list of URLs against a search term.
    Returns a list of (url, similarity_score) tuples.
    """

    # Generate embedding for the search term
    search_term_embedding = get_embedding(search_term, model, tokenizer)

    results = []
    for url in urls:
        text = extract_text_from_url(url)
        if text:
            text_embedding = get_embedding(text, model, tokenizer)
            similarity = cosine_similarity(text_embedding, search_term_embedding)[0][0]
            results.append((url, similarity))
            st.write(f"Cosine similarity for {url}: {similarity}")
        else:
            st.write(f"Could not extract text from {url}")
            results.append((url, None))

    return results
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
    logo_url = "https://theseoconsultant.ai/wp-content/uploads/2024/12/cropped-theseoconsultant-logo-2.jpg"
    create_navigation_menu(logo_url)
    st.title("Cosine Similarity Competitor Analysis")

    # Input fields
    search_term = st.text_input("Enter Search Term:", "")
    urls_input = st.text_area("Enter URLs (one per line):",
                            """""")
    urls = [url.strip() for url in urls_input.splitlines() if url.strip()]


    if st.button("Calculate Similarity"):
        if not urls:
            st.warning("Please enter at least one URL.")
        else:
            # Initialize model (only once)
            tokenizer, model = initialize_model()

            with st.spinner("Calculating similarities..."):
                similarity_scores = calculate_overall_similarity(urls, search_term, model, tokenizer)

            # Prepare data for plotting
            urls_plot = [url for url, score in similarity_scores]
            scores_plot = [score if score is not None else 0 for url, score in similarity_scores]  # Replace None with 0 for plotting

            # Create the bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(urls_plot, scores_plot)
            ax.set_xlabel("URLs")
            ax.set_ylabel("Similarity Score")
            ax.set_title("Cosine Similarity of URLs to Search Term")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

            # Display results in a table
            data = {'URL': urls_plot, 'Similarity Score': scores_plot}
            df = pd.DataFrame(data)
            st.dataframe(df)

    st.markdown("---") # Add a horizontal line for visual separation
    st.markdown(
        "Powered by [The SEO Consultant.ai](https://theseoconsultant.ai)",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
