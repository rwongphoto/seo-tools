import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from PIL import Image, ImageDraw, ImageFont
import cairosvg
from io import BytesIO
#from IPython.display import Image as IPythonImage, display # Remove IPython dependency

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

@st.cache_resource
def initialize_model():
    """Initializes the BERT tokenizer and model."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    return tokenizer, model

def extract_content_from_url(url):
    """
    Extracts text from a URL using Selenium and BeautifulSoup, rendering JavaScript,
    and returns a list of content blocks (paragraphs, titles, list items).
    """
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

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

        # Extract paragraphs, titles, and list items
        content = []
        for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            text = tag.get_text(separator=" ", strip=True)
            if text:  # Only add if not empty
                content.append(text)

        return content

    except Exception as e:
        st.error(f"Error fetching or processing URL {url}: {e}")
        return None

def get_embedding(text, model, tokenizer):
    """Generates a BERT embedding for the given text."""
    tokenizer.pad_token = tokenizer.unk_token
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def color_code_text_by_similarity_png(content, search_term, image_width=800):
    """
    Color-codes the input content blocks (paragraphs, titles, list items) based on
    cosine similarity to a search term, creating a PNG image for Streamlit display.
    """

    tokenizer, model = initialize_model()

    # Generate embedding for the search term
    search_term_embedding = get_embedding(search_term, model, tokenizer)

    # Generate embeddings for each content block and calculate cosine similarities
    content_embeddings = []
    similarities = []
    for block in content:
        block_embedding = get_embedding(block, model, tokenizer)
        content_embeddings.append(block_embedding)
        similarity = cosine_similarity(block_embedding, search_term_embedding)[0][0]
        similarities.append(similarity)

    # Normalize similarity scores for color mapping
    min_similarity = min(similarities)
    max_similarity = max(similarities)
    max_similarity = max_similarity if max_similarity > min_similarity else min_similarity + 1e-6
    normalized_similarities = [(s - min_similarity) / (max_similarity - min_similarity)
                               for s in similarities]

    # --- PNG Generation ---
    font_size = 14
    try:
        font = ImageFont.truetype("FreeSans.ttf", font_size)  # Attempt to load from relative path

    except:

       font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", font_size) #linux

    line_height = font_size * 1.2  # Add some spacing
    padding = 10
    background_color = "white"
    text_color = "black"

    # Calculate image height based on content and width
    num_lines = 0
    for block in content:  # Iterate through content blocks
        words = block.split()
        current_line = ""
        for word in words:
            test_line = current_line + word + " "
            if font.getlength(test_line) > image_width - 2 * padding:
                num_lines += 1
                current_line = word + " "
            else:
                current_line = test_line
        num_lines += 1  # Add one for the last line

    image_height = int(num_lines * line_height + 2 * padding)

    # Create image
    image = Image.new("RGB", (image_width, image_height), color=background_color)
    draw = ImageDraw.Draw(image)

    y_position = padding
    x_position = padding

    for block, similarity in zip(content, normalized_similarities):
        # Determine color based on similarity
        color = "red" if similarity < 0.40 else "black" if similarity < 0.70 else "green"

        # Draw the content block on the image, wrapping if necessary
        words = block.split()
        current_line = ""
        for word in words:
            test_line = current_line + word + " "
            if font.getlength(test_line) > image_width - 2 * padding:
                draw.text((x_position, y_position), current_line, fill=color, font=font)
                y_position += line_height
                current_line = word + " "
            else:
                current_line = test_line
        draw.text((x_position, y_position), current_line, fill=color, font=font)  # Draw the last line
        y_position += line_height

    # Save the image to a BytesIO object for Streamlit
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    return img_byte_arr

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
            <img src="{logo_url}" width="250">
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
          margin-bottom: 20px;        /* Add space below the menu */
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
    st.markdown(
        """
        <style>
        .title {
            display: flex;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h1 class='title'>Cosine Similarity Score - Every Embedding</h1>", unsafe_allow_html=True)

    logo_url = "https://theseoconsultant.ai/wp-content/uploads/2024/12/cropped-theseoconsultant-logo-2.jpg"
    create_navigation_menu(logo_url)

    # Input fields
    url = st.text_input("Enter URL:", "https://www.rwongphoto.com/gallery/california-pictures/")
    search_term = st.text_input("Enter Search Term:", "california photography")

    if st.button("Calculate Similarity and Color Code"):
        with st.spinner("Extracting Content and Calculating Similarities..."):
            content = extract_content_from_url(url)

            if content:
                try:
                    image_png = color_code_text_by_similarity_png(content, search_term)

                    st.image(image_png, caption="Color-Coded Text Based on Similarity", use_column_width=True)
                except Exception as e:
                    st.error(f"Error generating and displaying image: {e}")
            else:
                st.warning("Could not extract content from the URL.")

    st.markdown("By: [The SEO Consultant.ai](https://theseoconsultant.ai)", unsafe_allow_html=True)

if __name__ == "__main__":
    main()