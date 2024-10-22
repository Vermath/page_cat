# app.py
import streamlit as st
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
from openai import OpenAI  # Import OpenAI class
from bs4 import BeautifulSoup
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Function to fetch sitemap URLs with increased timeout and retry logic
def fetch_sitemap_urls(domain, max_retries=3):
    main_sitemap_url = f"https://{domain}/sitemap.xml"
    sitemap_urls = []
    for attempt in range(max_retries):
        try:
            logger.debug(f"Fetching main sitemap URL: {main_sitemap_url}, Attempt: {attempt + 1}")
            response = requests.get(main_sitemap_url, timeout=30)
            response.raise_for_status()
            tree = ET.fromstring(response.content)
            namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            sitemap_elements = tree.findall('ns:sitemap', namespaces=namespace)
            if sitemap_elements:
                sitemap_urls = [sitemap.find('ns:loc', namespaces=namespace).text for sitemap in sitemap_elements]
            else:
                sitemap_urls = [main_sitemap_url]
            logger.debug(f"Sitemap URLs found: {sitemap_urls}")
            break  # Exit retry loop on success
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed to fetch sitemap: {e}")
            if attempt < max_retries - 1:
                logger.info("Retrying...")
                time.sleep(2)  # Wait before retrying
            else:
                logger.error("Max retries reached while fetching sitemap URLs. Exiting.")
                return sitemap_urls
        except ET.ParseError as e:
            logger.error(f"Error parsing XML from sitemap: {e}")
            return sitemap_urls
    return sitemap_urls

# Function to identify specific post sitemaps
def identify_post_sitemaps(sitemap_urls):
    logger.debug(f"Identifying post sitemaps from: {sitemap_urls}")
    relevant_sitemaps = [url for url in sitemap_urls if 'post' in url.lower()]
    logger.debug(f"Post sitemaps identified: {relevant_sitemaps}")
    return relevant_sitemaps

# Function to fetch all URLs from sitemaps concurrently
def fetch_all_post_urls(post_sitemaps, max_workers=5):
    all_post_urls = []
    logger.debug(f"Fetching all post URLs from sitemaps: {post_sitemaps}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_urls_from_sitemap, ps_url): ps_url for ps_url in post_sitemaps}
        for future in as_completed(futures):
            sitemap_url = futures[future]
            try:
                urls = future.result()
                logger.debug(f"Fetched {len(urls)} URLs from sitemap {sitemap_url}")
                all_post_urls.extend(urls)
            except Exception as e:
                logger.error(f"Error fetching URLs from sitemap {sitemap_url}: {e}")
    return all_post_urls

# Function to fetch all URLs from a sitemap with retry logic
def fetch_urls_from_sitemap(sitemap_url, max_retries=3):
    urls = []
    for attempt in range(max_retries):
        try:
            logger.debug(f"Fetching URLs from sitemap: {sitemap_url}, Attempt: {attempt + 1}")
            response = requests.get(sitemap_url, timeout=30)
            response.raise_for_status()
            tree = ET.fromstring(response.content)
            namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            url_elements = tree.findall('ns:url', namespaces=namespace)
            urls = [url.find('ns:loc', namespaces=namespace).text for url in url_elements]
            logger.debug(f"Found {len(urls)} URLs in sitemap {sitemap_url}")
            break  # Exit retry loop on success
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} to fetch {sitemap_url} failed: {e}")
            if attempt < max_retries - 1:
                logger.info("Retrying...")
                time.sleep(2)
            else:
                logger.error(f"Failed to fetch {sitemap_url} after {max_retries} attempts.")
                break
        except ET.ParseError as e:
            logger.error(f"Error parsing XML from {sitemap_url}: {e}")
            break
    return urls

# Function to extract text from a webpage
def extract_text(session, url):
    try:
        logger.debug(f"Extracting text from URL: {url}")
        response = session.get(url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        text = soup.get_text(separator=' ', strip=True)
        logger.debug(f"Extracted text length from {url}: {len(text)}")
        return text
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error fetching {url}: {e}")
        return ""
    except Exception as e:
        logger.warning(f"Error processing {url}: {e}")
        return ""

# Function to truncate text to fit within token limits
def truncate_text(text, max_tokens=128000):
    max_chars = max_tokens * 4
    if len(text) > max_chars:
        logger.debug(f"Truncating text from length {len(text)} to {max_chars}")
        return text[:max_chars]
    return text

# Function to evaluate content with OpenAI API
def evaluate_content(content, theme, client):
    truncated_content = truncate_text(content)
    prompt = f"Determine if the following content is related to the theme '{theme}'. Respond with 'Yes' or 'No'.\n\nContent:\n{truncated_content}"
    try:
        logger.debug(f"Sending prompt to OpenAI API for evaluation. Prompt length: {len(prompt)}")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=10,
            temperature=0,
        )
        logger.debug(f"OpenAI API response: {response}")
        answer = response.choices[0].message.content.strip().lower()
        logger.debug(f"Received answer from OpenAI API: {answer}")
        return "yes" in answer
    except Exception as e:
        logger.warning(f"Unexpected error during content evaluation: {e}")
        return False

# Helper function to process each URL
def process_url(session, url, theme, client):
    logger.debug(f"Processing URL: {url}")
    text = extract_text(session, url)
    if not text:
        logger.debug(f"No text extracted from URL: {url}")
        return url, False
    is_relevant = evaluate_content(text, theme, client)
    logger.debug(f"URL {url} relevance result: {is_relevant}")
    return url, is_relevant

# Main function
def main():
    st.title("🔍 Sitemap Content Evaluator")

    # Read OpenAI API Key from secrets
    openai_api_key = st.secrets.get("openai_api_key")

    if not openai_api_key:
        st.error("OpenAI API Key not found in secrets.toml. Please add it and restart the app.")
        return

    # Initialize OpenAI client
    client = OpenAI(
        api_key=openai_api_key,
    )

    # Inputs for domain and theme
    st.header("Input Parameters")
    domain = st.text_input("Enter the domain (e.g., recipetineats.com):")
    theme = st.text_input("Enter the specific theme to search for:")

    if st.button("🚀 Start Processing"):
        if not domain or not theme:
            st.error("Please provide both the domain and the theme.")
            return

        logger.info(f"Starting processing for domain: {domain} with theme: {theme}")

        # Fetch sitemap URLs
        st.info("Fetching sitemap URLs...")
        sitemap_urls = fetch_sitemap_urls(domain)
        if not sitemap_urls:
            st.error("No sitemap URLs found. Exiting.")
            return

        st.success(f"Found {len(sitemap_urls)} sitemap URLs in the main sitemap.")
        logger.debug(f"Sitemap URLs: {sitemap_urls}")

        # Identify post sitemaps
        st.info("Identifying post sitemaps...")
        post_sitemaps = identify_post_sitemaps(sitemap_urls)
        if not post_sitemaps:
            st.error("No post sitemaps identified. Exiting.")
            return

        st.success(f"Identified {len(post_sitemaps)} post sitemap(s):")
        for ps in post_sitemaps:
            st.write(f"- {ps}")
        logger.debug(f"Post sitemaps: {post_sitemaps}")

        # Fetch all URLs from identified post sitemaps
        st.info("Fetching URLs from post sitemaps...")
        all_post_urls = fetch_all_post_urls(post_sitemaps)
        all_post_urls = list(set(all_post_urls))  # Remove duplicates

        st.success(f"Total post URLs found: {len(all_post_urls)}")
        logger.debug(f"All post URLs: {all_post_urls}")

        if not all_post_urls:
            st.error("No post URLs to process. Exiting.")
            return

        # Optionally limit the number of URLs processed
        MAX_URLS = 5000  # Adjust as needed
        all_post_urls = all_post_urls[:MAX_URLS]
        logger.info(f"Processing up to {MAX_URLS} URLs.")

        # Evaluating pages
        st.info("Evaluating pages...")

        # Create a session to reuse HTTP connections
        session = requests.Session()

        # Define the maximum number of worker threads
        max_workers = 10  # Adjust based on your system's capabilities and API rate limits

        relevant_urls = []

        # Initialize progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_urls = len(all_post_urls)
        logger.info(f"Total URLs to process: {total_urls}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(process_url, session, url, theme, client): url for url in all_post_urls
            }

            for idx, future in enumerate(as_completed(future_to_url)):
                url = future_to_url[future]
                try:
                    url, is_relevant = future.result()
                    logger.debug(f"Result for URL {url}: {is_relevant}")
                    if is_relevant:
                        relevant_urls.append(url)
                except Exception as e:
                    logger.error(f"Error processing {url}: {e}")
                    st.warning(f"Error processing {url}: {e}")

                # Update progress
                progress = (idx + 1) / total_urls
                progress_bar.progress(progress)
                status_text.text(f"Processing URLs: {idx + 1}/{total_urls}")

        logger.info(f"Found {len(relevant_urls)} relevant URLs.")

        # Display results
        if relevant_urls:
            st.success(f"\nFound {len(relevant_urls)} relevant pages.")
            # Create a DataFrame and allow user to download CSV
            df = pd.DataFrame(relevant_urls, columns=["URL"])
            csv_buffer = BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            st.download_button(
                label="📥 Download relevant pages as CSV",
                data=csv_buffer,
                file_name='relevant_pages.csv',
                mime='text/csv',
            )

            st.header("Relevant Pages")
            st.dataframe(df)
            logger.debug(f"Relevant URLs DataFrame:\n{df}")
        else:
            st.warning("\nNo relevant pages found.")

if __name__ == "__main__":
    main()