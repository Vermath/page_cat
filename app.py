# app.py
import streamlit as st
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from urllib.parse import urlparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
from bs4 import BeautifulSoup
from io import BytesIO

# Function to fetch sitemap URLs with increased timeout and retry logic
def fetch_sitemap_urls(domain, max_retries=3):
    main_sitemap_url = f"https://{domain}/sitemap.xml"
    sitemap_urls = []
    for attempt in range(max_retries):
        try:
            response = requests.get(main_sitemap_url, timeout=30)
            response.raise_for_status()
            tree = ET.fromstring(response.content)
            namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            sitemap_elements = tree.findall('ns:sitemap', namespaces=namespace)
            if sitemap_elements:
                sitemap_urls = [sitemap.find('ns:loc', namespaces=namespace).text for sitemap in sitemap_elements]
            else:
                sitemap_urls = [main_sitemap_url]
            break  # Exit retry loop on success
        except requests.exceptions.RequestException as e:
            st.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                st.info("Retrying...")
                time.sleep(2)  # Wait before retrying
            else:
                st.error("Max retries reached. Exiting.")
                return sitemap_urls
        except ET.ParseError as e:
            st.error(f"Error parsing XML: {e}")
            return sitemap_urls
    return sitemap_urls

# Function to identify specific post sitemaps
def identify_post_sitemaps(sitemap_urls):
    """
    Identifies all post sitemaps by checking if 'post' is in the sitemap URL.
    """
    relevant_sitemaps = [url for url in sitemap_urls if 'post' in url.lower()]
    return relevant_sitemaps

# Function to fetch all URLs from sitemaps concurrently
def fetch_all_post_urls(post_sitemaps, max_workers=5):
    all_post_urls = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_urls_from_sitemap, ps_url) for ps_url in post_sitemaps]
        for future in as_completed(futures):
            urls = future.result()
            all_post_urls.extend(urls)
    return all_post_urls

# Function to fetch all URLs from a sitemap with retry logic
def fetch_urls_from_sitemap(sitemap_url, max_retries=3):
    urls = []
    for attempt in range(max_retries):
        try:
            response = requests.get(sitemap_url, timeout=30)
            response.raise_for_status()
            tree = ET.fromstring(response.content)
            namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            url_elements = tree.findall('ns:url', namespaces=namespace)
            urls = [url.find('ns:loc', namespaces=namespace).text for url in url_elements]
            break  # Exit retry loop on success
        except requests.exceptions.RequestException as e:
            st.warning(f"Attempt {attempt + 1} to fetch {sitemap_url} failed: {e}")
            if attempt < max_retries - 1:
                st.info("Retrying...")
                time.sleep(2)
            else:
                st.error(f"Failed to fetch {sitemap_url} after {max_retries} attempts.")
                break
    return urls

# Function to extract text from a webpage
def extract_text(session, url):
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser', from_encoding='utf-8')
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return text
    except requests.exceptions.RequestException as e:
        st.warning(f"Error fetching {url}: {e}")
        return ""
    except Exception as e:
        st.warning(f"Error processing {url}: {e}")
        return ""

# Function to truncate text to fit within token limits
def truncate_text(text, max_tokens=128000):
    max_chars = max_tokens * 4
    if len(text) > max_chars:
        return text[:max_chars]
    return text

# Function to evaluate content with OpenAI API
def evaluate_content(content, theme, openai_api_key):
    truncated_content = truncate_text(content)
    prompt = f"Determine if the following content is related to the theme '{theme}'. Respond with 'Yes' or 'No'.\n\nContent:\n{truncated_content}"
    try:
        response = openai.ChatCompletion.create(
            api_key=openai_api_key,
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=10,
            temperature=0,
        )
        answer = response.choices[0].message.content.strip().lower()
        return "yes" in answer
    except openai.error.OpenAIError as e:
        st.warning(f"OpenAI API error: {e}")
        return False
    except Exception as e:
        st.warning(f"Unexpected error during content evaluation: {e}")
        return False

# Helper function to process each URL
def process_url(session, url, theme, openai_api_key):
    text = extract_text(session, url)
    if not text:
        return url, False
    is_relevant = evaluate_content(text, theme, openai_api_key)
    return url, is_relevant

# Streamlit App
def main():
    st.title("🔍 Sitemap Content Evaluator")

    # Read OpenAI API Key from secrets
    openai_api_key = st.secrets.get("openai_api_key")

    if not openai_api_key:
        st.error("OpenAI API Key not found in secrets.toml. Please add it and restart the app.")
        st.stop()

    # Inputs for domain and theme
    st.header("Input Parameters")
    domain = st.text_input("Enter the domain (e.g., recipetineats.com):")
    theme = st.text_input("Enter the specific theme to search for:")

    if st.button("🚀 Start Processing"):
        if not domain or not theme:
            st.error("Please provide both the domain and the theme.")
            st.stop()

        # Fetch sitemap URLs
        st.info("Fetching sitemap URLs...")
        sitemap_urls = fetch_sitemap_urls(domain)
        if not sitemap_urls:
            st.error("No sitemap URLs found. Exiting.")
            st.stop()

        st.success(f"Found {len(sitemap_urls)} sitemap URLs in the main sitemap.")

        # Identify post sitemaps
        st.info("Identifying post sitemaps...")
        post_sitemaps = identify_post_sitemaps(sitemap_urls)
        if not post_sitemaps:
            st.error("No post sitemaps identified. Exiting.")
            st.stop()

        st.success(f"Identified {len(post_sitemaps)} post sitemap(s):")
        for ps in post_sitemaps:
            st.write(f"- {ps}")

        # Fetch all URLs from identified post sitemaps
        st.info("Fetching URLs from post sitemaps...")
        all_post_urls = fetch_all_post_urls(post_sitemaps)
        all_post_urls = list(set(all_post_urls))  # Remove duplicates

        st.success(f"Total post URLs found: {len(all_post_urls)}")

        if not all_post_urls:
            st.error("No post URLs to process. Exiting.")
            st.stop()

        # Optionally limit the number of URLs processed
        MAX_URLS = 2000  # Adjust as needed
        all_post_urls = all_post_urls[:MAX_URLS]

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

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(process_url, session, url, theme, openai_api_key): url for url in all_post_urls
            }

            total_futures = len(future_to_url)
            for idx, future in enumerate(as_completed(future_to_url)):
                url = future_to_url[future]
                try:
                    url, is_relevant = future.result()
                    if is_relevant:
                        relevant_urls.append(url)
                except Exception as e:
                    st.warning(f"Error processing {url}: {e}")

                # Update progress
                progress = (idx + 1) / total_futures
                progress_bar.progress(progress)
                status_text.text(f"Processing URLs: {idx + 1}/{total_futures}")

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
        else:
            st.warning("\nNo relevant pages found.")

if __name__ == "__main__":
    main()
