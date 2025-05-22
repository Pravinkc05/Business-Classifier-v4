# -*- coding: utf-8 -*-
"""
Created on Tue May 20 06:37:04 2025

@author: Sahil
"""


import streamlit as st
import pandas as pd
import numpy as np
import requests
from openai import OpenAI
import openai
import time

# ---- Streamlit App Setup ----
st.title("Business Classifier")

# ---- Input Fields ----
input_type = st.radio("Search by:", ["Business Name", "Website URL"])

col1, col2 = st.columns(2)

with col1:
    label = "Enter Business Name:" if input_type == "Business Name" else "Enter Website URL:"
    user_input = st.text_input(label)

with col2:
    state_input = ""
    if input_type == "Business Name":
        state_input = st.text_input("Enter State:")

# ---- Buttons ----
col3, col4 = st.columns(2)
with col3:
    submit = st.button("Submit")
with col4:
    clear = st.button("Clear")

if clear:
    st.rerun()

# ---- Load Reference Data ----
@st.cache_data
def load_reference_data():
    return pd.read_excel(r"C:\Python project\IRMI Cross-Reference Guide.xlsx")

ref_data = load_reference_data()

# ---- Brave Search API ----
BRAVE_API_KEY = st.secrets["BRAVE_API_KEY"]

def search_website_brave(query):
    if not BRAVE_API_KEY:
        st.error("Brave API key is missing.")
        return None

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "x-subscription-token": BRAVE_API_KEY
    }
    params = {
        "q": query,   # <-- Do NOT encode this
        "count": 1
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        results = response.json().get("web", {}).get("results", [])
        if results:
            return results[0].get("url")
    except Exception as e:
        st.error(f"Brave search failed: {e}")
    return None

# ---- Web Scraping ----

SCRAPER_API_KEY = st.secrets["SCRAPER_API_KEY"]

def get_website_text(url):
    try:
        api_url = f"https://api.scraperapi.com?api_key={SCRAPER_API_KEY}&url={url}&render=true"
        response = requests.get(api_url, timeout=60)
        response.raise_for_status()
        text = response.text

        if "not available in your region" in text.lower() or len(text.strip()) < 100:
            st.warning("ScraperAPI returned restricted or empty content.")
            return ""

        return text[:4000]

    except Exception as e:
        st.error(f"ScraperAPI failed: {e}")
        return ""

# ---- OpenAI Setup ----
openai.api_key = st.secrets["OPENAI_API_KEY"]

client = OpenAI(api_key=openai.api_key)

def summarize_business(content: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert insurance underwriter."},
                {"role": "user", "content": f"Based on this content, describe the business operation in one or two lines: {content}"}
            ],
            temperature=0.4,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI summarization failed: {e}")
        return ""

# ---- Cosine Similarity & Matching ----
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_best_match(input_description, state_input, client):
    try:
        df = pd.read_pickle(r"C:\Python project\descriptions_with_embeddings.pkl")
        df.columns = df.columns.str.strip()

        response = client.embeddings.create(
            input=[input_description],
            model="text-embedding-3-small"
        )
        input_embedding = response.data[0].embedding

        df["similarity"] = df["embedding"].apply(lambda x: cosine_similarity(x, input_embedding))
        top = df.sort_values("similarity", ascending=False).iloc[0]

        state = state_input.strip().upper()
        if state in df.columns and pd.notna(top[state]):
            class_code = top[state]
            code_source = f"{state} Class Code"
        else:
            class_code = top.get("NCCI", "N/A")
            code_source = "NCCI Class Code"

        return {
            "Classification Wording": top["Classification Wording"],
            "NAICS": top["NAICS"],
            "Class Code": class_code,
            "Code Source": code_source
        }

    except Exception as e:
        st.error(f"Matching failed: {e}")
        return None

# ---- Main Logic ----
if submit and user_input:
    status_text = st.empty()
    status_text.info("Processing...")

    url = user_input
    if input_type == "Business Name":
        query = f"{user_input} {state_input}" if state_input else user_input
        url = search_website_brave(query)

    if url:
        st.markdown(f"**Website:** {url}")
        raw_text = get_website_text(url)

        if raw_text:
            summary = summarize_business(raw_text)
            if summary:
                st.markdown("### Description of Operations")
                st.write(summary)

                match = find_best_match(summary, state_input, client)
                if match:
                    # Format NAICS and Class Code as integers
                    naics = f"{int(match['NAICS'])}" if pd.notna(match['NAICS']) else "N/A"
                    class_code = f"{int(match['Class Code'])}" if pd.notna(match['Class Code']) else "N/A"

                    st.markdown("### Classification Match")
                    st.markdown(f"""
                    <div style='border-left: 4px solid #28a745; background:#d4edda; padding:10px; border-radius:5px'>
                        <b>Classification Wording:</b> {match['Classification Wording']}<br>
                        <b>NAICS:</b> {naics}<br>
                        <b>{match['Code Source']}:</b> {class_code}
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.warning("No suitable classification found.")
            else:
                st.warning("Couldn't summarize the content.")
        else:
            st.warning("No readable content found on the website.")
    else:
        st.error("Could not determine a website for the input provided.")

    status_text.success("Done")
