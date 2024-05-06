import streamlit as st
import os
from PyPDF2 import PdfReader
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

os.getenv("OPENAI_API_KEY")
# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_metadata(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    document_metadata = pdf_reader.metadata
    
    title = document_metadata.get("/Title")
    author = document_metadata.get("/Author")
    keywords = document_metadata.get("/Keywords")

    return title, author, keywords

def summarize_pdf(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)

    summaries = []
    for chunk in chunks:
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt="Summarize the following text:\n" + chunk,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.5,
        )
        chunk_summary = response.choices[0].text.strip()
        summaries.append(chunk_summary)

    combined_summary = " ".join(summaries)
    return combined_summary

def categorize_pdf(text, use_summary=True, metadata=None):
    if use_summary:
        text = summarize_pdf(text)  # Summarize first if needed

    if metadata and metadata[0]:
        context = metadata[0]  # Prioritize title from metadata
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
        chunks = text_splitter.split_text(text)

        categories = []
        for chunk in chunks:
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt="Categorize the following text:\n" + chunk,
                max_tokens=10,
                n=1,
                stop=None,
                temperature=0.7,
            )
            chunk_categories = response.choices[0].text.strip().splitlines()
            categories.extend(chunk_categories)

        # Combine categories to extract a single context
        unique_categories = list(set(categories))
        combined_categories = []
        for category in unique_categories:
            count = categories.count(category)
            combined_categories.extend([category] * count)  # Add multiple times based on frequency

        most_frequent_category = max(set(combined_categories), key=combined_categories.count)
        context = most_frequent_category

    return context

# Streamlit app structure
st.title("PDF Context Extractor (using OpenAI)")

uploaded_file = st.file_uploader("Upload PDF file", type="pdf")

if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    title, author, keywords = extract_metadata(uploaded_file)
    st.write("Title:", title)
    st.write("Author:", author)
    if keywords:
        st.write("Keywords:", keywords)

    context = categorize_pdf(text, use_summary=True, metadata=(title, author, keywords))
    st.write("Context:", context)

    # Option to display summary
    if st.button("Show Summary"):
        summary = summarize_pdf(text)
        st.write("Summary:", summary)