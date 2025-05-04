import streamlit as st
from transformers import pipeline

# Set page configuration
st.set_page_config(page_title="QA with RoBERTa", layout="centered")

# Title
st.title("🤖 Question Answering with RoBERTa (Hugging Face Transformers)")
st.write("Input any context and ask a question. The model will extract the answer from the context.")

# Load QA pipeline with caching to avoid reloading
@st.cache_resource
def load_qa_pipeline():
    return pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        tokenizer="deepset/roberta-base-squad2"
    )

qa_pipeline = load_qa_pipeline()

# Input fields
context = st.text_area("📄 Enter Context Paragraph", height=250, placeholder="Paste your context here...")

question = st.text_input("❓ Enter Your Question", placeholder="Type your question...")

# Run QA when button is clicked
if st.button("Get Answer"):
    if context.strip() == "" or question.strip() == "":
        st.warning("⚠️ Please provide both a context and a question.")
    else:
        result = qa_pipeline(question=question, context=context)
        st.success(f"**Answer:** {result['answer']}")
        st.info(f"**Confidence Score:** {result['score']:.4f}")
