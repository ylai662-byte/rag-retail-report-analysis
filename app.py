
import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


CHUNK_SIZE = 500
CHUNK_OVERLAP = 150
TOP_K = 5
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-5.6"


st.set_page_config(
    page_title="AI-Powered Retail Report Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("AI-Powered Retail Report Analysis")

st.write(
    "Upload a retail industry report and ask business questions "
    "using Retrieval-Augmented Generation (RAG)."
)

st.caption(
    f"Retrieval configuration: {CHUNK_SIZE}-character chunks, "
    f"{CHUNK_OVERLAP}-character overlap, Top-K = {TOP_K}."
)


# OpenAI API Key
api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password"
)

if not api_key:
    st.sidebar.info(
        "Enter your OpenAI API key to analyze a report."
    )


@st.cache_resource
def build_rag_components(file_bytes, api_key):

    pdf_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:
            temp_file.write(file_bytes)
            pdf_path = temp_file.name

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

    finally:
        if pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )

    return documents, chunks, retriever


# Upload PDF
uploaded_file = st.file_uploader(
    "Upload Retail Report",
    type="pdf"
)


if uploaded_file is not None and api_key:

    file_bytes = uploaded_file.getvalue()

    try:
        documents, chunks, retriever = build_rag_components(
            file_bytes,
            api_key
        )

        llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=api_key
        )

        prompt = ChatPromptTemplate.from_template(
            """Use the following retrieved context to answer the question.

Answer only from the provided context.
If the answer is not in the context, say you do not know.

Context:
{context}

Question:
{question}
"""
        )

        st.success(
            f"Report processed successfully: "
            f"{len(documents)} pages and {len(chunks)} chunks."
        )

        question = st.text_input(
            "Ask a Business Question",
            placeholder="What are the main challenges facing the retail industry?"
        )

        if st.button("Analyze Report"):

            if question:

                with st.spinner("Analyzing the report..."):

                    # Retrieve only once
                    source_documents = retriever.invoke(question)

                    context = "\n\n".join(
                        doc.page_content
                        for doc in source_documents
                    )

                    formatted_prompt = prompt.invoke({
                        "context": context,
                        "question": question
                    })

                    response = llm.invoke(
                        formatted_prompt
                    ).content

                st.subheader("AI Answer")
                st.write(response)

                st.subheader("Supporting Sources")

                for i, doc in enumerate(
                    source_documents,
                    start=1
                ):

                    page = doc.metadata.get(
                        "page",
                        "N/A"
                    )

                    if page != "N/A":
                        page += 1

                    with st.expander(
                        f"Source {i} — Page {page}"
                    ):
                        st.write(doc.page_content)

            else:
                st.warning(
                    "Please enter a business question."
                )

    except Exception as e:
        st.error(
            f"An error occurred while processing the report: {e}"
        )


elif uploaded_file is not None and not api_key:

    st.warning(
        "Please enter your OpenAI API key in the sidebar."
    )