# utils/loader.py
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.common_utils import load_config

import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------
# Load Document and Create Chunks
# ----------------------------------------
def load_and_create_chunks(folder_path = "./data/raw/insurance_docs", chunk_size = 200, chunk_overlap = 50):
    if not os.path.exists(folder_path):
        raise ValueError(f"Path {folder_path} does not exists. Please create path and add docs")

    docs = []

    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        if file.endswith(".txt") or file.endswith(".md"):
            loader = TextLoader(path)
        elif file.endswith(".pdf"):
            loader  = PyPDFLoader(path)
        else:
            continue

        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    chunks = splitter.split_documents(docs)
    print(f"✅ Loaded {len(docs)} docs → {len(chunks)} chunks")
    return chunks