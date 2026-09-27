# utils/common_utils.py
import os
import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

import warnings
warnings.filterwarnings("ignore")

load_dotenv()
# ----------------------------------------
# Load Config
# ----------------------------------------
def load_config(config_path = "config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# ----------------------------------------
# Load Emdedding Model (OpenAI or Gemini)
# ----------------------------------------
def get_embedding_model(config: dict):
    provider = config["llm"]["provider"]

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("'OPENAI_API_KEY' is not present in .env")
        return OpenAIEmbeddings(
            api_key = api_key,
            model = config["embedding"]["openai_model"]
        )
    elif provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("'GOOGLE_API_KEY' is not present in .env")
        return GoogleGenerativeAIEmbeddings(
            google_api_key = api_key,
            model = config["embedding"]["gemini_model"]
        )
    else:
        raise ValueError("Provider must be one of: openai | gemini")

# ----------------------------------------
# Load Chat LLM (OpenAI or Gemini)
# ----------------------------------------
def get_llm(config: dict):
    provider = config["llm"]["provider"]
    temperature = config["llm"]["temperature"]

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("'OPENAI_API_KEY' is not present in .env")
        return ChatOpenAI(
            api_key = api_key, 
            model = config["llm"]["model_openai"], 
            temperature = temperature
        )
    elif provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("'GOOGLE_API_KEY' is not present in .env")
        return ChatGoogleGenerativeAI(
            google_api_key = api_key, 
            model = config["llm"]["model_gemini"], 
            temperature = temperature
        )
    else:
        raise ValueError("Provider must be one of: openai | gemini")

# ----------------------------------------
# Unify Documents
# ----------------------------------------
def unify_docs(docs_or_pairs):
    """
    Accepts either:
      - list[Document]
      - list[(Document, score)]
    Returns list[Document]
    """
    if not docs_or_pairs:
        return []
    if hasattr(docs_or_pairs[0], "page_content"):
        return docs_or_pairs
    return [d for d, _ in docs_or_pairs]

# ----------------------------------------
# Prepare Context
# ----------------------------------------
def make_context(docs, max_chars = 3000):
    """
    Concatenate page_content into a single context string,
    truncated to max_chars for cost/speed while demoing.
    """
    text = "\n\n---\n\n".join(d.page_content.strip() for d in docs)
    return text[:max_chars]