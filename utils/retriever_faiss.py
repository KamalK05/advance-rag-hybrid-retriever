# utils/retrieve_faiss.py
from langchain_community.vectorstores import FAISS
from utils.common_utils import load_config, get_embedding_model
import os

import warnings
warnings.filterwarnings("ignore")

INDEX_NAME = "index"  # langchain FAISS default -> creates index.faiss + index.pkl

# -----------------------------------------------
# Get Index Path
# -----------------------------------------------
def get_index_path(provider, config: dict) -> str:
    return (
        config["vectordb"]["faiss_openai"]
        if provider == "openai"
        else config["vectordb"]["faiss_gemini"]
    )

# -----------------------------------------------
# Verify if Index File Exists
# -----------------------------------------------
def _is_index_file_exists(index_path: str, index_name: str = INDEX_NAME) -> bool:
    faiss_path = os.path.join(index_path, f"{index_name}.faiss")
    pkl_path = os.path.join(index_path, f"{index_name}.pkl")
    return os.path.exists(faiss_path) and os.path.exists(pkl_path)

# -----------------------------------------------
# Load Documents into FAISS and Create Retriever
# -----------------------------------------------
def create_retriever(chunks, provider: str, config: dict):
    """Build a fresh FAISS index from chunks."""
    index_path = get_index_path(provider, config)
    embedding_model  = get_embedding_model(config)

    os.makedirs(index_path, exist_ok = True)

    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(index_path, index_name = INDEX_NAME)

    print(f"✅ Created FAISS index ({provider}) at: {index_path}")
    return vectorstore.as_retriever(search_kwargs={"k": config["retrieval"]["top_k"]})

# -----------------------------------------------
# Load Documents From FAISS and Return Retriever
# -----------------------------------------------
def load_retriever(provider: str, config: dict):
    """Load an existing FAISS index."""
    index_path = get_index_path(provider, config)
    embedding_model  = get_embedding_model(config)

    vectorstore = FAISS.load_local(
        index_path,
        embedding_model,
        index_name = INDEX_NAME,
        allow_dangerous_deserialization = True
    )
    print(f"✅ Loaded FAISS index ({provider}) from: {index_path}")
    return vectorstore.as_retriever(search_kwargs={"k": config["retrieval"]["top_k"]})

# -----------------------------------------------
# Get Retriever (CREATE or LOAD)
# -----------------------------------------------
def get_retriever(config_path = "config.yaml", chunks_if_needed = None):
    """
    Robust loader:
      - If index files missing -> create (requires chunks)
      - If load fails -> rebuild (if chunks provided), else raise a clear error
    """
    config = load_config(config_path)
    provider = config["llm"]["provider"]
    index_path = get_index_path(provider, config)

    if not _is_index_file_exists(index_path):
        print(f"⚠️ No FAISS index found for '{provider}' at {index_path}. Creating a new one...")
        if chunks_if_needed is None:
            raise RuntimeError(
                f"FAISS files not found at {index_path}. "
                f"Please pass chunks_if_needed to build the index."
            )
        return create_retriever(chunks_if_needed, provider, config)
    else:
        try:
            return load_retriever(provider, config)
        except Exception as e:
            print(f"⚠️ Failed to load FAISS index at {index_path}: {e}")
            if chunks_if_needed is None:
                raise RuntimeError(
                    "Existing FAISS index appears corrupted or incompatible, "
                    "and no chunks were provided to rebuild it. "
                    "Please delete the folder and rerun with chunks."
                )
            print("🔧 Rebuilding FAISS index from chunks...")
            return create_retriever(chunks_if_needed, provider, config)





