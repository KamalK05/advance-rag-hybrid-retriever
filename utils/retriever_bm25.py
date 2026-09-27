# utils/retrieve_bm25.py
from rank_bm25 import BM25Okapi
import re
from utils.common_utils import load_config
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------
# Load Document and Create Chunks
# ----------------------------------------
def simple_tokenizer(text: str):
    """
    Basic tokenizer suitable for BM25. 
    You can enhance later (stopwords, lemmatization, etc.).
    """
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    '''
    This line returns a list of all words in text, ignoring punctuation and splitting on whitespace or non-word characters. 
    For example, "Hello, world!" → ["hello", "world"] (after lowercasing).
    '''
    return tokens

# ----------------------------------------
# BM25 Retriever Class
# ----------------------------------------
class BM25Retriever:
    def __init__(self, documents, config_path = "config.yaml"):
        self.config  = load_config(config_path)
        self.documents = documents
        self.corpus = [simple_tokenizer(doc.page_content) for doc in documents]

        self.bm25 = BM25Okapi(
            self.corpus,
            k1 = self.config["bm25"]["k1"],
            b = self.config["bm25"]["b"],
        )

# ----------------------------------------
# Retrieve Documents
# ----------------------------------------
    def get_top_k(self, query, k = None):
        if k is None:
            k = self.config["hybrid"]["k_sparse"]

        query_tokens = simple_tokenizer(query)

        scores = self.bm25.get_scores(query_tokens)

        top_k_idx = np.argsort(scores)[::-1][:k]
        '''
        scores = np.array([0.2, 0.8, 0.5, 0.1])
        np.argsort(scores) → array([3, 0, 2, 1])
        np.argsort(scores)[::-1] → array([1, 2, 0, 3])
        np.argsort(scores)[::-1][:2] → array([1, 2])
        '''

        results = []
        for idx in top_k_idx:
            results.append((self.documents[idx], float(scores[idx])))

        return results
