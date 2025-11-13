"""Embedding generation for review text using sentence-transformers"""

from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingGenerator:
    """Generate embeddings for review text"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        print(f"Loaded embedding model: {model_name}")
    
    def embed_text(self, text):
        """Generate embedding for single text"""
        if not text or not text.strip():
            return np.zeros(384)
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts):
        """Generate embeddings for batch of texts"""
        processed = [t if t and t.strip() else " " for t in texts]
        return self.model.encode(processed, convert_to_numpy=True, show_progress_bar=True, batch_size=32)
    
    def embed_reviews(self, df):
        """Generate embeddings for all reviews with text"""
        text_reviews = df[df['has_text']]
        if len(text_reviews) == 0:
            return np.array([]), text_reviews
        
        print(f"Generating embeddings for {len(text_reviews)} reviews...")
        return self.embed_batch(text_reviews['caption'].tolist()), text_reviews
