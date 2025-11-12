"""Embedding generation for review text using sentence-transformers"""

from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingGenerator:
    """Generate embeddings for review text"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize embedding model (384-dim, fast, good quality)"""
        self.model = SentenceTransformer(model_name)
        self.dimension = 384
        print(f"Loaded embedding model: {model_name}")
    
    def embed_text(self, text):
        """Generate embedding for single text"""
        if not text or len(text.strip()) == 0:
            return np.zeros(self.dimension)
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts):
        """Generate embeddings for batch of texts"""
        # Handle empty texts
        processed_texts = [t if t and len(t.strip()) > 0 else " " for t in texts]
        embeddings = self.model.encode(
            processed_texts, 
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )
        return embeddings
    
    def embed_reviews(self, df):
        """Generate embeddings for all reviews with text"""
        # Only embed reviews that have text
        text_reviews = df[df['has_text']]
        
        if len(text_reviews) == 0:
            return np.array([])
        
        print(f"Generating embeddings for {len(text_reviews)} reviews...")
        texts = text_reviews['caption'].tolist()
        embeddings = self.embed_batch(texts)
        
        return embeddings, text_reviews

