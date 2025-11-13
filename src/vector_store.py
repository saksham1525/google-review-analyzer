"""ChromaDB vector store for review embeddings"""

import chromadb
import uuid


class ReviewVectorStore:
    """Manages ChromaDB for storing and querying review embeddings"""
    
    def __init__(self, persist_directory="./chroma_db"):
        """Initialize ChromaDB client"""
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=chromadb.Settings(anonymized_telemetry=False)
        )
        self.collection = None
        print(f"ChromaDB initialized at: {persist_directory}")
    
    def create_collection(self, collection_name="reviews"):
        """Create collection and delete existing if present"""
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        self.collection = self.client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
        print(f"Created collection: {collection_name}")
        return self.collection
    
    def add_reviews(self, embeddings, reviews_df):
        """Add review embeddings with metadata to ChromaDB"""
        if not self.collection:
            raise ValueError("Collection not created. Call create_collection() first.")
        
        metadatas = [{
            'rating': float(row['rating']),
            'sentiment': str(row.get('sentiment', 'UNKNOWN')),
            'username': str(row['username']),
            'relative_date': str(row.get('relative_date', '')),
            'text_length': int(row.get('text_length', 0))
        } for _, row in reviews_df.iterrows()]
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=reviews_df['caption'].tolist(),
            metadatas=metadatas,
            ids=[str(uuid.uuid4()) for _ in range(len(reviews_df))]
        )
        print(f"Added {len(reviews_df)} reviews to vector store")
    
    def search(self, query_embedding, top_k=15, filters=None):
        """Search for similar reviews"""
        if not self.collection:
            raise ValueError("Collection not created.")
        
        where = {}
        if filters:
            if 'rating' in filters:
                where['rating'] = {'$gte': filters['rating']}
            if 'sentiment' in filters:
                where['sentiment'] = filters['sentiment']
        
        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where or None
        )
    
    def get_collection_stats(self):
        """Get collection statistics"""
        return {"count": self.collection.count() if self.collection else 0}
