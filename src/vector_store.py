"""ChromaDB vector store for review embeddings"""

import chromadb
from chromadb.config import Settings
import uuid


class ReviewVectorStore:
    """Manages ChromaDB for storing and querying review embeddings"""
    
    def __init__(self, persist_directory="./chroma_db"):
        """Initialize ChromaDB client"""
        self.persist_directory = persist_directory
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        self.collection = None
        print(f"ChromaDB initialized at: {persist_directory}")
    
    def create_collection(self, collection_name="reviews"):
        """Create or get collection"""
        # Delete existing collection if it exists
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        print(f"Created collection: {collection_name}")
        return self.collection
    
    def add_reviews(self, embeddings, reviews_df):
        """Add review embeddings to ChromaDB with metadata"""
        if self.collection is None:
            raise ValueError("Collection not created. Call create_collection() first.")
        
        # Prepare data for ChromaDB
        ids = [str(uuid.uuid4()) for _ in range(len(reviews_df))]
        documents = reviews_df['caption'].tolist()
        
        # Prepare metadata (ChromaDB requires all values to be strings, ints, or floats)
        metadatas = []
        for _, row in reviews_df.iterrows():
            metadata = {
                'rating': float(row['rating']),
                'sentiment': str(row.get('sentiment', 'UNKNOWN')),
                'username': str(row['username']),
                'relative_date': str(row.get('relative_date', '')),
                'text_length': int(row.get('text_length', 0))
            }
            metadatas.append(metadata)
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(ids)} reviews to vector store")
    
    def search(self, query_embedding, top_k=15, filters=None):
        """Search for similar reviews"""
        if self.collection is None:
            raise ValueError("Collection not created.")
        
        # Build where clause for filtering
        where_clause = None
        if filters:
            where_clause = {}
            if 'rating' in filters:
                where_clause['rating'] = {'$gte': filters['rating']}
            if 'sentiment' in filters:
                where_clause['sentiment'] = filters['sentiment']
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_clause if where_clause else None
        )
        
        return results
    
    def get_collection_stats(self):
        """Get statistics about the collection"""
        if self.collection is None:
            return {"count": 0}
        
        count = self.collection.count()
        return {"count": count}

