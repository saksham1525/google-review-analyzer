"""RAG Pipeline for review question answering"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import os
import time


class RAGPipeline:
    """Orchestrates RAG query: Question → Retrieve → Generate Answer"""
    
    def __init__(self, vector_store, embedder):
        """Initialize RAG pipeline with vector store and embedder"""
        self.vector_store = vector_store
        self.embedder = embedder
        
        # Initialize Gemini with LangChain
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv('GOOGLE_API_KEY'),
            temperature=0.7,
            timeout=120,  # 2 min timeout
            max_retries=3,  # Built-in retry
            convert_system_message_to_human=True
        )
        print("RAG Pipeline initialized with Gemini-2.5-Flash")
    
    def _format_context(self, search_results, max_reviews=15):
        """Format retrieved reviews for LLM"""
        docs, metas = search_results['documents'][0], search_results['metadatas'][0]
        return "\n\n".join([
            f"Review {i+1} [{meta.get('rating', 'N/A')}★, {meta.get('sentiment', 'UNKNOWN')}]: {doc[:300]}"
            for i, (doc, meta) in enumerate(zip(docs[:max_reviews], metas[:max_reviews]))
        ])
    
    def query(self, question, top_k=15, filters=None, df_stats=None):
        """Execute RAG: retrieve relevant reviews and generate answer"""
        # Retrieve
        query_embedding = self.embedder.embed_text(question)
        results = self.vector_store.search(query_embedding, top_k, filters)
        
        if not results['documents'][0]:
            return "I couldn't find relevant reviews to answer your question."
        
        # Format context
        context = self._format_context(results, top_k)
        stats_text = ""
        if df_stats:
            stats_text = f"Stats: {df_stats['total']} reviews | Avg: {df_stats['avg_rating']:.1f}/5 | {df_stats['positive']} Positive, {df_stats['neutral']} Neutral, {df_stats['negative']} Negative\n\n"
        
        # Generate answer
        system = SystemMessage(content="You are an expert at analyzing restaurant reviews. Answer based on provided reviews. Be specific and concise.")
        human = HumanMessage(content=f"{stats_text}REVIEWS:\n{context}\n\nQUESTION: {question}\n\nANSWER:")
        
        return self.llm.invoke([system, human]).content
