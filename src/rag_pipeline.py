"""RAG Pipeline for review question answering"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import os


class RAGPipeline:
    """Orchestrates RAG query: Question → Retrieve → Generate Answer"""
    
    def __init__(self, vector_store, embedder):
        """Initialize RAG pipeline with vector store and embedder"""
        self.vector_store = vector_store
        self.embedder = embedder
        
        # Initialize Gemini with LangChain
        api_key = os.getenv('GOOGLE_API_KEY')
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.7
        )
        print("RAG Pipeline initialized with Gemini-2.5-Flash")
    
    def retrieve_relevant_reviews(self, question, top_k=15, filters=None):
        """Retrieve most relevant reviews for a question"""
        # Generate query embedding
        query_embedding = self.embedder.embed_text(question)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )
        
        return results
    
    def format_context(self, search_results, max_reviews=15):
        """Format retrieved reviews into context for LLM"""
        documents = search_results['documents'][0]
        metadatas = search_results['metadatas'][0]
        
        context = []
        for i, (doc, meta) in enumerate(zip(documents[:max_reviews], metadatas[:max_reviews])):
            rating = meta.get('rating', 'N/A')
            sentiment = meta.get('sentiment', 'UNKNOWN')
            review_text = doc[:300] if len(doc) > 300 else doc
            context.append(f"Review {i+1} [{rating}★, {sentiment}]: {review_text}")
        
        return "\n\n".join(context)
    
    def query(self, question, top_k=15, filters=None, df_stats=None):
        """
        Execute RAG query: retrieve relevant reviews and generate answer
        
        Args:
            question: User's question
            top_k: Number of reviews to retrieve (default 15)
            filters: Optional filters (e.g., {'rating': 5, 'sentiment': 'POSITIVE'})
            df_stats: Optional dictionary with overall stats
        
        Returns:
            str: Generated answer based on retrieved reviews
        """
        # Retrieve relevant reviews
        results = self.retrieve_relevant_reviews(question, top_k, filters)
        
        if not results['documents'][0]:
            return "I couldn't find any relevant reviews to answer your question."
        
        # Format context
        context = self.format_context(results, max_reviews=top_k)
        
        # Create prompt
        system_prompt = SystemMessage(content="""You are an expert at analyzing restaurant reviews. 
Answer questions based on the provided reviews. Be specific, cite details, and keep responses concise.""")
        
        # Add stats if provided
        stats_text = ""
        if df_stats:
            stats_text = f"""STATISTICS:
Total Reviews: {df_stats.get('total', 'N/A')}
Average Rating: {df_stats.get('avg_rating', 'N/A'):.1f}/5
Sentiment: {df_stats.get('positive', 0)} Positive, {df_stats.get('neutral', 0)} Neutral, {df_stats.get('negative', 0)} Negative

"""
        
        human_prompt = HumanMessage(content=f"""{stats_text}REVIEWS:
{context}

QUESTION: {question}

ANSWER:""")
        
        # Generate answer
        response = self.llm.invoke([system_prompt, human_prompt])
        return response.content
