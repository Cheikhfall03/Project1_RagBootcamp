# Working imports for chain.py - Replace your problematic imports with these:

from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    LLMListwiseRerank,  # This is the reranking equivalent
    EmbeddingsFilter,   # Alternative fast reranking
    DocumentCompressorPipeline  # For combining multiple compressors
)
from sentence_transformers import CrossEncoder
from app.config import GROQ_API_KEY

# Option 1: Custom CrossEncoder Reranker (Recommended)
from typing import List, Sequence
from langchain_core.documents import Document
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor

class CustomCrossEncoderReranker(BaseDocumentCompressor):
    """Custom CrossEncoder reranker using sentence_transformers"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", top_n: int = 3):
        self.model = CrossEncoder(model_name)
        self.top_n = top_n
    
    def compress_documents(
        self, 
        documents: Sequence[Document], 
        query: str, 
        callbacks=None
    ) -> List[Document]:
        """Rerank documents using CrossEncoder"""
        if not documents:
            return []
        
        # Create pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Sort documents by score (descending)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_n documents
        return [doc for doc, score in scored_docs[:self.top_n]]

# Option 2: Using LLMListwiseRerank (Built-in LangChain reranker)
def create_llm_reranker(llm, top_n=3):
    """Create LLM-based reranker (requires a chat model with structured output)"""
    return LLMListwiseRerank.from_llm(llm, top_n=top_n)

# Option 3: Using EmbeddingsFilter (Fast alternative)
def create_embeddings_reranker(embeddings, similarity_threshold=0.76):
    """Create embeddings-based filter"""
    return EmbeddingsFilter(embeddings=embeddings, similarity_threshold=similarity_threshold)

# Complete reranking function that you can use in your chain
def rerank_documents(documents, query, method="custom", **kwargs):
    """
    Rerank documents using different methods
    
    Args:
        documents: List of documents to rerank
        query: Query string
        method: "custom", "llm", or "embeddings"
        **kwargs: Additional parameters for specific methods
    """
    
    if method == "custom":
        # Use custom CrossEncoder reranker
        model_name = kwargs.get("model_name", "BAAI/bge-reranker-base")
        top_n = kwargs.get("top_n", 3)
        reranker = CustomCrossEncoderReranker(model_name=model_name, top_n=top_n)
        return reranker.compress_documents(documents, query)
    
    elif method == "llm":
        # Use LLM-based reranker (requires chat model)
        llm = kwargs.get("llm")
        top_n = kwargs.get("top_n", 3)
        if llm is None:
            raise ValueError("LLM is required for llm method")
        reranker = LLMListwiseRerank.from_llm(llm, top_n=top_n)
        return reranker.compress_documents(documents, query)
    
    elif method == "embeddings":
        # Use embeddings-based filter
        embeddings = kwargs.get("embeddings")
        similarity_threshold = kwargs.get("similarity_threshold", 0.76)
        if embeddings is None:
            raise ValueError("Embeddings model is required for embeddings method")
        reranker = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=similarity_threshold)
        return reranker.compress_documents(documents, query)
    
    else:
        raise ValueError(f"Unknown reranking method: {method}")

# Example usage for your retrieve_hybrid_docs function
def retrieve_hybrid_docs(vectorstore, bm25_retriever, query, k=10, rerank_top_n=3):
    """
    Retrieve documents using hybrid search and rerank them
    """
    # Create ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vectorstore.as_retriever(search_kwargs={"k": k}), bm25_retriever],
        weights=[0.5, 0.5]
    )
    
    # Retrieve documents
    docs = ensemble_retriever.invoke(query)
    
    # Rerank using custom CrossEncoder
    reranked_docs = rerank_documents(docs, query, method="custom", top_n=rerank_top_n)
    
    return reranked_docs

# Alternative: Using ContextualCompressionRetriever (LangChain way)
def create_contextual_compression_retriever(base_retriever, rerank_method="custom", **kwargs):
    """
    Create a contextual compression retriever with reranking
    """
    if rerank_method == "custom":
        model_name = kwargs.get("model_name", "BAAI/bge-reranker-base")
        top_n = kwargs.get("top_n", 3)
        compressor = CustomCrossEncoderReranker(model_name=model_name, top_n=top_n)
    
    elif rerank_method == "llm":
        llm = kwargs.get("llm")
        top_n = kwargs.get("top_n", 3)
        compressor = LLMListwiseRerank.from_llm(llm, top_n=top_n)
    
    elif rerank_method == "embeddings":
        embeddings = kwargs.get("embeddings")
        similarity_threshold = kwargs.get("similarity_threshold", 0.76)
        compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=similarity_threshold)
    
    else:
        raise ValueError(f"Unknown rerank method: {rerank_method}")
    
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
