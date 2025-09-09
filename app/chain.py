from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableMap, RunnablePassthrough, RunnableLambda
from langchain.schema.runnable import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.vectorstore_simple import store_chunks, get_vectorstore, get_bm25_retriever
from langchain.retrievers import EnsembleRetriever
#from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from sentence_transformers import CrossEncoder
#from langchain.retrievers.document_compressors import CrossEncoderReranker
from app.config import GROQ_API_KEY


from typing import List
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever

def retrieve_hybrid_docs(query: str, vectorstore, bm25_retriever, top_k: int = 5) -> List[Document]:
    """
    Optimized hybrid document retrieval with proper error handling and modern API usage.
    """
    try:
        # Get documents using modern invoke() method
        semantic_docs = vectorstore.similarity_search(query, k=top_k)
        keyword_docs = bm25_retriever.invoke(query)[:top_k]
        
        # Create optimized ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[
                vectorstore.as_retriever(search_kwargs={"k": top_k}),
                bm25_retriever
            ],
            weights=[0.5, 0.5]
        )
        
        # Get combined results
        ensemble_docs = ensemble_retriever.invoke(query)[:top_k]
        
        # Efficient deduplication while preserving relevance order
        seen_contents = set()
        unique_docs = []
        for doc in semantic_docs + keyword_docs + ensemble_docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                unique_docs.append(doc)
                
        return unique_docs[:top_k*2]  # Return slightly more than requested
    
    except Exception as e:
        print(f"Error in hybrid retrieval: {e}")
        # Fallback to simple semantic search if hybrid fails
        return vectorstore.similarity_search(query, k=top_k)


def rerank_documents(query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
    """
    Stable implementation of document reranking with proper device handling.
    """
    try:
        # First try with explicit CPU device and memory optimizations
        model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu",
            automodel_args={
                'low_cpu_mem_usage': True,
                'torch_dtype': torch.float32  # Explicit dtype
            }
        )
        
        pairs = [[query, doc.page_content] for doc in documents]
        scores = model.predict(pairs)
        
        # Combine and sort documents by score
        scored_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return [doc for doc, _ in scored_docs[:top_k]]
        
    except Exception as e:
        print(f"Reranking failed: {e}")
        # Fallback: return first top_k documents
        return documents[:top_k]


def build_llm_chain():
    """
    Builds a streaming LLM chain using LangChain Runnables with Groq LLaMA model.
    """

    # Prompt template to inject context + user question
    prompt = PromptTemplate.from_template("""
    You are a helpful AI assistant. Answer the question based only on the context below.

    Context:
    {context}

    Question:
    {question}

    Answer:""")

    # Set up Groq's LLaMA 3 model with streaming enabled
    llm = ChatGroq(
        model="gemma2-9b-it",
        api_key=GROQ_API_KEY,
        streaming=True
    )

    # Chain: prepare prompt → pass to LLM → parse output
    chain = (
        RunnableMap({
            "context": lambda x: "\n\n".join([doc.page_content for doc in x["docs"]]),
            "question": lambda x: x["question"]
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
