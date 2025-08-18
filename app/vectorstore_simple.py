import os
import shutil
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
import pysqlite3
# Use the ChromaDB wrapper instead of direct import
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from chroma_wrapper_py import Chroma, Settings
# Try to import different embedding options
try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Persistent directory for Chroma vectorstore
PERSIST_DIR = "./chromastore"

# Force CPU usage and disable CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["NO_CUDA"] = "1"

def check_vectorstore_exists() -> bool:
    """
    Checks if the Chroma vectorstore exists in the persistent directory.
    
    Returns:
        bool: True if vectorstore exists, False otherwise
    """
    return os.path.exists(PERSIST_DIR) and len([f for f in os.listdir(PERSIST_DIR) if not f.startswith('.')]) > 0

def reset_vectorstore():
    """Reset the vectorstore by removing the directory"""
    if os.path.exists(PERSIST_DIR):
        try:
            shutil.rmtree(PERSIST_DIR)
            print("🗑️ Vectorstore reset successfully")
        except Exception as e:
            print(f"⚠️ Could not reset vectorstore: {e}")

def create_embeddings():
    """Create embeddings with fallback options"""
    
    # Option 1: Try OpenAI embeddings (most reliable)
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        try:
            print("🔄 Using OpenAI embeddings...")
            return OpenAIEmbeddings(model="text-embedding-3-small")
        except Exception as e:
            print(f"❌ OpenAI embeddings failed: {e}")
    
    # Option 2: Try HuggingFace embeddings with fixes
    if HUGGINGFACE_AVAILABLE:
        try:
            print("🔄 Using HuggingFace embeddings...")
            # Force CPU usage
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Create embeddings with minimal configuration
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': False,
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 1
                }
            )
            print("✅ HuggingFace embeddings created successfully")
            return embeddings
            
        except Exception as e:
            print(f"❌ HuggingFace embeddings failed: {e}")
            
            # Try alternative smaller model
            try:
                print("🔄 Trying alternative HuggingFace model...")
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
                    model_kwargs={'device': 'cpu'}
                )
                print("✅ Alternative HuggingFace model loaded successfully")
                return embeddings
            except Exception as e2:
                print(f"❌ Alternative model also failed: {e2}")
    
    # Option 3: Use fake embeddings as last resort
    print("⚠️ Using fake embeddings for testing - not suitable for production!")
    from langchain_community.embeddings import FakeEmbeddings
    return FakeEmbeddings(size=384)

def store_chunks(chunks: List[Document]):
    """Stores document chunks in ChromaDB with embeddings."""
    try:
        # Create embeddings
        embeddings = create_embeddings()
        if embeddings is None:
            raise Exception("Failed to create any embeddings model")
        
        # Check if vectorstore exists and is valid
        vectorstore_exists = check_vectorstore_exists()
        
        if vectorstore_exists:
            try:
                print("📂 Loading existing vectorstore...")
                vectorstore = Chroma(
                    persist_directory=PERSIST_DIR,
                    embedding_function=embeddings
                )
                # Test the connection
                vectorstore._collection.count()
                vectorstore.add_documents(chunks)
                print(f"✅ Added {len(chunks)} chunks to existing vectorstore")
            except Exception as e:
                print(f"❌ Error with existing vectorstore: {e}")
                print("🔄 Resetting and creating new vectorstore...")
                reset_vectorstore()
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=PERSIST_DIR
                )
                print(f"✅ Created new vectorstore with {len(chunks)} chunks")
        else:
            # Create new vectorstore
            print("🆕 Creating new vectorstore...")
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=PERSIST_DIR
            )
            print(f"✅ Created new vectorstore with {len(chunks)} chunks")
        
        return vectorstore
        
    except Exception as e:
        print(f"❌ Error in store_chunks: {e}")
        # Try to reset and recreate
        try:
            print("🔄 Attempting to reset and recreate vectorstore...")
            reset_vectorstore()
            embeddings = create_embeddings()
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=PERSIST_DIR
            )
            print(f"✅ Successfully recreated vectorstore with {len(chunks)} chunks")
            return vectorstore
        except Exception as e2:
            print(f"❌ Final attempt failed: {e2}")
            raise e2

def get_bm25_retriever(chunks: List[Document]):
    """Initializes BM25Retriever with proper configuration."""
    try:
        if not chunks:
            print("⚠️ No chunks provided for BM25, creating empty retriever")
            # Create a dummy document for BM25 initialization
            dummy_doc = Document(page_content="dummy", metadata={})
            return BM25Retriever.from_documents([dummy_doc], k=4)
        return BM25Retriever.from_documents(chunks, k=4)
    except Exception as e:
        print(f"❌ Error creating BM25 retriever: {e}")
        return None

def get_vectorstore():
    """Loads existing Chroma vectorstore."""
    try:
        if not check_vectorstore_exists():
            print("❌ No existing vectorstore found")
            return None
            
        embeddings = create_embeddings()
        if embeddings is None:
            raise Exception("Failed to create embeddings model")
        
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
        
        # Test the connection
        try:
            count = vectorstore._collection.count()
            print(f"✅ Loaded existing vectorstore with {count} documents")
            return vectorstore
        except Exception as e:
            print(f"❌ Vectorstore connection test failed: {e}")
            print("🔄 Resetting corrupted vectorstore...")
            reset_vectorstore()
            return None
        
    except Exception as e:
        print(f"❌ Error loading vectorstore: {e}")
        return None
