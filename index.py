import sys
import os


# Now import everything else
import streamlit as st
from app.loaders import load_and_chunk_pdf
from app.vectorstore_simple import (
    store_chunks, 
    get_vectorstore, 
    get_bm25_retriever,  # Add this
    check_vectorstore_exists  # And this if needed
)
from app.chain import build_llm_chain, retrieve_hybrid_docs, rerank_documents
from app.pdf_handler import upload_pdfs

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["NO_CUDA"] = "1"

# DOIT ÊTRE LA PREMIÈRE COMMANDE STREAMLIT
st.set_page_config(
    page_title="📄 ZoomYourQueryAI - Chat with PDF", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensuite seulement vous pouvez mettre le reste du code
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["NO_CUDA"] = "1"

# ===========================================
# CONFIGURATION API (déplacé après set_page_config)
# ===========================================
st.sidebar.title("🔑 Configuration API")

# Section pour entrer la clé Groq
groq_api_key = st.sidebar.text_input(
    "Entrez votre clé API Groq",
    type="password",
    help="Obtenez votre clé sur https://console.groq.com/",
    key="groq_api_key_input"
)

if not groq_api_key:
    st.sidebar.warning("Veuillez entrer votre clé API pour utiliser l'application")
    st.stop()
else:
    os.environ["GROQ_API_KEY"] = groq_api_key

st.sidebar.success("Clé API configurée avec succès!")
# CSS Professionnel avec animations simplifiées
st.markdown("""
<style>
/* Police moderne et lisible */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Variables CSS pour la cohérence */
:root {
    --primary: #4361ee;
    --primary-dark: #3a0ca3;
    --secondary: #4895ef;
    --accent: #f72585;
    --light: #f8f9fa;
    --dark: #212529;
    --gray: #6c757d;
    --light-gray: #e9ecef;
    --success: #4cc9f0;
    --warning: #f8961e;
    --danger: #ef233c;
    --border-radius: 12px;
    --box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    --transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}

/* Reset et styles de base */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    line-height: 1.6;
    color: var(--dark);
    background-color: #f5f7fa;
}

/* Conteneur principal */
.stApp {
    background-color: #f5f7fa;
    min-height: 100vh;
}

/* En-tête */
.main-header {
    text-align: center;
    padding: 2rem 1rem 3rem;
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    color: white;
    margin-bottom: 2rem;
    border-radius: 0 0 20px 20px;
    box-shadow: var(--box-shadow);
}

.main-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.main-header p {
    font-size: 1.1rem;
    opacity: 0.9;
    max-width: 700px;
    margin: 0 auto;
}

/* Cartes */
.card {
    background: white;
    border-radius: var(--border-radius);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--box-shadow);
    transition: var(--transition);
    border: 1px solid rgba(0, 0, 0, 0.05);
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
}

.card-title {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: var(--primary);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Zone de téléchargement */
.upload-zone {
    border: 2px dashed var(--gray);
    border-radius: var(--border-radius);
    padding: 2rem;
    text-align: center;
    background: rgba(255, 255, 255, 0.5);
    transition: var(--transition);
    cursor: pointer;
    margin-bottom: 1.5rem;
}

.upload-zone:hover {
    border-color: var(--primary);
    background: rgba(67, 97, 238, 0.05);
}

/* Messages */
.message {
    padding: 1rem;
    border-radius: var(--border-radius);
    margin-bottom: 1rem;
    font-weight: 500;
}

.message-success {
    background-color: rgba(76, 201, 240, 0.1);
    color: #0a9396;
    border-left: 4px solid var(--success);
}

.message-info {
    background-color: rgba(72, 149, 239, 0.1);
    color: var(--secondary);
    border-left: 4px solid var(--secondary);
}

.message-warning {
    background-color: rgba(248, 150, 30, 0.1);
    color: var(--warning);
    border-left: 4px solid var(--warning);
}

.message-error {
    background-color: rgba(239, 35, 60, 0.1);
    color: var(--danger);
    border-left: 4px solid var(--danger);
}

/* Section de question/réponse */
.question-section {
    margin-bottom: 2rem;
}

.answer-section {
    background: white;
    border-radius: var(--border-radius);
    padding: 1.5rem;
    margin-top: 1rem;
    box-shadow: var(--box-shadow);
    border-left: 4px solid var(--primary);
}

/* Inputs */
.stTextInput>div>div>input {
    border-radius: var(--border-radius) !important;
    padding: 0.75rem 1rem !important;
    border: 1px solid var(--light-gray) !important;
    transition: var(--transition) !important;
}

.stTextInput>div>div>input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(67, 97, 238, 0.2) !important;
}

/* Boutons */
.stButton>button {
    border-radius: var(--border-radius) !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 500 !important;
    transition: var(--transition) !important;
    background-color: var(--primary) !important;
}

.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 15px rgba(67, 97, 238, 0.3) !important;
}

/* Métriques */
.metric {
    text-align: center;
    padding: 1rem;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary);
    margin-bottom: 0.25rem;
}

.metric-label {
    color: var(--gray);
    font-size: 0.9rem;
}

/* Pied de page */
.footer {
    text-align: center;
    padding: 2rem 1rem;
    margin-top: 3rem;
    color: var(--gray);
    font-size: 0.9rem;
    border-top: 1px solid var(--light-gray);
}

/* Animation de chargement */
@keyframes spin {
    to { transform: rotate(360deg); }
}

.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(67, 97, 238, 0.2);
    border-radius: 50%;
    border-top-color: var(--primary);
    animation: spin 1s ease-in-out infinite;
    margin-right: 8px;
}

/* Responsive */
@media (max-width: 768px) {
    .main-header h1 {
        font-size: 2rem;
    }
    
    .main-header p {
        font-size: 1rem;
    }
    
    .card {
        padding: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)

# Custom header
st.markdown("""
<div class="main-header">
    <h1>🚀 ZoomYourQueryAI</h1>
    <p>Intelligent Document Analysis with Advanced AI</p>
</div>
""", unsafe_allow_html=True)

UPLOAD_DIR = "uploaded_files"
#PERSIST_DIR = "./chroma_store"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# INITIALISATION DE L'ÉTAT STREAMLIT
if 'vectorstore_initialized' not in st.session_state:
    st.session_state.vectorstore_initialized = False
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'bm25' not in st.session_state:
    st.session_state.bm25 = None
if 'current_model_info' not in st.session_state:
    st.session_state.current_model_info = {
        'llm': 'Not Initialized',
        'embeddings': 'Not Initialized',
        'vector_db': 'ChromaDB',
        'reranker': 'Not Initialized'
    }

# NOUVELLE LIGNE :
PERSIST_DIR = "./faiss_store"

# Dans la fonction update_model_info(), remplacez :
# 'vector_db': 'ChromaDB',
# PAR :
# 'vector_db': 'FAISS',

# Voici la fonction update_model_info() mise à jour complète :
def update_model_info():
    """Met à jour les informations sur les modèles utilisés"""
    model_info = {
        'llm': 'Not Available',
        'embeddings': 'Not Available',
        'vector_db': 'FAISS',  # ← Changement ici
        'reranker': 'Simple Ranking'
    }
    
    # Détection du LLM utilisé
    if 'GROQ_API_KEY' in os.environ:
        model_info['llm'] = 'Groq (Llama3)'
    else:
        try:
            from app.chain import OPENAI_AVAILABLE, OLLAMA_AVAILABLE
            if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
                model_info['llm'] = 'GPT-3.5-turbo'
            elif OLLAMA_AVAILABLE:
                model_info['llm'] = 'Llama3.2:3b'
        except:
            model_info['llm'] = 'Not Available'
    
    # Détection des embeddings utilisés
    try:
        from app.vectorstore_simple import OPENAI_AVAILABLE as EMB_OPENAI, HUGGINGFACE_AVAILABLE
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            model_info['embeddings'] = 'text-embedding-3-small'
        elif HUGGINGFACE_AVAILABLE:
            model_info['embeddings'] = 'all-MiniLM-L6-v2'
        else:
            model_info['embeddings'] = 'FakeEmbeddings (Test)'
    except:
        model_info['embeddings'] = 'HuggingFace (Default)'
    
    # Détection du reranker
    try:
        from app.chain import CROSSENCODER_AVAILABLE
        if CROSSENCODER_AVAILABLE:
            model_info['reranker'] = 'ms-marco-MiniLM-L-6-v2'
    except:
        model_info['reranker'] = 'Simple Ranking'
    
    st.session_state.current_model_info = model_info
    return model_info
def check_existing_vectorstore():
    """Vérifie si un vectorstore existe déjà"""
    return os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR)

def initialize_vectorstore():
    """Initialise le vectorstore depuis les données existantes"""
    try:
        print("🔄 Initialisation du vectorstore...")
        if check_vectorstore_exists():
            print("📂 Vectorstore trouvé, chargement...")
            vectorstore = get_vectorstore()
            if vectorstore is not None:
                st.session_state.vectorstore = vectorstore
                # Charger BM25 avec les chunks en cache
                bm25 = get_bm25_retriever([])
                if bm25 is not None:
                    st.session_state.bm25 = bm25
                    st.session_state.vectorstore_initialized = True
                    print("✅ Vectorstore initialisé avec succès")
                    return True
        print("❌ Aucun vectorstore trouvé")
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        st.error(f"Erreur lors du chargement du vectorstore existant: {str(e)}")
    return False

# Mettre à jour les informations des modèles au démarrage
update_model_info()

# Create columns for better layout
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    # STEP 1: Upload PDF Section
    st.markdown("""
    <div class="tech-card animate-on-scroll">
        <h2>📄 Document Upload</h2>
    </div>
    """, unsafe_allow_html=True)
    
    pdf_file, submitted = upload_pdfs()
    
    # STEP 2: Load + Index PDF if user submitted
    if pdf_file and submitted:
        file_path = os.path.join(UPLOAD_DIR, pdf_file.name)
        with open(file_path, "wb") as f:
            f.write(pdf_file.read())
        
        st.markdown(f"""
        <div class="success-message">
            ✅ Document uploaded successfully: <strong>{pdf_file.name}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🔍 Analyzing document structure..."):
            chunks = load_and_chunk_pdf(file_path)
            st.session_state.chunks = chunks
            
        st.markdown("""
        <div class="info-message">
            🎉 Document indexed successfully! Ready for intelligent queries.
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("💾 Building knowledge database..."):
            vectorstore = store_chunks(chunks)
            if vectorstore is not None:
                st.session_state.vectorstore = vectorstore
                bm25 = get_bm25_retriever(chunks)
                if bm25 is not None:
                    st.session_state.bm25 = bm25
                    st.session_state.vectorstore_initialized = True
                    update_model_info()

    elif not st.session_state.vectorstore_initialized:
        if initialize_vectorstore():
            st.markdown("""
            <div class="info-message">
                📚 Connected to existing knowledge database
            </div>
            """, unsafe_allow_html=True)
            update_model_info()
        else:
            st.markdown("""
            <div class="warning-message">
                ⚠️ Please upload a PDF document to begin analysis
            </div>
            """, unsafe_allow_html=True)
            st.stop()

    # STEP 3: Question Section
    st.markdown("""
    <div class="question-section animate-on-scroll">
        <h2>💬 Ask Your Question</h2>
    </div>
    """, unsafe_allow_html=True)
    
    query = st.text_input(
        "Question", 
        placeholder="What would you like to know about your document?", 
        key="question_input", 
        help="Ask anything about your uploaded document", 
        label_visibility="collapsed"
    )
    
    if query and st.session_state.vectorstore_initialized and st.session_state.vectorstore is not None:
        with st.spinner("🔍 Searching through document intelligence..."):
            retrieved_docs = retrieve_hybrid_docs(query, st.session_state.vectorstore, st.session_state.bm25)
        
        if not retrieved_docs:
            st.warning("No relevant content found for your query.")
        else:
            with st.spinner("📊 Optimizing results with AI ranking..."):
                reranked_docs = rerank_documents(query, retrieved_docs)
            
            chain = build_llm_chain()
            
            if chain is None:
                st.error("Error initializing language model.")
            else:
                st.markdown("""
                <div class="answer-section">
                    <h3>🤖 AI Analysis:</h3>
                </div>
                """, unsafe_allow_html=True)
                
                with st.spinner("🧠 Generating intelligent response..."):
                    try:
                        response = chain.invoke({"question": query, "docs": reranked_docs})
                        st.markdown(response)
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
    
    elif query and not st.session_state.vectorstore_initialized:
        st.warning("Please upload and index a PDF document first.")

with col2:
    # System Status
    st.markdown("""
    <div class="tech-card animate-on-scroll">
        <h3>📊 System Status</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.vectorstore_initialized:
        st.markdown("""
        <div class="metric-card">
            <div style="display: flex; align-items: center; justify-content: center; gap: 10px;">
                <span class="status-indicator status-ready"></span>
                <strong style="font-size: 1.1rem;">System Ready</strong>
            </div>
            <div style="margin-top: 12px; color: var(--success-color); font-size: 0.95rem; font-weight: 500;">
                ✅ All systems operational
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.chunks:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2.5rem; font-weight: 700; color: var(--primary-color); margin-bottom: 0.5rem;">
                    {len(st.session_state.chunks):,}
                </div>
                <div style="color: var(--text-secondary); font-size: 0.9rem; font-weight: 500;">
                    Document Chunks Indexed
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # AI Model Info
    model_info = st.session_state.current_model_info
    st.markdown(f"""
    <div class="tech-card animate-on-scroll" style="margin-top: 2rem;">
        <h3>🧠 AI Engine</h3>
        <div class="ai-engine-info">
            <div class="ai-info-row">
                <span class="ai-info-label">LLM Model:</span>
                <span class="ai-info-value">{model_info['llm']}</span>
            </div>
            <div class="ai-info-row">
                <span class="ai-info-label">Vector Database:</span>
                <span class="ai-info-value">{model_info['vector_db']}</span>
            </div>
            <div class="ai-info-row">
                <span class="ai-info-label">Embeddings:</span>
                <span class="ai-info-value">{model_info['embeddings']}</span>
            </div>
            <div class="ai-info-row">
                <span class="ai-info-label">Reranker:</span>
                <span class="ai-info-value">{model_info['reranker']}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>© 2024 ZoomYourQueryAI - Advanced Document Analysis</p>
</div>
""", unsafe_allow_html=True)