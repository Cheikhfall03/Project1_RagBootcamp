#!/bin/bash

# Configuration pour économiser la mémoire
export CUDA_VISIBLE_DEVICES=""
export NO_CUDA=1
export CUDA_LAUNCH_BLOCKING=1
export TF_CPP_MIN_LOG_LEVEL=2
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Limite la mémoire utilisée par Python
export PYTHONMALLOC=malloc
export MALLOC_TRIM_THRESHOLD_=100000

# Configuration Streamlit pour économiser les ressources
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=50
export STREAMLIT_SERVER_MAX_MESSAGE_SIZE=50

echo "🚀 Démarrage de ZoomYourQueryAI avec configuration optimisée..."
echo "💡 Configuration CPU uniquement activée"
echo "🔧 Limitations mémoire appliquées"

# Démarrer Streamlit avec configuration légère
streamlit run index.py \
  --server.maxUploadSize=50 \
  --server.maxMessageSize=50 \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false