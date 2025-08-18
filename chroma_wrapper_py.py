# app/chroma_wrapper.py - ChromaDB import wrapper with SQLite3 fix

import sys
import os

# SQLite3 compatibility fix for Streamlit Cloud
def setup_sqlite():
    try:
        # Force pysqlite3 to replace sqlite3
        import pysqlite3
        sys.modules['sqlite3'] = pysqlite3
        print("✅ SQLite3 replaced with pysqlite3")
        return True
    except ImportError:
        try:
            # Alternative approach
            __import__('pysqlite3')
            sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
            print("✅ SQLite3 replaced with pysqlite3 (alternative method)")
            return True
        except ImportError as e:
            print(f"❌ Failed to replace SQLite3: {e}")
            return False

# Apply the fix
setup_sqlite()

# Now safely import ChromaDB components
try:
    from langchain_chroma import Chroma
    from chromadb import Settings
    print("✅ Successfully imported ChromaDB components")
except ImportError as e:
    print(f"❌ Failed to import ChromaDB: {e}")
    # Fallback: you might want to use a different vector store
    Chroma = None
    Settings = None