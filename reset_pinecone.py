import sys
import os
import time

# Add backend to path to import config
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from pinecone import Pinecone
from config import settings

def reset_index():
    print(f"🔑 Using API Key: {settings.PINECONE_API_KEY[:5]}...")
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index_name = settings.PINECONE_INDEX_NAME
    
    print(f"Checking for index: {index_name}")
    try:
        indexes = pc.list_indexes()
        # Handle both object and string list returns depending on version
        index_names = [i.name if hasattr(i, 'name') else i for i in indexes]
        
        if index_name in index_names:
            print(f"WARNING: Index '{index_name}' found (likely with wrong dimensions).")
            print(f"Deleting index...")
            pc.delete_index(index_name)
            print("Index deleted successfully.")
            print("⏳ Waiting 10 seconds to ensure deletion propagates...")
            time.sleep(10)
            print("👉 Now run 'python backend/run.py index' to recreate it with dimension 1024.")
        else:
            print(f"INFO: Index '{index_name}' does not exist. You are ready to index!")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    reset_index()
