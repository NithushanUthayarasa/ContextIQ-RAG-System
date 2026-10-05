import sys, json
sys.path.append('C:/Users/uthay/Desktop/ContextIQ-RAG')
from app.vectorstore.chroma_store import ChromaVectorStore
store = ChromaVectorStore()
chunks = store.get_all_chunks()
print('TOTAL CHUNKS:', len(chunks))
print('FIRST 20 IDs:', json.dumps([c['chunk_id'] for c in chunks[:20]]))
