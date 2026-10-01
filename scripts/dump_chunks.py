import sys, json
sys.path.append('C:/Users/uthay/Desktop/ContextIQ-RAG')
from app.vectorstore.chroma_store import ChromaVectorStore
store = ChromaVectorStore()
chunks = store.get_all_chunks()
for i, c in enumerate(chunks[:30]):
    print(f"{i}\t{c['chunk_id']}\tSource:{c['metadata'].get('source','N/A')}\nSnippet: {c['text'].replace('\n',' ')[:200]}\n---")
