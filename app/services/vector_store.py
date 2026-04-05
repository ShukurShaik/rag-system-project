from langchain_community.vectorstores import FAISS

def create_vector_store(chunks, embeddings):
    return FAISS.from_documents(chunks, embeddings)

def retrieve_documents(vector_store, query, k=5):
    return vector_store.similarity_search(query, k=k)
