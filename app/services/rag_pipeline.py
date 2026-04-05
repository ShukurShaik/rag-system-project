from langchain_groq import ChatGroq
from app.services.document_loader import load_documents
from app.services.chunking import split_documents
from app.services.embedding import get_embeddings
from app.services.vector_store import create_vector_store, retrieve_documents
from app.config.settings import GROQ_API_KEY, MODEL_NAME, TEMPERATURE

def run_rag(query: str):
    documents = load_documents()
    chunks = split_documents(documents)
    embeddings = get_embeddings()
    vector_store = create_vector_store(chunks, embeddings)

    retrieved_docs = retrieve_documents(vector_store, query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=TEMPERATURE
    )

    prompt = f"""
    answer the question using only the context below.
    if the answer is not present, say "i don't know".

    context:
    {context}

    question:
    {query}
    """

    response = llm.invoke(prompt)
    return response.content
