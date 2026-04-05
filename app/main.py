from app.services.rag_pipeline import run_rag

if __name__ == "__main__":
    query = "what is append method with example"
    answer = run_rag(query)
    print("\nAnswer:\n", answer)
