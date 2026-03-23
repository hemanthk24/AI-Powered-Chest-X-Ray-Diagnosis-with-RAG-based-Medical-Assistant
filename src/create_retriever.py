from src.vector_store import create_embedding_model, load_vectordb
from src.docs_loader import get_chunks
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# 🔥 Metadata filter
def get_metadata_filter(disease, confidence, threshold=0.6):

    disease = disease.lower()

    # Normal → no filter
    if disease == "normal":
        return None

    # Low confidence → no filter
    if confidence < threshold:
        return None

    # Apply filter
    if disease == "pneumonia":
        return {"topic": {"$eq": "Pneumonia"}}

    if disease == "tuberculosis":
        return {"topic": {"$eq": "Tuberculosis"}}

    return None


# FINAL RETRIEVER FUNCTION
def get_retriever(vectordb, chunks, disease=None, confidence=None):

    # Get metadata filter
    metadata_filter = get_metadata_filter(disease, confidence)

    # Pinecone retriever
    search_kwargs = {
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.5
    }

    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    pinecone_retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )

    # BM25 retriever (no metadata filter)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 3

    # Hybrid retriever
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, pinecone_retriever],
        weights=[0.2,0.8]
    )

    return hybrid_retriever