from pinecone import Pinecone, ServerlessSpec
from pinecone import ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from src.docs_loader import get_chunks
import os
from dotenv import load_dotenv

load_dotenv()

# creating embeddind model
def create_embedding_model():
    embedding = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    return embedding

# stroing embeddings in pinecone
def ingest_to_pinecone(indexname,chunks,embedding_model):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = indexname

    if not pc.has_index(index_name):
        pc.create_index(
            name = index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    
    docsearch = PineconeVectorStore.from_documents(
        documents=chunks,
        index_name=index_name,
        embedding=embedding_model,
    )

    print("Data successfully uploaded to Pinecone")

# loadind existing vectorDB
def load_vectordb(indexname,embedding_model):
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=indexname,
        embedding=embedding_model
    )
    return vectorstore



if __name__ == "__main__":
    chunks = get_chunks()
    embedding = create_embedding_model()
    ingest_to_pinecone(indexname="medibot",chunks=chunks,embedding_model=embedding)