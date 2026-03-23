# ✅ Updated imports (NO langchain_classic)
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI

from src.vector_store import load_vectordb, create_embedding_model
from src.docs_loader import get_chunks
from src.create_retriever import get_retriever
from src.prompt import rag_prompt


# 🔥 Load ONCE (IMPORTANT - performance boost)
embedding = create_embedding_model()
vectordb = load_vectordb("medibot", embedding)
chunks = get_chunks()


# 🔥 LLM
def load_llm():
    return ChatOpenAI(
        temperature=0.3,
        model="gpt-4o-mini"   # updated param name
    )


# 🔥 Memory (global for session)
memory = ConversationBufferWindowMemory(
    k=5,
    memory_key="chat_history",
    return_messages=True
)


# 🚀 MAIN PIPELINE
def get_rag_chain(disease=None, confidence=None):

    # 1️⃣ Retriever (metadata filtering inside)
    retriever = get_retriever(
        vectordb=vectordb,
        chunks=chunks,
        disease=disease,
        confidence=confidence
    )

    # 2️⃣ LLM
    llm = load_llm()

    # 3️⃣ RAG Chain
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": rag_prompt
        },
        verbose=True
    )

    return rag_chain