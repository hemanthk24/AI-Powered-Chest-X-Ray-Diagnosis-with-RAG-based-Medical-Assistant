from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

# current path
current_path = Path(__file__).resolve()
project_root = current_path.parent.parent

#loading the file paths 
tb_path = project_root / "data" / "109.pdf"
p_path = project_root / "data" / "Pneumonia.pdf"

def create_documents(tb_path, p_path):
    loader1 = PyPDFLoader(tb_path)
    docs1 = loader1.load()

    loader2 = PyPDFLoader(p_path)
    docs2 = loader2.load()
    
    documents = []

    # metadata filtering
    for doc in docs1:
        doc.metadata["source"] = "Tuberculosis_doc"
        doc.metadata["topic"] = "Tuberculosis"
        doc.metadata["doc_type"] = "pdf"
    
    
    for doc in docs2:
        doc.metadata["source"] = "Pneumonia_docs"
        doc.metadata["topic"] = "Pneumonia"
        doc.metadata["doc_type"] = "pdf"
    
    documents.extend(docs1+docs2)
    return documents

# metadata filtering
def clean_metadata(doc):
    return {
        "source": doc.metadata.get("source", ""),
        "topic": doc.metadata.get("topic", ""),
        "doc_type": doc.metadata.get("doc_type", ""),
        "page": doc.metadata.get("page", None)
    }
    
# text splitting
def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
    )

    return splitter.split_documents(documents)

# getting chunks
def get_chunks():
    documents = create_documents(tb_path, p_path)

    cleaned_docs = []
    for doc in documents:
        doc.metadata = clean_metadata(doc)
        cleaned_docs.append(doc)

    chunks = split_docs(cleaned_docs)
    return chunks