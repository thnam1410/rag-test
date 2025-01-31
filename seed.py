import os
from rag.vector_store import VectorStore
from path import AppPath
from langchain.text_splitter import CharacterTextSplitter, TextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents.base import Document


def get_book_files():
    if not os.path.exists(AppPath.data_dir):
        raise FileNotFoundError(
            f"The directory {AppPath.data_dir} does not exist. Please check the path."
        )
    return [f for f in os.listdir(AppPath.data_dir) if f.endswith(".txt")]


def chunk_document(file_path: str, text_splitter: TextSplitter) -> list[Document]:
    loader = TextLoader(file_path)
    book_docs = loader.load()
    documents = []

    for doc in text_splitter.split_documents(book_docs):
        doc.metadata.update({
            "source": os.path.basename(file_path)
        })
        documents.append(doc)

    return documents


def store_documents(documents: list[Document], batch_size=10):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        print(f"Storing {len(batch)} chunks in ChromaDB...")
        db = VectorStore.db_instance()
        db.add_documents(documents=batch)


def seed_db():
    if VectorStore.is_db_exist():
        print(f"Vector store exists.")
        return

    print(f"Start seeding documents")
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )

    book_files = get_book_files()

    print("Num of book files:", len(book_files))

    for book_file in book_files:
        file_path = os.path.join(AppPath.data_dir, book_file)
        documents = chunk_document(file_path, text_splitter)
        store_documents(documents)
        print(f"Finished storing all chunks from {book_file}")
