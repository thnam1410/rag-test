import os

from langchain_community.vectorstores import Chroma
from rag.embedding import Embedding
from path import AppPath


class VectorStore:
    # Chroma vector database
    def db_instance():
        return Chroma(
            embedding_function=Embedding.model,
            persist_directory=AppPath.persistent_directory,
        )

    def get_retriever():
        return VectorStore.db_instance().as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )

    @classmethod
    def is_db_exist(cls):
        return os.path.exists(AppPath.persistent_directory)
