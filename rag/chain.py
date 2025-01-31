from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from rag.llm import LLM
from rag.prompts import Prompt
from rag.vector_store import VectorStore
from langchain_core.runnables import Runnable


class Chain:
    def create_chain() -> Runnable:
        # Create a history-aware retriever
        # This uses the LLM to help reformulate the question based on chat history
        history_aware_retriever = create_history_aware_retriever(
            LLM.model, VectorStore.get_retriever(), Prompt.contextualize_q_prompt
        )

        # Create a chain to combine documents for question answering
        # `create_stuff_documents_chain` feeds all retrieved context into the LLM
        question_answer_chain = create_stuff_documents_chain(
            LLM.model, Prompt.qa_prompt)

        # Create a retrieval chain that combines the history-aware retriever and the question answering chain
        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

        return rag_chain
