from rag.vector_store import VectorStore
from seed import seed_db
from app_env import AppEnv
from langchain_redis import RedisChatMessageHistory
from rag.chain import Chain
from agent.app_agent import AppAgent

session_id = "user_123"


def context_continuous_chat():
    redis = RedisChatMessageHistory(session_id, redis_url=AppEnv.REDIS_URL)

    while True:
        query = input("You: ")

        if query.lower() == "exit":
            break

        if (query.lower() == "_ch_"):
            print(redis.messages)
            continue

        if (query.lower() == "_clear_"):
            print(redis.clear())
            continue

        # Process the user's query through the retrieval chain
        result = Chain.create_chain().invoke(
            {"input": query, "chat_history": redis.messages})

        print(f"AI: {result['answer']}")

        # Update the chat history
        redis.add_user_message(query)
        redis.add_ai_message(result["answer"])


def agent_react_docstore_continous_chat():
    redis = RedisChatMessageHistory(session_id, redis_url=AppEnv.REDIS_URL)

    while True:
        query = input("You: ")

        if query.lower() == "exit":
            break

        if (query.lower() == "_ch_"):
            print(redis.messages)
            continue

        if (query.lower() == "_clear_"):
            print(redis.clear())
            continue

        # Process the user's query through the retrieval chain
        agent_executor = AppAgent().get_agent()

        response = agent_executor.invoke(
            {
                "input": query,
                "chat_history": redis.messages,
                "session_id": session_id,
            }
        )

        print(f"AI: {response['output']}")

        # Update history
        redis.add_user_message(query)
        redis.add_ai_message(response["output"])

    # Main function to start the continual chat
if __name__ == "__main__":
    if not VectorStore.is_db_exist():
        seed_db()
    agent_react_docstore_continous_chat()
