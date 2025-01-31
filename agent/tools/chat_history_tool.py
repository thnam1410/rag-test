from langchain_redis import RedisChatMessageHistory
from app_env import AppEnv
from langchain_core.tools import tool


@tool
def chat_history(
    session_id: str,
):
    """Useful for when you need to answer questions about chat history."""
    print(f"Session ID: {session_id}")

    redis = RedisChatMessageHistory(session_id, redis_url=AppEnv.REDIS_URL)
    chat_history = redis.messages

    # Access the arguments passed by the agent
    print(f"Chat history length: {len(chat_history)}")

    return {
        "chat_history": chat_history,
        "total_messages": len(chat_history),
        "session_info": {
            "session_id": session_id,
        }
    }
