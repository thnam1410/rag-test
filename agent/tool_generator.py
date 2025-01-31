
from langchain_core.tools import Tool
from rag.chain import Chain
from agent.tools.chat_history_tool import chat_history


class ToolGenerator:
    def get_tools() -> list[Tool]:
        return [
            Tool(
                name="Answer Question",
                func=lambda input, **kwargs: Chain.create_chain().invoke(
                    {
                        "input": input,
                        "chat_history": kwargs.get("chat_history", [])
                    }
                ),
                description="useful for when you need to answer questions about the context",
            ),
            chat_history
        ]
