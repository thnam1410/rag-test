from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.language_models import BaseChatModel

from agent.tool_generator import ToolGenerator
from rag.llm import LLM


class AppAgent:
    llm: BaseChatModel = LLM.model

    def __init__(self) -> None:
        pass

    @classmethod
    def invoke(cls, inputArgs: dict[str, any]):
        prompt = hub.pull("hwchase17/react")

        tools = ToolGenerator.get_tools()

        agent = create_react_agent(
            llm=cls.llm,
            tools=tools,
            prompt=prompt,
        )

        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,  # Handle any parsing errors gracefully
        )

        print("inputArgs:", inputArgs)
        return agent_executor.invoke(inputArgs)
