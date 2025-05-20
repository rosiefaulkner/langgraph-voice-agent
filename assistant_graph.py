from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver
from typing import List
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from tools import create_expense, delete_expense, list_expenses
from langgraph.prebuilt import ToolNode, tools_condition
from state import AgentState


load_dotenv()


class Agent:
    def __init__(
            self, 
            name: str = "Scout",
            system_prompt: str = "You are Scout, a helpful expense manager.",
            model: str = "gpt-4.1-mini-2025-04-14",
            tools: List[BaseTool] = [create_expense, delete_expense, list_expenses],
            ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.tools = tools

        self.llm = ChatOpenAI(name=self.name, model=model).bind_tools(tools=self.tools)
        self.graph = self.build_graph()

    def build_graph(self,) -> CompiledStateGraph:
        builder = StateGraph(AgentState)

        def assistant(state: AgentState):
            response = self.llm.invoke([SystemMessage(content=self.system_prompt)] + state.messages)
            state.messages.append(response)
            return state

        builder.add_node(assistant)
        builder.add_node(ToolNode(self.tools))

        builder.set_entry_point("assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition
        )
        builder.add_edge("tools", "assistant")

        return builder.compile(checkpointer=InMemorySaver())

    def draw_graph(self,):
        if self.graph is None:
            raise ValueError("Graph not built yet")
        from IPython.display import Image

        return Image(self.graph.get_graph().draw_mermaid_png())
    
agent_graph = Agent().build_graph()

if __name__ == "__main__":
    agent = Agent(name="Scout")
    agent.draw_graph()
