import logging
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver
from typing import List
from dotenv import load_dotenv

from state import AgentState
from mcps.local_servers.db import ExpenseCategory


load_dotenv()

# Configure logging to suppress HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


class Agent:
    def __init__(
            self,
            name: str = "Luna",
            model: str = "gpt-4.1-mini-2025-04-14",
            tools: List[BaseTool] = [],
            system_prompt: str = """You are Luna, the company's expense manager. You are responsible for managing employee expenses. You have access to the employee's expenses and can help them create, delete, and query expenses.

            Your messages are read aloud to the user, so respond in a way that is easy to understand when spoken. Be brief and to the point.

            When creating new expenses, you must classify the expense into one of the allowed categories below. If the expense does not fit into any of the categories, choose "other". If unsure, you can ask the customer for more information.

            <expense_categories>
            {expense_categories}
            </expense_categories>

            <db_schema>
            You have access to a database with the following schema:
            - customers (id, created_at, updated_at, first_name, last_name, email)
            - expenses (id, created_at, updated_at, name, description, category, amount, customer_id)
            </db_schema>

            The active customer_id is:
            {customer_id}
            """,
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
            """The main assistant node that uses the LLM to generate responses."""
            # inject customer_id from the state into the system prompt
            system_prompt = self.system_prompt.format(
                customer_id=state.customer_id,
                expense_categories=", ".join([c.value for c in ExpenseCategory])
                )

            response = self.llm.invoke([SystemMessage(content=system_prompt)] + state.messages)
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

agent = Agent()
agent_graph = agent.build_graph()

if __name__ == "__main__":
    agent.draw_graph()
