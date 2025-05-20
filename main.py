from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.graph import StateGraph
from typing import AsyncGenerator
import asyncio

from state import AgentState
from graph import Agent
from tools import session


async def stream_graph_response(
        input: AgentState, graph: StateGraph, config: dict = {}
        ) -> AsyncGenerator[str, None]:
    """
    Stream the response from the graph while parsing out tool calls.

    Args:
        input: The input for the graph.
        graph: The graph to run.
        config: The config to pass to the graph. Required for memory.

    Yields:
        A processed string from the graph's chunked response.
    """
    async for message_chunk, metadata in graph.astream(
        input=input,
        stream_mode="messages",
        config=config
        ):
        if isinstance(message_chunk, AIMessageChunk):
            if message_chunk.response_metadata:
                finish_reason = message_chunk.response_metadata.get("finish_reason", "")
                if finish_reason == "tool_calls":
                    yield "\n\n"

            if message_chunk.tool_call_chunks:
                tool_chunk = message_chunk.tool_call_chunks[0]

                tool_name = tool_chunk.get("name", "")
                args = tool_chunk.get("args", "")
                
                if tool_name:
                    tool_call_str = f"\n\n< TOOL CALL: {tool_name} >\n\n"
                if args:
                    tool_call_str = args

                yield tool_call_str
            else:
                yield message_chunk.content
            continue

async def main():
    agent = Agent(name="Scout")
    config = {"configurable": {"thread_id": "thread-1"}}

    # Initialize the input state outside the loop for the first turn
    current_input = AgentState(messages=[])

    while True:
        async for response in stream_graph_response(
            input = current_input, 
            graph = agent.graph, 
            config = config
            ):
            print(response, end="", flush=True)
            
        # After streaming, clear the current_input for the next iteration
        current_input = AgentState(messages=[])

        # Get the latest state to check if the graph ended
        thread_state = agent.graph.get_state(config=config)

        if thread_state.values.get("end", False): # Use .get() for safer access
            print("\nGraph finished.")
            break

        # Print expenses if available in the state
        if session.expenses:
            print("\n\nExpenses:", session.expenses)
        print("")


if __name__ == "__main__":
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())
