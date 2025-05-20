from langgraph.graph import add_messages
from pydantic import BaseModel
from typing import Annotated, List


class AgentState(BaseModel):
    messages: Annotated[List, add_messages] = []
    customer_id: str = ""
