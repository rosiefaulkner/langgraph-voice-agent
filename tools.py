from pydantic import BaseModel, Field
from langchain_core.tools import tool
from uuid import uuid4, UUID
from typing import List


class Expense(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    amount: float


class UserSession:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.expenses: List[Expense] = []


session = UserSession("Kenny")


@tool()
def create_expense(
        name: str,
        amount: float
):
    """Creates an expense"""
    expense = Expense(name=name, amount=amount)
    session.expenses.append(expense)
    return f"Successfully added expense: \n\n{expense.name} for {expense.amount}"


@tool()
def list_expenses():
    """Lists all expenses"""
    return f"Expenses:\n\n{session.expenses}"


@tool()
def delete_expense(
        id: UUID
):
    """Deletes an expense by id"""
    if not session.expenses:
        return "No expenses found"
    expense = next((expense for expense in session.expenses if expense.id == id), None)
    if not expense:
        return "Expense not found"
    session.expenses.remove(expense)
    return f"Successfully deleted expense: {expense.name}"
