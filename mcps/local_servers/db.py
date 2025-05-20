from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from typing import List, Optional
from sqlalchemy import ForeignKey, String, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session
from uuid import UUID, uuid4
from datetime import datetime
import os
from pydantic import BaseModel
from enum import StrEnum
import pandas as pd


load_dotenv()


# ----------------------------
# SQLAlchemy Models
# ----------------------------

class Base(DeclarativeBase):
     pass


class DBCustomer(Base):
    __tablename__ = "customers"

    id: Mapped[UUID] = mapped_column(primary_key=True, index=True, server_default=text("gen_random_uuid()"))
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(nullable=False, server_default=text("now()"), onupdate=datetime.now)
    first_name: Mapped[str] = mapped_column(String, nullable=False)
    last_name: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[str] = mapped_column(String, nullable=False)

    expenses: Mapped[List["DBExpense"]] = relationship(back_populates="customer", cascade="all, delete-orphan")


class DBExpense(Base):
    __tablename__ = "expenses"

    id: Mapped[UUID] = mapped_column(primary_key=True, index=True, server_default=text("gen_random_uuid()"))
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(nullable=False, server_default=text("now()"), onupdate=datetime.now)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    category: Mapped[str] = mapped_column(String, nullable=False, server_default=text("other"))
    amount: Mapped[float] = mapped_column(nullable=False)

    customer_id: Mapped[UUID] = mapped_column(ForeignKey("customers.id", ondelete="CASCADE"), nullable=False)
    customer: Mapped["DBCustomer"] = relationship(back_populates="expenses", foreign_keys=[customer_id])


# ----------------------------
# Pydantic Models
# ----------------------------

class Customer(BaseModel):
    id: UUID
    created_at: datetime
    updated_at: datetime
    first_name: str
    last_name: str
    email: str


class ExpenseCategory(StrEnum):
    MEALS = "meals"
    TRAVEL = "travel"
    LODGING = "lodging"
    ENTERTAINMENT = "entertainment"
    TRAINING = "training"
    GIFTS = "gifts"
    EDUCATION = "education"
    OFFICE_SUPPLIES = "office_supplies"
    OTHER = "other"


class Expense(BaseModel):
    id: UUID
    created_at: datetime
    updated_at: datetime
    name: str
    description: Optional[str]
    category: ExpenseCategory
    amount: float
    customer_id: UUID


# ----------------------------
# DB Session
# ----------------------------

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine(url=os.getenv("SUPABASE_URI"))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ----------------------------
# MCP Server
# ----------------------------

mcp = FastMCP("db")


@mcp.tool()
async def create_expense(
    customer_id: UUID,
    name: str,
    amount: float,
    category: ExpenseCategory,
    description: Optional[str],
    ) -> str:
    f"""Create a new expense for the customer_id.
    
    Args:
        customer_id: The customer_id to create the expense for.
        name: The name of the expense.
        amount: The amount of the expense.
        category: The category of the expense. Options are: {', '.join([c.value for c in ExpenseCategory])}
        description: An optional description of the expense that may include additional details not captured in the name or category.

    Returns:
        The created expense.
    """
    with SessionLocal() as session:
        new_expense = DBExpense(
            name=name,
            amount=amount,
            category=category.value,
            description=description,
            customer_id=customer_id,
            )
        session.add(new_expense)
        session.commit()
        session.refresh(new_expense)
    
    return Expense.model_validate(new_expense.__dict__).model_dump_json(indent=2)


@mcp.tool()
async def delete_expense(id: UUID) -> str:
    """Delete an expense by id.
    
    Args:
        id: The id of the expense to delete.

    Returns:
        The deleted expense.
    """
    with SessionLocal() as session:
        expense = session.query(DBExpense).filter(DBExpense.id == id).first()
        if not expense:
            return "Expense not found"
        session.delete(expense)
        session.commit()
    
    return Expense.model_validate(expense.__dict__).model_dump_json(indent=2)


@mcp.tool()
async def update_expense(
    id: UUID,
    customer_id: UUID,
    name: Optional[str],
    amount: Optional[float],
    category: Optional[ExpenseCategory],
    description: Optional[str],
    ) -> str:
    f"""Update an expense by id.
    
    Args:
        id: The id of the expense to update.
        customer_id: The customer_id to update the expense for.
        name: The name of the expense.
        amount: The amount of the expense.
        category: The category of the expense. Options are: {', '.join([c.value for c in ExpenseCategory])}
        description: An optional description of the expense that may include additional details not captured in the name or category.

    Returns:
        The updated expense.
    """
    with SessionLocal() as session:
        expense = session.query(DBExpense).filter(DBExpense.id == id and DBExpense.customer_id == customer_id).first()
        if not expense:
            return "Expense not found"
        
        if name:
            expense.name = name
        if amount:
            expense.amount = amount
        if category:
            expense.category = category.value
        if description:
            expense.description = description

        session.commit()
    
    return Expense.model_validate(expense.__dict__).model_dump_json(indent=2)


@mcp.tool()
async def list_expenses(customer_id: UUID) -> str:
    """List the customer's expenses.
    
    Args:
        customer_id: The customer_id to query expenses for.

    Returns:
        A list of expenses.
    """
    with SessionLocal() as session:
        expenses = session.query(DBExpense).filter(DBExpense.customer_id == customer_id)
        expenses = [Expense.model_validate(expense.__dict__).model_dump_json(indent=2) for expense in expenses]
    return f"[{', \n'.join(expenses)}]"


@mcp.tool()
async def query_db(query: str, customer_id: UUID) -> str:
    """Query the database using SQL.
    
    Args:
        query: A valid PostgreSQL query to run.
        customer_id: The active customer_id, used for RLS.

    Returns:
        The query results
    """
    with SessionLocal() as session:
        session.execute(text("SET app.customer_id = :customer_id"), {"customer_id": str(customer_id)})

        result = session.execute(text(query), {"customer_id": customer_id})
        
    return pd.DataFrame(result.all(), columns=result.keys()).to_json(orient="records", indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
