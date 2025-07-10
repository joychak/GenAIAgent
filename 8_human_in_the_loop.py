from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
import yfinance as yf
from langgraph.types import interrupt, Command
    

class State(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def get_stock_price(symbol: str) -> float:
    """Fetch the latest stock price for a given symbol."""
    stock = yf.Ticker(symbol)
    price = stock.history(period="1d")["Close"].iloc[-1]
    return price

@tool
def buy_stocks(symbol: str, quantity: int, total_cost: float) -> str:
    """Buy a specified quantity of stocks at the current price."""
    decision = interrupt(f"Do you want to buy {quantity} shares of {symbol} at a total cost of ${total_cost:.2f}?")
    if decision == "yes":
        return f"Bought {quantity} shares of {symbol} at a total cost of ${total_cost:.2f}."
    else:
        return "Buying stock declined by user."

def chatbot(state: State) -> State:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


load_dotenv()
memory = MemorySaver()

tools = [get_stock_price, buy_stocks]

llm = init_chat_model("gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

builder = StateGraph(State)

builder.add_node("chatbot_node", chatbot)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot_node")
builder.add_conditional_edges("chatbot_node", tools_condition)
builder.add_edge("tools", "chatbot_node")
builder.add_edge("chatbot_node", END)

graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "buy_thread"}}

# user asked for stock price
message = {"role": "user", "content": "What is the price of AMZN stock now?"}
state = graph.invoke({"messages": [message]}, config=config)
print(state["messages"][-1].content)

# user asked to buy stocks
message = {"role": "user", "content": "Buy 10 AMZN stock at current price"}
state = graph.invoke({"messages": [message]}, config=config)
# print(state["messages"][-1].content)
print(state.get("__interrupt__"))

decision = input("Approve (yes/no): ")
state = graph.invoke(Command(resume=decision), config=config)
print(state["messages"][-1].content)