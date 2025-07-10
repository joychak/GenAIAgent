from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.prompts import load_mcp_prompt
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import asyncio
import sys
import time

import logging
logger = logging.getLogger("langchain_mcp_adapters.client.MultiServerMCPClient")
logger.setLevel(logging.ERROR)

client = MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            "args": ["10_math_mcp_server.py"],
            "transport": "stdio",
        },
        "stock_price": {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        }
    }
)

class State(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def buy_stocks(symbol: str, quantity: int, total_cost: float) -> str:
    """Buy a specified quantity of stocks at the current price."""
    decision = interrupt(f"Do you want to buy {quantity} shares of {symbol} at a total cost of ${total_cost:.2f}?")
    if decision == "yes":
        return f"Bought {quantity} shares of {symbol} at a total cost of ${total_cost:.2f}."
    else:
        return "Buying stock declined by user."

async def create_graph(math_session, stock_price_session, memory):
    # tools = client.get_tools()
    math_tool = await load_mcp_tools(math_session)
    stock_tool = await load_mcp_tools(stock_price_session)
    tools = math_tool + stock_tool + [buy_stocks]
    
    llm = init_chat_model("gpt-4o-mini")
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = await load_mcp_prompt(math_session, "system_prompt")
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt[0].content),
        MessagesPlaceholder("messages")
    ])
   
    chat_llm = prompt_template | llm_with_tools
    
    def chatbot(state: State) -> State:
        return {"messages": [chat_llm.invoke({"messages": state["messages"]})]}
    
    builder = StateGraph(State)

    builder.add_node("chatbot_node", chatbot)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "chatbot_node")
    builder.add_conditional_edges("chatbot_node", tools_condition)
    builder.add_edge("tools", "chatbot_node")
    builder.add_edge("chatbot_node", END)

    graph = builder.compile(checkpointer=memory)

    return graph

async def animate_cursor():
    chars = "|/-\\"
    while True:
        for char in chars:
            sys.stdout.write(f"\r{char} Thinking...")
            sys.stdout.flush()
            await asyncio.sleep(0.1)  # Control animation speed

async def main():
    load_dotenv()
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "1234"}}
    async with client.session("math") as math_session, client.session("stock_price") as stock_price_session:
        
        agent = await create_graph(math_session, stock_price_session, memory)
        state = None
        while True:
            in_message = input("[You]: ")
            if in_message.lower() in {"exit", "quit"}:
                print("Exiting the chatbot.")
                break

            if state is None:
                state: State = {
                    "messages": [{"role": "user", "content": in_message}] 
                }
            else:
                state["messages"].append({"role": "user", "content": in_message})

            animation_task = asyncio.create_task(animate_cursor())
            state = await agent.ainvoke(state, config=config)
            animation_task.cancel()
            sys.stdout.write('\r\b')
            sys.stdout.flush()

            if state.get("__interrupt__"):
                decision = input("Approve (yes/no): ")

                animation_task = asyncio.create_task(animate_cursor())
                state = await agent.ainvoke(Command(resume=decision), config=config)
                animation_task.cancel()
                sys.stdout.write('\r\b')
                sys.stdout.flush()

            for char in "[Bot]: " + state["messages"][-1].content:
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(0.005)
            # print("Bot:", state["messages"][-1].content)
            print("\n-------------------------------------------------------\n")

if __name__ == "__main__":
    asyncio.run(main())
        