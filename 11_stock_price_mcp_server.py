from mcp.server.fastmcp import FastMCP
from langgraph.types import interrupt
import yfinance as yf

mcp = FastMCP("stock_price") #, log_level="ERROR") #, port=8001)

@mcp.tool()
def get_stock_price(symbol: str) -> float:
    """Fetch the latest stock price for a given symbol."""
    stock = yf.Ticker(symbol)
    price = stock.history(period="1d")["Close"].iloc[-1]
    return price

if __name__ == "__main__":
    mcp.run(transport="streamable-http")