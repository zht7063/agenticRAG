import asyncio
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient  
from typing import List
from config.settings import OPENAI_BASE_URL, OPENAI_API_KEY
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage


class MCPTime:
    def __init__(self):
        self.client_mcp_time = MultiServerMCPClient(
            {
                "time-server": {
                    "transport": "stdio",
                    "args": [
                        "mcp-server-time"
                    ],
                    "command": "uvx"
                }
            }
        )

    async def get_mcp_tools(self) -> List[BaseTool]:
        tools = await self.client_mcp_time.get_tools()
        return tools


async def test_mcp_time():
    """测试 MCPTime 获取工具列表"""
    mcp_time = MCPTime()
    tools = await mcp_time.get_mcp_tools()
    
    print(f"共发现 {len(tools)} 个工具:\n")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")
    
    agent = create_agent(
        model = init_chat_model(model="gpt-4o-mini", base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY),
        tools = tools
    )

    result = await agent.ainvoke(
        input = {"messages": [HumanMessage(content="""
            获取一下现在的北京时间是多少？
        """)]}
    )
    from pprint import pprint
    pprint(result)