import asyncio
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient  
from typing import List
from config.settings import OPENAI_BASE_URL, OPENAI_API_KEY
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage


class MCPWebSearch:
    def __init__(self):
        self.client_mcp_web_search = MultiServerMCPClient(
            {
                "bing-cn-mcp-server": {
                    "transport": "streamable_http",
                    "url": "https://mcp.api-inference.modelscope.net/0052091cc6334a/mcp"
                }
            }
        )

    async def get_mcp_tools(self) -> List[BaseTool]:
        tools = await self.client_mcp_web_search.get_tools()
        return tools


async def test_mcp_websearch(query: str):
    """测试 MCPWebSearch 获取工具列表"""
    mcp_web_search = MCPWebSearch()
    tools = await mcp_web_search.get_mcp_tools()
    
    print(f"共发现 {len(tools)} 个工具:\n")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")
    
    agent = create_agent(
        model = init_chat_model(model="gpt-4o-mini", base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY),
        tools = tools
    )

    result = await agent.ainvoke(
        input = {"messages": [HumanMessage(content=f"""
            调用你的搜索工具，搜索一下下面的内容：
            '''
                {query}
            '''
        """)]}
    )
    from pprint import pprint
    pprint(result)