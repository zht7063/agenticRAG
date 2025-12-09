import os
from typing import List

from dotenv import load_dotenv
from langchain_core.tools import Tool
from langchain_mcp_adapters.client import MultiServerMCPClient  
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from .mcp_wrapper import MCPWrapper

load_dotenv()

class MCPFetch(MCPWrapper):
    def __init__(self):
        self.mcp_client = MultiServerMCPClient  (
            {
                "fetch": {
                    "transport": "streamable_http",
                    "url": "https://mcp.api-inference.modelscope.net/ae877438d9ec4a/mcp"
                }
            }
        )


    async def get_tools(self) -> List[Tool]:
        tools = await self.mcp_client.get_tools()
        return tools



