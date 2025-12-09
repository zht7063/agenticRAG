from abc import ABC, abstractmethod
from typing import List
from langchain_core.tools import Tool


class MCPWrapper(ABC):
    @abstractmethod
    def get_tools(self) -> List[Tool]:
        """ 返回 MCP 工具集列表 """
        pass