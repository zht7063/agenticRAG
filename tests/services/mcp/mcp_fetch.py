"""
MCP Fetch 服务单元测试

测试 MCPFetch 类的工具获取和 Agent 集成功能：
1. 测试工具获取功能
2. 测试 Agent 调用 fetch 工具获取网页内容
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径，支持直接运行测试
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import dotenv
import pytest
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from src.services.mcp.mcp_fetch import MCPFetch

dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_mcp_fetch_get_tools():
    """测试 MCPFetch 工具获取功能"""
    mcp_fetch = MCPFetch()
    tools = await mcp_fetch.get_tools()
    
    assert tools is not None
    assert len(tools) > 0
    # 验证工具类型
    assert all(hasattr(tool, 'name') for tool in tools)


@pytest.mark.asyncio
async def test_mcp_fetch_agent_integration():
    """测试 MCPFetch 与 Agent 的集成"""
    mcp_fetch = MCPFetch()
    agent = create_agent(
        model=init_chat_model(
            model="gpt-4o-mini",
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        tools=await mcp_fetch.get_tools(),
        system_prompt="你是一个网页解析助手，你需要根据用户的问题，使用 fetch 工具获取网页内容，并返回给用户。",
    )

    resp = agent.invoke(
        input={
            "messages": [
                HumanMessage(content="请获取 https://www.baidu.com 的网页内容")
            ]
        }
    )
    
    assert resp is not None
    assert "messages" in resp
    assert len(resp["messages"]) > 0
    assert hasattr(resp["messages"][-1], "content")


if __name__ == "__main__":
    # 支持直接运行测试（用于调试）
    import asyncio
    
    async def run_tests():
        print("测试工具获取功能...")
        await test_mcp_fetch_get_tools()
        print("✓ 工具获取测试通过")
        
        print("\n测试 Agent 集成...")
        await test_mcp_fetch_agent_integration()
        print("✓ Agent 集成测试通过")
    
    asyncio.run(run_tests())
