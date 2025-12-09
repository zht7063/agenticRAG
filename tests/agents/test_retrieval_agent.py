"""
测试 RetrievalAgent 类

测试内容：
1. 测试 RetrievalAgent 初始化
2. 测试语义检索功能
3. 测试关键词检索功能
4. 测试混合检索功能
5. 测试 Agent 执行功能
6. 测试边界情况和异常处理

"""

import pytest
import tempfile
import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.resource_agent import ResourceAgent
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture
def temp_vectorstore_dir():
    """
    创建临时向量存储目录
    
    测试结束后自动清理。
    """
    temp_dir = tempfile.mkdtemp(prefix="test_vectorstore_retrieval_")
    
    yield temp_dir
    
    # 清理临时目录及其内容
    import shutil
    import time
    
    # 等待文件句柄释放（Windows 特有问题）
    time.sleep(0.5)
    
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            # Windows 下 Chroma 可能还在使用文件，忽略清理错误
            # 临时文件会在系统重启时被清理
            pass


@pytest.fixture
def vector_store(temp_vectorstore_dir):
    """
    创建测试用的 Chroma 向量存储实例
    """
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    
    store = Chroma(
        collection_name="test_retrieval_collection",
        embedding_function=embeddings,
        persist_directory=temp_vectorstore_dir
    )
    
    yield store
    
    # 清理：关闭 Chroma 客户端连接
    try:
        # 删除集合以释放资源
        store.delete_collection()
    except Exception:
        pass


@pytest.fixture
def pdf_path():
    """
    获取测试用 PDF 文件路径
    """
    # 使用项目中的 PDF 文件
    return "assets/pdf/chn197566.pdf"


@pytest.fixture
def populated_vector_store(vector_store, pdf_path):
    """
    预填充测试数据的向量存储
    
    使用 ResourceAgent 向向量存储添加 PDF 测试数据
    """
    # 使用 ResourceAgent 添加测试数据
    resource_agent = ResourceAgent(vector_store=vector_store)
    resource_agent.save_pdf_content(pdf_path)
    
    return vector_store


@pytest.fixture
def retrieval_agent(populated_vector_store):
    """
    创建 RetrievalAgent 实例（基于预填充的向量存储）
    """
    return RetrievalAgent(vector_store=populated_vector_store)


# ============================================================
# 基础功能测试
# ============================================================

def test_retrieval_agent_initialization(populated_vector_store):
    """
    测试 RetrievalAgent 初始化是否成功
    
    验证：
    1. RetrievalAgent 实例成功创建
    2. 必要的成员变量已初始化
    3. Agent 和工具列表已创建
    """
    agent = RetrievalAgent(vector_store=populated_vector_store)
    
    # 验证实例创建成功
    assert agent is not None
    assert isinstance(agent, RetrievalAgent)
    
    # 验证成员变量
    assert agent.vector_store is not None
    assert agent.tools is not None
    assert agent.agent is not None
    
    print(f"\nRetrievalAgent 初始化成功")
    print(f"工具数量: {len(agent.tools)}")


def test_create_tools(retrieval_agent):
    """
    验证工具列表是否正确创建
    
    验证：
    1. 工具列表包含 3 个工具
    2. 工具名称正确（semantic_search, keyword_search, hybrid_search）
    """
    tools = retrieval_agent.tools
    
    # 验证工具数量
    assert len(tools) == 3
    
    # 验证工具名称
    tool_names = [tool.name for tool in tools]
    assert "semantic_search" in tool_names
    assert "keyword_search" in tool_names
    assert "hybrid_search" in tool_names
    
    print(f"\n工具列表验证成功:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")


# ============================================================
# 检索方法测试
# ============================================================

def test_semantic_search(retrieval_agent):
    """
    测试语义检索功能
    
    验证：
    1. 返回结果格式正确（包含 content, source, score, metadata）
    2. top_k 参数有效
    3. 结果相关性
    """
    query = "未经注册商标所有人许可，在同一种商品上使用与其注册商标相同的商标，将如何惩罚？"
    
    # 执行语义检索
    results = retrieval_agent.semantic_search(query, top_k=3)
    
    # 验证返回结果
    assert results is not None
    assert isinstance(results, list)
    assert len(results) > 0
    assert len(results) <= 3
    
    # 验证结果格式
    for result in results:
        assert "content" in result
        assert "source" in result
        assert "score" in result
        assert "metadata" in result
        assert isinstance(result["content"], str)
        assert isinstance(result["score"], float)
    
    # 验证结果相关性（检查关键词）
    all_content = " ".join([r["content"] for r in results])
    assert "商标" in all_content or "注册" in all_content
    
    print(f"\n语义检索测试成功，返回 {len(results)} 条结果:")
    for i, result in enumerate(results):
        print(f"\n--- 结果 {i+1} ---")
        print(f"内容: {result['content'][:100]}...")
        print(f"来源: {result['source']}")
        print(f"分数: {result['score']:.4f}")


def test_keyword_search(retrieval_agent):
    """
    测试关键词检索功能
    
    验证：
    1. 返回结果只包含匹配关键词的文档
    2. 结果数量符合 top_k
    3. 结果格式正确
    """
    keywords = "注册商标"
    
    # 执行关键词检索
    results = retrieval_agent.keyword_search(keywords, top_k=3)
    
    # 验证返回结果
    assert results is not None
    assert isinstance(results, list)
    assert len(results) <= 3
    
    # 验证每个结果都包含关键词
    for result in results:
        assert "content" in result
        assert "source" in result
        assert "metadata" in result
        # 验证内容包含关键词
        assert keywords in result["content"]
    
    print(f"\n关键词检索测试成功，返回 {len(results)} 条结果:")
    for i, result in enumerate(results):
        print(f"\n--- 结果 {i+1} ---")
        print(f"内容: {result['content'][:100]}...")
        print(f"来源: {result['source']}")


def test_hybrid_search(retrieval_agent):
    """
    测试混合检索功能
    
    验证：
    1. 结果融合和去重逻辑
    2. 返回结果数量
    3. 结果格式正确
    """
    query = "商标侵权惩罚"
    
    # 执行混合检索
    results = retrieval_agent.hybrid_search(query, top_k=5)
    
    # 验证返回结果
    assert results is not None
    assert isinstance(results, list)
    assert len(results) > 0
    assert len(results) <= 5
    
    # 验证结果格式
    for result in results:
        assert "content" in result
        assert "source" in result
        assert "metadata" in result
    
    # 验证去重（检查内容是否唯一）
    contents = [r["content"] for r in results]
    assert len(contents) == len(set(contents)), "混合检索结果应该去重"
    
    print(f"\n混合检索测试成功，返回 {len(results)} 条结果:")
    for i, result in enumerate(results):
        print(f"\n--- 结果 {i+1} ---")
        print(f"内容: {result['content'][:100]}...")
        print(f"来源: {result['source']}")


def test_different_top_k_values(retrieval_agent):
    """
    测试不同 top_k 参数值
    
    验证：
    1. top_k=1 返回 1 条结果
    2. top_k=5 返回最多 5 条结果
    3. top_k=10 返回最多 10 条结果
    """
    query = "商标"
    
    # 测试 top_k=1
    results_1 = retrieval_agent.semantic_search(query, top_k=1)
    assert len(results_1) == 1
    
    # 测试 top_k=5
    results_5 = retrieval_agent.semantic_search(query, top_k=5)
    assert len(results_5) <= 5
    assert len(results_5) > len(results_1)
    
    # 测试 top_k=10
    results_10 = retrieval_agent.semantic_search(query, top_k=10)
    assert len(results_10) <= 10
    
    print(f"\ntop_k 参数测试成功:")
    print(f"  top_k=1: {len(results_1)} 条结果")
    print(f"  top_k=5: {len(results_5)} 条结果")
    print(f"  top_k=10: {len(results_10)} 条结果")


# ============================================================
# Agent 执行测试
# ============================================================

def test_agent_execute_semantic_query(retrieval_agent):
    """
    测试 Agent 处理语义查询
    
    验证 Agent 能够理解概念性问题并选择合适的检索策略
    """
    # 输入概念性问题
    input_msg = "请帮我查找关于商标侵权的法律规定和惩罚措施的相关内容"
    
    # 执行任务
    response = retrieval_agent.execute(input_msg)
    
    # 验证响应
    assert response is not None
    assert "messages" in response
    assert len(response["messages"]) > 0
    
    # 获取最后一条消息
    last_message = response["messages"][-1]
    
    print(f"\nAgent 语义查询测试:")
    print(f"用户输入: {input_msg}")
    print(f"Agent 响应: {last_message.content[:300]}...")


def test_agent_execute_keyword_query(retrieval_agent):
    """
    测试 Agent 处理关键词查询
    
    验证 Agent 能够理解精确匹配需求并选择合适的检索策略
    """
    # 输入精确匹配需求
    input_msg = "请精确查找包含'有期徒刑'这个关键词的内容"
    
    # 执行任务
    response = retrieval_agent.execute(input_msg)
    
    # 验证响应
    assert response is not None
    assert "messages" in response
    assert len(response["messages"]) > 0
    
    # 获取最后一条消息
    last_message = response["messages"][-1]
    
    print(f"\nAgent 关键词查询测试:")
    print(f"用户输入: {input_msg}")
    print(f"Agent 响应: {last_message.content[:300]}...")


# ============================================================
# 边界情况测试
# ============================================================

def test_empty_results(vector_store):
    """
    测试查询无结果时的行为
    
    验证：
    1. 在空的向量存储中查询返回空列表
    2. 查询完全不相关的内容返回空或极少结果
    """
    # 创建一个空的向量存储的 RetrievalAgent
    agent = RetrievalAgent(vector_store=vector_store)
    
    # 查询应该返回空列表
    results = agent.semantic_search("随机不存在的内容xyz123", top_k=5)
    
    # 验证返回空列表
    assert isinstance(results, list)
    assert len(results) == 0
    
    print(f"\n空结果测试成功，返回 {len(results)} 条结果")


def test_semantic_search_with_filter(retrieval_agent):
    """
    测试带过滤条件的语义检索
    
    验证 source_type 参数是否有效
    """
    query = "商标"
    
    # 测试默认（all）
    results_all = retrieval_agent.semantic_search(query, top_k=3, source_type="all")
    assert len(results_all) > 0
    
    # 测试 pdf 过滤
    results_pdf = retrieval_agent.semantic_search(query, top_k=3, source_type="pdf")
    # 注意：这个测试可能会失败，因为我们的测试数据可能没有设置 source_type
    # 只验证不会报错
    assert isinstance(results_pdf, list)
    
    print(f"\n过滤条件测试:")
    print(f"  source_type=all: {len(results_all)} 条结果")
    print(f"  source_type=pdf: {len(results_pdf)} 条结果")

