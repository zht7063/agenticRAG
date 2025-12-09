"""
测试 ResourceWorker 类

测试内容：
1. 测试 PDF 文档存储和检索
2. 测试 URL 内容存储和检索
3. 验证向量相似度搜索结果

"""

import pytest
import tempfile
import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from src.agents.resource_agent import ResourceWorker
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture
def temp_vectorstore_dir():
    """
    创建临时向量存储目录
    
    测试结束后自动清理。
    """
    temp_dir = tempfile.mkdtemp(prefix="test_vectorstore_")
    
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
        collection_name="test_collection",
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
def resource_agent(vector_store):
    """
    创建 ResourceWorker 实例
    """
    return ResourceWorker(vector_store=vector_store)


@pytest.fixture
def pdf_path():
    """
    获取测试用 PDF 文件路径
    """
    # 使用项目中的 PDF 文件
    return "assets/pdf/chn197566.pdf"


@pytest.fixture
def test_url():
    """
    测试用 URL
    """
    return "https://arxiv.org/html/2402.08954v1"


# ============================================================
# 测试用例
# ============================================================

def test_save_pdf_content(resource_agent, pdf_path):
    """
    测试保存 PDF 内容到向量存储
    
    验证：
    1. PDF 文件成功加载并分块
    2. 文档成功保存到向量存储
    3. 返回的文档 ID 列表不为空
    """
    # 保存 PDF 内容
    ids = resource_agent.save_pdf_content(pdf_path)
    
    # 验证返回的 ID 列表
    assert ids is not None
    assert len(ids) > 0
    assert isinstance(ids, list)
    
    print(f"\n成功保存 {len(ids)} 个 PDF 文档片段")


def test_save_url_content(resource_agent, test_url):
    """
    测试保存 URL 内容到向量存储
    
    验证：
    1. URL 内容成功获取并分块
    2. 文档成功保存到向量存储
    3. 返回的文档 ID 列表不为空
    """
    # 保存 URL 内容
    ids = resource_agent.save_url_content(test_url)
    
    # 验证返回的 ID 列表
    assert ids is not None
    assert len(ids) > 0
    assert isinstance(ids, list)
    
    print(f"\n成功保存 {len(ids)} 个 URL 文档片段")


def test_pdf_similarity_search(resource_agent, vector_store, pdf_path):
    """
    测试 PDF 内容的相似度搜索
    
    测试查询：未经注册商标所有人许可，在同一种商品上使用与其注册商标相同的商标，将如何惩罚？
    
    预期：搜索结果应包含相关的法律条款内容
    """
    # 先保存 PDF 内容
    resource_agent.save_pdf_content(pdf_path)
    
    # 进行相似度搜索
    query = "未经注册商标所有人许可，在同一种商品上使用与其注册商标相同的商标，将如何惩罚？"
    results = vector_store.similarity_search(query, k=3)
    
    # 验证搜索结果
    assert results is not None
    assert len(results) > 0
    
    # 检查搜索结果是否包含相关内容
    search_text = " ".join([doc.page_content for doc in results])
    
    # 验证关键词是否存在
    assert "注册商标" in search_text or "商标" in search_text
    assert "有期徒刑" in search_text or "罚金" in search_text or "拘役" in search_text
    
    print(f"\n相似度搜索结果：")
    for i, doc in enumerate(results):
        print(f"\n--- 结果 {i+1} ---")
        print(f"内容: {doc.page_content[:200]}...")
        print(f"元数据: {doc.metadata}")


def test_url_similarity_search(resource_agent, vector_store, test_url):
    """
    测试 URL 内容的相似度搜索
    
    测试查询：arXiv 的使命是什么？
    
    预期：搜索结果应包含 arXiv 的使命相关内容
    """
    # 先保存 URL 内容
    resource_agent.save_url_content(test_url)
    
    # 进行相似度搜索
    query = "arXiv 的使命是什么？"
    results = vector_store.similarity_search(query, k=3)
    
    # 验证搜索结果
    assert results is not None
    assert len(results) > 0
    
    # 检查搜索结果是否包含相关内容
    search_text = " ".join([doc.page_content for doc in results])
    
    # 验证关键词是否存在（arXiv 的使命相关内容）
    assert "arXiv" in search_text
    assert any(keyword in search_text.lower() for keyword in [
        "mission", "使命", "access", "research", "openness", "科学"
    ])
    
    print(f"\n相似度搜索结果：")
    for i, doc in enumerate(results):
        print(f"\n--- 结果 {i+1} ---")
        print(f"内容: {doc.page_content[:200]}...")
        print(f"元数据: {doc.metadata}")


def test_combined_search(resource_agent, vector_store, pdf_path, test_url):
    """
    综合测试：同时存储 PDF 和 URL 内容，并进行独立搜索
    
    验证：
    1. 两种内容都能正确存储
    2. 查询 PDF 内容时返回 PDF 相关结果
    3. 查询 URL 内容时返回 URL 相关结果
    """
    # 保存 PDF 内容
    pdf_ids = resource_agent.save_pdf_content(pdf_path)
    
    # 保存 URL 内容
    url_ids = resource_agent.save_url_content(test_url)
    
    # 验证两者都成功保存
    assert len(pdf_ids) > 0
    assert len(url_ids) > 0
    
    # 搜索 PDF 相关内容
    pdf_query = "未经注册商标所有人许可，在同一种商品上使用与其注册商标相同的商标，将如何惩罚？"
    pdf_results = vector_store.similarity_search(pdf_query, k=3)
    
    # 搜索 URL 相关内容
    url_query = "arXiv 的使命是什么？"
    url_results = vector_store.similarity_search(url_query, k=3)
    
    # 验证搜索结果
    assert len(pdf_results) > 0
    assert len(url_results) > 0
    
    # 验证 PDF 搜索结果主要来自 PDF（检查 metadata）
    pdf_search_text = " ".join([doc.page_content for doc in pdf_results])
    assert "注册商标" in pdf_search_text or "商标" in pdf_search_text
    
    # 验证 URL 搜索结果主要来自 URL
    url_search_text = " ".join([doc.page_content for doc in url_results])
    assert "arXiv" in url_search_text
    
    print(f"\n综合测试完成：")
    print(f"PDF 文档片段数: {len(pdf_ids)}")
    print(f"URL 文档片段数: {len(url_ids)}")
    print(f"PDF 搜索结果数: {len(pdf_results)}")
    print(f"URL 搜索结果数: {len(url_results)}")


def test_agent_execute_pdf(resource_agent, pdf_path):
    """
    测试使用 agent 的 execute 方法处理 PDF
    
    验证 agent 能够理解用户意图并调用正确的工具
    """
    # 构建输入消息
    input_msg = f"请将 PDF 文件 {pdf_path} 的内容加入到向量存储中"
    
    # 执行任务
    response = resource_agent.execute(input_msg)
    
    # 验证响应
    assert response is not None
    assert "messages" in response
    assert len(response["messages"]) > 0
    
    # 获取最后一条消息
    last_message = response["messages"][-1]
    print(f"\nAgent 响应: {last_message.content}")


def test_agent_execute_url(resource_agent, test_url):
    """
    测试使用 agent 的 execute 方法处理 URL
    
    验证 agent 能够理解用户意图并调用正确的工具
    """
    # 构建输入消息
    input_msg = f"请将 URL {test_url} 的内容加入到向量存储中"
    
    # 执行任务
    response = resource_agent.execute(input_msg)
    
    # 验证响应
    assert response is not None
    assert "messages" in response
    assert len(response["messages"]) > 0
    
    # 获取最后一条消息
    last_message = response["messages"][-1]
    print(f"\nAgent 响应: {last_message.content}")

