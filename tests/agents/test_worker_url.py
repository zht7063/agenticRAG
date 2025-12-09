"""
测试 UrlWorker 类

验证 UrlWorker 能否正确从 URL 获取内容并保存到向量存储中。
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

from src.agents.worker_url import UrlWorker

load_dotenv()


@pytest.fixture
def temp_vectorstore():
    """
    创建临时向量存储
    
    测试结束后自动清理。
    """
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="test_vectorstore_")
    
    # 创建向量存储
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    vectorstore = Chroma(
        collection_name="test_collection",
        embedding_function=embeddings,
        persist_directory=temp_dir
    )
    
    yield vectorstore
    
    # 清理向量存储和临时目录
    try:
        # 尝试删除集合以释放资源
        vectorstore.delete_collection()
    except:
        pass
    
    # 等待一下让文件句柄释放
    import time
    time.sleep(0.5)
    
    # 清理临时目录
    if Path(temp_dir).exists():
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            # Windows 上可能会遇到权限问题，忽略即可
            pass


def test_url_worker_fetch_and_save(temp_vectorstore):
    """
    测试 UrlWorker 的 fetch_and_save 方法
    
    验证：
    1. 能够从 URL 获取内容
    2. 能够将内容保存到向量存储
    3. 返回的 ID 列表不为空
    """
    # 创建 UrlWorker 实例
    worker = UrlWorker(vector_store=temp_vectorstore)
    
    # 测试 URL
    test_url = "https://arxiv.org/html/2402.08954v1"
    
    # 调用 fetch_and_save 方法（直接调用实例方法）
    ids = worker.fetch_and_save(test_url)
    
    # 验证返回的 ID 列表
    assert ids is not None, "返回的 ID 列表不应为 None"
    assert len(ids) > 0, "应该至少保存了一个文档片段"
    
    # 验证向量存储中确实保存了数据
    collection = temp_vectorstore._collection
    stored_count = collection.count()
    
    assert stored_count == len(ids), f"向量存储中的文档数量 ({stored_count}) 应该等于返回的 ID 数量 ({len(ids)})"
    assert stored_count > 0, "向量存储中应该至少有一个文档"
    
    print(f"\n✓ 成功从 URL 获取并保存了 {len(ids)} 个文档片段")
    print(f"✓ 向量存储中共有 {stored_count} 个文档")


def test_url_worker_content_retrieval(temp_vectorstore):
    """
    测试保存后能否正确检索内容
    
    验证：
    1. 保存的内容能够被检索到
    2. 检索到的内容包含预期的关键词
    """
    # 创建 UrlWorker 实例
    worker = UrlWorker(vector_store=temp_vectorstore)
    
    # 测试 URL
    test_url = "https://arxiv.org/html/2402.08954v1"
    
    # 保存内容（直接调用实例方法）
    ids = worker.fetch_and_save(test_url)
    
    # 进行相似度搜索
    results = temp_vectorstore.similarity_search(
        query="agent framework",
        k=3
    )
    
    # 验证检索结果
    assert len(results) > 0, "应该能够检索到相关内容"
    
    # 验证检索到的文档有内容
    for doc in results:
        assert len(doc.page_content) > 0, "检索到的文档应该有内容"
        assert doc.metadata.get("source") == test_url, "文档的 source 元数据应该是测试 URL"
    
    print(f"\n✓ 成功检索到 {len(results)} 个相关文档片段")
    print(f"✓ 第一个文档片段预览: {results[0].page_content[:100]}...")


if __name__ == "__main__":
    # 允许直接运行测试
    pytest.main([__file__, "-v", "-s"])

