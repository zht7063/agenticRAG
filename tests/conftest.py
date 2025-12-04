"""
pytest 配置和共享 fixtures

提供测试所需的通用配置和资源：

1. 临时数据库 fixture
2. 数据库连接 fixture
3. Schema 初始化 fixture
4. Repository fixtures
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.services.database.connection import DatabaseConnection
from src.services.database.schema import SchemaManager
from src.services.database.repository import (
    PaperRepository,
    CollectionRepository,
    NoteRepository,
    ExperimentRepository,
    InspirationRepository,
)


# ============================================================
# 数据库 Fixtures
# ============================================================

@pytest.fixture
def temp_db_path():
    """
    创建临时数据库文件路径
    
    测试结束后自动清理文件。
    """
    # 创建临时文件
    fd, path = tempfile.mkstemp(suffix=".db", prefix="test_scholar_")
    os.close(fd)
    
    yield path
    
    # 清理临时文件
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def db_connection(temp_db_path):
    """
    创建测试用数据库连接
    
    每个测试使用独立的临时数据库，测试结束后自动清理。
    """
    # 重置单例，确保每个测试使用新的连接
    DatabaseConnection.reset_instance()
    
    db = DatabaseConnection(temp_db_path)
    db.connect()
    
    yield db
    
    db.close()
    DatabaseConnection.reset_instance()


@pytest.fixture
def initialized_db(db_connection):
    """
    已初始化 Schema 的数据库连接
    
    包含所有表结构，可直接进行数据操作。
    """
    schema_manager = SchemaManager(db_connection)
    schema_manager.initialize_schema()
    
    return db_connection


# ============================================================
# Repository Fixtures
# ============================================================

@pytest.fixture
def paper_repo(initialized_db):
    """PaperRepository 实例"""
    return PaperRepository(initialized_db)


@pytest.fixture
def collection_repo(initialized_db):
    """CollectionRepository 实例"""
    return CollectionRepository(initialized_db)


@pytest.fixture
def note_repo(initialized_db):
    """NoteRepository 实例"""
    return NoteRepository(initialized_db)


@pytest.fixture
def experiment_repo(initialized_db):
    """ExperimentRepository 实例"""
    return ExperimentRepository(initialized_db)


@pytest.fixture
def inspiration_repo(initialized_db):
    """InspirationRepository 实例"""
    return InspirationRepository(initialized_db)


# ============================================================
# 测试数据 Fixtures
# ============================================================

@pytest.fixture
def sample_paper():
    """示例论文数据"""
    from src.services.database.repository import Paper
    return Paper(
        title="Attention Is All You Need",
        authors="Vaswani, Shazeer, Parmar, et al.",
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
        keywords="transformer, attention, neural network",
        publish_date="2017-06-12",
        venue="NeurIPS 2017",
        doi="10.48550/arXiv.1706.03762",
        url="https://arxiv.org/abs/1706.03762"
    )


@pytest.fixture
def sample_collection():
    """示例合集数据"""
    from src.services.database.repository import Collection
    return Collection(
        name="Transformer 相关论文",
        description="自注意力机制相关研究",
        tags="transformer,attention,deep learning"
    )


@pytest.fixture
def sample_note():
    """示例笔记数据"""
    from src.services.database.repository import Note
    return Note(
        content="关键创新: 完全基于自注意力机制，摒弃了循环和卷积结构",
        note_type="highlight",
        page_number=1
    )


@pytest.fixture
def sample_experiment():
    """示例实验数据"""
    from src.services.database.repository import Experiment
    return Experiment(
        name="BERT 微调实验",
        description="在自定义数据集上微调 BERT 模型",
        parameters='{"learning_rate": 2e-5, "epochs": 3, "batch_size": 32}',
        status="planned"
    )


@pytest.fixture
def sample_inspiration():
    """示例灵感数据"""
    from src.services.database.repository import Inspiration
    return Inspiration(
        title="多模态 RAG 优化方向",
        content="结合视觉和文本信息进行检索增强",
        priority="high",
        status="new"
    )

