"""
Retrieval Worker - 检索 Worker

负责文献检索和向量搜索，是 ScholarRAG 的核心检索能力：

1. 向量检索
   - 基于语义相似度的文档检索
   - 支持 PDF 和 HTML 来源的向量库
   - 可配置 top-k 和相似度阈值

2. 混合检索
   - 关键词检索 + 向量语义检索的融合
   - 结果去重和重排序
   - 多源检索结果整合

3. Query 处理
   - Query Rewrite 优化长难查询
   - 关键词提取和扩展
   - 多轮对话上下文感知

4. 结果处理
   - 检索结果排序和筛选
   - 来源信息附加（文档名、页码、章节）
   - 相关度评分

触发场景:
- 用户提问涉及已存储的文献内容
- Master 判断需要本地知识库支持
- 自反思阶段需要补充上下文
"""

from .worker_base import BaseWorker


class RetrievalWorker(BaseWorker):
    """检索 Worker，负责文献检索和向量搜索"""
    
    def __init__(self):
        super().__init__(name="retrieval")
        self.vector_store = None  # 向量存储服务实例
    
    def execute(self, task: dict) -> dict:
        """
        执行检索任务
        
        Args:
            task: {
                "type": "retrieval",
                "query": str,           # 检索查询
                "top_k": int,           # 返回数量
                "sources": list,        # 指定检索源 ["pdf", "html"]
                "threshold": float      # 相似度阈值
            }
        """
        pass
    
    def semantic_search(self, query: str, top_k: int = 5) -> list:
        """语义向量检索"""
        pass
    
    def keyword_search(self, query: str, top_k: int = 5) -> list:
        """关键词检索"""
        pass
    
    def hybrid_search(self, query: str, top_k: int = 5) -> list:
        """混合检索（语义 + 关键词）"""
        pass
    
    def rewrite_query(self, query: str, context: list = None) -> str:
        """Query Rewrite 优化查询"""
        pass

