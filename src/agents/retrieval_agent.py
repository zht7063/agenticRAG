"""
检索代理：

根据用户的问题，从指定的向量存储中检索相关内容，并返回给用户。

"""
from langchain_core.tools import tool
from .agent_base import BaseAgent
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from typing import List, Dict
from langchain_core.messages import HumanMessage

import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

sys_prompt = """

你是一个专业的文档检索助手，你的任务是根据用户的查询需求，从向量存储库中检索相关的文档内容。

你可以使用以下检索工具：

1. semantic_search - 语义检索，适合理解用户意图和概念相似度检索
2. keyword_search - 关键词检索，适合精确匹配特定术语或短语
3. hybrid_search - 混合检索，结合语义和关键词，适合复杂查询

你的工作步骤如下：

1. 理解用户的查询意图和需求
2. 判断使用哪种检索策略最合适：
   - 如果用户需要概念性、语义理解的内容，使用 semantic_search
   - 如果用户需要精确匹配特定关键词，使用 keyword_search
   - 如果查询比较复杂或需要综合考虑，使用 hybrid_search
3. 调用合适的工具进行检索
4. 整理检索结果，以清晰、结构化的方式返回给用户
5. 如果结果为空，告知用户未找到相关内容

"""

class RetrievalAgent(BaseAgent):
    def __init__(self, vector_store: Chroma):
        """
        初始化检索代理
        
        Args:
            vector_store: Chroma 向量存储实例
        """
        self.vector_store = vector_store
        
        # 创建工具列表
        self.tools = self._create_tools()
        
        # 创建 Agent
        self.agent = create_agent(
            model = init_chat_model(
                model = os.getenv("OPENAI_MODEL"),
                base_url = os.getenv("OPENAI_BASE_URL"),
                api_key = os.getenv("OPENAI_API_KEY"),
            ),
            tools = self.tools,
            system_prompt = sys_prompt,
        )
        
        logger.info("RetrievalAgent 初始化完成")
    
    def execute(self, input_msg: str):
        """
        执行检索任务
        
        Args:
            input_msg: 用户的查询输入
            
        Returns:
            检索结果和说明
        """
        logger.info(f"Retrieval Agent 执行任务: {input_msg}")
        resp = self.agent.invoke(
            input = {
                "messages": [
                    HumanMessage(content=input_msg)
                ]
            })
        logger.info(f"Retrieval Agent 完成任务")
        return resp
    
    def semantic_search(self, query: str, top_k: int = 5, source_type: str = "all") -> List[Dict]:
        """
        语义检索
        
        根据查询文本在向量库中进行语义相似度检索，
        返回最相关的文档片段及其来源信息。
        
        Args:
            query: 查询文本
            top_k: 返回结果数量，默认 5
            source_type: 来源类型，可选 "pdf", "html", "all"
            
        Returns:
            检索结果列表，每项包含 content, source, score 字段
        """
        # 构建过滤条件
        filter_dict = None
        if source_type != "all":
            filter_dict = {"source_type": source_type}
        
        # 执行相似度检索
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=filter_dict
        )
        
        # 格式化结果
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "score": float(score),
                "metadata": doc.metadata
            })
        
        logger.info(f"语义检索完成，返回 {len(formatted_results)} 条结果")
        return formatted_results
    
    def keyword_search(self, keywords: str, top_k: int = 5) -> List[Dict]:
        """
        关键词检索
        
        根据关键词在文档库中进行精确匹配检索。
        
        Args:
            keywords: 搜索关键词
            top_k: 返回结果数量，默认 5
            
        Returns:
            检索结果列表
        """
        # 使用 Chroma 的 similarity_search 配合关键词过滤
        results = self.vector_store.similarity_search(
            query=keywords,
            k=top_k * 2,  # 获取更多结果用于过滤
        )
        
        # 格式化结果并进行关键词过滤
        formatted_results = []
        for doc in results:
            # 简单的关键词相关性判断（包含关键词的结果）
            if keywords.lower() in doc.page_content.lower():
                formatted_results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "metadata": doc.metadata
                })
        
        logger.info(f"关键词检索完成，返回 {len(formatted_results[:top_k])} 条结果")
        return formatted_results[:top_k]
    
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        混合检索
        
        结合语义检索和关键词检索，对结果进行融合和重排序。
        
        Args:
            query: 查询文本
            top_k: 返回结果数量，默认 5
            
        Returns:
            融合后的检索结果列表
        """
        # 执行语义检索
        semantic_results = self.semantic_search(query, top_k=top_k*2)
        
        # 执行关键词检索
        keyword_results = self.keyword_search(query, top_k=top_k*2)
        
        # 结果融合去重（基于内容）
        seen_content = set()
        merged_results = []
        
        # 优先语义检索结果
        for result in semantic_results:
            if result["content"] not in seen_content:
                seen_content.add(result["content"])
                merged_results.append(result)
        
        # 补充关键词检索结果
        for result in keyword_results:
            if result["content"] not in seen_content:
                seen_content.add(result["content"])
                merged_results.append(result)
        
        logger.info(f"混合检索完成，返回 {len(merged_results[:top_k])} 条结果")
        return merged_results[:top_k]
    
    def _create_tools(self) -> List:
        """
        创建检索工具列表
        
        将检索方法包装为 LangChain 工具
        
        Returns:
            工具列表
        """
        @tool
        def semantic_search(query: str, top_k: int = 5) -> List[Dict]:
            """
            语义检索工具：根据查询文本进行语义相似度检索
            
            Args:
                query: 查询文本
                top_k: 返回结果数量，默认 5
                
            Returns:
                检索结果列表
            """
            return self.semantic_search(query, top_k)
        
        @tool
        def keyword_search(keywords: str, top_k: int = 5) -> List[Dict]:
            """
            关键词检索工具：根据关键词进行精确匹配检索
            
            Args:
                keywords: 搜索关键词
                top_k: 返回结果数量，默认 5
                
            Returns:
                检索结果列表
            """
            return self.keyword_search(keywords, top_k)
        
        @tool
        def hybrid_search(query: str, top_k: int = 5) -> List[Dict]:
            """
            混合检索工具：结合语义和关键词检索，适合复杂查询
            
            Args:
                query: 查询文本
                top_k: 返回结果数量，默认 5
                
            Returns:
                融合后的检索结果列表
            """
            return self.hybrid_search(query, top_k)
        
        return [semantic_search, keyword_search, hybrid_search]
