"""
主控代理（Master Agent）：

负责用户交互、意图分析、任务规划和 Worker 调度。
接收用户问题后，Master 会选择合适的 Worker Agent 来收集信息，
然后整合所有信息，生成高质量的最终回答返回给用户。

本文件用户：AI Agent 开发者

结构与功能概述：
- 定义了 MasterAgent 类，是整个 Agentic RAG 系统的核心协调者
- 将各个专业 Worker（SQLAgent、RetrievalAgent、ResourceAgent）注册为工具
- 使用 LangChain 的 create_agent 实现智能调度
- 负责信息整合和最终答案的生成
- 实现自反思机制，确保答案质量

使用方式：
1. 在项目根目录的 .env 文件中配置 OPENAI_API_KEY、OPENAI_BASE_URL 和 OPENAI_MODEL
2. 确保 config/settings.py 中的配置正确（数据库路径、向量存储配置等）
3. 创建 MasterAgent 实例，系统会自动创建所有 Worker Agent（sql_worker、RetrievalAgent、ResourceAgent）
4. 调用 execute 方法，传入用户问题，即可获得完整的回答
"""

from .agent_base import BaseAgent
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import os
from dotenv import load_dotenv
from loguru import logger
from config.settings import (
    SQLITE_DB_PATH,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
    OPENAI_API_KEY,
    OPENAI_BASE_URL
)
from .sql_agent import sql_worker
from .retrieval_agent import RetrievalAgent
from .resource_agent import ResourceAgent

load_dotenv()

sys_prompt = """
你是一个智能学术助手 Master Agent，负责为用户提供专业的学术研究相关帮助。

你的核心能力：
1. 理解和分析用户的学术问题
2. 调度专业的 Worker Agent 收集信息
3. 整合多源信息并生成高质量回答

你可以调用的 Worker Agent：

- **query_database**: SQL Agent，用于查询本地数据库
  - 适用场景：用户询问已保存的论文、笔记、实验记录等结构化数据
  - 示例：查找某个主题的论文、获取实验结果统计等

- **search_documents**: Retrieval Agent，用于检索文档内容
  - 适用场景：用户需要了解已存储文档的具体内容和细节
  - 示例：某篇论文的具体方法、相关文献的观点等

- **process_resource**: Resource Agent，用于处理新资源
  - 适用场景：用户提供 PDF 文件或 URL，需要添加到知识库
  - 示例：处理新上传的论文、抓取网页内容等

工作流程：

1. **理解意图**：分析用户问题，判断需要哪些信息来源
2. **调度 Worker**：根据需求调用相应的 Worker Agent 获取信息
   - 可以串行调用多个 Worker
   - 可以根据前一个 Worker 的结果决定是否需要调用其他 Worker
3. **整合信息**：汇总所有 Worker 返回的信息
4. **生成答案**：基于收集的信息，生成准确、完整、有条理的回答
5. **自反思**：评估答案质量
   - 如果信息不足，考虑是否需要补充检索
   - 如果存在不确定性，明确告知用户

生成答案的原则：

- **忠实于信息**：严格基于 Worker 返回的信息进行回答，不编造内容
- **明确边界**：如果信息不完整或不存在，明确告知用户
- **结构化组织**：将回答组织成清晰的结构，便于理解
- **学术专业性**：保持学术性的语言风格，准确使用专业术语
- **标注来源**：在适当时引用信息来源，增强可信度
- **补充建议**：必要时提供进一步的研究方向或操作建议

记住：你不仅是信息的搬运工，更是智能的协调者和专业的答案生成者。
"""


class MasterAgent(BaseAgent):
    """
    主控代理
    
    负责用户交互、Worker 调度和答案生成。
    在初始化时自动创建所有 Worker Agent（sql_worker、RetrievalAgent、ResourceAgent），
    配置从 config/settings.py 读取。
    """
    
    def __init__(self):
        """
        初始化主控代理
        
        自动创建所有 Worker Agent（sql_worker、RetrievalAgent、ResourceAgent），
        配置从 config/settings.py 读取。
        """
        # 从环境变量获取配置
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY 未设置，请在 .env 文件中配置")
        
        # 初始化 LLM 模型
        self.model = init_chat_model(
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
        
        # 创建 Embeddings 实例（用于向量存储）
        embeddings = OpenAIEmbeddings(
            api_key=api_key,
            base_url=base_url
        )
        
        # 创建 Chroma 向量存储实例
        vector_store = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY
        )
        
        # 将 SQLite 路径转换为 URI 格式
        db_uri = f"sqlite:///{SQLITE_DB_PATH}"
        
        # 创建所有 Worker Agent
        self.sql_worker = sql_worker(db_url=db_uri)
        self.retrieval_agent = RetrievalAgent(vector_store=vector_store)
        self.resource_agent = ResourceAgent(vector_store=vector_store)
        
        logger.info("已创建所有 Worker Agent：sql_worker、RetrievalAgent、ResourceAgent")
        
        # 创建工具列表
        self.tools = self._create_tools()
        
        # 创建 Agent
        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=sys_prompt,
        )
        
        logger.info(f"MasterAgent 初始化完成，使用模型: {model}，注册工具数: {len(self.tools)}")
    
    def execute(self, user_input: str) -> Dict[str, Any]:
        """
        执行用户请求
        
        Args:
            user_input: 用户的问题或请求
            
        Returns:
            包含回答和元数据的字典
        """
        logger.info(f"MasterAgent 收到用户请求: {user_input}")
        
        # 调用 Agent 处理请求
        response = self.agent.invoke(
            input={
                "messages": [
                    HumanMessage(content=user_input)
                ]
            }
        )
        
        # 提取最终答案
        final_answer = response['messages'][-1].content if response.get('messages') else str(response)
        
        logger.info(f"MasterAgent 完成任务，生成回答长度: {len(final_answer)} 字符")
        
        return response
    
    def _create_tools(self) -> List:
        """
        创建工具列表
        
        将各个 Worker Agent 包装为 LangChain 工具，供 Master Agent 调用
        
        Returns:
            工具列表
        """
        tools = []
        
        # SQL Worker 工具
        if self.sql_worker:
            @tool
            def query_database(query: str) -> str:
                """
                查询本地数据库，获取论文、笔记、实验等结构化数据
                
                Args:
                    query: 自然语言查询描述
                    
                Returns:
                    查询结果
                """
                logger.info(f"MasterAgent 调用 SQL Worker: {query}")
                result = self.sql_worker.execute(query)
                return str(result['messages'][-1].content) if result.get('messages') else str(result)
            
            tools.append(query_database)
        
        # Retrieval Agent 工具
        if self.retrieval_agent:
            @tool
            def search_documents(query: str) -> str:
                """
                在已存储的文档中检索相关内容，支持语义搜索和关键词搜索
                
                Args:
                    query: 搜索查询
                    
                Returns:
                    检索到的相关文档内容
                """
                logger.info(f"MasterAgent 调用 Retrieval Agent: {query}")
                result = self.retrieval_agent.execute(query)
                return str(result['messages'][-1].content) if result.get('messages') else str(result)
            
            tools.append(search_documents)
        
        # Resource Agent 工具
        if self.resource_agent:
            @tool
            def process_resource(resource_info: str) -> str:
                """
                处理并添加新的资源到知识库，支持 PDF 文件和 URL
                
                Args:
                    resource_info: 资源信息（PDF 路径或 URL）
                    
                Returns:
                    处理结果
                """
                logger.info(f"MasterAgent 调用 Resource Agent: {resource_info}")
                result = self.resource_agent.execute(resource_info)
                return str(result['messages'][-1].content) if result.get('messages') else str(result)
            
            tools.append(process_resource)
        
        return tools
