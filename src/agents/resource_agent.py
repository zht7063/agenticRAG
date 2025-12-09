"""
资源代理：

调用工具获取资源，并将其加入到指定的向量存储中。

"""
from langchain_core.tools import tool
from .worker_base import BaseWorker
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from src.services.datasource.html_toolkit import HTMLToolkit
from src.services.datasource.pdf_toolkit import PDFToolkit
from typing import List
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

sys_prompt = """

你是一个资源处理专家，你的任务是根据用户提供的资源（URL 或 PDF 文件），获取并解析内容，将解析后得到的内容加入到向量存储中。

对于 URL 资源，使用 save_url_content 工具；对于 PDF 文件，使用 save_pdf_content 工具。

你的工作步骤如下：

1. 判断用户提供的是哪种资源；
2. 根据资源类型，使用对应的工具存储资源；
3. 返回存储结果。

"""

class ResourceWorker(BaseWorker):
    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store
        
        # 创建工具列表，将实例方法包装为工具
        self.tools = self._create_tools()

        self.agent = create_agent(
            model = init_chat_model(
                model = os.getenv("OPENAI_MODEL"),
                base_url = os.getenv("OPENAI_BASE_URL"),
                api_key = os.getenv("OPENAI_API_KEY"),
            ),
            tools = self.tools,
            system_prompt = sys_prompt,
        )

    def execute(self, input_msg: str):
        """ 执行任务

        Args:
            input_msg: 输入消息
        """
        logger.info(f"Resource Worker 执行任务: {input_msg}")
        resp = self.agent.invoke(
            input = {
                "messages": [
                    HumanMessage(content=input_msg)
                ]
            })
        logger.info(f"Resource Worker 执行任务: {resp['messages'][-1].content}")
        return resp
    
    def save_pdf_content(self, pdf_path: str) -> List[str]:
        """
        根据 pdf 路径获取 pdf 内容并保存到向量存储中
        
        Args:
            pdf_path: pdf 路径
            
        Returns:
            保存的文档 ID 列表
        """
        splits = PDFToolkit().get_splits(pdf_path = pdf_path)
        ids = self.vector_store.add_documents(splits)
        logger.info(f"成功保存 {len(ids)} 个文档片段到向量存储")
        return ids


    def save_url_content(self, url: str) -> List[str]:
        """
        根据 url 获取 html 内容并保存到向量存储中
        
        Args:
            url: 网页 URL
            
        Returns:
            保存的文档 ID 列表
        """
        splits = HTMLToolkit().get_splits(url = url)  # 调用工具包 fetch html 内容
        ids = self.vector_store.add_documents(splits) # 调用向量存储工具包 add documents
        logger.info(f"成功保存 {len(ids)} 个文档片段到向量存储")
        return ids
    
    def _create_tools(self) -> List:
        """创建资源处理工具列表
        
        将 save_url_content 和 save_pdf_content 方法包装为 LangChain 工具
        
        Returns:
            工具列表
        """
        @tool
        def save_url_content(url: str) -> List[str]:
            """根据 URL 获取网页内容并保存到向量存储中
            
            Args:
                url: 网页地址
                
            Returns:
                保存的文档 ID 列表
            """
            return self.save_url_content(url)
        
        @tool
        def save_pdf_content(pdf_path: str) -> List[str]:
            """根据 PDF 路径获取文件内容并保存到向量存储中
            
            Args:
                pdf_path: PDF 文件路径
                
            Returns:
                保存的文档 ID 列表
            """
            return self.save_pdf_content(pdf_path)
        
        return [save_url_content, save_pdf_content]
