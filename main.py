"""
ScholarRAG - 科研文献智能助手

应用入口文件，提供：

1. CLI 交互模式
   - 命令行对话界面
   - 支持多轮对话
   - 命令解析（/help, /clear, /quit）

2. 初始化流程
   - 配置加载和验证
   - 数据库初始化
   - Agent 系统启动

3. 主要功能入口
   - 文献问答
   - 文档添加
   - 数据库查询

使用方式:
    python main.py              # 启动 CLI 交互模式
    python main.py --init       # 初始化数据库
    python main.py --add <path> # 添加文档
"""

import argparse
import asyncio
from pathlib import Path

from config.settings import ensure_directories, SQLITE_DB_PATH
from src.agents import MasterAgent
from src.services.database import DatabaseConnection, SchemaManager
from src.utils.helpers import setup_logging, get_logger


def init_database():
    """初始化数据库"""
    pass


def add_document(path: str):
    """添加文档到知识库"""
    pass


def cli_chat():
    """CLI 交互模式"""
    pass


def print_welcome():
    """打印欢迎信息"""
    welcome = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║   ScholarRAG - 科研文献智能助手                          ║
    ║                                                          ║
    ║   命令:                                                  ║
    ║     /help  - 显示帮助信息                                ║
    ║     /clear - 清空对话历史                                ║
    ║     /quit  - 退出程序                                    ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(welcome)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ScholarRAG - 科研文献智能助手")
    parser.add_argument("--init", action="store_true", help="初始化数据库")
    parser.add_argument("--add", type=str, help="添加文档（PDF 路径或 URL）")
    
    args = parser.parse_args()
    
    # 确保目录存在
    ensure_directories()
    
    # 配置日志
    setup_logging()
    
    if args.init:
        init_database()
    elif args.add:
        add_document(args.add)
    else:
        print_welcome()
        cli_chat()


if __name__ == "__main__":
    main()
