"""
services 模块 - 服务层

本模块包含 ScholarRAG 系统的核心服务：

1. datasource - 数据源服务
   - PDFVectorStore: PDF 文档向量存储
   - HTMLVectorStore: HTML 网页向量存储
   - SQLVectorStore: SQL Schema 语义化存储

2. database - 数据库服务
   - connection: SQLite 连接管理
   - schema: 数据库表结构定义
   - repository: 数据访问层（CRUD 操作）

3. mcp - MCP 协议服务
   - MCPWebSearch: 网络搜索服务
   - MCPFetch: 网页抓取服务
   - MCPTime: 时间服务
"""

