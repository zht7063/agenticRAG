"""
SQL Worker - 数据库查询 Worker

负责 SQLite 数据库的查询和操作：

1. NL2SQL 转换
   - 将自然语言查询转换为 SQL 语句
   - 支持复杂查询（JOIN、聚合、子查询）
   - SQL 语法验证和安全检查

2. 论文元数据查询
   - 按条件检索论文信息
   - 文献合集管理查询
   - 作者、关键词、时间等多维度筛选

3. 研究数据管理
   - 研究笔记的 CRUD 操作
   - 实验记录的查询和更新
   - 研究灵感的检索和关联

4. Schema 感知
   - 自动获取数据库结构信息
   - 根据 Schema 生成准确的 SQL
   - 字段类型和约束感知

触发场景:
- 用户查询涉及结构化数据（论文列表、实验数据等）
- 需要按特定条件筛选文献
- 管理研究笔记和实验记录
"""

from .worker_base import BaseWorker


class SQLWorker(BaseWorker):
    """SQL Worker，负责数据库查询和操作"""
    
    def __init__(self):
        super().__init__(name="sql")
        self.db_connection = None  # 数据库连接
        self.schema_info = None    # Schema 缓存
    
    def execute(self, task: dict) -> dict:
        """
        执行数据库查询任务
        
        Args:
            task: {
                "type": "sql",
                "query": str,           # 自然语言查询
                "operation": str,       # 操作类型 ["select", "insert", "update", "delete"]
                "table": str,           # 目标表（可选）
                "raw_sql": str          # 直接执行的 SQL（可选）
            }
        """
        pass
    
    def nl2sql(self, natural_query: str) -> str:
        """将自然语言转换为 SQL"""
        pass
    
    def execute_sql(self, sql: str) -> list:
        """执行 SQL 并返回结果"""
        pass
    
    def get_schema(self) -> dict:
        """获取数据库 Schema 信息"""
        pass
    
    def validate_sql(self, sql: str) -> bool:
        """验证 SQL 语法和安全性"""
        pass
    
    def query_papers(self, conditions: dict) -> list:
        """查询论文元数据"""
        pass
    
    def query_notes(self, paper_id: int = None) -> list:
        """查询研究笔记"""
        pass
    
    def query_experiments(self, status: str = None) -> list:
        """查询实验记录"""
        pass

