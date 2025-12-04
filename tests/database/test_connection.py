"""
DatabaseConnection 单元测试

测试数据库连接管理的核心功能：
1. 连接建立和关闭
2. 单例模式
3. SQL 执行
4. 事务管理
5. 工具方法
"""

import pytest
import sqlite3
from pathlib import Path

from src.services.database.connection import DatabaseConnection


class TestDatabaseConnection:
    """数据库连接基础功能测试"""
    
    def test_connect_creates_database_file(self, temp_db_path):
        """测试连接时自动创建数据库文件"""
        DatabaseConnection.reset_instance()
        
        db = DatabaseConnection(temp_db_path)
        db.connect()
        
        assert Path(temp_db_path).exists()
        db.close()
    
    def test_connect_returns_connection(self, db_connection):
        """测试 connect 返回有效连接"""
        assert db_connection.is_connected
    
    def test_close_disconnects(self, temp_db_path):
        """测试 close 断开连接"""
        DatabaseConnection.reset_instance()
        
        db = DatabaseConnection(temp_db_path)
        db.connect()
        assert db.is_connected
        
        db.close()
        assert not db.is_connected
    
    def test_connect_without_path_raises_error(self):
        """测试无路径连接抛出错误"""
        DatabaseConnection.reset_instance()
        
        db = DatabaseConnection(None)
        
        with pytest.raises(ValueError, match="数据库路径未指定"):
            db.connect()


class TestSingletonPattern:
    """单例模式测试"""
    
    def test_singleton_returns_same_instance(self, temp_db_path):
        """测试单例返回相同实例"""
        DatabaseConnection.reset_instance()
        
        db1 = DatabaseConnection(temp_db_path)
        db2 = DatabaseConnection(temp_db_path)
        
        assert db1 is db2
        
        DatabaseConnection.reset_instance()
    
    def test_reset_instance_clears_singleton(self, temp_db_path):
        """测试 reset_instance 清除单例"""
        DatabaseConnection.reset_instance()
        
        db1 = DatabaseConnection(temp_db_path)
        db1.connect()
        
        DatabaseConnection.reset_instance()
        
        db2 = DatabaseConnection(temp_db_path)
        
        # 新实例应该是不同的对象（虽然 Python 可能复用内存地址）
        # 主要验证 _initialized 被重置
        assert not db2.is_connected


class TestSQLExecution:
    """SQL 执行测试"""
    
    def test_execute_creates_table(self, db_connection):
        """测试执行建表语句"""
        db_connection.execute("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name TEXT
            )
        """)
        db_connection.commit()
        
        assert db_connection.table_exists("test_table")
    
    def test_execute_with_params(self, db_connection):
        """测试参数化查询"""
        db_connection.execute("""
            CREATE TABLE test_params (id INTEGER, value TEXT)
        """)
        db_connection.execute(
            "INSERT INTO test_params (id, value) VALUES (?, ?)",
            (1, "test_value")
        )
        db_connection.commit()
        
        row = db_connection.fetchone("SELECT * FROM test_params WHERE id = ?", (1,))
        assert row["value"] == "test_value"
    
    def test_executemany_batch_insert(self, db_connection):
        """测试批量插入"""
        db_connection.execute("""
            CREATE TABLE test_batch (id INTEGER, name TEXT)
        """)
        
        data = [(1, "a"), (2, "b"), (3, "c")]
        db_connection.executemany(
            "INSERT INTO test_batch (id, name) VALUES (?, ?)",
            data
        )
        db_connection.commit()
        
        count = db_connection.fetchval("SELECT COUNT(*) FROM test_batch")
        assert count == 3
    
    def test_executescript_multiple_statements(self, db_connection):
        """测试执行多条语句脚本"""
        script = """
            CREATE TABLE script_test1 (id INTEGER);
            CREATE TABLE script_test2 (id INTEGER);
            CREATE INDEX idx_test1 ON script_test1(id);
        """
        db_connection.executescript(script)
        
        assert db_connection.table_exists("script_test1")
        assert db_connection.table_exists("script_test2")


class TestQueryMethods:
    """查询方法测试"""
    
    @pytest.fixture(autouse=True)
    def setup_test_table(self, db_connection):
        """为每个测试创建测试表"""
        db_connection.execute("""
            CREATE TABLE query_test (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value INTEGER
            )
        """)
        db_connection.executemany(
            "INSERT INTO query_test (name, value) VALUES (?, ?)",
            [("alpha", 10), ("beta", 20), ("gamma", 30)]
        )
        db_connection.commit()
    
    def test_fetchall_returns_all_rows(self, db_connection):
        """测试 fetchall 返回所有行"""
        rows = db_connection.fetchall("SELECT * FROM query_test")
        assert len(rows) == 3
    
    def test_fetchone_returns_single_row(self, db_connection):
        """测试 fetchone 返回单行"""
        row = db_connection.fetchone("SELECT * FROM query_test WHERE name = ?", ("beta",))
        assert row["value"] == 20
    
    def test_fetchone_returns_none_when_not_found(self, db_connection):
        """测试 fetchone 无结果返回 None"""
        row = db_connection.fetchone("SELECT * FROM query_test WHERE name = ?", ("nonexistent",))
        assert row is None
    
    def test_fetchval_returns_single_value(self, db_connection):
        """测试 fetchval 返回单个值"""
        total = db_connection.fetchval("SELECT SUM(value) FROM query_test")
        assert total == 60
    
    def test_fetchval_returns_none_when_empty(self, db_connection):
        """测试 fetchval 无结果返回 None"""
        result = db_connection.fetchval(
            "SELECT value FROM query_test WHERE name = ?", 
            ("nonexistent",)
        )
        assert result is None


class TestTransactionManagement:
    """事务管理测试"""
    
    def test_commit_persists_changes(self, db_connection):
        """测试 commit 持久化更改"""
        db_connection.execute("CREATE TABLE commit_test (id INTEGER)")
        db_connection.execute("INSERT INTO commit_test VALUES (1)")
        db_connection.commit()
        
        # 验证数据已持久化
        count = db_connection.fetchval("SELECT COUNT(*) FROM commit_test")
        assert count == 1
    
    def test_rollback_reverts_changes(self, db_connection):
        """测试 rollback 回滚更改"""
        db_connection.execute("CREATE TABLE rollback_test (id INTEGER)")
        db_connection.commit()
        
        db_connection.execute("INSERT INTO rollback_test VALUES (1)")
        db_connection.rollback()
        
        # 验证数据已回滚
        count = db_connection.fetchval("SELECT COUNT(*) FROM rollback_test")
        assert count == 0


class TestContextManager:
    """上下文管理器测试"""
    
    def test_context_manager_auto_connects(self, temp_db_path):
        """测试上下文管理器自动连接"""
        DatabaseConnection.reset_instance()
        
        with DatabaseConnection(temp_db_path) as db:
            assert db.is_connected
    
    def test_context_manager_auto_commits_on_success(self, temp_db_path):
        """测试正常退出时自动提交"""
        DatabaseConnection.reset_instance()
        
        with DatabaseConnection(temp_db_path) as db:
            db.execute("CREATE TABLE ctx_test (id INTEGER)")
            db.execute("INSERT INTO ctx_test VALUES (1)")
        
        # 重新连接验证数据已提交
        DatabaseConnection.reset_instance()
        db2 = DatabaseConnection(temp_db_path)
        db2.connect()
        
        count = db2.fetchval("SELECT COUNT(*) FROM ctx_test")
        assert count == 1
        
        db2.close()
    
    def test_context_manager_closes_on_exit(self, temp_db_path):
        """测试退出时自动关闭连接"""
        DatabaseConnection.reset_instance()
        
        db = DatabaseConnection(temp_db_path)
        with db:
            pass
        
        assert not db.is_connected


class TestUtilityMethods:
    """工具方法测试"""
    
    def test_table_exists_returns_true(self, db_connection):
        """测试 table_exists 存在返回 True"""
        db_connection.execute("CREATE TABLE exists_test (id INTEGER)")
        db_connection.commit()
        
        assert db_connection.table_exists("exists_test")
    
    def test_table_exists_returns_false(self, db_connection):
        """测试 table_exists 不存在返回 False"""
        assert not db_connection.table_exists("nonexistent_table")
    
    def test_get_table_names(self, db_connection):
        """测试获取所有表名"""
        db_connection.execute("CREATE TABLE table_a (id INTEGER)")
        db_connection.execute("CREATE TABLE table_b (id INTEGER)")
        db_connection.commit()
        
        tables = db_connection.get_table_names()
        assert "table_a" in tables
        assert "table_b" in tables
    
    def test_get_table_info(self, db_connection):
        """测试获取表结构信息"""
        db_connection.execute("""
            CREATE TABLE info_test (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                value REAL
            )
        """)
        db_connection.commit()
        
        info = db_connection.get_table_info("info_test")
        
        assert len(info) == 3
        
        column_names = [col["name"] for col in info]
        assert "id" in column_names
        assert "name" in column_names
        assert "value" in column_names

