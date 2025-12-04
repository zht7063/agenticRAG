"""
SchemaManager 单元测试

测试数据库 Schema 管理的核心功能：
1. Schema 初始化
2. 表结构验证
3. 版本管理
4. Schema 信息查询
"""

import pytest

from src.services.database.schema import SchemaManager, CURRENT_SCHEMA_VERSION


class TestSchemaInitialization:
    """Schema 初始化测试"""
    
    def test_initialize_schema_creates_all_tables(self, db_connection):
        """测试初始化创建所有核心表"""
        schema_manager = SchemaManager(db_connection)
        schema_manager.initialize_schema()
        
        expected_tables = [
            "papers",
            "collections",
            "collection_papers",
            "notes",
            "experiments",
            "inspirations",
            "schema_version"
        ]
        
        for table in expected_tables:
            assert schema_manager.table_exists(table), f"表 {table} 应该存在"
    
    def test_initialize_schema_is_idempotent(self, db_connection):
        """测试多次初始化是安全的（幂等性）"""
        schema_manager = SchemaManager(db_connection)
        
        # 第一次初始化
        schema_manager.initialize_schema()
        tables_after_first = schema_manager.get_all_tables()
        
        # 第二次初始化
        schema_manager.initialize_schema()
        tables_after_second = schema_manager.get_all_tables()
        
        assert tables_after_first == tables_after_second
    
    def test_initialize_schema_records_version(self, db_connection):
        """测试初始化记录 Schema 版本"""
        schema_manager = SchemaManager(db_connection)
        schema_manager.initialize_schema()
        
        version = schema_manager.get_schema_version()
        assert version == CURRENT_SCHEMA_VERSION


class TestSchemaVersionManagement:
    """Schema 版本管理测试"""
    
    def test_get_schema_version_returns_none_before_init(self, db_connection):
        """测试初始化前版本为 None"""
        schema_manager = SchemaManager(db_connection)
        
        # 未初始化时，schema_version 表不存在
        version = schema_manager.get_schema_version()
        assert version is None
    
    def test_get_schema_version_returns_current_after_init(self, initialized_db):
        """测试初始化后返回当前版本"""
        schema_manager = SchemaManager(initialized_db)
        
        version = schema_manager.get_schema_version()
        assert version == CURRENT_SCHEMA_VERSION


class TestSchemaVerification:
    """Schema 验证测试"""
    
    def test_verify_schema_valid(self, initialized_db):
        """测试验证完整 Schema"""
        schema_manager = SchemaManager(initialized_db)
        
        result = schema_manager.verify_schema()
        
        assert result["is_valid"] is True
        assert len(result["missing_tables"]) == 0
        assert result["schema_version"] == CURRENT_SCHEMA_VERSION
    
    def test_verify_schema_detects_missing_tables(self, db_connection):
        """测试验证检测缺失的表"""
        schema_manager = SchemaManager(db_connection)
        
        # 只创建部分表
        db_connection.execute("CREATE TABLE papers (id INTEGER PRIMARY KEY)")
        db_connection.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
        db_connection.commit()
        
        result = schema_manager.verify_schema()
        
        assert result["is_valid"] is False
        assert "collections" in result["missing_tables"]
        assert "notes" in result["missing_tables"]


class TestTableInformation:
    """表信息查询测试"""
    
    def test_get_all_tables(self, initialized_db):
        """测试获取所有表名"""
        schema_manager = SchemaManager(initialized_db)
        
        tables = schema_manager.get_all_tables()
        
        assert "papers" in tables
        assert "collections" in tables
        assert "notes" in tables
    
    def test_table_exists_returns_true(self, initialized_db):
        """测试表存在返回 True"""
        schema_manager = SchemaManager(initialized_db)
        
        assert schema_manager.table_exists("papers")
        assert schema_manager.table_exists("notes")
    
    def test_table_exists_returns_false(self, initialized_db):
        """测试表不存在返回 False"""
        schema_manager = SchemaManager(initialized_db)
        
        assert not schema_manager.table_exists("nonexistent_table")
    
    def test_get_table_info(self, initialized_db):
        """测试获取表结构信息"""
        schema_manager = SchemaManager(initialized_db)
        
        info = schema_manager.get_table_info("papers")
        
        # 验证关键列存在
        column_names = [col["name"] for col in info]
        assert "id" in column_names
        assert "title" in column_names
        assert "authors" in column_names
        assert "abstract" in column_names
        assert "keywords" in column_names


class TestTableStructure:
    """表结构验证测试"""
    
    def test_papers_table_structure(self, initialized_db):
        """测试 papers 表结构"""
        schema_manager = SchemaManager(initialized_db)
        info = schema_manager.get_table_info("papers")
        
        columns = {col["name"]: col for col in info}
        
        # 验证必要列
        assert "id" in columns
        assert "title" in columns
        assert "authors" in columns
        assert "abstract" in columns
        assert "keywords" in columns
        assert "publish_date" in columns
        assert "venue" in columns
        assert "doi" in columns
        assert "url" in columns
        assert "pdf_path" in columns
        assert "vector_ids" in columns
        assert "created_at" in columns
        assert "updated_at" in columns
        
        # 验证 id 是主键
        assert columns["id"]["pk"] == 1
    
    def test_collections_table_structure(self, initialized_db):
        """测试 collections 表结构"""
        schema_manager = SchemaManager(initialized_db)
        info = schema_manager.get_table_info("collections")
        
        columns = {col["name"]: col for col in info}
        
        assert "id" in columns
        assert "name" in columns
        assert "description" in columns
        assert "tags" in columns
        assert "created_at" in columns
    
    def test_notes_table_structure(self, initialized_db):
        """测试 notes 表结构"""
        schema_manager = SchemaManager(initialized_db)
        info = schema_manager.get_table_info("notes")
        
        columns = {col["name"]: col for col in info}
        
        assert "id" in columns
        assert "paper_id" in columns
        assert "content" in columns
        assert "note_type" in columns
        assert "page_number" in columns
        assert "created_at" in columns
    
    def test_experiments_table_structure(self, initialized_db):
        """测试 experiments 表结构"""
        schema_manager = SchemaManager(initialized_db)
        info = schema_manager.get_table_info("experiments")
        
        columns = {col["name"]: col for col in info}
        
        assert "id" in columns
        assert "name" in columns
        assert "description" in columns
        assert "parameters" in columns
        assert "results" in columns
        assert "related_papers" in columns
        assert "status" in columns
    
    def test_inspirations_table_structure(self, initialized_db):
        """测试 inspirations 表结构"""
        schema_manager = SchemaManager(initialized_db)
        info = schema_manager.get_table_info("inspirations")
        
        columns = {col["name"]: col for col in info}
        
        assert "id" in columns
        assert "title" in columns
        assert "content" in columns
        assert "source_papers" in columns
        assert "priority" in columns
        assert "status" in columns


class TestSchemaManagement:
    """Schema 管理操作测试"""
    
    def test_drop_table(self, initialized_db):
        """测试删除单个表"""
        schema_manager = SchemaManager(initialized_db)
        
        # 创建一个测试表
        initialized_db.execute("CREATE TABLE drop_test (id INTEGER)")
        initialized_db.commit()
        
        assert schema_manager.table_exists("drop_test")
        
        schema_manager.drop_table("drop_test")
        
        assert not schema_manager.table_exists("drop_test")
    
    def test_drop_all_tables(self, initialized_db):
        """测试删除所有表"""
        schema_manager = SchemaManager(initialized_db)
        
        # 确认表存在
        assert len(schema_manager.get_all_tables()) > 0
        
        schema_manager.drop_all_tables()
        
        # 所有表应该被删除
        assert len(schema_manager.get_all_tables()) == 0


class TestMigration:
    """Schema 迁移测试"""
    
    def test_migrate_initializes_when_no_schema(self, db_connection):
        """测试无 Schema 时迁移执行初始化"""
        schema_manager = SchemaManager(db_connection)
        
        assert schema_manager.get_schema_version() is None
        
        schema_manager.migrate(CURRENT_SCHEMA_VERSION)
        
        assert schema_manager.get_schema_version() == CURRENT_SCHEMA_VERSION
    
    def test_migrate_skips_when_up_to_date(self, initialized_db):
        """测试已是最新版本时跳过迁移"""
        schema_manager = SchemaManager(initialized_db)
        
        current_version = schema_manager.get_schema_version()
        
        # 迁移到当前版本应该不做任何事
        schema_manager.migrate(current_version)
        
        assert schema_manager.get_schema_version() == current_version

