"""
数据库查看脚本 - 展示数据库内容

用于查看演示数据库中的数据，验证数据填充结果。

功能：
1. 显示所有论文列表
2. 显示合集及其包含的论文
3. 显示笔记内容
4. 显示实验记录
5. 显示研究灵感

使用方式：
    python scripts/view_database.py [数据库路径]
    
    不指定路径时默认使用: data/scholar_demo.db
"""

import sys
from pathlib import Path
from typing import Optional

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.database.connection import DatabaseConnection
from src.services.database.repository import (
    PaperRepository, CollectionRepository, NoteRepository,
    ExperimentRepository, InspirationRepository
)
from src.utils.helpers.logger import get_logger

logger = get_logger("view_database")


def print_separator(char="=", length=80):
    """打印分隔线"""
    print(char * length)


def print_section_header(title: str):
    """打印章节标题"""
    print()
    print_separator()
    print(f"📚 {title}")
    print_separator()


def view_papers(paper_repo: PaperRepository):
    """查看所有论文"""
    print_section_header("论文列表")
    
    papers = paper_repo.get_all()
    
    if not papers:
        print("暂无论文数据")
        return
    
    print(f"\n共 {len(papers)} 篇论文:\n")
    
    for i, paper in enumerate(papers, 1):
        print(f"{i}. 【{paper.id}】 {paper.title}")
        print(f"   作者: {paper.authors[:80]}{'...' if len(paper.authors) > 80 else ''}")
        print(f"   发表: {paper.venue} ({paper.publish_date})")
        print(f"   关键词: {paper.keywords}")
        if paper.url:
            print(f"   链接: {paper.url}")
        print()


def view_collections(collection_repo: CollectionRepository, paper_repo: PaperRepository):
    """查看所有合集"""
    print_section_header("文献合集")
    
    collections = collection_repo.get_all()
    
    if not collections:
        print("暂无合集数据")
        return
    
    print(f"\n共 {len(collections)} 个合集:\n")
    
    for i, collection in enumerate(collections, 1):
        paper_count = collection_repo.get_paper_count(collection.id)
        print(f"{i}. 【合集 {collection.id}】 {collection.name} ({paper_count} 篇论文)")
        print(f"   描述: {collection.description}")
        print(f"   标签: {collection.tags}")
        
        # 显示合集中的论文
        papers = collection_repo.get_papers(collection.id)
        if papers:
            print(f"   包含论文:")
            for paper in papers:
                print(f"     - [{paper.id}] {paper.title}")
        print()


def view_notes(note_repo: NoteRepository, paper_repo: PaperRepository):
    """查看所有笔记"""
    print_section_header("研究笔记")
    
    notes = note_repo.get_all()
    
    if not notes:
        print("暂无笔记数据")
        return
    
    print(f"\n共 {len(notes)} 条笔记:\n")
    
    # 按笔记类型分组
    note_types = {"highlight": "📌 重点标注", "comment": "💬 评论", "question": "❓ 问题"}
    
    for note_type, type_name in note_types.items():
        type_notes = [n for n in notes if n.note_type == note_type]
        if type_notes:
            print(f"\n{type_name} ({len(type_notes)} 条):")
            print("-" * 70)
            for note in type_notes:
                paper = paper_repo.get_by_id(note.paper_id) if note.paper_id else None
                paper_title = paper.title if paper else "独立笔记"
                
                print(f"\n  📄 {paper_title}")
                if note.page_number:
                    print(f"  📍 第 {note.page_number} 页")
                print(f"  📝 {note.content}")


def view_experiments(experiment_repo: ExperimentRepository):
    """查看所有实验"""
    print_section_header("实验记录")
    
    experiments = experiment_repo.get_all()
    
    if not experiments:
        print("暂无实验数据")
        return
    
    print(f"\n共 {len(experiments)} 个实验:\n")
    
    # 按状态分组
    status_map = {
        "planned": "📝 计划中",
        "running": "🔄 进行中",
        "completed": "✅ 已完成"
    }
    
    for status, status_name in status_map.items():
        status_exps = [e for e in experiments if e.status == status]
        if status_exps:
            print(f"\n{status_name} ({len(status_exps)} 个):")
            print("-" * 70)
            for exp in status_exps:
                print(f"\n  🔬 {exp.name}")
                print(f"  📋 {exp.description}")
                if exp.parameters:
                    print(f"  ⚙️  参数: {exp.parameters}")
                if exp.results:
                    print(f"  📊 结果: {exp.results}")
                if exp.related_papers:
                    print(f"  🔗 关联论文: {exp.related_papers}")


def view_inspirations(inspiration_repo: InspirationRepository):
    """查看所有灵感"""
    print_section_header("研究灵感")
    
    inspirations = inspiration_repo.get_all()
    
    if not inspirations:
        print("暂无灵感数据")
        return
    
    print(f"\n共 {len(inspirations)} 条灵感:\n")
    
    # 按优先级分组
    priority_map = {
        "high": "🔥 高优先级",
        "medium": "⭐ 中优先级",
        "low": "💡 低优先级"
    }
    
    for priority, priority_name in priority_map.items():
        priority_ideas = [i for i in inspirations if i.priority == priority]
        if priority_ideas:
            print(f"\n{priority_name} ({len(priority_ideas)} 条):")
            print("-" * 70)
            for idea in priority_ideas:
                status_icon = {"new": "🆕", "exploring": "🔍", "archived": "📦"}.get(idea.status, "")
                print(f"\n  {status_icon} {idea.title}")
                print(f"  📝 {idea.content}")
                if idea.source_papers:
                    print(f"  🔗 来源论文: {idea.source_papers}")


def view_statistics(
    paper_repo: PaperRepository,
    collection_repo: CollectionRepository,
    note_repo: NoteRepository,
    experiment_repo: ExperimentRepository,
    inspiration_repo: InspirationRepository
):
    """显示统计信息"""
    print_section_header("数据统计")
    
    print(f"""
📊 数据总览:
  
  论文数量:   {paper_repo.count()} 篇
  合集数量:   {len(collection_repo.get_all())} 个
  笔记数量:   {len(note_repo.get_all())} 条
  实验数量:   {len(experiment_repo.get_all())} 个
  灵感数量:   {len(inspiration_repo.get_all())} 条

📈 详细统计:

  笔记类型分布:
    - 重点标注: {len(note_repo.get_by_type('highlight'))} 条
    - 评论:     {len(note_repo.get_by_type('comment'))} 条
    - 问题:     {len(note_repo.get_by_type('question'))} 条

  实验状态分布:
    - 已完成:   {len(experiment_repo.get_by_status('completed'))} 个
    - 进行中:   {len(experiment_repo.get_by_status('running'))} 个
    - 计划中:   {len(experiment_repo.get_by_status('planned'))} 个

  灵感优先级分布:
    - 高优先级: {len(inspiration_repo.get_by_priority('high'))} 条
    - 中优先级: {len(inspiration_repo.get_by_priority('medium'))} 条
    - 低优先级: {len(inspiration_repo.get_by_priority('low'))} 条
    """)


def main():
    """主函数"""
    # 获取数据库路径
    if len(sys.argv) > 1:
        db_path = Path(sys.argv[1])
    else:
        db_path = project_root / "data" / "scholar_demo.db"
    
    if not db_path.exists():
        print(f"❌ 错误: 数据库文件不存在: {db_path}")
        print(f"\n请先运行 seed_database.py 生成演示数据库:")
        print(f"  python scripts/seed_database.py")
        return
    
    print_separator("=", 80)
    print("📚 ScholarRAG 数据库查看器")
    print_separator("=", 80)
    print(f"数据库路径: {db_path}")
    
    # 连接数据库
    DatabaseConnection.reset_instance()
    db = DatabaseConnection(str(db_path))
    db.connect()
    
    # 创建 Repository 实例
    paper_repo = PaperRepository(db)
    collection_repo = CollectionRepository(db)
    note_repo = NoteRepository(db)
    experiment_repo = ExperimentRepository(db)
    inspiration_repo = InspirationRepository(db)
    
    # 显示各类数据
    view_statistics(paper_repo, collection_repo, note_repo, experiment_repo, inspiration_repo)
    view_papers(paper_repo)
    view_collections(collection_repo, paper_repo)
    view_notes(note_repo, paper_repo)
    view_experiments(experiment_repo)
    view_inspirations(inspiration_repo)
    
    # 结束
    print_separator()
    print("✨ 查看完成！")
    print_separator()
    
    # 关闭连接
    db.close()


if __name__ == "__main__":
    main()

