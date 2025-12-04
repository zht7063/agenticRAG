"""
Repository 单元测试

测试各 Repository 的 CRUD 操作：
1. PaperRepository - 论文元数据
2. CollectionRepository - 文献合集
3. NoteRepository - 研究笔记
4. ExperimentRepository - 实验记录
5. InspirationRepository - 研究灵感
"""

import pytest

from src.services.database.repository import (
    Paper, Collection, Note, Experiment, Inspiration,
    PaperRepository, CollectionRepository, NoteRepository,
    ExperimentRepository, InspirationRepository
)


# ============================================================
# PaperRepository 测试
# ============================================================

class TestPaperRepository:
    """论文元数据仓储测试"""
    
    def test_create_paper(self, paper_repo, sample_paper):
        """测试创建论文"""
        paper_id = paper_repo.create(sample_paper)
        
        assert paper_id is not None
        assert paper_id > 0
    
    def test_get_by_id(self, paper_repo, sample_paper):
        """测试根据 ID 获取论文"""
        paper_id = paper_repo.create(sample_paper)
        
        retrieved = paper_repo.get_by_id(paper_id)
        
        assert retrieved is not None
        assert retrieved.id == paper_id
        assert retrieved.title == sample_paper.title
        assert retrieved.authors == sample_paper.authors
    
    def test_get_by_id_not_found(self, paper_repo):
        """测试获取不存在的论文"""
        retrieved = paper_repo.get_by_id(99999)
        assert retrieved is None
    
    def test_get_all(self, paper_repo):
        """测试获取所有论文"""
        # 创建多篇论文
        for i in range(3):
            paper_repo.create(Paper(title=f"Paper {i}", authors=f"Author {i}"))
        
        papers = paper_repo.get_all()
        
        assert len(papers) == 3
    
    def test_get_all_with_pagination(self, paper_repo):
        """测试分页获取论文"""
        for i in range(10):
            paper_repo.create(Paper(title=f"Paper {i}"))
        
        # 获取前 5 条
        first_page = paper_repo.get_all(limit=5, offset=0)
        assert len(first_page) == 5
        
        # 获取后 5 条
        second_page = paper_repo.get_all(limit=5, offset=5)
        assert len(second_page) == 5
    
    def test_search_by_title(self, paper_repo):
        """测试按标题搜索"""
        paper_repo.create(Paper(title="Attention Is All You Need"))
        paper_repo.create(Paper(title="BERT: Pre-training of Deep Bidirectional Transformers"))
        paper_repo.create(Paper(title="GPT-3: Language Models are Few-Shot Learners"))
        
        results = paper_repo.search_by_title("Attention")
        
        assert len(results) == 1
        assert "Attention" in results[0].title
    
    def test_search_by_keywords(self, paper_repo):
        """测试按关键词搜索"""
        paper_repo.create(Paper(title="Paper A", keywords="transformer, attention"))
        paper_repo.create(Paper(title="Paper B", keywords="cnn, convolution"))
        paper_repo.create(Paper(title="Paper C", keywords="rnn, lstm"))
        
        results = paper_repo.search_by_keywords("transformer")
        
        assert len(results) == 1
        assert "transformer" in results[0].keywords
    
    def test_search_by_author(self, paper_repo):
        """测试按作者搜索"""
        paper_repo.create(Paper(title="Paper A", authors="Vaswani, Shazeer"))
        paper_repo.create(Paper(title="Paper B", authors="Devlin, Chang"))
        
        results = paper_repo.search_by_author("Vaswani")
        
        assert len(results) == 1
        assert "Vaswani" in results[0].authors
    
    def test_search_full_text(self, paper_repo):
        """测试全文搜索"""
        paper_repo.create(Paper(
            title="Deep Learning",
            abstract="Neural networks are powerful",
            keywords="deep learning",
            authors="Hinton"
        ))
        paper_repo.create(Paper(title="Machine Learning", keywords="ml"))
        
        # 搜索标题
        results = paper_repo.search("Deep")
        assert len(results) == 1
        
        # 搜索摘要
        results = paper_repo.search("Neural")
        assert len(results) == 1
        
        # 搜索关键词
        results = paper_repo.search("learning")
        assert len(results) == 2
    
    def test_update_paper(self, paper_repo, sample_paper):
        """测试更新论文"""
        paper_id = paper_repo.create(sample_paper)
        
        # 更新论文
        paper = paper_repo.get_by_id(paper_id)
        paper.title = "Updated Title"
        paper.venue = "Updated Venue"
        
        success = paper_repo.update(paper)
        
        assert success
        
        # 验证更新
        updated = paper_repo.get_by_id(paper_id)
        assert updated.title == "Updated Title"
        assert updated.venue == "Updated Venue"
    
    def test_delete_paper(self, paper_repo, sample_paper):
        """测试删除论文"""
        paper_id = paper_repo.create(sample_paper)
        
        success = paper_repo.delete(paper_id)
        
        assert success
        assert paper_repo.get_by_id(paper_id) is None
    
    def test_count(self, paper_repo):
        """测试论文计数"""
        assert paper_repo.count() == 0
        
        paper_repo.create(Paper(title="Paper 1"))
        paper_repo.create(Paper(title="Paper 2"))
        
        assert paper_repo.count() == 2
    
    def test_exists(self, paper_repo, sample_paper):
        """测试论文存在检查"""
        paper_id = paper_repo.create(sample_paper)
        
        assert paper_repo.exists(paper_id)
        assert not paper_repo.exists(99999)


# ============================================================
# CollectionRepository 测试
# ============================================================

class TestCollectionRepository:
    """文献合集仓储测试"""
    
    def test_create_collection(self, collection_repo, sample_collection):
        """测试创建合集"""
        coll_id = collection_repo.create(sample_collection)
        
        assert coll_id is not None
        assert coll_id > 0
    
    def test_get_by_id(self, collection_repo, sample_collection):
        """测试根据 ID 获取合集"""
        coll_id = collection_repo.create(sample_collection)
        
        retrieved = collection_repo.get_by_id(coll_id)
        
        assert retrieved is not None
        assert retrieved.name == sample_collection.name
    
    def test_get_all(self, collection_repo):
        """测试获取所有合集"""
        collection_repo.create(Collection(name="Collection 1"))
        collection_repo.create(Collection(name="Collection 2"))
        
        collections = collection_repo.get_all()
        
        assert len(collections) == 2
    
    def test_add_paper_to_collection(self, collection_repo, paper_repo, sample_paper):
        """测试添加论文到合集"""
        paper_id = paper_repo.create(sample_paper)
        coll_id = collection_repo.create(Collection(name="Test Collection"))
        
        success = collection_repo.add_paper(coll_id, paper_id)
        
        assert success
        assert collection_repo.get_paper_count(coll_id) == 1
    
    def test_add_paper_duplicate_ignored(self, collection_repo, paper_repo, sample_paper):
        """测试重复添加论文被忽略"""
        paper_id = paper_repo.create(sample_paper)
        coll_id = collection_repo.create(Collection(name="Test Collection"))
        
        collection_repo.add_paper(coll_id, paper_id)
        collection_repo.add_paper(coll_id, paper_id)  # 重复添加
        
        assert collection_repo.get_paper_count(coll_id) == 1
    
    def test_remove_paper_from_collection(self, collection_repo, paper_repo, sample_paper):
        """测试从合集移除论文"""
        paper_id = paper_repo.create(sample_paper)
        coll_id = collection_repo.create(Collection(name="Test Collection"))
        
        collection_repo.add_paper(coll_id, paper_id)
        assert collection_repo.get_paper_count(coll_id) == 1
        
        success = collection_repo.remove_paper(coll_id, paper_id)
        
        assert success
        assert collection_repo.get_paper_count(coll_id) == 0
    
    def test_get_papers_in_collection(self, collection_repo, paper_repo):
        """测试获取合集中的论文"""
        # 创建论文
        paper1_id = paper_repo.create(Paper(title="Paper 1"))
        paper2_id = paper_repo.create(Paper(title="Paper 2"))
        
        # 创建合集并添加论文
        coll_id = collection_repo.create(Collection(name="Test Collection"))
        collection_repo.add_paper(coll_id, paper1_id)
        collection_repo.add_paper(coll_id, paper2_id)
        
        # 获取合集论文
        papers = collection_repo.get_papers(coll_id)
        
        assert len(papers) == 2
        titles = [p.title for p in papers]
        assert "Paper 1" in titles
        assert "Paper 2" in titles
    
    def test_update_collection(self, collection_repo, sample_collection):
        """测试更新合集"""
        coll_id = collection_repo.create(sample_collection)
        
        collection = collection_repo.get_by_id(coll_id)
        collection.name = "Updated Name"
        collection.description = "Updated Description"
        
        success = collection_repo.update(collection)
        
        assert success
        
        updated = collection_repo.get_by_id(coll_id)
        assert updated.name == "Updated Name"
    
    def test_delete_collection(self, collection_repo, sample_collection):
        """测试删除合集"""
        coll_id = collection_repo.create(sample_collection)
        
        success = collection_repo.delete(coll_id)
        
        assert success
        assert collection_repo.get_by_id(coll_id) is None


# ============================================================
# NoteRepository 测试
# ============================================================

class TestNoteRepository:
    """研究笔记仓储测试"""
    
    def test_create_note(self, note_repo, sample_note):
        """测试创建笔记"""
        note_id = note_repo.create(sample_note)
        
        assert note_id is not None
        assert note_id > 0
    
    def test_create_note_with_paper(self, note_repo, paper_repo, sample_paper):
        """测试创建关联论文的笔记"""
        paper_id = paper_repo.create(sample_paper)
        
        note = Note(
            paper_id=paper_id,
            content="Important finding",
            note_type="highlight"
        )
        note_id = note_repo.create(note)
        
        retrieved = note_repo.get_by_id(note_id)
        assert retrieved.paper_id == paper_id
    
    def test_get_by_paper(self, note_repo, paper_repo, sample_paper):
        """测试获取论文的所有笔记"""
        paper_id = paper_repo.create(sample_paper)
        
        # 创建多条笔记
        note_repo.create(Note(paper_id=paper_id, content="Note 1", note_type="highlight"))
        note_repo.create(Note(paper_id=paper_id, content="Note 2", note_type="comment"))
        note_repo.create(Note(paper_id=None, content="Note 3", note_type="question"))  # 独立笔记
        
        notes = note_repo.get_by_paper(paper_id)
        
        assert len(notes) == 2
    
    def test_get_by_type(self, note_repo):
        """测试按类型获取笔记"""
        note_repo.create(Note(content="Highlight 1", note_type="highlight"))
        note_repo.create(Note(content="Highlight 2", note_type="highlight"))
        note_repo.create(Note(content="Comment 1", note_type="comment"))
        
        highlights = note_repo.get_by_type("highlight")
        
        assert len(highlights) == 2
    
    def test_update_note(self, note_repo, sample_note):
        """测试更新笔记"""
        note_id = note_repo.create(sample_note)
        
        note = note_repo.get_by_id(note_id)
        note.content = "Updated content"
        note.note_type = "comment"
        
        success = note_repo.update(note)
        
        assert success
        
        updated = note_repo.get_by_id(note_id)
        assert updated.content == "Updated content"
        assert updated.note_type == "comment"
    
    def test_delete_note(self, note_repo, sample_note):
        """测试删除笔记"""
        note_id = note_repo.create(sample_note)
        
        success = note_repo.delete(note_id)
        
        assert success
        assert note_repo.get_by_id(note_id) is None


# ============================================================
# ExperimentRepository 测试
# ============================================================

class TestExperimentRepository:
    """实验记录仓储测试"""
    
    def test_create_experiment(self, experiment_repo, sample_experiment):
        """测试创建实验"""
        exp_id = experiment_repo.create(sample_experiment)
        
        assert exp_id is not None
        assert exp_id > 0
    
    def test_get_by_id(self, experiment_repo, sample_experiment):
        """测试根据 ID 获取实验"""
        exp_id = experiment_repo.create(sample_experiment)
        
        retrieved = experiment_repo.get_by_id(exp_id)
        
        assert retrieved is not None
        assert retrieved.name == sample_experiment.name
    
    def test_get_all(self, experiment_repo):
        """测试获取所有实验"""
        experiment_repo.create(Experiment(name="Exp 1"))
        experiment_repo.create(Experiment(name="Exp 2"))
        
        experiments = experiment_repo.get_all()
        
        assert len(experiments) == 2
    
    def test_get_by_status(self, experiment_repo):
        """测试按状态获取实验"""
        experiment_repo.create(Experiment(name="Planned 1", status="planned"))
        experiment_repo.create(Experiment(name="Running 1", status="running"))
        experiment_repo.create(Experiment(name="Completed 1", status="completed"))
        
        planned = experiment_repo.get_by_status("planned")
        running = experiment_repo.get_by_status("running")
        
        assert len(planned) == 1
        assert len(running) == 1
    
    def test_update_status(self, experiment_repo, sample_experiment):
        """测试更新实验状态"""
        exp_id = experiment_repo.create(sample_experiment)
        
        success = experiment_repo.update_status(exp_id, "running")
        
        assert success
        
        updated = experiment_repo.get_by_id(exp_id)
        assert updated.status == "running"
    
    def test_update_experiment(self, experiment_repo, sample_experiment):
        """测试更新实验"""
        exp_id = experiment_repo.create(sample_experiment)
        
        exp = experiment_repo.get_by_id(exp_id)
        exp.results = '{"accuracy": 0.95}'
        exp.status = "completed"
        
        success = experiment_repo.update(exp)
        
        assert success
        
        updated = experiment_repo.get_by_id(exp_id)
        assert updated.results == '{"accuracy": 0.95}'
    
    def test_delete_experiment(self, experiment_repo, sample_experiment):
        """测试删除实验"""
        exp_id = experiment_repo.create(sample_experiment)
        
        success = experiment_repo.delete(exp_id)
        
        assert success
        assert experiment_repo.get_by_id(exp_id) is None


# ============================================================
# InspirationRepository 测试
# ============================================================

class TestInspirationRepository:
    """研究灵感仓储测试"""
    
    def test_create_inspiration(self, inspiration_repo, sample_inspiration):
        """测试创建灵感"""
        idea_id = inspiration_repo.create(sample_inspiration)
        
        assert idea_id is not None
        assert idea_id > 0
    
    def test_get_by_id(self, inspiration_repo, sample_inspiration):
        """测试根据 ID 获取灵感"""
        idea_id = inspiration_repo.create(sample_inspiration)
        
        retrieved = inspiration_repo.get_by_id(idea_id)
        
        assert retrieved is not None
        assert retrieved.title == sample_inspiration.title
    
    def test_get_all(self, inspiration_repo):
        """测试获取所有灵感"""
        inspiration_repo.create(Inspiration(title="Idea 1"))
        inspiration_repo.create(Inspiration(title="Idea 2"))
        
        ideas = inspiration_repo.get_all()
        
        assert len(ideas) == 2
    
    def test_get_by_priority(self, inspiration_repo):
        """测试按优先级获取灵感"""
        inspiration_repo.create(Inspiration(title="High 1", priority="high"))
        inspiration_repo.create(Inspiration(title="High 2", priority="high"))
        inspiration_repo.create(Inspiration(title="Medium 1", priority="medium"))
        
        high_priority = inspiration_repo.get_by_priority("high")
        
        assert len(high_priority) == 2
    
    def test_get_by_status(self, inspiration_repo):
        """测试按状态获取灵感"""
        inspiration_repo.create(Inspiration(title="New 1", status="new"))
        inspiration_repo.create(Inspiration(title="Exploring 1", status="exploring"))
        
        new_ideas = inspiration_repo.get_by_status("new")
        
        assert len(new_ideas) == 1
    
    def test_update_status(self, inspiration_repo, sample_inspiration):
        """测试更新灵感状态"""
        idea_id = inspiration_repo.create(sample_inspiration)
        
        success = inspiration_repo.update_status(idea_id, "exploring")
        
        assert success
        
        updated = inspiration_repo.get_by_id(idea_id)
        assert updated.status == "exploring"
    
    def test_update_inspiration(self, inspiration_repo, sample_inspiration):
        """测试更新灵感"""
        idea_id = inspiration_repo.create(sample_inspiration)
        
        idea = inspiration_repo.get_by_id(idea_id)
        idea.content = "Updated content"
        idea.priority = "low"
        
        success = inspiration_repo.update(idea)
        
        assert success
        
        updated = inspiration_repo.get_by_id(idea_id)
        assert updated.content == "Updated content"
        assert updated.priority == "low"
    
    def test_delete_inspiration(self, inspiration_repo, sample_inspiration):
        """测试删除灵感"""
        idea_id = inspiration_repo.create(sample_inspiration)
        
        success = inspiration_repo.delete(idea_id)
        
        assert success
        assert inspiration_repo.get_by_id(idea_id) is None


# ============================================================
# 集成测试
# ============================================================

class TestRepositoryIntegration:
    """Repository 集成测试"""
    
    def test_paper_collection_relationship(
        self, paper_repo, collection_repo, note_repo
    ):
        """测试论文、合集、笔记的关联关系"""
        # 创建论文
        paper = Paper(
            title="Test Paper",
            authors="Test Author",
            keywords="test, integration"
        )
        paper_id = paper_repo.create(paper)
        
        # 创建合集并添加论文
        collection = Collection(name="Test Collection")
        coll_id = collection_repo.create(collection)
        collection_repo.add_paper(coll_id, paper_id)
        
        # 为论文添加笔记
        note = Note(
            paper_id=paper_id,
            content="Test note",
            note_type="comment"
        )
        note_id = note_repo.create(note)
        
        # 验证关联关系
        papers_in_coll = collection_repo.get_papers(coll_id)
        assert len(papers_in_coll) == 1
        assert papers_in_coll[0].id == paper_id
        
        notes_for_paper = note_repo.get_by_paper(paper_id)
        assert len(notes_for_paper) == 1
        assert notes_for_paper[0].id == note_id
    
    def test_cascade_delete_collection(
        self, paper_repo, collection_repo
    ):
        """测试删除合集时关联关系被级联删除"""
        # 创建论文和合集
        paper_id = paper_repo.create(Paper(title="Test Paper"))
        coll_id = collection_repo.create(Collection(name="Test Collection"))
        collection_repo.add_paper(coll_id, paper_id)
        
        # 删除合集
        collection_repo.delete(coll_id)
        
        # 论文应该仍然存在
        assert paper_repo.exists(paper_id)
        
        # 合集应该被删除
        assert collection_repo.get_by_id(coll_id) is None

