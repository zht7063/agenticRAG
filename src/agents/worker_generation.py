"""
Generation Worker - 生成 Worker

负责基于检索结果生成高质量的回答和文档：

1. 答案生成
   - 基于检索上下文生成准确回答
   - 支持多轮对话的连贯性
   - 附带来源引用

2. 文献综述生成
   - 基于多篇论文生成结构化综述
   - 支持按主题、方法、时间线组织
   - 自动生成参考文献列表

3. 研究建议生成
   - 分析研究空白和未来方向
   - 跨论文的方法对比
   - 创新点发现和建议

4. 内容总结
   - 单篇论文摘要生成
   - 多文档信息整合
   - 关键观点提取

5. 质量控制
   - 幻觉检测和避免
   - 事实一致性验证
   - 置信度评估

触发场景:
- 需要生成最终回答给用户
- 用户请求文献综述或总结
- 需要研究建议和灵感
"""

from .worker_base import BaseWorker


class GenerationWorker(BaseWorker):
    """生成 Worker，负责答案生成和文献综述"""
    
    def __init__(self):
        super().__init__(name="generation")
        self.llm = None  # LLM 实例
    
    def execute(self, task: dict) -> dict:
        """
        执行生成任务
        
        Args:
            task: {
                "type": "generation",
                "generation_type": str,  # 生成类型 ["answer", "summary", "review", "suggestion"]
                "query": str,            # 用户问题
                "context": list,         # 检索到的上下文
                "format": str            # 输出格式 ["text", "markdown"]
            }
        """
        pass
    
    def generate_answer(self, query: str, context: list) -> str:
        """生成问答回复"""
        pass
    
    def generate_summary(self, documents: list) -> str:
        """生成文档摘要"""
        pass
    
    def generate_review(self, papers: list, topic: str) -> str:
        """生成文献综述"""
        pass
    
    def generate_suggestion(self, papers: list, research_area: str) -> str:
        """生成研究建议"""
        pass
    
    def check_hallucination(self, answer: str, context: list) -> bool:
        """检测答案是否存在幻觉"""
        pass
    
    def add_citations(self, answer: str, sources: list) -> str:
        """为答案添加引用标注"""
        pass

