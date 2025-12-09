from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Agent 基类
    
    定义所有 Agent 的通用接口
    """
    @abstractmethod
    def execute(self, input_msg: str):
        """
        执行任务
        
        Args:
            input_msg: 输入消息
        """
        pass


class BaseWorker(ABC):
    """
    Worker 基类（已弃用，保留用于兼容性）
    
    请使用 BaseAgent 代替
    """
    @abstractmethod
    def execute(self):
        """
        执行任务
        """
        pass
