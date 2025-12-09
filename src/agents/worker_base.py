from abc import ABC, abstractmethod

class BaseWorker():
    @abstractmethod
    def execute(self):
        """
        执行任务
        """
        pass
