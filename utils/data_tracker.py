import os
import json
from datetime import datetime


class DataTracker:
    """
    数据跟踪器，用于记录数据流大小、步数和扩展次数
    """
    
    def __init__(self, save_dir=None):
        """
        初始化数据跟踪器
        
        Args:
            save_dir: 保存记录的目录路径
        """
        self.data_flow_size = 0  # 数据流大小 = 加载数据个数累加值
        self.expansion_count = 0  # 扩展次数
        self.save_dir = save_dir
        self.records = []  # 记录列表
        
        # 创建保存目录（如果不存在）
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def update_data_flow(self, batch_size):
        """
        更新数据流大小
        
        Args:
            batch_size: 当前批次的数据大小
        """
        self.data_flow_size += batch_size
    
    def increment_expansion(self):
        """
        增加扩展次数
        """
        self.expansion_count += 1
    
    def get_steps(self):
        """
        计算步数
        
        Returns:
            步数 = 数据流大小 ÷ 10 向下取整
        """
        return self.data_flow_size // 10
    
    def record_state(self, task_id=None, batch_idx=None):
        """
        记录当前状态
        
        Args:
            task_id: 当前任务ID
            batch_idx: 当前批次索引
        """
        state = {
            "timestamp": datetime.now().isoformat(),
            "data_flow_size": self.data_flow_size,
            "steps": self.get_steps(),
            "expansion_count": self.expansion_count,
            "task_id": task_id,
            "batch_idx": batch_idx
        }
        self.records.append(state)
        return state
    
    def save_records(self, filename="data_tracker.json"):
        """
        保存记录到文件
        
        Args:
            filename: 保存的文件名
        """
        if not self.save_dir:
            return
        
        save_path = os.path.join(self.save_dir, filename)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)
        
        print(f"数据跟踪记录已保存到: {save_path}")
    
    def get_current_state(self):
        """
        获取当前状态
        
        Returns:
            当前状态字典
        """
        return {
            "data_flow_size": self.data_flow_size,
            "steps": self.get_steps(),
            "expansion_count": self.expansion_count
        }
