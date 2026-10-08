"""
多轮对话记忆管理模块——支持多会话隔离
"""
import uuid
import logging
from datetime import datetime, timedelta
from langchain.memory import ConversationBufferWindowMemory

logger = logging.getLogger(__name__)

class ConversationSession:
    """ 单个会话的记忆管理 """
    def __init__(self, session_id: str, window_size: int = 5):
        self.session_id = session_id
        self.create_at = datetime.now()
        self.last_active = datetime.now()
        self.memory = ConversationBufferWindowMemory(
            k=window_size,  # k参数是指窗口大小 即最多保留多少条对话记录
            memory_key="chat_history",  # memory_key是指对话历史的键名
            return_messages=True,   # return_messages是指是否返回消息列表
            output_key="answer" # output_key是指输出的键名
        )
        self.metadata: dict = {}

    def add_user_message(self, message: str):
        """ 添加用户消息 """
        self.memory.chat_memory.add_user_message(message)
        self.last_active = datetime.now()

    def add_ai_message(self, message: str):
        """ 添加AI消息 """
        self.memory.chat_memory.add_ai_message(message)
        self.last_active = datetime.now()

    def get_memory(self):
        """ 获取当前会话的记忆 """
        return self.memory.chat_memory.messages

    def clear(self):
        """ 清空当前会话的记忆 """
        self.memory.clear()
        self.last_active = datetime.now()

    def to_dict(self):
        """ 转换为字典 """
        messages = []
        for msg in self.memory.chat_memory.messages:
            messages.append({
                "role": msg.type,
                "content": msg.content
            })
        return {
            "session_id": self.session_id,
            "create_at": self.create_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "messages": messages,
            "metadata": self.metadata
        }
    
class MemoryManager:
    """ 多会话记忆管理 """
    def __init__(self, window_size: int = 5, session_ttl_minutes: int = 60):
        self.sessions: dict = {}
        self.window_size = window_size
        self.session_ttl = timedelta(minutes=session_ttl_minutes)
        logger.info(f"记忆管理器初始化 window_size={window_size}, session_ttl={session_ttl_minutes} minutes")

    def create_session(self, session_id: str):
        """ 创建新会话 """
        if session_id is None:
            # uuid4() 是一个全局唯一标识符 当然也可以使用时间戳
            session_id = str(uuid.uuid4())
        if session_id in self.sessions:
            logger.warning(f"会话 {session_id} 已存在，将返回现有会话")
            return session_id
        session = ConversationSession(session_id, self.window_size)
        self.sessions[session_id] = session
        logger.info(f"创建新会话: {session_id}")
        return session_id
    
    def get_session(self, session_id: str):
        """ 获取会话 """
        session = self.sessions.get(session_id)
        if session:
            session.last_active = datetime.now()
        return session
    
    def get_or_create_session(self, session_id: str):
        """ 获取或创建会话 """
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        new_session_id = self.create_session(session_id)
        return self.sessions[new_session_id]
    
    def add_exchange(self, session_id: str, question: str, answer: str):
        """ 添加一轮对话记录 """
        session = self.get_or_create_session(session_id)
        session.add_user_message(question)
        session.add_ai_message(answer)
        logger.info(f"添加对话记录: 会话 {session_id}, 用户: {question}, AI: {answer}")

    def get_chat_history(self, session_id: str):
        """ 获取会话的聊天历史 """
        session = self.get_session(session_id)
        if not session:
            return []
        return session.get_memory()
    
    def clear_session(self, session_id: str):
        """ 清空会话 """
        if session_id in self.sessions:
            self.sessions[session_id].clear()
            logger.info(f"清空会话: {session_id}")

    def delete_session(self, session_id: str):
        """ 删除会话 """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"删除会话: {session_id}")

    def cleanup_old_sessions(self):
        """ 清理过期会话 """
        now = datetime.now()
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if now - session.last_active > self.session_ttl
        ]
        for session_id in expired_sessions:
            self.delete_session(session_id)
        if expired_sessions:
            logger.info(f"清理过期会话: 已删除 {len(expired_sessions)} 个会话")
        return len(expired_sessions)

    def list_sessions(self):
        """ 列出所有会话 """
        return [session.to_dict() for session in self.sessions.values()]
    
    def get_default_memory(self):
        """ 获取默认会话的记忆 """
        default_session = self.get_or_create_session("default")
        return default_session.memory
    


# 全局单例
_memory_manager_instance = None

def get_memory_manager():
    """ 获取以及管理器单例 """
    global _memory_manager_instance
    if _memory_manager_instance is None:
        _memory_manager_instance = MemoryManager()
    return _memory_manager_instance
