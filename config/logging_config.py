"""
日志配置模块
为什么配置日志？——因为日志可以帮助我们更好的去了解程序的运行情况，快速定位出现问题的地方。
日志的级别——DEBUG  INFO   WARNING   ERROR   CRITICAL
写好了日志配置模块之后，在主程序中调用，在其他的任意模块中可以继承全局日志。
"""

import logging
import sys
from config.settings import settings
from logging.handlers import RotatingFileHandler

def setup_logging():
    """"配置全局日志的函数"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(console_handler)

    # 文件处理器 (带轮转)
    # 设置日志文件的大小，当达到大小的时候，会自动轮转
    file_handler = RotatingFileHandler(
        settings.LOG_FILE, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,  # 保留5个备份
        encoding="utf-8"
        )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(file_handler)

    # 减少第三方库的日志噪声 （可选）
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    # 测试日志
    root_logger.info(f"日志系统初始化完成, 级别是 {settings.LOG_LEVEL}")
    