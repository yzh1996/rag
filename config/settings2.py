"""
settings.py 配置管理模块 —— 事用Pydantic Settings 实现类型安全的配置。

.env文件只是存储配置的原材料，它本身不会被程序自动识别的，必须有代码显式的读取他。

Pydantic Settings 是一个用于管理应用程序配置的工具，它可以从多种来源加载配置，包括环境变量、命令行参数、.env文件等。
它还可以自动验证配置的类型和格式，并提供默认值和环境变量名称的映射。

最简单的理解，不设置的话，你需要再所有需要加载环境变量的文件中写代码，一处修改处处修改，很麻烦。
"""
from pydantic_settings import *
from pathlib import Path
from pydantic import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    API_V1_STR: str = ""
    SECRET_KEY: str = "8d9d09f0f6c34601a64761f5c96f0000"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    SERVER_NAME: str = "127.0.0.1"
    SERVER_PORT: int = 8000
    DATABASE_URL: str = "sqlite:///./test.db"
    DATABASE_TABLE_PREFIX: str = "t_"
    DATABASE_ECHO: bool = False
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_TIMEOUT: int = 30
    DATABASE_CHARSET: str = "utf8mb4"
    DATABASE_COLLATE: str = "utf8mb4_general_ci"
    DATABASE_ENGINE: str = "InnoDB"
    DATABASE_CHARSET_COLLATE: str = "utf8mb4_general_ci"
    DATABASE_ECHO: bool = False
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_TIMEOUT: int = 30





