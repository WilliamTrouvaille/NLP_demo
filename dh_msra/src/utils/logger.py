import logging
import sys
from pathlib import Path
from datetime import datetime

from src.utils import Loader


class LoggerHandler:
    """
    单例模式的日志封装类
    功能：支持控制台+文件双输出、自动创建日志目录、动态时间戳命名
    用法：直接调用LoggerHandler().debug/info/warning/error/critical方法
    """
    logger = None
    _instance = None

    def __new__(cls, level=logging.DEBUG,
                fmt='[%(asctime)s] [%(levelname)s] %(message)s'):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            # 初始化日志器
            cls.logger = logging.getLogger("LoggerHandler")
            cls.logger.setLevel(level)

            # 创建日志目录
            root = Loader.get_root_dir()
            log_dir = Path(root) / "logs"
            log_dir.mkdir(exist_ok=True)
            Path(log_dir).mkdir(exist_ok=True)

            # 定义格式器
            formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

            # 控制台处理器
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)

            # 文件处理器（按日期命名）
            log_file = Path(log_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)

            # 添加处理器
            cls.logger.addHandler(console_handler)
            cls.logger.addHandler(file_handler)

        return cls._instance

    @classmethod
    def debug(cls, msg):
        cls.logger.debug(msg)

    @classmethod
    def info(cls, msg):
        cls.logger.info(msg)

    @classmethod
    def warning(cls, msg):
        cls.logger.warning(msg)

    @classmethod
    def error(cls, msg):
        cls.logger.error(msg)

    @classmethod
    def critical(cls, msg):
        cls.logger.critical(msg)
