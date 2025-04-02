import logging
import sys
from pathlib import Path
from typing import Optional, Dict
from logging.handlers import RotatingFileHandler
from loader import Loader


class Logger:
    """增强型日志管理器"""

    _configured = False  # 确保配置只执行一次

    def __init__(self):
        self.config = Loader.load()
        self._setup_logging()

    def _setup_logging(self):
        """根据配置初始化日志系统"""
        if Logger._configured:
            return

        log_cfg = self.config.get('logging', {})

        # 基础配置
        logging.basicConfig(
            level=log_cfg.get('level', 'INFO'),
            format=log_cfg.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            handlers=self._create_handlers(log_cfg)
        )

        # 特殊库的日志级别控制
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

        Logger._configured = True

    def _create_handlers(self, log_cfg: Dict) -> list:
        """创建处理器集合"""
        handlers = []

        # 控制台处理器
        if log_cfg.get('console', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self._get_formatter(log_cfg))
            handlers.append(console_handler)

        # 文件处理器（带滚动）
        if log_cfg.get('file', False):
            log_dir = Path(Loader.build_path(log_cfg['dir']))
            log_dir.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                filename=log_dir / 'run.log',
                maxBytes=log_cfg.get('max_size', 10 * 1024 * 1024),  # 默认10MB
                backupCount=log_cfg.get('backup_count', 5)
            )
            file_handler.setFormatter(self._get_formatter(log_cfg))
            handlers.append(file_handler)

        return handlers

    @staticmethod
    def _get_formatter(log_cfg: Dict) -> logging.Formatter:
        """获取格式器"""
        return logging.Formatter(
            fmt=log_cfg.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            datefmt=log_cfg.get('datefmt', '%Y-%m-%d %H:%M:%S')
        )

    @property
    def configured(self):
        return self._configured


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取预配置的logger对象

    Args:
        name: 日志命名空间，通常使用 __name__
    """
    # 延迟初始化确保配置加载完成
    if not Logger.configured:
        Logger()
    return logging.getLogger(name or __name__)