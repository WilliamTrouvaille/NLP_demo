from .config_loader import ConfigLoader
import logging
from logging import Logger
from .progress import TrainingProgress

# 初始化日志配置
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# 定义包级别logger
_package_logger = logging.getLogger(__name__)
_package_logger.info("Utils package initialized")


def get_logger(name: str = None) -> Logger:
    """获取预配置的logger对象
    :rtype: object
    """
    return logging.getLogger(name or __name__)


__all__ = ['ConfigLoader', 'get_logger','TrainingProgress']
