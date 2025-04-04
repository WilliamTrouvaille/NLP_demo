import yaml
from pathlib import Path


class Loader:
    """加载工具类（跨模块复用）"""

    @staticmethod
    def get_root_dir() -> Path:
        """获取项目根目录（自动定位）"""
        # 通过当前文件路径回溯定位项目根目录（src/utils -> ../../）
        current_dir = Path(__file__).parent
        return current_dir.parent.parent  # 项目根目录

    @classmethod
    def load(cls, config_name='config.yaml') -> dict:
        """加载配置文件"""
        config_path = cls.get_root_dir() / 'config' / config_name
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        cls._validate_paths(config)
        return config

    @classmethod
    def build_path(cls, *path_parts) -> str:
        """构建基于项目根目录的绝对路径"""
        return str(cls.get_root_dir().joinpath(*path_parts))

    @staticmethod
    def _validate_paths(config: dict):
        """校验路径配置是否存在"""
        # 示例：检查训练数据路径
        train_path = config['data']['train_path']
        if not Path(train_path).exists():
            raise FileNotFoundError(f"训练数据文件不存在：{train_path}")
