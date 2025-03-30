import yaml
from pathlib import Path


class ConfigLoader:
    """配置加载工具类（跨模块复用）"""

    @staticmethod
    def load(config_name='config.yaml'):
        """
        自动定位项目根目录加载配置
        Args:
            config_name: 配置文件名
        Returns:
            包含配置项的字典
        """
        # 通过文件路径回溯定位项目根目录
        current_dir = Path(__file__).parent
        config_path = current_dir.parent.parent / 'config' / config_name
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
