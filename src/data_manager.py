# src/data_manager.py
import logging
import sys
from pathlib import Path
from typing import Union, List, Optional
import pandas as pd
import chardet
from ruamel.yaml import YAML

# 配置日志格式
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class BaseDataLoader:
    """基础数据加载类，包含核心数据加载功能"""

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """
        初始化数据加载器

        :param base_dir: 数据根目录路径，默认自动定位到项目data/raw
        """
        self.base_dir = self._validate_base_dir(base_dir)
        self.logger = logging.getLogger(f"{__name__}.BaseDataLoader")
        self.logger.info(f"数据目录初始化为: {self.base_dir}")

    def _validate_base_dir(self, base_dir: Optional[Union[str, Path]]) -> Path:
        """验证并返回有效数据目录路径"""
        if base_dir:
            path = Path(base_dir)
            if not path.exists():
                raise FileNotFoundError(f"指定数据目录不存在: {path}")
            return path

        # 自动定位到项目根目录的data/raw（关键路径修正）
        current_script = Path(__file__).resolve()
        project_root = current_script.parent.parent
        default_dir = project_root / "data" / "raw"

        if not default_dir.exists():
            available_dirs = "\n".join([str(p) for p in project_root.glob("*")])
            raise FileNotFoundError(
                f"默认数据目录不存在: {default_dir}\n"
                f"项目根目录内容:\n{available_dirs}"
            )

        return default_dir

    def _detect_encoding(self, file_path: Path) -> str:
        """自动检测文件编码"""
        try:
            with open(file_path, 'rb') as f:
                raw = f.read(10000)  # 读取前10KB用于编码检测
                result = chardet.detect(raw)
                self.logger.info(f"检测到文件编码: {result['encoding']} (置信度: {result['confidence']:.2%})")
                return result['encoding'] or 'utf-8'
        except Exception as e:
            self.logger.warning(f"编码检测失败: {str(e)}，使用utf-8")
            return 'utf-8'

    def load_csv(
            self,
            file_name: Optional[Union[str, List[str]]] = None,
            pattern: str = "*.csv"
    ) -> pd.DataFrame:
        """
        加载CSV文件

        :param file_name: 指定文件名或文件列表
        :param pattern: 文件匹配模式（当file_name为None时生效）
        :return: 合并后的DataFrame
        """
        file_paths = self._resolve_file_paths(file_name, pattern)

        dfs = []
        for fp in file_paths:
            if not fp.exists():
                available_files = "\n".join([f.name for f in self.base_dir.glob("*")])
                raise FileNotFoundError(
                    f"数据文件不存在: {fp}\n"
                    f"当前目录可用文件:\n{available_files}"
                )

            self.logger.info(f"正在加载文件: {fp.name}")
            try:
                df = pd.read_csv(
                    fp,
                    encoding=self._detect_encoding(fp),
                    on_bad_lines='warn',
                    engine='python'  # 提高中文兼容性
                )
                dfs.append(df)
            except UnicodeDecodeError as e:
                self.logger.error(f"文件解码失败: {fp}\n错误信息: {str(e)}")
                raise
            except pd.errors.ParserError as e:
                self.logger.error(f"CSV解析失败: {fp}\n错误信息: {str(e)}")
                raise

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def _resolve_file_paths(
            self,
            file_name: Optional[Union[str, List[str]]],
            pattern: str
    ) -> List[Path]:
        """解析文件路径列表"""
        if file_name:
            if isinstance(file_name, str):
                return [self.base_dir / file_name]
            return [self.base_dir / f for f in file_name]

        return list(self.base_dir.glob(pattern))


class ConfigurableDataLoader(BaseDataLoader):
    """支持YAML配置的高级数据加载器"""

    def __init__(self, config_name: str = "data_config.yaml"):
        """
        初始化配置加载器

        :param config_name: 配置文件名（位于项目根目录的configs目录）
        """
        self.yaml = YAML(typ='safe')
        self.yaml.allow_duplicate_keys = True
        super().__init__()
        self.config = self._load_config(config_name)
        self.logger = logging.getLogger(f"{__name__}.ConfigurableDataLoader")

    def _load_config(self, config_name: str) -> dict:
        """加载YAML配置文件"""
        config_path = self._get_config_path(config_name)

        if not config_path.exists():
            available_configs = "\n".join([
                f.name for f in config_path.parent.glob("*.yaml")
                if f.is_file()
            ])
            raise FileNotFoundError(
                f"配置文件 {config_name} 不存在于 {config_path.parent}\n"
                f"可用配置文件:\n{available_configs}"
            )

        with open(config_path, 'r', encoding='utf-8') as f:
            config = self.yaml.load(f)  # 现在可以正确访问self.yaml
            self.logger.info(f"成功加载配置文件: {config_path.name}")
            return config

    def _get_config_path(self, config_name: str) -> Path:
        """获取配置文件绝对路径（关键路径修正）"""
        current_script = Path(__file__).resolve()
        project_root = current_script.parent.parent  # 项目根目录
        return project_root / "configs" / config_name

    def load_dataset(self, name: str) -> pd.DataFrame:
        """
        根据配置名称加载数据集

        :param name: 数据配置名称（对应YAML中的键）
        """
        if 'datasets' not in self.config:
            available_keys = ", ".join(self.config.keys())
            raise KeyError(
                f"配置文件中缺少 'datasets' 主键\n"
                f"现有配置项: {available_keys}"
            )

        dataset_config = self.config['datasets'].get(name)
        if not dataset_config:
            available_datasets = ", ".join(self.config['datasets'].keys())
            raise KeyError(
                f"未找到 '{name}' 的配置，可用数据集: {available_datasets}"
            )

        self.logger.info(f"加载数据集配置: {name} -> {dataset_config}")
        return self.load_csv(
            file_name=dataset_config.get('files'),
            pattern=dataset_config.get('pattern', '*.csv')
        )


if __name__ == "__main__":
    # 测试代码
    try:
        logger.info("启动数据加载测试...")

        # 测试配置加载器
        config_loader = ConfigurableDataLoader()
        print("\n配置文件结构示例:", list(config_loader.config['datasets'].keys()))

        # 测试实际数据加载
        test_df = config_loader.load_dataset("hotel_reviews")
        print(f"\n成功加载数据示例:\n{test_df.head(3)}")
        print(f"共加载 {len(test_df)} 条记录")

        logger.info("数据加载测试通过")

    except Exception as e:
        logger.error("数据加载测试失败", exc_info=True)
        sys.exit(1)
