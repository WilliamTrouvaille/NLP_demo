from pathlib import Path
import pandas as pd
import logging
from typing import Union, List

from requests.compat import chardet


class DataLoader:
    def __init__(self, base_dir: str = None):
        self.logger = logging.getLogger(__name__)
        self.base_dir = self._validate_base_dir(base_dir)

    @staticmethod
    def _validate_base_dir(base_dir: str) -> Path:
        """验证数据目录有效性"""
        if base_dir:
            path = Path(base_dir)
        else:
            # 默认项目相对路径
            current_script = Path(__file__).resolve()
            path = current_script.parent.parent / "data" / "raw"

        if not path.exists():
            raise FileNotFoundError(f"数据目录不存在: {path}")
        return path

    def load_csv(self,
                 file_name: Union[str, List[str]] = None,
                 pattern: str = "*.csv") -> pd.DataFrame:
        """
        加载CSV文件（支持单文件/多文件/通配符）

        参数：
        file_name : 可选，指定文件名或列表
        pattern   : 文件匹配模式（当file_name为None时生效）
        """
        if file_name:
            if isinstance(file_name, str):
                file_paths = [self.base_dir / file_name]
            else:
                file_paths = [self.base_dir / f for f in file_name]
        else:
            file_paths = list(self.base_dir.glob(pattern))

        if not file_paths:
            raise ValueError(f"未找到符合{pattern}的数据文件")

        dfs = []
        for fp in file_paths:
            if not fp.exists():
                raise FileNotFoundError(f"文件不存在: {fp}")
            self.logger.info(f"正在加载 {fp.name}")
            dfs.append(pd.read_csv(fp, encoding=self._detect_encoding(fp)))


        return pd.concat(dfs, ignore_index=True)

    def _detect_encoding(self, file_path: Path) -> str:
        """自动检测文件编码"""
        try:
            with open(file_path, 'rb') as f:
                raw = f.read(10000)
                result = chardet.detect(raw)
                return result['encoding'] or 'utf-8'
        except Exception as e:
            self.logger.warning(f"编码检测失败，使用utf-8: {str(e)}")
            return 'utf-8'


# 使用示例
# if __name__ == "__main__":
#     loader = DataLoader()
#
#     # 方式1：加载指定文件
#     df1 = loader.load_csv(file_name="ChnSentiCorp_htl_all.csv")

    # 方式2：加载多个文件
    # df2 = loader.load_csv(file_name=["reviews1.csv", "reviews2.csv"])
    #
    # # 方式3：加载所有CSV文件
    # df3 = loader.load_csv()
    #
    # # 方式4：按通配符加载
    # df4 = loader.load_csv(pattern="hotel_*.csv")