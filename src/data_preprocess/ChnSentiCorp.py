from src.data_manager import ConfigurableDataLoader
import sys
import logging
import re
import jieba
import json
from pathlib import Path
from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).resolve().parent.parent))

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """文本预处理流水线（适配 label/review 列名）"""

    def __init__(self, stopwords_path: str = None):
        self.stopwords = self._load_stopwords(stopwords_path)
        self._init_jieba()

    def _init_jieba(self):
        jieba.setLogLevel('WARN')
        jieba.add_word("性价比高", freq=2000)
        jieba.add_word("服务态度差", freq=2000)

    def _load_stopwords(self, path: str) -> set:
        default_stopwords = {"的", "了", "和", "是", "就", "都"}
        if not path:
            return default_stopwords
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return {line.strip() for line in f}
        except FileNotFoundError:
            logger.warning("停用词文件未找到，使用默认列表")
            return default_stopwords

    def clean_text(self, text: str) -> str:
        """增强版文本清洗，处理空值"""
        if pd.isna(text):
            return ""

        text = str(text)

        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\w\u4e00-\u9fff]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def segment(self, text: str) -> List[str]:
        words = jieba.lcut(text)
        return [w for w in words if w not in self.stopwords and len(w) > 1]

    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("开始文本预处理...")

        # 过滤空值并重置索引
        df = df.dropna(subset=['review']).copy()
        df['review'] = df['review'].astype(str)  # 强制转换为字符串

        df['cleaned_text'] = df['review'].apply(self.clean_text)
        df['tokenized'] = df['cleaned_text'].apply(self.segment)

        processed_df = df[['tokenized', 'label']].copy()
        logger.info(f"预处理完成，示例数据:\n{processed_df.head(2)}")
        return processed_df


class DataSplitter:
    @staticmethod
    def split_data(df: pd.DataFrame, test_size: float = 0.2) -> dict:
        logger.info("划分数据集...")
        train_df, test_df = train_test_split(
            df, test_size=test_size, stratify=df['label'], random_state=42
        )
        return {"train": train_df, "test": test_df, "full": df}


def save_processed_data(data: dict, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("保存处理结果...")
    for name, df in data.items():
        file_path = output_path / f"{name}_processed.csv"
        df.to_csv(file_path, index=False, encoding='utf-8')

    json_path = output_path / "tokenized_data.json"
    json_data = {
        "train": data['train']['tokenized'].tolist(),
        "test": data['test']['tokenized'].tolist(),
        "labels": {
            "train": data['train']['label'].tolist(),
            "test": data['test']['label'].tolist()
        }
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False)


def main():
    try:
        loader = ConfigurableDataLoader()
        raw_df = loader.load_dataset("hotel_reviews")

        # 列名验证
        required_columns = {'review', 'label'}
        if not required_columns.issubset(raw_df.columns):
            missing = required_columns - set(raw_df.columns)
            raise KeyError(f"数据集中缺少必要列: {missing}")

        preprocessor = TextPreprocessor()
        processed_df = preprocessor.process_batch(raw_df)

        splitter = DataSplitter()
        data_dict = splitter.split_data(processed_df)

        project_root = Path(__file__).resolve().parent.parent.parent
        save_processed_data(data_dict, project_root / "data" / "processed")

        logger.info("预处理完成！")

    except Exception as e:
        logger.error("预处理失败", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

# class TextPreprocessor:
#     """文本预处理流水线"""
#
#     def __init__(self, stopwords_path: str = None):
#         """
#         初始化预处理器
#         :param stopwords_path: 停用词文件路径（默认使用内置停用词）
#         """
#         self.stopwords = self._load_stopwords(stopwords_path)
#         self._init_jieba()
#
#     def _init_jieba(self):
#         """初始化结巴分词"""
#         jieba.setLogLevel('WARN')  # 关闭jieba调试信息
#         # 添加领域专业词（示例）
#         jieba.add_word("性价比高", freq=2000, tag='n')
#         jieba.add_word("服务态度差", freq=2000, tag='n')
#
#     def _load_stopwords(self, path: str) -> set:
#         """加载停用词表"""
#         default_stopwords = {"的", "了", "和", "是", "就", "都", "而", "及", "与", "在"}  # 示例停用词
#         if not path:
#             return default_stopwords
#
#         try:
#             with open(path, 'r', encoding='utf-8') as f:
#                 return {line.strip() for line in f if line.strip()}
#         except FileNotFoundError:
#             logger.warning(f"停用词文件 {path} 未找到，使用默认停用词")
#             return default_stopwords
#
#     def clean_text(self, text: str) -> str:
#         """文本清洗"""
#         # 去除HTML标签
#         text = re.sub(r'<[^>]+>', '', text)
#         # 去除特殊符号
#         text = re.sub(r'[^\w\u4e00-\u9fff]', ' ', text)
#         # 合并连续空格
#         return re.sub(r'\s+', ' ', text).strip()
#
#     def segment(self, text: str) -> List[str]:
#         """中文分词与过滤"""
#         words = jieba.lcut(text)
#         return [w for w in words if w not in self.stopwords and len(w) > 1]
#
#     def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
#         """批量处理数据"""
#         logger.info("开始文本预处理...")
#
#         # 清洗文本
#         df['cleaned_text'] = df['review'].apply(self.clean_text)
#
#         # 分词处理
#         df['tokenized'] = df['cleaned_text'].apply(self.segment)
#
#         logger.info(f"预处理完成，示例数据:\n{df[['tokenized', 'label']].head(2)}")
#         return df
#
#
# class DataSplitter:
#     """数据集划分器"""
#
#     @staticmethod
#     def split_data(df: pd.DataFrame, test_size: float = 0.2) -> dict:
#         """
#         划分训练集/测试集
#         :return: 包含三个数据集的字典
#         """
#         logger.info("划分数据集...")
#
#         # 确保数据分布均衡
#         train_df, test_df = train_test_split(
#             df,
#             test_size=test_size,
#             stratify=df['label'],
#             random_state=42
#         )
#
#         return {
#             "train": train_df,
#             "test": test_df,
#             "full": df
#         }
#
#
# def save_processed_data(data: dict, output_dir: str):
#     """保存处理后的数据"""
#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)
#
#     logger.info("保存处理结果...")
#     for name, df in data.items():
#         file_path = output_path / f"{name}_processed.csv"
#         df.to_csv(file_path, index=False, encoding='utf-8')
#         logger.info(f"已保存 {name} 数据集至 {file_path}")
#
#     # 保存分词后的JSON格式（供深度学习模型使用）
#     json_path = output_path / "tokenized_data.json"
#     json_data = {
#         "train": data['train']['tokenized'].tolist(),
#         "test": data['test']['tokenized'].tolist(),
#         "labels": {
#             "train": data['train']['label'].tolist(),
#             "test": data['test']['label'].tolist()
#         }
#     }
#     with open(json_path, 'w', encoding='utf-8') as f:
#         json.dump(json_data, f, ensure_ascii=False)
#     logger.info(f"已保存分词数据至 {json_path}")
#
#
# def main():
#     """主处理流程"""
#     try:
#         # 初始化配置加载器
#         loader = ConfigurableDataLoader()
#
#         # 加载原始数据
#         raw_df = loader.load_dataset("hotel_reviews")
#
#         # 文本预处理
#         preprocessor = TextPreprocessor(stopwords_path="configs/cn_stopwords.txt")
#         processed_df = preprocessor.process_batch(raw_df)
#
#         # 划分数据集
#         splitter = DataSplitter()
#         data_dict = splitter.split_data(processed_df)
#
#         # 保存结果
#         project_root = Path(__file__).resolve().parent.parent.parent
#         output_dir = project_root / "data" / "processed"
#         save_processed_data(data_dict, output_dir)
#
#         logger.info("预处理流程完成！")
#
#     except Exception as e:
#         logger.error("预处理流程失败", exc_info=True)
#         sys.exit(1)
#
#
# if __name__ == '__main__':
#     main()
