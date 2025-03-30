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
    def __init__(self, stopwords_path: str = None):
        self.stopwords = self._load_stopwords(stopwords_path)
        self._init_jieba()

    @staticmethod
    def _init_jieba():
        jieba.setLogLevel('WARN')
        jieba.add_word("性价比高", freq=2000)
        jieba.add_word("服务态度差", freq=2000)

    @staticmethod
    def _load_stopwords(path: str) -> set:
        default_stopwords = {"的", "了", "和", "是", "就", "都"}
        if not path:
            return default_stopwords
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return {line.strip() for line in f}
        except FileNotFoundError:
            logger.warning("停用词文件未找到，使用默认列表")
            return default_stopwords

    @staticmethod
    def clean_text(text: str) -> str:
        """文本清洗，并处理空值"""
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

        df = df.dropna(subset=['review']).copy()
        df['review'] = df['review'].astype(str)

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