import os
import re
from pathlib import Path

import jieba
import pandas as pd

from utils import get_logger
from utils.config_loader import ConfigLoader

logger = get_logger()


class DataCleaner:
    """数据清洗处理类"""

    def __init__(self, stopwords_path):
        self.stopwords = self._load_stopwords(stopwords_path)
        logger.info(f"已加载 {len(self.stopwords)} 个停用词")

    def _load_stopwords(self, path):
        """加载停用词表"""
        with open(path, 'r', encoding='utf-8') as f:
            return set([line.strip() for line in f])

    def _text_clean(self, text):
        """文本清洗方法"""
        # 去除@提及和表情符号
        text = re.sub(r'@\S+', '', text)
        text = re.sub(r'$$.*?$$', '', text)
        # 保留中文和基本标点
        text = re.sub(r'[^\u4e00-\u9fa5。！？，]', ' ', text)
        return text.strip()

    def _chinese_seg(self, text):
        """中文分词方法"""
        return [word for word in jieba.cut(text) if
                word not in self.stopwords and len(word) > 1]


class DataPreprocessor:
    """数据预处理流水线"""

    def __init__(self):
        self.config = ConfigLoader.load()
        self.raw_path = Path(self.config['raw_path'])
        self.processed_path = Path(self.config['processed_path'])
        self.cleaner = DataCleaner(Path(self.config['stopwords_path']))
        logger.info("数据预处理器初始化完成")

    def _load_data(self):
        """加载原始数据"""
        logger.info(f"正在从 {self.raw_path} 加载原始数据")
        df = pd.read_csv(self.raw_path)
        return df.dropna(subset=['label', 'review'])

    def _process_batch(self, df):
        """执行批处理"""
        logger.info("开始文本清洗与分词处理")
        df['cleaned'] = df['review'].apply(self.cleaner._text_clean)
        df['segmented'] = df['cleaned'].apply(self.cleaner._chinese_seg)
        return df[['label', 'segmented']]

    def execute(self):
        """执行完整流程"""
        os.makedirs(self.processed_path.parent, exist_ok=True)

        raw_df = self._load_data()
        logger.info(f"成功加载 {len(raw_df)} 条原始记录")

        processed_df = self._process_batch(raw_df)
        logger.info(f"完成处理，有效记录数：{len(processed_df)} 条")

        processed_df.to_csv(self.processed_path, index=False)
        logger.info(f"处理结果已保存至 {self.processed_path}")

        splitter = DatasetSplitter(self.config)
        splitter.execute_split()


class DatasetSplitter:
    """数据集划分类"""

    def __init__(self, config):
        self.processed_path = Path(config['processed_path'])
        self.split_ratios = config['split_ratio']
        self.output_dir = Path(config['split_output_dir'])
        self.random_seed = config.get('random_seed', 42)
        logger.info(f"数据集划分器初始化完成，划分比例：{self.split_ratios}")

    def _validate_ratios(self):
        """验证划分比例有效性"""
        if not sum(self.split_ratios.values()) == 1.0:
            logger.error("划分比例之和必须等于1.0")
            raise ValueError("Invalid split ratios summation")
        if not all(0 < v < 1 for v in self.split_ratios.values()):
            logger.error("各划分比例必须介于0和1之间")
            raise ValueError("Invalid split ratio range")

    def _load_processed_data(self):
        """加载预处理后的数据"""
        logger.info(f"正在从 {self.processed_path} 加载已处理数据")
        return pd.read_csv(self.processed_path)

    def _stratified_split(self, df):
        """分层抽样划分数据集"""
        from sklearn.model_selection import train_test_split

        # 先划分训练集和临时集合
        train_df, temp_df = train_test_split(
            df,
            test_size=1 - self.split_ratios['train'],
            stratify=df['label'],
            random_state=self.random_seed
        )

        # 再划分验证集和测试集
        val_ratio = self.split_ratios['val'] / (self.split_ratios['val'] + self.split_ratios['test'])
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1 - val_ratio,
            stratify=temp_df['label'],
            random_state=self.random_seed
        )

        return train_df, val_df, test_df

    def execute_split(self):
        """执行数据集划分"""
        self._validate_ratios()
        os.makedirs(self.output_dir, exist_ok=True)

        full_df = self._load_processed_data()
        logger.info(f"成功加载 {len(full_df)} 条已处理数据")

        # 执行分层抽样
        train_df, val_df, test_df = self._stratified_split(full_df)

        # 保存结果
        train_path = self.output_dir / 'train.csv'
        val_path = self.output_dir / 'val.csv'
        test_path = self.output_dir / 'test.csv'

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"数据集划分完成，训练集：{len(train_df)}条 | 验证集：{len(val_df)}条 | 测试集：{len(test_df)}条")
        logger.info(f"文件已保存至：{self.output_dir}")


def main():
    processor = DataPreprocessor()
    processor.execute()


if __name__ == '__main__':
    main()
