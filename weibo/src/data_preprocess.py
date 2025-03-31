import os
import re
from pathlib import Path

import jieba
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import get_logger
from utils.config_loader import ConfigLoader

logger = get_logger(__name__)


class DataCleaner:
    """数据清洗处理类，负责文本的清洗和分词"""

    def __init__(self, config):
        self.stopwords = self._load_resource(
            Path(config['stopwords_path']),
            loader=lambda f: set(line.strip() for line in f)
        )
        self.network_phrases = self._load_resource(
            Path(config['network_dict_path']),
            loader=lambda f: dict(line.strip().split('=', 1) for line in f)
        )
        self.emoticon_map = self._load_resource(
            Path(config['emoticon_path']),
            loader=lambda f: dict(line.strip().split('=', 1) for line in f)
        )
        logger.info(f"已加载 {len(self.stopwords)} 个停用词, {len(self.network_phrases)} 个网络用语, {len(self.emoticon_map)} 个表情映射")

    def _load_resource(self, path, loader):
        """
        通用资源加载方法
        :param path: 文件路径
        :param loader: 资源加载函数，接受文件对象并返回解析后的数据
        :return: 解析后的资源数据
        """
        if not path.exists():
            raise FileNotFoundError(f"文件未找到：{path}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return loader(f)
        except Exception as e:
            logger.error(f"加载文件 {path} 时出错: {str(e)}")
            raise

    def _preserve_emoticons(self, text):
        """保留并标准化表情符号"""
        replaced = re.sub(r'$$([^$$]+)\]', lambda m: f'<EMO_{self.emoticon_map.get(m.group(1), m.group(1))}>', text)
        return re.sub(r'(<EMO_[^>]+>)\s+(?=<EMO_)', r'\1', replaced)

    def _replace_network_phrases(self, text):
        """替换网络用语到标准形式"""
        for phrase, standard in self.network_phrases.items():
            text = text.replace(phrase, standard)
        return text

    def clean_text(self, text):
        """改进版清洗流程"""
        text = self._preserve_emoticons(text)
        text = self._replace_network_phrases(text)
        text = re.sub(r'@\S+\s?', '[USER] ', text)  # 统一用户提及
        text = re.sub(r'http\S+', '[URL]', text)  # 保留URL标记
        text = re.sub(r'([，。！？；]){2,}', r'\1', text)  # 去除重复标点
        text = re.sub(r'\s+', ' ', text).strip()  # 去除多余空格
        return text

    def segment(self, text):
        """改进版分词方法"""
        protected = []

        def protect(match):
            protected.append(match.group(0))
            return f' PROTECTED_{len(protected) - 1} '

        temp_text = re.sub(r'<(EMO_[^>]+|URL|USER)>', protect, text)
        words = jieba.lcut(temp_text)

        final_words = []
        for w in words:
            if w.startswith('PROTECTED_'):
                idx = int(w.split('_')[1])
                final_words.append(protected[idx])
            else:
                if w not in self.stopwords and len(w) > 1:
                    final_words.append(w)

        return final_words


class DataPreprocessor:
    """数据预处理流水线，负责数据的加载、清洗、分词和保存"""

    def __init__(self):
        self.config = ConfigLoader.load()
        self.raw_path = Path(self.config['raw_path'])
        self.processed_path = Path(self.config['processed_path'])
        self.cleaner = DataCleaner(self.config)
        logger.info("数据预处理器初始化完成")

    def _load_data(self):
        """加载原始数据"""
        logger.info(f"正在从 {self.raw_path} 加载原始数据")
        df = pd.read_csv(self.raw_path)
        return df.dropna(subset=['label', 'review'])

    def _process_batch(self, df):
        """执行批处理"""
        logger.info("开始文本清洗与分词处理")
        df['cleaned'] = df['review'].apply(self.cleaner.clean_text)
        df['segmented'] = df['cleaned'].apply(self.cleaner.segment)
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
    """数据集划分类，负责将数据集划分为训练集、验证集和测试集"""

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
        train_df, temp_df = train_test_split(
            df,
            test_size=1 - self.split_ratios['train'],
            stratify=df['label'],
            random_state=self.random_seed
        )

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

        train_df, val_df, test_df = self._stratified_split(full_df)

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
