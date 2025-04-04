"""
数据预处理模块
功能：
1. 加载原始数据集（已经）
2. 划分训练集/验证集/测试集
3. 保存划分后的数据集
"""
import random
from pathlib import Path

from tqdm.auto import tqdm

from src.utils import Loader, LoggerHandler


class DataProcessor:
    """数据预处理管道"""

    def __init__(self, config: dict):
        """
        初始化处理器
        Args:
            config: 从配置文件加载的字典（data部分）
        """
        # 动态生成路径
        root = Loader.get_root_dir()
        self.raw_path = Path(root) / config['data']['raw_path']
        self.processed_dir = Path(root) / config['data']['processed_dir']

        # 确保目录存在
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # 确保目录存在
        self.raw_path = Path(config['data']['raw_path'])
        self.processed_dir = Path(config['data']['processed_dir'])
        self.splits = config['data']['splits']
        self.seed = config.get('seed', 42)
        self.shuffle = config.get('shuffle', True)

        # 初始化中间数据容器
        self.raw_sentences = []  # 原始句子集合
        self.splitted_data = {}  # 划分后的数据集

    def _load_raw_sentences(self):
        """加载原始数据"""
        LoggerHandler().info(f"开始加载原始数据：{self.raw_path}")

        current_sentence = []
        with open(self.raw_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="加载原始数据", unit=' lines'):
                line = line.strip()

                # 遇到空行时提交当前句子
                if not line:
                    if current_sentence:
                        self.raw_sentences.append(current_sentence)
                        current_sentence = []
                    continue

                # 解析字符和标签
                parts = line.split('\t')
                if len(parts) != 2:
                    LoggerHandler().warning(f"异常数据格式：{line}")
                    continue

                current_sentence.append((parts[0], parts[1]))

            # 处理文件末尾未提交的句子
            if current_sentence:
                self.raw_sentences.append(current_sentence)

        LoggerHandler().info(f"原始数据加载完成，共 {len(self.raw_sentences)} 个句子")

    def _split_dataset(self):
        """数据集划分"""
        LoggerHandler().info("开始数据集划分...")

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self.raw_sentences)

        total = len(self.raw_sentences)
        splits = self.splits

        # 计算划分点
        train_end = int(total * splits['train'])
        val_end = train_end + int(total * splits['val'])

        self.splitted_data = {
            'train': self.raw_sentences[:train_end],
            'val': self.raw_sentences[train_end:val_end],
            'test': self.raw_sentences[val_end:]
        }

        LoggerHandler().info(
            f"数据集划分完成 | "
            f"Train: {len(self.splitted_data['train'])} | "
            f"Val: {len(self.splitted_data['val'])} | "
            f"Test: {len(self.splitted_data['test'])}"
        )

    def _save_processed_data(self):
        """保存处理后的数据"""
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        for split_name, data in self.splitted_data.items():
            path = self.processed_dir / f"{split_name}.txt"
            with open(path, 'w', encoding='utf-8') as f:
                for sentence in tqdm(data, desc=f"保存 {split_name} 数据"):
                    for char, tag in sentence:
                        f.write(f"{char}\t{tag}\n")
                    f.write("\n")  # 句子间空行分隔
            LoggerHandler().info(f"{split_name} 数据集已保存至：{path}")

    def run(self):
        """启动处理流程"""
        self._load_raw_sentences()
        self._split_dataset()
        self._save_processed_data()


def main():
    # 从配置文件加载参数
    config = Loader.load()

    # 创建并运行处理器
    processor = DataProcessor(config)
    processor.run()


if __name__ == '__main__':
    main()
