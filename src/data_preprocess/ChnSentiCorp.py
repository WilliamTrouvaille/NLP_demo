import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import jieba
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_manager import ConfigurableDataLoader


def main():
    # 加载数据集
    loader = ConfigurableDataLoader("data_config.yaml")
    # hotel_df = config_loader.load_dataset("hotel_reviews")
    # labels = hotel_df["label"].tolist()
    # review = hotel_df["review"].tolist()

    print(loader.load_dataset("hotel_reviews").head())


if __name__ == '__main__':
    main()
