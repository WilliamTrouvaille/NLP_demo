import sys
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pylab import mpl
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# 配置项目路径
sys.path.append(str(Path(__file__).resolve().parent.parent))
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估工具类"""

    def __init__(self, model_path: str, config: dict):
        """
        :param model_path: 模型文件路径
        :param config: 与训练一致的配置字典
        """
        self.model = load_model(model_path)
        self.config = config
        self.tokenizer = Tokenizer(num_words=config['max_vocab'])

    def load_test_data(self) -> tuple:
        """加载测试数据"""
        data_path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "tokenized_data.json"

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data['test'], np.array(data['labels']['test'])

    def preprocess_data(self, texts: list) -> np.ndarray:
        """数据预处理（保持与训练一致）"""
        # 必须使用与训练相同的词汇表
        train_data_path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "train_processed.csv"
        train_df = pd.read_csv(train_data_path)
        self.tokenizer.fit_on_texts(train_df['tokenized'].apply(eval).apply(' '.join))

        sequences = self.tokenizer.texts_to_sequences(
            [' '.join(tokens) for tokens in texts]
        )
        return pad_sequences(sequences, maxlen=self.config['max_len'])

    def evaluate(self):
        """执行完整评估流程"""
        # 加载数据
        X_test_texts, y_test = self.load_test_data()

        # 预处理
        X_test_pad = self.preprocess_data(X_test_texts)

        # 预测
        y_pred = self.model.predict(X_test_pad)
        y_pred_bin = (y_pred > 0.5).astype(int)

        # 生成报告
        report = classification_report(y_test, y_pred_bin, target_names=['负面', '正面'])
        print("分类报告:\n", report)

        # 可视化混淆矩阵
        self.plot_confusion_matrix(y_test, y_pred_bin)

        return report

    def plot_confusion_matrix(self, y_true, y_pred):
        """绘制混淆矩阵"""

        mpl.rcParams["font.sans-serif"] = ["SimHei"]
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['负面', '正面'],
                    yticklabels=['负面', '正面'])
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.savefig(Path(__file__).parent.parent.parent / "docs" / "confusion_matrix.png")
        plt.close()


if __name__ == "__main__":
    # 配置需与训练完全一致
    train_config = {
        'max_vocab': 10000,
        'max_len': 200
    }

    # 模型路径
    model_path = Path(__file__).resolve().parent.parent.parent / "models" / "pretrained" / "text_classifier.keras"

    try:
        logger.info("启动模型评估...")
        evaluator = ModelEvaluator(model_path, train_config)
        report = evaluator.evaluate()
        logger.info("评估完成，结果已保存至 docs/confusion_matrix.png")

    except Exception as e:
        logger.error("评估流程失败", exc_info=True)
        sys.exit(1)
