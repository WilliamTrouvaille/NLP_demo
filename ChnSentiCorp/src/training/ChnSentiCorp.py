import sys
import logging
import json
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).resolve().parent.parent))


# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class TextClassifierTrainer:
    """文本分类模型训练器"""

    def __init__(self, config: dict):
        """
        初始化训练器
        :param config: 训练配置字典
        """
        self.config = config
        self.tokenizer = Tokenizer(num_words=config['max_vocab'])
        self.model = None

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """加载预处理数据"""
        data_path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "tokenized_data.json"

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        X_train = data['train']
        X_test = data['test']
        y_train = np.array(data['labels']['train'])
        y_test = np.array(data['labels']['test'])

        return X_train, X_test, y_train, y_test

    def preprocess_data(self, X_train: list, X_test: list) -> Tuple[np.ndarray, np.ndarray]:
        """数据向量化处理"""
        # 构建词汇表
        self.tokenizer.fit_on_texts(X_train)

        # 转换为序列
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_test_seq = self.tokenizer.texts_to_sequences(X_test)

        # 填充序列
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.config['max_len'])
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.config['max_len'])

        return X_train_pad, X_test_pad

    def build_model(self) -> Sequential:
        """构建LSTM模型"""
        model = Sequential([
            Embedding(
                input_dim=self.config['max_vocab'],
                output_dim=self.config['embed_dim'],
            ),
            LSTM(self.config['lstm_units'], return_sequences=True),
            Dropout(self.config['dropout_rate']),
            LSTM(self.config['lstm_units']),
            Dense(self.config['dense_units'], activation='relu'),
            Dropout(self.config['dropout_rate']),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        return model

    def train(self):
        """执行训练流程"""
        # 加载数据
        X_train, X_test, y_train, y_test = self.load_data()

        # 数据预处理
        X_train_pad, X_test_pad = self.preprocess_data(X_train, X_test)

        # 构建模型
        self.model = self.build_model()

        # 准备模型保存路径
        model_dir = Path(__file__).resolve().parent.parent.parent / "models" / "pretrained"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "text_classifier.keras"

        # 回调函数
        callbacks = [
            EarlyStopping(patience=3, restore_best_weights=True),
            ModelCheckpoint(
                filepath=model_path,
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]

        # 训练模型
        history = self.model.fit(
            X_train_pad, y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_data=(X_test_pad, y_test),
            callbacks=callbacks
        )

        return history


def main():
    """主训练流程"""

    # 根据CPU核心数调整
    tf.config.threading.set_inter_op_parallelism_threads(4)

    # 确保已安装CUDA和cuDNN
    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    # 训练配置
    train_config = {
        'max_vocab': 10000,  # 词汇表大小
        'max_len': 200,  # 序列最大长度
        'embed_dim': 128,  # 词向量维度
        'lstm_units': 64,  # LSTM单元数
        'dense_units': 32,  # 全连接层单元数
        'dropout_rate': 0.5,  # Dropout比例
        'batch_size': 64,  # 批大小
        'epochs': 10  # 训练轮数
    }

    try:
        logger.info("启动模型训练...")

        # 初始化训练器
        trainer = TextClassifierTrainer(train_config)

        # 执行训练
        history = trainer.train()

        logger.info(f"训练完成，最佳验证准确率: {max(history.history['val_accuracy']):.2%}")

    except Exception as e:
        logger.error("训练流程失败", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
