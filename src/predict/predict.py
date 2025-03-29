from src.data_preprocess.ChnSentiCorp import TextPreprocessor
from src.data_manager import ConfigurableDataLoader
from tensorflow.keras.preprocessing.text import Tokenizer
import sys
import logging
from pathlib import Path
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.append(str(Path(__file__).resolve().parent.parent))

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """交互式情感分析器"""

    def __init__(self):
        # 加载配置
        self.config_loader = ConfigurableDataLoader("data_config.yaml")
        model_path = Path(__file__).resolve().parent.parent.parent / "models" / "pretrained" / "text_classifier.keras"

        # 加载模型和预处理工具
        self.model = load_model(model_path)
        self.preprocessor = TextPreprocessor()
        self.tokenizer = self._load_tokenizer()

    def _load_tokenizer(self):
        """加载与训练一致的tokenizer"""
        # 从训练数据重建tokenizer
        train_data_path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "train_processed.csv"
        train_df = pd.read_csv(train_data_path)

        # 正确解析tokenized列
        train_texts = train_df['tokenized'].apply(
            lambda x: ' '.join(eval(x))  # 将字符串转换为列表再合并
        )

        tokenizer = Tokenizer(
            num_words=self.config_loader.config['model']['max_vocab']  # 正确键名
        )
        tokenizer.fit_on_texts(train_texts)
        return tokenizer

    def predict(self, text: str) -> dict:
        """执行预测"""
        # 预处理
        cleaned_text = self.preprocessor.clean_text(text)
        tokens = self.preprocessor.segment(cleaned_text)

        # 向量化
        sequence = self.tokenizer.texts_to_sequences([' '.join(tokens)])
        padded = pad_sequences(sequence, maxlen=self.config_loader.config['model']['max_len'])

        # 预测
        prob = self.model.predict(padded, verbose=0)[0][0]
        return {
            "text": text,
            "sentiment": "正面" if prob > 0.5 else "负面",
            "confidence": round(float(prob if prob > 0.5 else 1 - prob), 4)
        }


def main():
    """交互式预测入口"""
    try:
        analyzer = SentimentAnalyzer()
        logger.info("情感分析器已加载，输入 'exit' 退出")

        while True:
            text = input("\n请输入要分析的文本：")
            if text.lower() == 'exit':
                break

            if not text.strip():
                print("输入不能为空！")
                continue

            result = analyzer.predict(text)
            print(f"\n分析结果：{result['sentiment']} (置信度: {result['confidence']:.2%})")

    except Exception as e:
        logger.error("预测失败", exc_info=True)


if __name__ == "__main__":
    main()
