# test.py
import logging
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pathlib import Path
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, BertTokenizer)
from utils.config_loader import ConfigLoader
from utils.progress import TrainingProgress

logger = logging.getLogger(__name__)
plt.style.use('ggplot')


class ModelTester:
    """模型测试与分析器"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigLoader.load(config_path)['training']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.test_loader = None
        self._prepare_components()

    def _prepare_components(self):
        """初始化测试组件"""
        # 加载分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.config['bert_model'])

        # 加载模型
        model_path = Path(self.config['save_dir']) / "bert_model_202504010140.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config['bert_model'],
            num_labels=2
        ).to(self.device)

        # 加载训练好的权重
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"成功加载模型：{model_path}")

        # 准备测试数据
        test_set = BertTextDataset(
            self.config['test_path'],
            self.tokenizer,
            self.config['max_length']
        )
        self.test_loader = DataLoader(
            test_set,
            batch_size=self.config['batch_size']
        )

    def evaluate(self, save_dir: str = "./results"):
        """执行完整评估流程"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # 获取预测结果
        true_labels, pred_labels, probabilities = self._get_predictions()

        # 计算各项指标
        metrics = {
            'accuracy': accuracy_score(true_labels, pred_labels),
            'roc_auc': roc_auc_score(true_labels, probabilities[:, 1]),
            'classification_report': classification_report(
                true_labels, pred_labels, output_dict=True
            )
        }

        # 保存评估结果
        self._save_results(metrics, save_dir)

        # 生成可视化图表
        self._plot_confusion_matrix(true_labels, pred_labels, save_dir)
        self._plot_roc_curve(true_labels, probabilities, save_dir)

        logger.info(f"评估结果已保存至 {save_dir}")
        return metrics

    def _get_predictions(self):
        """获取模型预测结果"""
        self.model.eval()
        true_labels, pred_labels, probabilities = [], [], []

        with torch.no_grad():
            for batch in self.test_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].cpu().numpy()

                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                true_labels.extend(labels)
                pred_labels.extend(preds)
                probabilities.extend(probs)

        return np.array(true_labels), np.array(pred_labels), np.array(probabilities)

    def _save_results(self, metrics: dict, save_dir: str):
        """保存评估结果到文件"""
        # 保存数值指标
        with open(f"{save_dir}/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        # 保存分类报告
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        report_df.to_csv(f"{save_dir}/classification_report.csv", index=True)

        # 打印重要指标
        logger.info(f"测试准确率: {metrics['accuracy']:.4f}")
        logger.info(f"AUC-ROC: {metrics['roc_auc']:.4f}")

    def _plot_confusion_matrix(self, y_true, y_pred, save_dir):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['负面', '正面'],
                    yticklabels=['负面', '正面'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(f"{save_dir}/confusion_matrix.png")
        plt.close()

    def _plot_roc_curve(self, y_true, y_prob, save_dir):
        """绘制ROC曲线"""
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = roc_auc_score(y_true, y_prob[:, 1])

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(f"{save_dir}/roc_curve.png")
        plt.close()


class BertTextDataset(Dataset):
    """BERT文本数据集（与训练时相同实现）"""

    def __init__(self, data_path: str, tokenizer: BertTokenizer, max_length: int = 128):
        self.df = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = self._preprocess_text()
        self.labels = torch.LongTensor(self.df['label'].values)

    def _preprocess_text(self):
        return self.df['segmented'].apply(eval).apply(' '.join).tolist()

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.df)

def main():
    # 初始化日志和测试器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        tester = ModelTester()
        metrics = tester.evaluate()
        logger.info("模型测试完成！")

    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        raise


if __name__ == '__main__':
    main()

