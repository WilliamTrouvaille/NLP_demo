import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
)
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification

from training import NERDataset
from utils import Loader, LoggerHandler

# 初始化日志和配置
config = Loader.load()
logger = LoggerHandler()  # 初始化日志器
root = Loader.get_root_dir()

plt.rcParams['font.sans-serif'] = ["WenQuanYi Micro Hei"]
plt.rcParams['axes.unicode_minus'] = False


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, model_path, test_data_path):
        self.device = torch.device(config['training']['device'])
        self.tokenizer = BertTokenizerFast.from_pretrained(config['model']['pretrained'])
        # 先初始化测试数据集
        self.test_dataset = NERDataset(test_data_path, self.tokenizer, max_length=config['training']['max_seq_length'])
        # 再加载模型
        model_path = Path(root) / "models" / model_path
        self.model = self._load_model(model_path)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False
        )

    def _load_model(self, model_path):
        """加载训练好的模型"""
        logger.info(f"加载模型：{model_path}")
        model = BertForTokenClassification.from_pretrained(
            config['model']['pretrained'],
            num_labels=config['model']['num_labels'],
            id2label={v: k for k, v in self.test_dataset.label2id.items()}
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        logger.info("模型加载完成")
        return model

    def evaluate(self):
        """执行评估"""
        logger.info("开始模型评估...")
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in self.test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                mask = (batch['labels'] != -100)

                # 收集预测和真实标签
                all_preds.extend(preds[mask].cpu().numpy())
                all_labels.extend(batch['labels'][mask].cpu().numpy())

        # 计算评估指标
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        logger.info(f"准确率: {accuracy:.4f}")
        logger.info(f"F1 值: {f1:.4f}")

        # 打印分类报告
        logger.info("分类报告：")
        print(classification_report(all_labels, all_preds, target_names=list(self.test_dataset.label2id.keys())))

        # 绘制混淆矩阵
        self.plot_confusion_matrix(all_labels, all_preds)

        # 绘制损失曲线

        if os.path.exists("logs/training_loss.log"):
            self.plot_loss_curve()

    def plot_confusion_matrix(self, true_labels, pred_labels):
        """绘制混淆矩阵"""
        logger.info("绘制混淆矩阵...")
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.test_dataset.label2id.keys(),
                    yticklabels=self.test_dataset.label2id.keys())
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.savefig(Path(root) / "docs/confusion_matrix.png")
        plt.close()
        logger.info(f"混淆矩阵已保存至 {Path(root)}/docs/confusion_matrix.png")

    def plot_loss_curve(self):
        """绘制损失曲线"""
        logger.info("绘制损失曲线...")
        with open(Path(root) / "logs/training_loss.log", "r") as f:
            lines = f.readlines()
            train_loss = [float(line.split()[1]) for line in lines if "Train Loss" in line]
            val_loss = [float(line.split()[1]) for line in lines if "Val Loss" in line]

        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='训练损失')
        plt.plot(val_loss, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.title('训练和验证损失曲线')
        plt.legend()
        plt.savefig(Path(root) / "docs/loss_curve.png")
        plt.close()
        logger.info(f"损失曲线已保存至 {Path(root)}/docs/loss_curve.png")


def main():
    try:
        logger.info("启动模型评估任务...")
        model_path = "macbert-ner-epoch6-20250404_231713.bin"
        test_data_path = Path(config['data']['processed_dir']) / 'test.txt'
        evaluator = ModelEvaluator(model_path, test_data_path)
        evaluator.evaluate()
        logger.info("模型评估任务完成")
    except Exception as e:
        logger.error(f"模型评估任务失败：{e}")
        raise


if __name__ == "__main__":
    main()
