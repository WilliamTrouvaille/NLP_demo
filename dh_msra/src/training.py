from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification

from utils import Loader, LoggerHandler, TrainingProgress

# 初始化日志和配置
config = Loader.load()
logger = LoggerHandler()  # 初始化日志器
root = Loader.get_root_dir()


class NERDataset(Dataset):
    """命名实体识别数据集类"""

    def __init__(self, data_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        data_path = Path(root) / data_path

        self.sentences, self.labels = self._load_data(data_path)
        self.label2id = self._create_label_map()
        logger.info(f"数据集加载完成，共 {len(self.sentences)} 个句子")

    def _create_label_map(self):
        """创建标签到ID的映射表"""
        unique_labels = set()
        for tags in self.labels:
            unique_labels.update(tags)
        label2id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        logger.info(f"创建标签映射表，共 {len(label2id)} 个标签")
        return label2id

    def _load_data(self, path):
        """加载预处理后的数据"""
        sentences, labels = [], []
        current_sentence, current_labels = [], []

        # 确保路径存在
        path = Path(path)
        if not path.exists():
            logger.error(f"数据文件 {path} 不存在，请先运行数据预处理脚本。")
            raise FileNotFoundError(f"数据文件 {path} 不存在，请先运行数据预处理脚本。")

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_sentence:
                        sentences.append(current_sentence)
                        labels.append(current_labels)
                        current_sentence, current_labels = [], []
                    continue
                char, tag = line.split('\t')
                current_sentence.append(char)
                current_labels.append(tag)

        return sentences, labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label_ids = [self.label2id.get(tag, 0) for tag in self.labels[idx]]

        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            is_split_into_words=True,
            return_tensors='pt'
        )

        # 对齐标签（处理subword问题）
        aligned_labels = []
        word_ids = encoding.word_ids()
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:  # 特殊token
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:  # 新词
                aligned_labels.append(label_ids[word_idx])
                previous_word_idx = word_idx
            else:  # 同词的后续subword
                aligned_labels.append(-100)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.LongTensor(aligned_labels)
        }


class NERTrainer:
    """命名实体识别训练器"""

    def __init__(self, dry_run=False):
        self.config = config
        self.device = torch.device(config['training']['device'])
        self.tokenizer = BertTokenizerFast.from_pretrained(config['model']['pretrained'])
        self.logger = logger
        self.model = None  # 初始化时先不加载模型
        self.dataset = None  # 初始化时先不加载数据集
        self.dry_run = dry_run  # 是否启用 dry_run 模式
        self.logger.info("NERTrainer 初始化完成")

    def _init_model(self):
        """初始化模型"""
        self.logger.info("开始初始化模型...")
        model = BertForTokenClassification.from_pretrained(
            config['model']['pretrained'],
            num_labels=config['model']['num_labels'],
            id2label={v: k for k, v in self.dataset.label2id.items()}
        )
        model = model.to(self.device)
        self.logger.info("模型初始化完成")
        return model

    def prepare_data(self):
        """准备数据集"""
        self.logger.info("开始准备数据集...")
        # 加载完整数据集
        data_path = Path(config['data']['processed_dir']) / 'train.txt'
        full_dataset = NERDataset(
            data_path,
            self.tokenizer,
            max_length=config['training']['max_seq_length']
        )

        # 划分训练/验证集
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(config['data']['seed'])
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size']
        )
        self.dataset = full_dataset  # 保持标签映射的引用

        # 初始化模型
        self.model = self._init_model()
        self.logger.info("数据集准备完成")

    def train(self):
        """执行训练流程"""
        self.logger.info("开始训练流程...")
        optimizer = AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        progress = TrainingProgress(
            total_epochs=config['training']['num_epochs'],
            train_loader_len=len(self.train_loader),
            val_loader_len=len(self.val_loader),
            metrics=['loss', 'acc', 'f1']
        )

        best_score = 0
        for epoch in range(config['training']['num_epochs']):
            epoch_progress = progress.epoch_progress(epoch + 1)

            # 训练阶段
            self.model.train()
            train_metrics = defaultdict(float)
            batch_progress = progress.batch_progress('train', epoch + 1)

            for step, batch in enumerate(self.train_loader):
                if self.dry_run:
                    self.logger.info(f"Dry run: 跳过训练步骤 {step + 1}")
                    continue

                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # 计算指标
                preds = torch.argmax(outputs.logits, dim=-1)
                mask = (batch['labels'] != -100)
                correct = (preds[mask] == batch['labels'][mask]).sum().item()
                total = mask.sum().item()

                train_metrics['loss'] += loss.item()
                train_metrics['acc'] += correct / total if total > 0 else 0

                # 更新进度条
                batch_progress.update(1)
                progress.update_metrics(
                    batch_progress,
                    {'loss': loss.item(), 'acc': correct / total if total > 0 else 0}
                )

            # 验证阶段
            val_metrics = self.evaluate()

            # 保存最佳模型
            if val_metrics['f1'] > best_score:
                best_score = val_metrics['f1']
                self.save_model(epoch + 1)

            # 更新epoch进度条
            epoch_progress.set_postfix({
                'train_loss': np.mean(train_metrics['loss']),
                'val_f1': val_metrics['f1'],
                'best_f1': best_score
            })
            epoch_progress.update(1)

        self.logger.info("训练流程完成")

    def evaluate(self):
        """模型评估"""
        self.logger.info("开始模型评估...")
        self.model.eval()
        metrics = defaultdict(float)
        progress = TrainingProgress(
            total_epochs=1,
            train_loader_len=len(self.val_loader),
            metrics=['loss', 'acc', 'f1']
        )
        batch_progress = progress.batch_progress('val')

        with torch.no_grad():
            for batch in self.val_loader:
                if self.dry_run:
                    self.logger.info(f"Dry run: 跳过评估步骤")
                    continue

                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)

                # 计算基础指标
                loss = outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                mask = (batch['labels'] != -100)
                correct = (preds[mask] == batch['labels'][mask]).sum().item()
                total = mask.sum().item()

                metrics['loss'] += loss
                metrics['acc'] += correct / total if total > 0 else 0

                # 更新进度条
                batch_progress.update(1)
                progress.update_metrics(
                    batch_progress,
                    {'loss': loss, 'acc': correct / total if total > 0 else 0}
                )

        self.logger.info("模型评估完成")
        return {k: v / len(self.val_loader) for k, v in metrics.items()}

    def save_model(self, epoch):
        """保存模型"""
        self.logger.info("开始保存模型...")
        model_dir = Path(config['logging']['model_dir'])
        model_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"macbert-ner-epoch{epoch}-{timestamp}.bin"

        torch.save(self.model.state_dict(), model_dir / filename)
        self.logger.info(f"模型已保存至：{filename}")


def main():
    try:
        logger.info("启动训练任务...")
        trainer = NERTrainer(dry_run=True)
        trainer.prepare_data()
        trainer.train()
        logger.info("训练任务完成")
    except Exception as e:
        logger.error(f"训练任务失败：{e}")
        raise


if __name__ == "__main__":
    main()
