import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from utils.config_loader import ConfigLoader
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """文本数据集类"""

    def __init__(self, data_path, vocab=None):
        self.df = pd.read_csv(data_path)
        self.texts = self._process_text()
        self.labels = torch.LongTensor(self.df['label'].values)

        # 动态构建词汇表（仅在训练集初始化时）
        self.vocab = vocab or self._build_vocab()
        logger.info(f"数据集加载完成，样本数：{len(self)} | 词汇量：{len(self.vocab)}")

    def _process_text(self):
        """将分词结果转换为索引序列"""
        return [eval(seq) for seq in self.df['segmented']]

    def _build_vocab(self):
        """构建词汇映射表"""
        vocab = {'<pad>': 0, '<unk>': 1}
        for seq in self.texts:
            for word in seq:
                if word not in vocab:
                    vocab[word] = len(vocab)
        return vocab

    def __getitem__(self, idx):
        indices = [self.vocab.get(word, 1) for word in self.texts[idx]]  # 未知词处理
        return torch.LongTensor(indices), self.labels[idx]

    def __len__(self):
        return len(self.df)


class TextClassifier(nn.Module):
    """文本分类模型基类"""

    def __init__(self, config, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config['embed_dim'])
        self.dropout = nn.Dropout(config['dropout'])
        self.classifier = nn.Linear(config['hidden_dim'], 2)

    def forward(self, x):
        raise NotImplementedError


class LSTMModel(TextClassifier):
    """LSTM分类模型"""

    def __init__(self, config, vocab_size):
        super().__init__(config, vocab_size)
        self.lstm = nn.LSTM(config['embed_dim'],
                            config['hidden_dim'],
                            num_layers=config['num_layers'],
                            bidirectional=config['bidirectional'])

        final_hidden_dim = config['hidden_dim'] * 2 if config['bidirectional'] else config['hidden_dim']
        self.classifier = nn.Linear(final_hidden_dim, 2)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded.permute(1, 0, 2))  # (seq_len, batch, dim)

        # 处理双向LSTM的最终状态
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]

        return self.classifier(hidden)


class Trainer:
    """模型训练控制器"""

    def __init__(self, config):
        self.config = config['training']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化组件
        self.model = self._init_model()
        self.optimizer = self._init_optimizer()
        self.criterion = nn.CrossEntropyLoss()
        self.vocab = None

        # 训练状态跟踪
        self.best_val_acc = 0.0
        self.early_stop_counter = 0

    def _init_model(self):
        """根据配置初始化模型"""
        train_data = TextDataset(self.config['train_path'])
        model_config = self.config['model']

        model_types = {
            'lstm': LSTMModel,
        }

        model = model_types[model_config['type']](
            model_config,
            vocab_size=len(train_data.vocab)
        ).to(self.device)

        # 保存词汇表
        torch.save(train_data.vocab, Path(self.config['save_dir']) / 'vocab.pt')
        return model

    def _init_optimizer(self):
        """初始化优化器"""
        optimizers = {
            'adam': optim.Adam,
            'sgd': optim.SGD
        }
        return optimizers[self.config['optimizer']](
            self.model.parameters(),
            lr=self.config['lr']
        )

    def _collate_fn(self, batch):
        """动态填充批次数据"""
        inputs, labels = zip(*batch)
        lengths = torch.LongTensor([len(x) for x in inputs])
        padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        return padded.to(self.device), torch.stack(labels).to(self.device), lengths

    def train_epoch(self, loader):
        """单个训练周期"""
        self.model.train()
        total_loss = 0.0

        for inputs, labels, _ in loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def evaluate(self, loader):
        """模型评估"""
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels, _ in loader:
                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return accuracy_score(all_labels, all_preds)

    def run(self):
        """执行完整训练流程"""
        # 初始化数据加载器
        train_set = TextDataset(self.config['train_path'], vocab=self.vocab)
        val_set = TextDataset(self.config['val_path'], vocab=self.vocab)

        train_loader = DataLoader(train_set,
                                  batch_size=self.config['batch_size'],
                                  collate_fn=self._collate_fn,
                                  shuffle=True)

        val_loader = DataLoader(val_set,
                                batch_size=self.config['batch_size'],
                                collate_fn=self._collate_fn)

        # 训练循环
        for epoch in range(1, self.config['epochs'] + 1):
            train_loss = self.train_epoch(train_loader)
            val_acc = self.evaluate(val_loader)

            logger.info(f"Epoch {epoch:02d} | 训练损失: {train_loss:.4f} | 验证准确率: {val_acc:.4f}")

            # 早停与模型保存
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.early_stop_counter = 0
                self._save_checkpoint(epoch)
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.config['patience']:
                    logger.info(f"早停触发，最佳验证准确率：{self.best_val_acc:.4f}")
                    break

    def _save_checkpoint(self, epoch):
        """保存模型检查点"""
        save_path = Path(self.config['save_dir']) / f"best_model_{datetime.now().strftime('%Y%m%d%H%M')}.pt"
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'val_acc': self.best_val_acc
        }, save_path)
        logger.info(f"发现新的最佳模型，已保存至 {save_path}")


def main():
    config = ConfigLoader.load()
    trainer = Trainer(config)
    trainer.run()


if __name__ == '__main__':
    main()
