import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, BertTokenizer,
                          get_linear_schedule_with_warmup)

from utils.config_loader import ConfigLoader
from utils.progress import TrainingProgress

logger = logging.getLogger(__name__)


class BertTextDataset:
    """BERT文本数据集处理类"""

    def __init__(self, data_path: str, tokenizer: BertTokenizer, max_length: int = 128):
        logger.info(f"初始化数据集，路径: {data_path}")
        try:
            self.df = pd.read_csv(data_path)
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.texts = self._preprocess_text()
            self.labels = torch.LongTensor(self.df['label'].values)
            logger.info(f"成功加载 {len(self)} 条样本")
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            raise

    def _preprocess_text(self) -> List[str]:
        """文本预处理流水线"""
        logger.debug("执行文本预处理")
        return self.df['segmented'].apply(eval).apply(' '.join).tolist()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本的编码"""
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

    def __len__(self) -> int:
        return len(self.df)


class BertTrainer:
    """BERT模型训练控制器"""

    def __init__(self, config: Dict):
        logger.info("初始化BERT训练器")
        self.config = config['training']
        self._validate_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化核心组件
        self.tokenizer = self._init_tokenizer()
        self.model = self._init_model()
        self.optimizer, self.scheduler = self._init_optimizer()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.early_stopping = EarlyStopping(patience=self.config['patience'])
        self.progress = None  # 进度条控制器
        self.current_epoch = 1

        logger.info(f"训练设备: {self.device}")
        logger.debug(f"训练配置: {self.config}")

    def _validate_config(self):
        """验证配置参数有效性"""
        required_keys = ['bert_model', 'batch_size', 'epochs', 'lr']
        for key in required_keys:
            if key not in self.config:
                logger.error(f"缺失必要配置项: {key}")
                raise ValueError(f"配置文件中缺少 {key}")

    def _init_tokenizer(self) -> BertTokenizer:
        """初始化BERT分词器"""
        logger.info(f"加载BERT分词器: {self.config['bert_model']}")
        try:
            return BertTokenizer.from_pretrained(
                self.config['bert_model'],
                cache_dir="./model_cache"
            )
        except Exception as e:
            logger.error(f"分词器加载失败: {str(e)}")
            raise

    def _init_model(self) -> torch.nn.Module:
        """初始化BERT分类模型"""
        logger.info(f"加载预训练模型: {self.config['bert_model']}")
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config['bert_model'],
                num_labels=2,
                output_attentions=False,
                output_hidden_states=False
            ).to(self.device)

            if self.config.get('freeze_bert', False):
                logger.info("冻结BERT基础层参数")
                for param in model.bert.parameters():
                    param.requires_grad = False
            return model
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise

    def _init_optimizer(self) -> Tuple[torch.optim.Optimizer, LambdaLR]:
        """初始化优化器和学习率调度器"""
        logger.info("初始化优化器")
        try:
            # 参数分组策略
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in self.model.named_parameters()
                               if not any(nd in n for nd in no_decay)],
                    'weight_decay': self.config['weight_decay']
                },
                {
                    'params': [p for n, p in self.model.named_parameters()
                               if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0
                }
            ]

            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.config['lr'],
                eps=self.config['adam_epsilon']
            )

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config['warmup_steps'],
                num_training_steps=self.config['total_steps']
            )
            return optimizer, scheduler
        except Exception as e:
            logger.error(f"优化器初始化失败: {str(e)}")
            raise

    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """创建数据加载器"""
        logger.info("创建数据加载器")
        try:
            train_set = BertTextDataset(
                self.config['train_path'],
                self.tokenizer,
                self.config['max_length']
            )
            val_set = BertTextDataset(
                self.config['val_path'],
                self.tokenizer,
                self.config['max_length']
            )
            test_set = BertTextDataset(
                self.config['test_path'],
                self.tokenizer,
                self.config['max_length']
            )

            return (
                DataLoader(train_set, batch_size=self.config['batch_size'], shuffle=True),
                DataLoader(val_set, batch_size=self.config['batch_size']),
                DataLoader(test_set, batch_size=self.config['batch_size'])
            )
        except Exception as e:
            logger.error(f"数据加载器创建失败: {str(e)}")
            raise

    def _compute_metrics(self, preds: np.ndarray, labels: np.ndarray) -> Dict:
        """计算综合评估指标"""
        logger.debug("计算评估指标")
        return {
            'acc': accuracy_score(labels, preds),
            'f1': f1_score(labels, preds, average='macro'),
            'roc_auc': roc_auc_score(labels, preds),
            'confusion_matrix': confusion_matrix(labels, preds),
            'classification_report': classification_report(labels, preds, output_dict=True)
        }

    def train_epoch(self, train_loader: DataLoader) -> float:
        """执行单个训练周期"""
        logger.info(f"开始第 {self.current_epoch} 轮训练")
        self.model.train()
        total_loss = 0.0

        # 初始化进度条
        epoch_pbar = self.progress.epoch_progress(self.current_epoch)
        batch_pbar = self.progress.batch_progress('train', self.current_epoch)

        try:
            for step, batch in enumerate(train_loader, 1):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()

                # 更新进度条
                metrics = {
                    'loss': loss.item(),
                    'lr': self.optimizer.param_groups[0]['lr']
                }
                self.progress.update_metrics(batch_pbar, metrics)
                batch_pbar.update()

                # 每10步记录日志
                if step % 10 == 0:
                    logger.debug(f"训练进度: {step}/{len(train_loader)} | 当前损失: {loss.item():.4f}")

            avg_loss = total_loss / len(train_loader)
            logger.info(f"第 {self.current_epoch} 轮训练完成，平均损失: {avg_loss:.4f}")
            return avg_loss

        except Exception as e:
            logger.error(f"训练过程中出现异常: {str(e)}")
            raise
        finally:
            batch_pbar.close()
            epoch_pbar.close()

    def evaluate(self, data_loader: DataLoader, mode: str = 'val') -> Tuple[float, Dict]:
        """模型评估流程"""
        logger.info(f"执行 {mode} 评估")
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        # 初始化进度条
        eval_pbar = self.progress.batch_progress(mode)

        try:
            with torch.no_grad():
                for batch in data_loader:
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                    labels = batch['labels'].to(self.device)

                    outputs = self.model(**inputs, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits

                    total_loss += loss.item()
                    preds = torch.argmax(logits, dim=1).cpu().numpy()

                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())
                    eval_pbar.update()

            avg_loss = total_loss / len(data_loader)
            metrics = self._compute_metrics(np.array(all_preds), np.array(all_labels))
            logger.info(f"{mode} 评估完成，平均损失: {avg_loss:.4f} | 准确率: {metrics['acc']:.4f}")
            return avg_loss, metrics

        except Exception as e:
            logger.error(f"评估过程中出现异常: {str(e)}")
            raise
        finally:
            eval_pbar.close()

    def run(self):
        """执行完整训练流程"""
        logger.info("启动训练流程")
        try:
            train_loader, val_loader, test_loader = self._create_dataloaders()

            # 初始化进度条组件
            self.progress = TrainingProgress(
                total_epochs=self.config['epochs'],
                train_loader_len=len(train_loader),
                val_loader_len=len(val_loader),
                metrics=['loss', 'acc', 'f1'],
                colours=self.config.get('progress_colors', None)
            )

            for epoch in range(1, self.config['epochs'] + 1):
                self.current_epoch = epoch
                start_time = time.time()

                # 训练阶段
                train_loss = self.train_epoch(train_loader)

                # 验证阶段
                val_loss, val_metrics = self.evaluate(val_loader)

                # 记录训练历史
                self._update_history(epoch, train_loss, val_loss, val_metrics)

                # 早停判断
                if self._check_early_stopping(val_loss, val_metrics):
                    break

                # 保存最佳模型
                if val_metrics['f1'] > self.early_stopping.best_metrics.get('f1', 0):
                    self._save_checkpoint(epoch)
                    logger.info(f"发现新的最佳模型，已保存至指定路径")

            # 最终测试
            logger.info("启动最终测试")
            _, test_metrics = self.evaluate(test_loader, mode='test')
            self._log_final_results(test_metrics)

        except KeyboardInterrupt:
            logger.warning("用户中断训练，尝试保存当前模型...")
            self._save_checkpoint(self.current_epoch, is_interrupted=True)
        except Exception as e:
            logger.error(f"训练流程异常终止: {str(e)}")
            raise

    def _update_history(self, epoch: int, train_loss: float, val_loss: float, metrics: Dict):
        """更新训练历史记录"""
        self.history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            **metrics
        })
        logger.debug(f"更新训练历史记录: Epoch {epoch}")

    def _check_early_stopping(self, val_loss: float, metrics: Dict) -> bool:
        """执行早停判断"""
        stop_metrics = {
            'loss': val_loss,
            'acc': metrics['acc'],
            'f1': metrics['f1']
        }
        if self.early_stopping(stop_metrics):
            logger.info(f"早停触发于第 {self.current_epoch} 轮，最佳F1: {self.early_stopping.best_metrics['f1']:.4f}")
            return True
        return False

    def _save_checkpoint(self, epoch: int, is_interrupted: bool = False):
        """保存模型检查点"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d%H%M')
            suffix = "_interrupted" if is_interrupted else ""
            save_path = Path(self.config['save_dir']) / f"bert_model_{timestamp}{suffix}.pt"

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_metrics': self.early_stopping.best_metrics
            }, save_path)

            logger.info(f"模型检查点已保存至: {save_path}")
        except Exception as e:
            logger.error(f"模型保存失败: {str(e)}")

    def _log_final_results(self, test_metrics: Dict):
        """记录最终测试结果"""
        logger.info("\n=== 最终测试结果 ===")
        logger.info(f"测试准确率: {test_metrics['acc']:.4f}")
        logger.info(f"测试F1分数: {test_metrics['f1']:.4f}")
        logger.info(f"测试AUC-ROC: {test_metrics['roc_auc']:.4f}")
        logger.info("\n分类报告:\n" + classification_report(
            test_metrics['classification_report']['0']['support'],
            None,
            target_names=['负面', '正面'],
            digits=4
        ))
        logger.info(f"混淆矩阵:\n{test_metrics['confusion_matrix']}")


class EarlyStopping:
    """改进的早停机制"""

    def __init__(self, patience: int = 5, delta: float = 0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_metrics = {'loss': float('inf'), 'f1': 0, 'acc': 0}

    def __call__(self, val_metrics: Dict) -> bool:
        improve_conditions = [
            val_metrics['loss'] < self.best_metrics['loss'] - self.delta,
            val_metrics['f1'] > self.best_metrics['f1'] + self.delta,
            val_metrics['acc'] > self.best_metrics['acc'] + self.delta
        ]

        if any(improve_conditions):
            self.best_metrics = val_metrics.copy()
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def main():
    """主函数入口"""
    try:
        logger.info("程序启动，加载配置...")
        config = ConfigLoader.load()
        trainer = BertTrainer(config)
        trainer.run()
    except Exception as e:
        logger.error(f"程序异常终止: {str(e)}")
        raise


if __name__ == '__main__':
    main()
