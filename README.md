# NLP demo

本demo用于我对于自然语言处理NLP的一些练习和实验

## ChnSentiCorp

+ 数据集来源：[酒店评论数据情感](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/ChnSentiCorp_htl_all/intro.ipynb)

![confusion_matrix](./ChnSentiCorp/docs/confusion_matrix.png)

## 微博weibo_senti

1. 项目概况
   + **数据集来源**：[微博评论情感标注](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/weibo_senti_100k/intro.ipynb)
   + 原始数据规模：10万条标注数据
   + 第一轮准确率：51.29%（接近随机猜测，判定为失败）
2. 失败总结
   1. **数据预处理关键失误**：一刀切的删除所有非中文字符所以把表情删了，表情其实是表达情绪的最直观的方法之一
   2. **模型架构选型失误：**LSTM对短文本建模失效（微博平均长度15.8字，窗口效应显著），注意力机制缺失（未能捕捉"但是"等转折词影响）
   3. **训练策略失误：**早停机制失效（验证集仅用准确率单指标，未监测F1-score波动），学习率机制僵化（未考虑损失曲面特性）
   4. **评估体系结构性缺陷：**指标单一化（仅依赖准确率忽视AUC-ROC，且未建立混淆矩阵分析）
