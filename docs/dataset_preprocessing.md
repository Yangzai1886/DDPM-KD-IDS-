# 数据集预处理指南

## 概述

本文档描述了DM-DC框架中使用的数据集的预处理流程，包括CICIDS2017、CICIDS2018和TonIoT数据集。

## 数据集信息

### CICIDS2017
- **来源**: Canadian Institute for Cybersecurity
- **样本数**: 2,830,743
- **特征数**: 78
- **攻击类型**: 15类（包括正常流量）
- **下载链接**: [官方页面](https://www.unb.ca/cic/datasets/ids-2017.html)

### CICIDS2018  
- **来源**: Canadian Institute for Cybersecurity
- **样本数**: 16,233,347
- **特征数**: 80
- **攻击类型**: 15类（包括正常流量）
- **下载链接**: [官方页面](https://www.unb.ca/cic/datasets/ids-2018.html)

### TonIoT
- **来源**: University of New South Wales
- **样本数**: 461,043
- **特征数**: 45
- **攻击类型**: 7类（包括正常流量）
- **下载链接**: [官方页面](https://research.unsw.edu.au/projects/toniot-datasets)

## 预处理流程

### 步骤1: 特征提取

```bash
python scripts/data_preprocessing/01_feature_extraction.py --dataset cicids2017