```
## 多模态模型超参数

### 数据预处理
- `img_size`: 32 - 生成的RGB图像尺寸
- `n_clusters`: 3 - 特征分组聚类数
- `feature_selection_threshold`: 75 - 选择的特征数量

### DDPM训练参数
- `ddpm_epochs`: 150 - DDPM训练轮数
- `ddpm_target_count`: 2000 - 每个少数类目标样本数
- `table_ddpm_samples`: 1200 - 表格DDPM生成样本数
- `learning_rate`: 2e-4 - 学习率
- `num_train_timesteps`: 1000 - 训练时间步数

### 多模态训练参数
- `batch_size`: 64 - 批次大小
- `teacher_epochs`: 20 - 教师模型训练轮数
- `student_epochs`: 20 - 学生模型训练轮数
- `k_folds`: 5 - 交叉验证折数
- `temperature`: 3.0 - 知识蒸馏温度
- `alpha`: 0.5 - 知识蒸馏损失权重

### 优化器参数
- `optimizer`: AdamW
- `weight_decay`: 1e-5
- `lr_scheduler`: 无

## 基准模型超参数

### 通用参数
- `img_size`: 224 - 输入图像尺寸
- `batch_size`: 64 - 批次大小
- `num_epochs`: 20 - 训练轮数
- `learning_rate`: 0.001 - 学习率

### 模型特定参数
- 分类器头根据数据集类别数调整
```