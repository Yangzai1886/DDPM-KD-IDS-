## DIFFUSION MULTIMODAL DISTILLATION COLLABORATION: A Generative  Equilibrium Framework for Efficient Vehicle Networking Intrusion  Detection

## 扩散多模态蒸馏协作：面向高效车联网入侵检测的生成式均衡框架

## 🏗️ GitHub仓库结构

text

```
IDS-Multimodal-DDPM/
├── README.md                          # 项目主文档
├── requirements.txt                   # 依赖包列表
├── environment.yml                    # Conda环境配置
├── config/
│   ├── datasets.yaml                  # 数据集配置
│   ├── experiments.yaml               # 实验超参数
│   └── model_architectures.yaml       # 模型架构配置
├── scripts/
│   ├── data_preprocessing/            # 数据预处理脚本
│   │   ├── 01_feature_extraction.py
│   │   ├── 02_feature_grouping.py
│   │   ├── 03_image_generation.py
│   │   └── 04_data_balancing.py
│   ├── models/                        # 模型训练脚本
│   │   ├── train_multimodal.py
│   │   ├── train_baseline_models.py
│   │   └── knowledge_distillation.py
│   └── evaluation/                    # 评估脚本
│       ├── evaluate_models.py
│       ├── generate_results.py
│       └── statistical_tests.py
├── src/                               # 源代码
│   ├── data/
│   │   ├── datasets.py
│   │   ├── preprocessing.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── multimodal.py
│   │   ├── ddpm.py
│   │   ├── baselines.py
│   │   └── distillation.py
│   ├── utils/
│   │   ├── config.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── experiments/
│       ├── base_experiment.py
│       ├── multimodal_experiment.py
│       └── baseline_experiment.py
├── notebooks/                         # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_result_analysis.ipynb
├── tests/                             # 单元测试
│   ├── test_data.py
│   ├── test_models.py
│   └── test_utils.py
└── docs/                              # 文档
    ├── dataset_preprocessing.md
    ├── hyperparameter_documentation.md
    └── reproduction_guide.md
```



## 📋 流程

### 1. 环境设置和依赖管理

**requirements.txt:**

txt

```
torch>=1.9.0
torchvision>=0.10.0
scikit-learn>=0.24.0
pandas>=1.3.0
numpy>=1.21.0
opencv-python>=4.5.0
Pillow>=8.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
diffusers>=0.3.0
efficientnet-pytorch>=0.7.0
scipy>=1.7.0
joblib>=1.0.0
```



### 2. 数据集预处理脚本

**scripts/data_preprocessing/01_feature_extraction.py:**

python

```
import argparse
import yaml
from src.data.preprocessing import DataPreprocessor

def main():
    parser = argparse.ArgumentParser(description='数据集特征提取')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['cicids2017', 'cicids2018', 'toniot'],
                       help='数据集名称')
    parser.add_argument('--config', type=str, default='config/datasets.yaml',
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化预处理器
    preprocessor = DataPreprocessor(config[args.dataset])
    
    # 执行预处理流程
    print(f"开始处理 {args.dataset} 数据集...")
    features, labels = preprocessor.load_and_preprocess()
    
    # 保存预处理结果
    preprocessor.save_processed_data(features, labels)
    print("预处理完成!")

if __name__ == "__main__":
    main()
```



### 3. 超参数文档

**docs/hyperparameter_documentation.md:**

markdown

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

### 通用参数
- `img_size`: 224 - 输入图像尺寸
- `batch_size`: 64 - 批次大小
- `num_epochs`: 20 - 训练轮数
- `learning_rate`: 0.001 - 学习率

### 模型特定参数
- 分类器头根据数据集类别数调整
```



### 4. 主运行脚本

**scripts/run_experiment.py:**

python

```
#!/usr/bin/env python3


import argparse
import yaml
import torch
import random
import numpy as np
from src.experiments.multimodal_experiment import MultimodalExperiment
from src.experiments.baseline_experiment import BaselineExperiment

def set_seed(seed=42):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='运行入侵检测实验')
    parser.add_argument('--experiment', type=str, required=True,
                       choices=['multimodal', 'baseline', 'all'],
                       help='实验类型')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cicids2017', 'cicids2018', 'toniot', 'all'],
                       help='数据集名称')
    parser.add_argument('--config', type=str, default='config/experiments.yaml',
                       help='实验配置文件')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"开始实验: {args.experiment}")
    print(f"数据集: {args.dataset}")
    print(f"随机种子: {args.seed}")
    print(f"设备: {args.device}")
    
    # 运行实验
    if args.experiment in ['multimodal', 'all']:
        multimodal_exp = MultimodalExperiment(config['multimodal'])
        multimodal_exp.run(args.dataset)
    
    if args.experiment in ['baseline', 'all']:
        baseline_exp = BaselineExperiment(config['baseline'])
        baseline_exp.run(args.dataset)

if __name__ == "__main__":
    main()
```



### 5. 可重现性配置

**config/experiments.yaml:**

yaml

```
multimodal:
  data_preprocessing:
    img_size: 32
    n_clusters: 3
    feature_selection:
      method: "mutual_information"
      n_features: 75
      n_groups: 3
    
  ddpm:
    image_ddpm:
      epochs: 100
      target_count: 2000
      batch_size: 32
      learning_rate: 2e-4
      timesteps: 1000
    
    table_ddpm:
      samples_per_class: 1200
      timesteps: 300
      learning_rate: 1e-3
      epochs: 30
  
  training:
    batch_size: 64
    teacher_epochs: 20
    student_epochs: 20
    k_folds: 5
    learning_rate: 0.001
    weight_decay: 1e-5
    
  knowledge_distillation:
    temperature: 3.0
    alpha: 0.5
    loss_weights:
      ce: 0.5
      kd: 0.5

baseline:
  models:
    - "mobilenetv3"
    - "shufflenet" 
    - "alexnet"
    - "efficientnet-lite"
    - "resnet50"
  
  training:
    img_size: 224
    batch_size: 64
    epochs: 20
    learning_rate: 0.001
    k_folds: 5
  
  evaluation:
    metrics:
      - "accuracy"
      - "precision"
      - "recall" 
      - "f1_score"
      - "inference_time"
      - "model_size"

seeds:
  data_splitting: 42
  model_initialization: 42
  training: 42
```



### 6. 运行命令示例

bash

```
# 1. 设置环境
conda env create -f environment.yml
conda activate ids-multimodal

# 2. 数据预处理
python scripts/data_preprocessing/01_feature_extraction.py --dataset cicids2017
python scripts/data_preprocessing/02_feature_grouping.py --dataset cicids2017
python scripts/data_preprocessing/03_image_generation.py --dataset cicids2017

# 3. 运行多模态实验
python scripts/run_experiment.py --experiment multimodal --dataset cicids2017 --seed 42

# 4. 运行基准实验
python scripts/run_experiment.py --experiment baseline --dataset cicids2017 --seed 42

# 5. 生成最终结果
python scripts/evaluation/generate_results.py --dataset cicids2017 --output results/

```
