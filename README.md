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

# 4. 运行基准实验
python scripts/run_experiment.py --experiment baseline --dataset cicids2017 --seed 42

# 5. 生成最终结果
python scripts/evaluation/generate_results.py --dataset cicids2017 --output results/
```
