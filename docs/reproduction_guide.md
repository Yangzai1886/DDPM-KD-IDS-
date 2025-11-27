安装必要环境后

配置路径

```
cicids2017:
  csv_path: "your/path/to/cicids2017/com2017fina_dataset.csv"
  # ... 其他路径
```

处理数据集

```
# 预处理单个数据集
python scripts/run_preprocessing_pipeline.py --dataset cicids2017

# 预处理所有数据集
python scripts/run_preprocessing_pipeline.py --dataset all
```

运行实验

```
# 运行完整多模态实验
python scripts/experiments/run_multimodal.py --dataset cicids2017

# 使用自定义配置
python scripts/experiments/run_multimodal.py \
  --dataset cicids2017 \
  --config config/custom_experiment.yaml \
  --seed 42
```

运行基准实验

```
# 运行所有基准模型
python scripts/experiments/run_baseline.py --dataset cicids2017

# 运行特定基准模型
python scripts/experiments/run_baseline.py \
  --dataset cicids2017 \
  --models resnet50 efficientnet-lite
```

分析结果

```
# 生成结果报告
python scripts/evaluation/generate_results.py --dataset cicids2017

# 统计检验
python scripts/evaluation/statistical_tests.py --dataset cicids2017 --metric accuracy
```