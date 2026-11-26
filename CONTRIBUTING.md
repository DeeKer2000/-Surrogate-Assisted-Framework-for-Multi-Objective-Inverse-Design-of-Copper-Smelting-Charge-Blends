# Contributing / 复现约定

## 只想跑通预测

```bash
pip install -r requirements.txt
export TABPFN_TOKEN=tabpfn_sk_xxx        # Windows: $env:TABPFN_TOKEN="..."
python ml_pipeline/step1_train_tabpfn.py --targets key
```

数据集需自行放置到 `data/<dataset>/ml_dataset.csv`（16 输入 + 29 输出），
数据集清单与格式见 `README.md` 的「数据准备」一节。

## 提交前自查

1. 不要提交 `data/`、`experiments/`、`*.ckpt`、`*.tabpfn_fit`、`*.csv`、`*.xlsx`。
2. 不要把 `TABPFN_TOKEN`、绝对路径（`E:\...`、`C:\Users\...`）写进代码；
   token 走环境变量，路径走 `ml_pipeline/config.py`。
3. 改完脚本请本地跑一次：

   ```bash
   python -m unittest discover tests
   ```

## 命名与目录约定

| 目录 | 内容 |
|---|---|
| `scripts/` | 单文件、可直接运行的实验脚本（每个脚本顶部集中放参数） |
| `ml_pipeline/` | 多步骤流水线：统一由 `config.py` 驱动，`run_all.ps1` 串起来 |
| `tests/` | 纯函数级单元测试，不需要 GPU 与数据集 |
