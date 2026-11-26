"""打印 run.bat 的菜单。

中文只写在这里、不写进 .bat：cmd.exe 会按当前控制台代码页解释批处理文件里的
非 ASCII 字节，UTF-8 的中文会被拆成野命令（实测报 'ig' / 'TARGET' 之类的错）。
Python 这边自己管编码，所以中文放这儿最稳。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from common import load_config  # noqa: E402

print("=" * 66)
print("  铜熔炼智能配矿  -  一键运行")
print("  要换数据集：改 ml_pipeline/config.py 里的 ACTIVE_DATASET")
print("=" * 66)
print()

try:
    cfg = load_config()
    print(f"  当前数据集 : {cfg.dataset}")
    print(f"  数据文件   : {cfg.data_path}")
    print(f"  结果目录   : {cfg.root}")
    print(f"  运行标识   : {cfg.run}")
except BaseException as exc:                                    # noqa: BLE001
    print(f"  [警告] 读 config.py 失败: {type(exc).__name__}: {exc}")

print()
print("-" * 66)
print("   1   步骤1   TabPFN 训练+预测（全部29目标）")
print("   1k  步骤1   同上，只跑4个关键目标（约16分钟）")
print("   3   步骤3   五次划分评价（表3/表S1）")
print("   4   步骤4   全目标 SHAP（较慢，建议先只跑 5）")
print("   5   步骤5   四关键指标 SHAP 论文图")
print("   5b  步骤5b  有符号 SHAP（蜂群图/waterfall/依赖图）")
print("   6   步骤6   学习曲线")
print("   7   步骤7   论文图2/图4/图5 + 运行汇总（秒级）")
print("-" * 66)
print("   配矿优化不在此菜单：见 scripts/optimization/nsga2_optimization.py")
print("   （固定 1200 ℃ / 21 g O2，只求 11 个配比 + SiO2 熔剂）")
print("-" * 66)
print("   多个编号用逗号分隔，例如 1k,7   或   1,7")
print("   直接回车 = 1k（复现四项关键指标结果）；q = 退出")
print()
