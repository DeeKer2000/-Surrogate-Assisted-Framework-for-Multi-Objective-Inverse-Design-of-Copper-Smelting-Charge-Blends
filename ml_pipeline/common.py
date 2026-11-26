"""统一实验流程的公共模块：配置加载、路径解析、共享常量与通用工具。

所有 step*.py 都只从这里取路径和参数，任何脚本里都不再出现写死的数据集路径。
换数据集 / 换结果目录，只改 config.py。

典型用法：
    from common import load_config, setup_env, setup_console, split_data

    cfg = load_config()                 # 读取 config.py（可用 --config 指定别的文件）
    setup_env(cfg); setup_console()
    df, x, y = load_dataset(cfg)
    x_tr, x_te, y_tr, y_te, train_idx, test_idx = split_data(cfg, x, y)
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
#                              共享常量
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.py"

#: 16 个输入特征（顺序必须与训练数据一致）
INPUT_COLS = [
    "T(℃)", "P(atm)", "Cu(g)", "S(g)", "Fe(g)", "SiO2(g)",
    "Pb(g)", "Zn(g)", "MgO(g)", "Al2O3(g)", "Ni(g)", "Sb(g)",
    "Bi(g)", "CaO(g)", "As(g)", "O2(g)",
]

#: 29 个输出目标
OUTPUT_COLS = [
    # 冰铜 (wt.%)
    "matte_Cu_pct", "matte_Fe_pct", "matte_S_pct",
    "matte_Ni_pct", "matte_Zn_pct", "matte_Pb_pct", "matte_As_pct",
    "matte_mass_g",
    # 渣 (wt.%)
    "slag_Cu_pct", "slag_FeO_pct", "slag_SiO2_pct", "slag_FeO_SiO2_ratio",
    "slag_Al2O3_pct", "slag_CaO_pct", "slag_MgO_pct", "slag_ZnO_pct",
    "slag_mass_g",
    # 气相 (mol分数)
    "gas_SO2_mol", "gas_S2_mol", "gas_SO_mol", "gas_SSO_mol", "gas_SO3_mol",
    "gas_Zn_mol", "gas_PbS_mol", "gas_CuS_mol", "gas_Pb_mol", "gas_Sb_mol",
    "gas_As_mol",
    "gas_mass_g",
]

#: 论文四项关键指标
KEY_TARGETS = ["matte_Cu_pct", "slag_Cu_pct", "slag_FeO_pct", "slag_SiO2_pct"]

#: 论文补充材料汇总的 24 项基础组分指标
TARGET_GROUPS_24 = {
    "matte": ["Cu", "Fe", "S", "Ni", "Zn", "Pb"],
    "slag": ["Cu", "FeO", "SiO2", "Al2O3", "CaO", "MgO", "ZnO"],
    "gas": ["SO2", "S2", "SO", "SSO", "SO3", "Zn", "PbS", "CuS", "Pb", "Sb", "As"],
}
TARGETS_24 = [
    f"{phase}_{species}_{'mol' if phase == 'gas' else 'pct'}"
    for phase, species_list in TARGET_GROUPS_24.items()
    for species in species_list
]

TARGET_LABELS = {
    "matte_Cu_pct": "Matte Cu",
    "slag_Cu_pct": "Slag Cu",
    "slag_FeO_pct": "Slag FeO",
    "slag_SiO2_pct": r"Slag SiO$_2$",
}

FEATURE_LABELS = {
    "T(℃)": "T", "P(atm)": "P", "Cu(g)": "Cu", "S(g)": "S", "Fe(g)": "Fe",
    "SiO2(g)": "SiO$_2$", "Pb(g)": "Pb", "Zn(g)": "Zn", "MgO(g)": "MgO",
    "Al2O3(g)": "Al$_2$O$_3$", "Ni(g)": "Ni", "Sb(g)": "Sb", "Bi(g)": "Bi",
    "CaO(g)": "CaO", "As(g)": "As", "O2(g)": "O$_2$",
}

#: 论文热力图/柱状图使用的 13 项炉料组成（不含温度、压力、氧量）
CHARGE_FEATURES = [
    "Cu(g)", "S(g)", "Fe(g)", "SiO2(g)", "Pb(g)", "Zn(g)", "MgO(g)",
    "Al2O3(g)", "Ni(g)", "Sb(g)", "Bi(g)", "CaO(g)", "As(g)",
]

BASELINE_ALGORITHMS = [
    "XGBoost", "LightGBM", "GradientBoosting", "CatBoost", "RandomForest",
]
ALGORITHM_ORDER = ["TabPFN", *BASELINE_ALGORITHMS]


# ============================================================================
#                              配置
# ============================================================================

class ConfigError(RuntimeError):
    """配置文件有误。"""


def _f(value, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"配置项 {name} 需要数字，实际为 {value!r}") from exc


def _i(value, name: str) -> int:
    return int(_f(value, name))


def _str_list(value, name: str) -> list[str]:
    """接受列表，也接受 'a, b, c' 形式的字符串（方便命令行传参）。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return parse_list(str(value))


def parse_list(value: str) -> list[str]:
    """把 'a, b, c' 解析为 ['a','b','c']；空值返回 []。"""
    if value is None:
        return []
    return [item.strip() for item in str(value).replace("，", ",").split(",") if item.strip()]


#: TabPFN 支持的 fit_mode 取值
FIT_MODES = ("fit_preprocessors", "fit_with_cache", "low_memory", "batched")

_FIT_MODE_ALIASES = {
    "default": "fit_preprocessors",
    "preprocessors": "fit_preprocessors",
    "fit_preprocessors": "fit_preprocessors",
    "cache": "fit_with_cache",
    "fit_with_cache": "fit_with_cache",
    "low_memory": "low_memory",
    "batched": "batched",
}


def normalize_fit_mode(value, default: str) -> str:
    """把命令行的 --fit-mode 归一化成 TabPFN 的取值。

    接受简写：cache = fit_with_cache（快，数值略有差异）；
    default = fit_preprocessors（与步骤1、2 口径一致）。
    """
    if value in (None, ""):
        return default
    key = str(value).strip().lower().replace("-", "_")
    if key not in _FIT_MODE_ALIASES:
        raise ConfigError(
            f"未知的 fit_mode: {value!r}；可选 cache（=fit_with_cache）、"
            f"default（=fit_preprocessors），或 {' / '.join(FIT_MODES)}"
        )
    return _FIT_MODE_ALIASES[key]


@dataclass
class Config:
    """一次实验的全部设置与派生路径。"""

    config_path: Path
    dataset: str
    dataset_label: str
    run: str
    data_path: Path
    root: Path

    test_size: float
    seeds: list[int]
    random_state: int

    model_path: Path
    device: str
    n_estimators: int
    model_seed: int
    predict_batch_size: int
    predict_batch_floor: int
    token: str
    hf_endpoint: str

    baseline_algorithms: list[str]
    baseline_targets: str
    baseline_n_estimators: int
    baseline_random_state: int

    shap_targets: str
    shap_n_background: int
    shap_n_explain: int
    shap_budget: int
    shap_source: str

    learning_curve_sizes: list[int]
    fonts: list[str]
    cache: dict[str, bool] = field(default_factory=dict)
    optimization: dict = field(default_factory=dict)
    raw: dict[str, dict[str, str]] = field(default_factory=dict)

    def fit_mode_for(self, step: str) -> str:
        """该步骤应使用的 TabPFN fit_mode。

        'fit_with_cache' 会预计算训练上下文的 K/V，多目标/多划分时快很多，
        但与默认的 'fit_preprocessors' 不是数值等价的（R² 在第 4～6 位小数上不同）。
        要进论文表格的步骤默认关闭缓存，详见 config.py 第 5 节。
        """
        return "fit_with_cache" if self.cache.get(step, False) else "fit_preprocessors"

    @property
    def model_label(self) -> str:
        """预训练权重的可读名称，用于汇总文档（'auto' 时显示为自动下载）。"""
        return "auto（首次运行自动下载）" if str(self.model_path) == "auto" else self.model_path.name

    # ---------- 结果目录（全部位于 root 之下）----------
    @property
    def tabpfn_dir(self) -> Path:
        return self.root / "tabpfn" / self.run

    @property
    def models_dir(self) -> Path:
        return self.tabpfn_dir / "saved_models"

    @property
    def predictions_dir(self) -> Path:
        return self.tabpfn_dir / "predictions"

    @property
    def tabpfn_figures_dir(self) -> Path:
        return self.tabpfn_dir / "figures"

    @property
    def baselines_dir(self) -> Path:
        return self.root / "baselines" / self.run

    @property
    def multiseed_dir(self) -> Path:
        return self.root / "multiseed" / self.run

    @property
    def shap_dir(self) -> Path:
        return self.root / "shap" / self.run

    @property
    def shap_all_dir(self) -> Path:
        return self.shap_dir / "all_targets"

    @property
    def shap_key_dir(self) -> Path:
        return self.shap_dir / "key_targets"

    @property
    def learning_curve_dir(self) -> Path:
        return self.root / "learning_curve" / self.run

    @property
    def response_dir(self) -> Path:
        """工况响应面与可达域（步骤11 响应面、步骤12 联合逆向求解）。"""
        return self.root / "response" / self.run

    @property
    def optimization_dir(self) -> Path:
        return self.root / "optimization" / self.run

    @property
    def figures_dir(self) -> Path:
        """论文风格成图（图2、图3 等）。"""
        return self.root / "figures"

    def all_dirs(self) -> list[Path]:
        return [
            self.root, self.tabpfn_dir, self.models_dir, self.predictions_dir,
            self.tabpfn_figures_dir, self.baselines_dir, self.multiseed_dir,
            self.shap_all_dir, self.shap_key_dir, self.learning_curve_dir,
            self.optimization_dir, self.response_dir, self.figures_dir,
        ]

    def ensure_dirs(self) -> None:
        for path in self.all_dirs():
            path.mkdir(parents=True, exist_ok=True)

    # ---------- 目标集合 ----------
    def resolve_targets(self, spec=None) -> list[str]:
        """把 'all' / 'key' / '24' / 目标名解析为目标列表（始终按 OUTPUT_COLS 顺序）。

        关键字 all/key/24 不区分大小写；目标名大小写不敏感
        （matte_Cu_pct 与 matte_cu_pct 都能用）；也可以直接传列表。
        """
        if isinstance(spec, (list, tuple, set)):
            raw = ",".join(str(item) for item in spec)
        else:
            raw = str(spec if spec is not None else "all")
        keyword = raw.strip().lower()
        if keyword in ("all", "全部", "29"):
            return list(OUTPUT_COLS)
        if keyword in ("key", "key4", "关键", "四项关键指标"):
            return list(KEY_TARGETS)
        if keyword in ("24", "targets24"):
            return list(TARGETS_24)
        # 目标名大小写不敏感匹配（matte_Cu_pct / matte_cu_pct 都能用）
        lookup = {name.lower(): name for name in OUTPUT_COLS}
        requested = {lookup.get(item.lower(), item) for item in parse_list(raw)}
        unknown = sorted(requested - set(OUTPUT_COLS))
        if unknown:
            raise ConfigError(f"未知目标变量: {unknown}")
        return [t for t in OUTPUT_COLS if t in requested]

    def describe(self) -> str:
        return (
            f"配置文件 : {self.config_path}\n"
            f"数据集   : {self.dataset}  ({self.dataset_label or '未填写说明'})\n"
            f"数据文件 : {self.data_path}\n"
            f"结果根目录: {self.root}\n"
            f"运行名称 : {self.run}"
        )


def _load_module(path: Path):
    """把配置文件当作 Python 模块加载，这样配置里可以直接写列表、表达式和注释。"""
    if not path.exists():
        raise ConfigError(f"配置文件不存在: {path}")
    spec = importlib.util.spec_from_file_location("ml_pipeline_user_config", path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"无法加载配置文件: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ConfigError(f"配置文件执行出错: {path}\n  {type(exc).__name__}: {exc}") from exc
    return module


def _abs(value, name: str) -> Path:
    if value in (None, ""):
        raise ConfigError(f"配置项 {name} 不能为空")
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def load_config(path: str | Path | None = None) -> Config:
    """读取配置；命令行的 --config/--dataset/--run/--root 会覆盖配置内容。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None,
                        help="配置文件路径（默认 ml_pipeline/config.py）")
    parser.add_argument("--dataset", default=None, help="临时切换数据集名称")
    parser.add_argument("--run", default=None, help="临时指定运行名称")
    parser.add_argument("--root", default=None, help="临时指定结果根目录")
    known, _ = parser.parse_known_args()

    cfg_path = Path(known.config) if known.config else (Path(path) if path else DEFAULT_CONFIG)
    module = _load_module(cfg_path)

    datasets = getattr(module, "DATASETS", None)
    if not isinstance(datasets, dict) or not datasets:
        raise ConfigError(f"{cfg_path.name} 缺少 DATASETS 字典")

    dataset_name = (known.dataset or getattr(module, "ACTIVE_DATASET", "") or "").strip()
    if dataset_name not in datasets:
        raise ConfigError(f"DATASETS 里没有数据集 {dataset_name!r}；可选: {', '.join(datasets)}")
    entry = dict(datasets[dataset_name])

    sections: dict[str, dict] = {
        name: dict(getattr(module, name.upper(), None) or {})
        for name in ("split", "model", "baselines", "shap", "learning_curve",
                     "cache_acceleration", "optimization", "misc")
    }
    # 数据集字典里与段落同名的子字典，只对该数据集生效
    for key, value in entry.items():
        if isinstance(value, dict) and key in sections:
            sections[key].update(value)

    split, model = sections["split"], sections["model"]
    base, shap = sections["baselines"], sections["shap"]
    lc, misc = sections["learning_curve"], sections["misc"]
    cache = {k: bool(v) for k, v in sections["cache_acceleration"].items()}
    optimization = sections["optimization"]

    data_path = _abs(entry.get("data"), f"DATASETS[{dataset_name}]['data']")
    root = _abs(known.root or entry.get("root"), f"DATASETS[{dataset_name}]['root']")
    # "auto" 表示交给 tabpfn 自动从缓存/网络解析预训练权重，不做路径拼接
    raw_model_path = model.get("path")
    model_path = (Path("auto") if str(raw_model_path).strip() == "auto"
                  else _abs(raw_model_path, "MODEL['path']"))

    sizes = [int(v) for v in (lc.get("sizes") or [])]

    return Config(
        config_path=cfg_path.resolve(),
        dataset=dataset_name,
        dataset_label=str(entry.get("label", "") or "").strip(),
        run=(known.run or getattr(module, "RUN", "run1") or "run1").strip(),
        data_path=data_path,
        root=root,
        test_size=_f(split.get("test_size", 0.2), "SPLIT['test_size']"),
        seeds=[_i(v, "SPLIT['seeds']") for v in (split.get("seeds") or [42])],
        random_state=_i(split.get("random_state", 42), "SPLIT['random_state']"),
        model_path=model_path,
        device=str(model.get("device", "cuda")).strip(),
        n_estimators=_i(model.get("n_estimators", 8), "MODEL['n_estimators']"),
        model_seed=_i(model.get("model_seed", 0), "MODEL['model_seed']"),
        predict_batch_size=_i(model.get("predict_batch_size", 1000), "MODEL['predict_batch_size']"),
        predict_batch_floor=_i(model.get("predict_batch_floor", 250), "MODEL['predict_batch_floor']"),
        token=str(model.get("token", "") or "").strip(),
        hf_endpoint=str(model.get("hf_endpoint", "") or "").strip(),
        baseline_algorithms=(_str_list(base.get("algorithms"), "BASELINES['algorithms']")
                             or list(BASELINE_ALGORITHMS)),
        baseline_targets=base.get("targets", "key"),
        baseline_n_estimators=_i(base.get("n_estimators", 100), "BASELINES['n_estimators']"),
        baseline_random_state=_i(base.get("random_state", 0), "BASELINES['random_state']"),
        shap_targets=shap.get("targets", "key"),
        shap_n_background=_i(shap.get("n_background", 200), "SHAP['n_background']"),
        shap_n_explain=_i(shap.get("n_explain", 50), "SHAP['n_explain']"),
        shap_budget=_i(shap.get("budget", 256), "SHAP['budget']"),
        shap_source=str(shap.get("source", "cache")).strip(),
        learning_curve_sizes=sizes,
        fonts=(_str_list(misc.get("fonts"), "MISC['fonts']")
               or ["SimHei", "Microsoft YaHei"]),
        cache=cache,
        optimization=optimization,
        raw=sections,
    )


def add_extra_args() -> argparse.Namespace:
    """子脚本可用：解析 --config/--dataset/--run/--root 之外的参数。"""
    parser = argparse.ArgumentParser(add_help=False)
    args, _ = parser.parse_known_args()
    return args


def setup_env(cfg: Config | None = None) -> None:
    """设置 HuggingFace 门控模型所需的 token 与镜像地址（必须在导入 tabpfn 前调用）。"""
    if cfg is not None:
        if cfg.token:
            os.environ.setdefault("TABPFN_TOKEN", cfg.token)
        if cfg.hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", cfg.hf_endpoint)
    # token 只从环境变量读取；config 里留空时不会覆盖用户已设的值
    if not os.environ.get("TABPFN_TOKEN"):
        print("[提示] 未检测到 TABPFN_TOKEN：TabPFN v3 是 HuggingFace 门控模型，"
              "请先接受许可并设置该环境变量。", flush=True)


def setup_console() -> None:
    """让中文日志在 GBK 控制台下也能正常输出。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================================
#                              通用工具
# ============================================================================

def human(seconds: float) -> str:
    seconds = int(round(max(seconds, 0)))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def banner(title: str, cfg: Config | None = None) -> None:
    print("=" * 74)
    print(title)
    if cfg is not None:
        print("-" * 74)
        print(cfg.describe())
    print("=" * 74, flush=True)


def load_dataset(cfg: Config, warn_rows: int | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """读取数据集，做必要的列与有限性检查。"""
    if not cfg.data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {cfg.data_path}")
    df = pd.read_csv(cfg.data_path, encoding="utf-8-sig")
    missing = [c for c in INPUT_COLS + OUTPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"数据集缺少列: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    x = df[INPUT_COLS].to_numpy(dtype=float)
    y = df[OUTPUT_COLS].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("输入特征中存在非有限值，请先清洗数据。")
    if not np.isfinite(y).all():
        raise ValueError("输出目标中存在非有限值，请先清洗数据。")
    print(f"数据: {len(df)} 行, {len(df.columns)} 列 "
          f"({len(INPUT_COLS)} 输入 + {len(OUTPUT_COLS)} 输出)", flush=True)
    if warn_rows is not None and len(df) != warn_rows:
        print(f"注意: 预期 {warn_rows} 行，实际 {len(df)} 行。", flush=True)
    return df, x, y


def split_data(cfg: Config, x: np.ndarray, y: np.ndarray, seed: int | None = None):
    """按配置做一次划分，返回 (x_tr, x_te, y_tr, y_te, train_idx, test_idx)。"""
    from sklearn.model_selection import train_test_split

    seed = cfg.random_state if seed is None else seed
    idx = np.arange(len(x))
    train_idx, test_idx = train_test_split(
        idx, test_size=cfg.test_size, random_state=seed
    )
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx], train_idx, test_idx


def predict_batched(reg, x: np.ndarray, batch_size: int, floor: int = 250,
                    progress=None) -> np.ndarray:
    """分批预测；显存不足时自动折半重试。

    TabPFN 单次 predict 就是几十秒到几分钟，中间没有迭代过程可报。
    传 progress 回调可以拿到"第几批/共几批"，用来给用户一个心跳。
    回调签名：progress(批次序号, 批次总数, 已处理行数, 总行数, 本批秒数)
    """
    import torch

    parts: list[np.ndarray] = []
    size = max(int(batch_size), 1)
    total = len(x)
    n_batches = max(1, -(-total // size))
    start = 0
    index = 0
    while start < total:
        t_batch = timer()
        try:
            parts.append(reg.predict(x[start:start + size]))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if size <= floor:
                raise
            size = max(floor, size // 2)
            n_batches = max(1, -(-total // size))
            print(f"    显存不足，预测批量降为 {size}", flush=True)
            continue
        start += size
        index += 1
        if progress is not None:
            progress(index, n_batches, min(start, total), total, timer() - t_batch)
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
#                    论文柱状图统一风格（白线 / 间隔）
# ---------------------------------------------------------------------------
#: 六算法配色，取自手稿原图3（paper_revisions/0910_five_split/plot_figure3_legacy_style.py）
PAPER_BAR_COLORS = ["#4b4b7f", "#c9a3b8", "#8fbd8c", "#b9b9b9", "#d5aa72", "#7c8cbd"]


def style_paper_bars(ax, axis: str = "y") -> None:
    """柱状图统一风格：浅色网格线**画在柱子上层**，横穿柱体（手稿原图3 的做法）。

    关键在 `set_axisbelow("line")` —— 网格位于柱体(patches)之上、折线(lines)之下；
    若用默认的 True，网格会被压到柱子背后，柱子上就看不到"白线"了。
    """
    ax.set_axisbelow("line")
    ax.grid(True, axis=axis, color="#dddddd", linewidth=0.5, alpha=0.72)


def apply_cjk_fonts(cfg: Config | None = None) -> None:
    """给含中文的图设置字体（写在图里出现中文的脚本开头）。

    matplotlib 只用字体列表里第一个能解析到的字体，**不会逐字符回退**：它没有的
    字形会直接画成豆腐块 ▯。所以这里让覆盖最全的 Microsoft YaHei 排第一
    （汉字 + ² ³ − ℃ ± × 都有），SimHei 兜底汉字，DejaVu Sans 兜底数学符号。

    两个已知坑，改图时注意：
      1. 不要把 Arial 放在中文图前面（实测 Arial 优先时中文全变 ▯）；
      2. 下标 ₂ 连 YaHei 都没有，图里要写 SiO$_2$ 而不是字面量 SiO₂。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fonts = (cfg.fonts if cfg else None) or ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = list(fonts)
    plt.rcParams["axes.unicode_minus"] = False
    # $^2$ / $_2$ 这类 mathtext 用自带字体的数学符号，不受上面字体覆盖范围影响
    plt.rcParams["mathtext.fontset"] = "dejavusans"


class ResumeTable:
    """可断点续跑的结果表：按 (键) 判重，逐行落盘，出错不丢已完成结果。"""

    def __init__(self, path: Path, fields: list[str], keys: list[str]):
        self.path = Path(path)
        self.fields = fields
        self.keys = keys
        if self.path.exists():
            frame = pd.read_csv(self.path)
            before = len(frame)
            frame = frame.drop_duplicates(self.keys, keep="last").reset_index(drop=True)
            if len(frame) != before:
                print(f"注意: {self.path.name} 中有 {before - len(frame)} 行重复，已保留最后一次结果。",
                      flush=True)
            self.frame = frame
        else:
            self.frame = pd.DataFrame(columns=fields)

    def has(self, **key_values) -> bool:
        if self.frame.empty:
            return False
        mask = pd.Series(True, index=self.frame.index)
        for name, value in key_values.items():
            mask &= self.frame[name] == value
        return bool(mask.any())

    def add(self, **row) -> pd.Series:
        record = pd.Series(row)
        self.frame = pd.concat([self.frame, record.to_frame().T], ignore_index=True)
        self.flush()
        return record

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_csv(self.path, index=False, encoding="utf-8-sig")

    def reindex(self, column: str, order: list[str]) -> pd.DataFrame:
        present = [v for v in order if v in set(self.frame[column])]
        frame = self.frame.set_index(column).loc[present].reset_index()
        self.frame = frame
        self.flush()
        return frame

    def __len__(self) -> int:
        return len(self.frame)


def timer() -> float:
    return time.perf_counter()
