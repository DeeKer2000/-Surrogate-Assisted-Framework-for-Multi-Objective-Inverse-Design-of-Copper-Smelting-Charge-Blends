"""
配矿优化（NSGA-II 双目标 + FeO/SiO₂约束）
用法: conda run -n tabpfn --env PYTHONIOENCODING=utf-8 python scripts/optimization/nsga2_optimization.py

优化目标：
  1. 最小化 slag_Cu_pct（渣含铜损失）
  2. 冰铜品位接近 60 wt%（|matte_Cu - 60| → min）
约束：
  - 渣FeO/SiO₂ ∈ [1.15, 1.30]
输出：experiments/3000_fixed_temp/optimization/
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# 门控模型 token 从环境变量读取（$env:TABPFN_TOKEN / export TABPFN_TOKEN），不要写进仓库
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

# ============================================================
#                    路径配置
# ============================================================

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_PATH = PROJECT_ROOT / 'data' / '3000_fixed_temp' / 'ml_dataset.csv'
MODELS_DIR = PROJECT_ROOT / 'experiments' / '3000_fixed_temp' / 'tabpfn' / '20260622_2247' / 'saved_models'
OPT_DIR = PROJECT_ROOT / 'experiments' / '3000_fixed_temp' / 'optimization'
OUTPUT_DIR = OPT_DIR / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda'

# ============================================================
#                    矿料数据
# ============================================================

# 11种矿料的化学组成（干基，wt%）
# 数据来源：data/矿源.xlsx
# 列含义：Cu(铜), S(硫), Fe(铁), SiO2(二氧化硅), Pb(铅), Zn(锌),
#         MgO(氧化镁), Al2O3(氧化铝), CaO(氧化钙),
#         Ni(镍), Sb(锑), Bi(铋), As(砷)
ORE_COMPOSITION = pd.DataFrame({
    'name':    ['炎鑫', '汇祥永金', '嘉能可ET', '金源矿冶', '阿舍勒', '路源',
                '外矿BOZ', '外矿AKT', '和鑫', '渣精矿', '白银'],
    'Cu':      [20.57, 23.12, 16.07, 22.94, 18.99, 21.29, 21.30, 23.60, 21.60, 17.40, 17.49],
    'S':       [22.45,  8.22, 14.95, 22.23, 43.45, 24.91, 28.96, 29.93, 30.45,  6.42, 25.75],
    'Fe':      [24.28,  7.60, 13.10, 24.39, 27.10, 19.61, 25.92, 26.10, 33.18, 34.49, 21.48],
    'SiO2':    [ 9.25, 34.82, 29.01, 11.02,  2.35, 15.33,  5.63,  4.80,  3.41, 14.19, 16.75],
    'Pb':      [ 0.05,  0.01,  0.04,  0.00,  1.04,  0.85,  0.07,  0.00,  0.05,  1.10,  0.03],
    'Zn':      [ 0.27,  0.07,  0.01,  0.53,  3.73,  4.67,  0.64,  0.26,  0.17,  1.21,  0.12],
    'MgO':     [ 0.50,  1.00,  1.22,  6.36,  0.14,  2.25,  0.49,  0.23,  1.62,  0.34,  1.19],
    'Al2O3':   [ 2.17,  7.29,  5.07,  0.47,  0.70,  1.55,  2.20,  2.11,  0.43,  0.82,  4.06],
    'CaO':     [ 0.92,  8.17,  0.60,  0.23,  0.18,  0.70,  0.52,  0.44,  0.02,  0.46,  6.09],
    'Ni':      [ 0.01,  0.02,  0.02,  0.77,  0.04,  0.00,  0.02,  0.01,  0.95,  0.01,  0.01],
    'Sb':      [ 0.10,  0.42,  0.18,  0.08,  0.05,  0.10,  0.03,  0.05,  0.05,  0.01,  0.04],
    'Bi':      [ 0.01,  0.02,  0.01,  0.01,  0.05,  0.01,  0.01,  0.01,  0.01,  0.01,  0.02],
    'As':      [ 0.01,  0.01,  0.01,  0.01,  1.45,  0.00,  0.01,  0.01,  0.01,  0.03,  0.03],
})

N_ORES = len(ORE_COMPOSITION)

# 模型输入列（与 train_tabpfn.py 一致）
# 含义：T(温度℃), P(压力atm), Cu/S/Fe/SiO2/Pb/Zn/MgO/Al2O3/Ni/Sb/Bi/CaO/As(各元素质量g), O2(氧气g)
INPUT_COLS = [
    'T(℃)', 'P(atm)', 'Cu(g)', 'S(g)', 'Fe(g)', 'SiO2(g)',
    'Pb(g)', 'Zn(g)', 'MgO(g)', 'Al2O3(g)', 'Ni(g)', 'Sb(g)',
    'Bi(g)', 'CaO(g)', 'As(g)', 'O2(g)',
]

# 从模型输入列派生质量特征顺序，避免矿源表列顺序与模型输入顺序不一致。
MASS_INPUT_COMPONENTS = [col.removesuffix('(g)') for col in INPUT_COLS[2:-1]]

# 固定工艺参数（所有优化方案共用）
T_DEFAULT = 1200.0   # 熔炼温度，单位：℃
P_DEFAULT = 1.0      # 炉内压力，单位：atm（标准大气压）
O2_DEFAULT = 21.0    # 氧气供给量，单位：g（对应空气气氛）

# 总投料量（克），用于将 wt% 转换为克
TOTAL_CHARGE = 100.0

# ============================================================
#                    优化参数
# ============================================================

POP_SIZE = 100       # 种群大小（每代个体数）
N_GEN = 200          # 最大迭代代数（遗传算法进化轮数上限）
SEED = 42            # 随机种子（保证结果可复现）
PATIENCE = 30        # 早停止耐心值：连续 PATIENCE 代无改善则提前终止

# ---- 目标函数（2个目标）+ 1个约束 ----
# f1：最小化渣含铜（slag_Cu_pct）—— 渣含铜越低，铜的损失越小，经济效益越高
# f2：冰铜品位接近60 wt%（|matte_Cu - 60|）—— 控制品位在理想值附近
# 约束：渣FeO/SiO₂ ∈ [1.15, 1.30] —— 控制渣的流动性和铜渣分离效率

# Fe/SiO₂ 最优范围：渣中FeO与SiO₂的质量比，控制渣的流动性和铜渣分离效率
FESIO2_LO = 1.15     # Fe/SiO₂ 下限（低于此值渣过酸，腐蚀炉衬）
FESIO2_HI = 1.3      # Fe/SiO₂ 上限（高于此值渣过碱，黏度增大）

# 冰铜品位目标值
MATTE_CU_TARGET = 60.0  # 冰铜品位目标（wt%）

# ============================================================
#                    代码
# ============================================================

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from pymoo.termination import get_termination


def ratios_to_features(ratios, sio2_flux_ratio):
    """将矿料配比和SiO₂熔剂比例转换为模型输入特征向量（16维）。

    转换流程：
      1. 用配比对11种矿料的化学组成加权求和，得到混合炉料综合组成（wt%）
      2. 加入SiO₂熔剂，重新归一化
      3. 将wt%转换为克（总投料量100g）
      4. 补充固定工艺参数（温度、压力、氧气），构成16维向量

    Args:
        ratios: shape (11,) 各矿料质量配比，总和为1
        sio2_flux_ratio: SiO₂熔剂添加比例（0~0.15），即每1g矿料额外添加的SiO₂克数

    Returns:
        features: shape (16,) 模型输入特征，顺序与 INPUT_COLS 一致
    """
    # 严格按模型训练时的输入顺序提取13种质量特征。
    ore_comp = ORE_COMPOSITION[MASS_INPUT_COMPONENTS].values

    # 步骤1：混合炉料综合组成（wt%），按配比加权平均
    blend_pct = ratios @ ore_comp  # shape (13,)

    # 步骤2：添加 SiO₂ 熔剂（加入到SiO2列，index=3），然后重新归一化
    total_ratio = 1.0 + sio2_flux_ratio
    blend_pct_with_flux = blend_pct.copy()
    sio2_idx = MASS_INPUT_COMPONENTS.index('SiO2')
    blend_pct_with_flux[sio2_idx] += sio2_flux_ratio * 100  # SiO2列额外加上熔剂质量
    blend_pct_with_flux = blend_pct_with_flux / total_ratio  # 归一化（总质量增加）

    # 步骤3：将wt%转换为克（总投料量100g）
    charge_g = blend_pct_with_flux * TOTAL_CHARGE / 100.0

    # 步骤4：构造16维输入向量 = [温度, 压力, 13种元素质量(g), 氧气量(g)]
    features = np.array([
        T_DEFAULT, P_DEFAULT,
        charge_g[0],  # Cu（铜）质量，g
        charge_g[1],  # S（硫）质量，g
        charge_g[2],  # Fe（铁）质量，g
        charge_g[3],  # SiO2（二氧化硅）质量，g
        charge_g[4],  # Pb（铅）质量，g
        charge_g[5],  # Zn（锌）质量，g
        charge_g[6],  # MgO（氧化镁）质量，g
        charge_g[7],  # Al2O3（氧化铝）质量，g
        charge_g[8],  # Ni（镍）质量，g
        charge_g[9],  # Sb（锑）质量，g
        charge_g[10], # Bi（铋）质量，g
        charge_g[11], # CaO（氧化钙）质量，g
        charge_g[12], # As（砷）质量，g
        O2_DEFAULT,   # O2（氧气）质量，g
    ], dtype=np.float32)

    return features


def check_metallurgical_constraints(ratios, sio2_flux_ratio):
    """检查冶金成分约束（罚函数前置过滤）。

    这些约束确保混合炉料的化学组成在FactSage训练数据的覆盖范围内，
    避免模型外推导致预测不可靠。

    Returns:
        True: 满足所有约束，方案可行
        False: 违反任一约束，方案不可行
    """
    ore_comp = ORE_COMPOSITION[['Cu', 'S', 'Fe', 'SiO2']].values
    blend_pct = ratios @ ore_comp

    total_ratio = 1.0 + sio2_flux_ratio
    blend_pct[3] += sio2_flux_ratio * 100
    blend_pct = blend_pct / total_ratio

    cu, s, fe, sio2 = blend_pct

    # 混合炉料成分约束（wt%范围，与FactSage训练数据覆盖范围一致）
    if not (14 <= cu <= 26):       # Cu（铜）含量
        return False
    if not (8 <= s <= 38):         # S（硫）含量
        return False
    if not (8 <= fe <= 36):        # Fe（铁）含量
        return False
    if not (3 <= sio2 <= 40):      # SiO2（二氧化硅）含量，SiO₂辅料可灵活添加
        return False

    # Fe/SiO₂ 比约束（宽松范围，用于前置过滤）
    if sio2 > 0:
        fesio2 = fe / sio2
        if not (0.8 <= fesio2 <= 4.0):
            return False

    return True


class OreBlendingProblem(Problem):
    """配矿优化问题（2目标 + FeO/SiO₂约束）。

    决策变量（12维）：
      - x[0:11] = 11种矿料的质量配比（归一化后总和为1）
      - x[11]   = SiO₂熔剂添加比例（0~0.15）

    目标函数（2个）：
      - f1: min slag_Cu_pct（渣含铜含量，wt%）—— 铜损失越小越好
      - f2: min |matte_Cu_pct - 60|（冰铜品位偏离目标60 wt%的程度）

    不等式约束（2个）：
      - g1: slag_FeO/SiO₂ ≥ 1.15（渣不能过酸）
      - g2: slag_FeO/SiO₂ ≤ 1.30（渣不能过碱）

    冶金成分约束（Cu/S/Fe/SiO₂范围）通过罚函数在check_metallurgical_constraints中处理。
    """

    def __init__(self, models):
        self.models = models  # dict: target_name -> model（TabPFN代理模型字典）
        n_var = N_ORES + 1  # 12维：11种矿料配比 + 1个SiO₂熔剂比例
        n_obj = 2            # 2个目标
        n_ieq_constr = 2     # 2个不等式约束：FeO/SiO₂下限和上限

        # 决策变量边界
        xl = np.zeros(n_var)       # 下界：所有配比 ≥ 0
        xu = np.ones(n_var)        # 上界：配比 ≤ 1
        xu[-1] = 0.15              # SiO₂熔剂比例上界（最多加15%）

        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr,
                         xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        n = X.shape[0]
        # F: 目标函数矩阵，shape (n, 2)
        #   F[i, 0] = slag_Cu_pct（渣含铜），越小越好
        #   F[i, 1] = |matte_Cu_pct - 60|（冰铜品位偏离目标），越接近0越好
        # G: 不等式约束矩阵，shape (n, 2)，>=0 表示满足
        #   G[i, 0] = slag_FeO_SiO2 - 1.15  （FeO/SiO₂ ≥ 下限）
        #   G[i, 1] = 1.3 - slag_FeO_SiO2    （FeO/SiO₂ ≤ 上限）
        F = np.full((n, 2), 1e6)
        G = np.full((n, 2), -1.0)

        # 收集有效个体的索引和特征，批量预测
        valid_indices = []
        valid_features = []

        for i in range(n):
            ore_ratios_raw = X[i, :N_ORES]
            sio2_flux = X[i, -1]

            total = ore_ratios_raw.sum()
            if total < 1e-10:
                continue
            ore_ratios = ore_ratios_raw / total

            if not check_metallurgical_constraints(ore_ratios, sio2_flux):
                F[i] = [10.0, 10.0]
                G[i] = [1.0, 1.0]
                continue

            features = ratios_to_features(ore_ratios, sio2_flux)
            valid_indices.append(i)
            valid_features.append(features)

        # 批量预测
        if valid_features:
            X_batch = np.array(valid_features, dtype=np.float32)
            try:
                pred_slag_cu = np.array(
                    self.models['slag_Cu_pct'].predict(X_batch)).flatten()
                pred_matte_cu = np.array(
                    self.models['matte_Cu_pct'].predict(X_batch)).flatten()
                pred_fesio2 = np.array(
                    self.models['slag_FeO_SiO2_ratio'].predict(X_batch)).flatten()

                for idx, i in enumerate(valid_indices):
                    F[i] = [float(pred_slag_cu[idx]),                                    # f1: min slag_Cu
                            abs(float(pred_matte_cu[idx]) - MATTE_CU_TARGET)]            # f2: target 60% matte grade
                    G[i] = [FESIO2_LO - float(pred_fesio2[idx]),                         # g1: FeO/SiO2 >= lower  (g1 <= 0 means feasible)
                            float(pred_fesio2[idx]) - FESIO2_HI]                         # g2: FeO/SiO2 <= upper  (g2 <= 0 means feasible)
            
            except Exception as e:
                print(f"\n  [DEBUG] Prediction ERROR: {e}")
                for i in valid_indices:
                    F[i] = [10.0, 10.0]
                    G[i] = [1.0, 1.0]

        out["F"] = F
        out["G"] = G


def load_models():
    """加载 TabPFN 模型。

    load_fitted_tabpfn_model 会从 init_params.json 读取 model_path（相对路径），
    需要将其替换为绝对路径后重新打包，避免 CWD 依赖。
    """
    import tempfile, zipfile, json, joblib
    from tabpfn import TabPFNRegressor
    from tabpfn.model_loading import _extract_archive, InferenceEngine

    # 预训练权重：默认 'auto'（由 tabpfn 自动下载/走缓存），
    # 也可用环境变量 TABPFN_MODEL_PATH 指向本地 ckpt
    MODEL_PATH_ABS = os.environ.get('TABPFN_MODEL_PATH', 'auto')

    model_names = ['slag_Cu_pct', 'matte_Cu_pct', 'slag_FeO_SiO2_ratio']
    models = {}
    for name in model_names:
        path = MODELS_DIR / f'{name}.tabpfn_fit'
        print(f"  Loading model: {name}...")

        # 手动加载，替换 model_path 为绝对路径
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            _extract_archive(path, tmp)

            with (tmp / "init_params.json").open() as f:
                params = json.load(f)

            params.pop("__class_name__")
            # 关键：将 model_path 替换为绝对路径
            params["model_path"] = MODEL_PATH_ABS
            params["device"] = DEVICE

            est = TabPFNRegressor(**params)
            est._initialize_model_variables()

            fitted_attrs = joblib.load(tmp / "fitted_attrs.joblib")
            for key, value in fitted_attrs.items():
                setattr(est, key, value)

            est.executor_ = InferenceEngine.load_state(
                tmp / "executor_state.joblib", est.models_
            )
            est.to(DEVICE)

        models[name] = est
        print(f"    OK")

    return models


class EarlyStopping:
    """早停止回调：连续 PATIENCE 代无改善则终止优化。"""

    def __init__(self, patience=PATIENCE):
        self.patience = patience
        self.best = float('inf')
        self.counter = 0

    def __call__(self, algorithm):
        # 获取当前最优目标函数值
        F = algorithm.pop.get("F")
        current_best = float(np.min(F))
        if current_best < self.best - 1e-6:
            self.best = current_best
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            print(f"\n  Early stopping: no improvement for {self.patience} generations")
            return True
        return False


def run_optimization(models):
    """运行优化（NSGA-II 双目标优化）。"""
    problem = OreBlendingProblem(models)

    # 默认使用 NSGA-II（多目标优化）
    # 如需切换为单目标 GA，改为：algorithm = GA(...)
    algorithm = NSGA2(
        pop_size=POP_SIZE,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", N_GEN)

    print(f"\n  Population: {POP_SIZE}")
    print(f"  Max generations: {N_GEN}, early stopping patience: {PATIENCE}")
    print(f"  Decision vars: {N_ORES} ore ratios + SiO2 flux = {N_ORES+1} vars")
    print(f"  Objectives (2):")
    print(f"    f1 = min(slag_Cu_pct)")
    print(f"    f2 = min(|matte_Cu - {MATTE_CU_TARGET}|)")
    print(f"  Constraint: FeO/SiO2 in [{FESIO2_LO}, {FESIO2_HI}]")
    print(f"\n  Starting optimization...")

    res = minimize(
        problem,
        algorithm,
        termination,
        callback=EarlyStopping(patience=PATIENCE),
        seed=SEED,
        verbose=True,
    )

    return res


def extract_results(res, models):
    """提取 Pareto 最优解集，并重新预测 matte_Cu / FeO/SiO₂ 用于展示。"""
    X_opt = np.atleast_2d(res.X)  # (n_solutions, n_var)
    F_opt = np.atleast_2d(res.F)  # (n_solutions, 2)

    # DEBUG: print F_opt stats
    print("  [DEBUG] res.F shape:", F_opt.shape)
    slag_cu_f = F_opt[:, 0]
    print("  [DEBUG] slag_Cu(F): min=%.4f max=%.4f mean=%.4f" % (
        float(np.min(slag_cu_f)), float(np.max(slag_cu_f)), float(np.mean(slag_cu_f))))
    n_pen = int(np.sum(slag_cu_f > 5))
    print("  [DEBUG] Penalty (F>5): %d/%d" % (n_pen, F_opt.shape[0]))
    if hasattr(res, "G") and res.G is not None:
        G_opt = np.atleast_2d(res.G)
        # pymoo defines constraints as G <= 0.  Using >= 0 here made the
        # diagnostic report every feasible solution as infeasible.
        n_feas = int(np.sum((G_opt[:, 0] <= 0) & (G_opt[:, 1] <= 0)))
        print("  [DEBUG] G feasible: %d/%d" % (n_feas, G_opt.shape[0]))


    # 批量预测 matte_Cu 和 Fe/SiO₂ 用于展示
    valid_indices = []
    valid_features = []
    for i in range(X_opt.shape[0]):
        ore_ratios_raw = X_opt[i, :N_ORES]
        sio2_flux = X_opt[i, -1]
        total = ore_ratios_raw.sum()
        if total < 1e-10:
            continue
        ore_ratios = ore_ratios_raw / total
        valid_indices.append(i)
        valid_features.append(ratios_to_features(ore_ratios, sio2_flux))

    pred_matte_cu = {}
    pred_fesio2 = {}
    if valid_features:
        X_batch = np.array(valid_features, dtype=np.float32)
        try:
            mc = np.array(models['matte_Cu_pct'].predict(X_batch)).flatten()
            fs = np.array(models['slag_FeO_SiO2_ratio'].predict(X_batch)).flatten()
            for idx, i in enumerate(valid_indices):
                pred_matte_cu[i] = float(mc[idx])
                pred_fesio2[i] = float(fs[idx])
        except Exception:
            pass

    results = []
    for i in range(X_opt.shape[0]):
        ore_ratios_raw = X_opt[i, :N_ORES]
        sio2_flux = X_opt[i, -1]
        total = ore_ratios_raw.sum()
        if total < 1e-10:
            continue
        ore_ratios = ore_ratios_raw / total

        slag_cu = float(F_opt[i, 0])
        matte_cu_deviation = float(F_opt[i, 1])
        matte_cu = pred_matte_cu.get(i, 0.0)
        fesio2_pred = pred_fesio2.get(i, 0.0)

        # 生成与模型及 FactSage 验证完全一致的 16 维最终输入。
        # TOTAL_CHARGE=100 g，因此13项质量特征的数值也等于最终入炉料中的 wt%。
        features = ratios_to_features(ore_ratios, sio2_flux)
        input_values = dict(zip(INPUT_COLS, features))
        ore_charge_g = TOTAL_CHARGE / (1.0 + sio2_flux)
        pure_sio2_flux_g = TOTAL_CHARGE * sio2_flux / (1.0 + sio2_flux)

        entry = {
            'solution_idx': i,
            'slag_Cu_pct': slag_cu,
            'matte_Cu_pct': matte_cu,
            'obj_matte_deviation': matte_cu_deviation,
            'slag_FeO_SiO2_ratio': fesio2_pred,
            'SiO2_flux': sio2_flux,
            'ore_charge_g': ore_charge_g,
            'pure_SiO2_flux_g': pure_sio2_flux_g,
        }

        # 直接保存温度、压力、13项组分质量及氧气量，便于逐方案输入FactSage复核。
        entry.update({column: float(input_values[column]) for column in INPUT_COLS})
        entry.update({
            'blend_Cu': float(input_values['Cu(g)']),
            'blend_S': float(input_values['S(g)']),
            'blend_Fe': float(input_values['Fe(g)']),
            'blend_SiO2': float(input_values['SiO2(g)']),
        })

        # 各矿料配比
        for j, name in enumerate(ORE_COMPOSITION['name']):
            entry[f'ratio_{name}'] = ore_ratios[j]

        results.append(entry)

    return pd.DataFrame(results)


def select_representative_solutions(df, n=5):
    """从满足 FeO/SiO₂ 约束的 Pareto 解中选取 n 个代表性方案。"""
    # 过滤出满足 FeO/SiO₂ 约束的 Pareto 解
    mask = ((df['slag_FeO_SiO2_ratio'] >= FESIO2_LO) &
            (df['slag_FeO_SiO2_ratio'] <= FESIO2_HI))
    df_feasible = df[mask].copy()

    if len(df_feasible) == 0:
        print("  Warning: no metallurgically feasible solutions, using all")
        df_feasible = df

    print(f"  Feasible (Fe/SiO2+matte_Cu feasible): {len(df_feasible)}/{len(df)}")

    if len(df_feasible) <= n:
        return df_feasible

    # 按 slag_Cu 排序，均匀选取
    df_sorted = df_feasible.sort_values('slag_Cu_pct').reset_index(drop=True)
    indices = np.linspace(0, len(df_sorted) - 1, n, dtype=int)
    return df_sorted.iloc[indices].reset_index(drop=True)


def generate_comparison(df_rep, output_dir=None):
    """生成优化方案 vs 人工方案对比（TabPFN预测的年度平均人工配料结果）。"""
    manual_path = OPT_DIR / 'manual_blend_prediction.csv'
    if manual_path.exists():
        manual_df = pd.read_csv(manual_path)
        manual = {
            'slag_Cu_pct': manual_df['slag_Cu_pct'].iloc[0],
            'matte_Cu_pct': manual_df['matte_Cu_pct'].iloc[0],
            'slag_FeO_SiO2_ratio': manual_df['slag_FeO_SiO2_ratio'].iloc[0],
        }
    else:
        manual = {
            'slag_Cu_pct': 0.65,
            'matte_Cu_pct': 58.0,
            'slag_FeO_SiO2_ratio': 1.35,
        }

    # 使用代表方案列表中间的实际方案（方案 C）进行对比。
    # 分别按列取中位数可能拼出一个并不存在的“组合方案”，从而使图表中的
    # FeO/SiO2 与表中推荐配方不一致。
    representative = df_rep.iloc[len(df_rep) // 2]
    opt = {
        'slag_Cu_pct': representative['slag_Cu_pct'],
        'matte_Cu_pct': representative['matte_Cu_pct'],
        'slag_FeO_SiO2_ratio': representative['slag_FeO_SiO2_ratio'],
    }

    return manual, opt


def plot_pareto_front(df_all, df_rep, output_dir):
    """生成带渣 FeO/SiO₂ 色条、目标带和推荐方案的 Pareto 前沿图。"""
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 9,
        'axes.linewidth': 0.9,
        'axes.unicode_minus': False,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'pdf.fonttype': 42,
        'svg.fonttype': 'none',
    })

    feasible = df_all[
        df_all['slag_FeO_SiO2_ratio'].between(
            FESIO2_LO, FESIO2_HI, inclusive='both'
        )
    ].copy()
    if feasible.empty:
        raise ValueError('No feasible Pareto points within the FeO/SiO2 range.')

    fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)

    # 与旧版论文图一致：用色条表达渣型，并突出冰铜品位目标区。
    target_band = 0.10
    ax.axhspan(MATTE_CU_TARGET - target_band,
               MATTE_CU_TARGET + target_band,
               color='#D4544A', alpha=0.08, zorder=0)
    ax.axhline(MATTE_CU_TARGET, color='#D4544A', linestyle='--',
               linewidth=1.0, alpha=0.9,
               label=f'Matte Cu target = {MATTE_CU_TARGET:.0f} wt%')

    scatter = ax.scatter(
        feasible['slag_Cu_pct'],
        feasible['matte_Cu_pct'],
        c=feasible['slag_FeO_SiO2_ratio'],
        cmap='viridis',
        s=42,
        alpha=0.92,
        edgecolors='white',
        linewidths=0.45,
        rasterized=True,
        zorder=2,
    )

    # 代表性方案标注
    labels = ['A', 'B', 'C', 'D', 'E']
    annotation_offsets = [(8, 8), (9, 8), (8, 7), (7, -18), (8, 8)]
    for idx, (_, row) in enumerate(df_rep.iterrows()):
        if idx >= len(labels):
            break
        ax.annotate(labels[idx],
                    xy=(row['slag_Cu_pct'], row['matte_Cu_pct']),
                    xytext=annotation_offsets[idx], textcoords='offset points',
                    fontsize=8.5, fontweight='bold', color='#D4544A',
                    zorder=5)

    recommended = df_rep.iloc[len(df_rep) // 2]
    ax.scatter(recommended['slag_Cu_pct'], recommended['matte_Cu_pct'],
               marker='*', s=270, facecolor='#E4584D', edgecolor='black',
               linewidth=0.8, label='Recommended solution', zorder=4)

    x_span = feasible['slag_Cu_pct'].max() - feasible['slag_Cu_pct'].min()
    ax.set_xlim(feasible['slag_Cu_pct'].min() - 0.05 * x_span,
                feasible['slag_Cu_pct'].max() + 0.05 * x_span)
    ax.set_ylim(min(MATTE_CU_TARGET - target_band,
                    feasible['matte_Cu_pct'].min()) - 0.02,
                max(MATTE_CU_TARGET + target_band,
                    feasible['matte_Cu_pct'].max()) + 0.02)
    ax.set_xlabel('Slag Cu / wt%', fontsize=10)
    ax.set_ylabel('Matte Cu / wt%', fontsize=10)
    ax.set_title('NSGA-II Pareto Front for Intelligent Blending',
                 fontsize=11.5)
    ax.grid(True, color='#DDDDDD', linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, loc='lower right', frameon=False)

    colorbar = fig.colorbar(scatter, ax=ax, pad=0.03)
    colorbar.set_label('Slag FeO/SiO$_2$', fontsize=9.5)

    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f'fig_pareto_front.{ext}', dpi=300,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  → fig_pareto_front.png/pdf")


def plot_comparison_bar(df_rep, manual, opt, output_dir):
    """生成带目标线和目标区间的人工方案—优化方案三联对比图。"""
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 9,
        'axes.linewidth': 0.9,
        'axes.unicode_minus': False,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'pdf.fonttype': 42,
        'svg.fonttype': 'none',
    })

    metrics = ['slag_Cu_pct', 'matte_Cu_pct', 'slag_FeO_SiO2_ratio']
    labels = ['Slag Cu / wt%', 'Matte Cu / wt%', 'Slag FeO/SiO2']

    fig, axes = plt.subplots(
        1, 3, figsize=(8.2, 3.05), constrained_layout=True
    )
    bar_colors = ['#AAB8DC', '#4B4B7F']

    for ax, metric, label in zip(axes, metrics, labels):
        v_manual = manual[metric]
        v_opt = opt[metric]

        bars = ax.bar([0, 1], [v_manual, v_opt],
                      color=bar_colors, edgecolor='#222222',
                      linewidth=0.6, width=0.54, zorder=2)

        y_ranges = {
            'slag_Cu_pct': (0, 0.62),
            'matte_Cu_pct': (0, 65),
            'slag_FeO_SiO2_ratio': (0, 1.75),
        }
        y_lo, y_hi = y_ranges[metric]
        label_pad = (y_hi - y_lo) * 0.015
        for bar, val in zip(bars, [v_manual, v_opt]):
            ax.text(bar.get_x() + bar.get_width()/2, val + label_pad,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7.6)

        if metric == 'slag_Cu_pct':
            improvement = (v_manual - v_opt) / v_manual * 100
            ax.text(0.5, 0.94, f'{improvement:.1f}% reduction',
                    transform=ax.transAxes, ha='center', va='top', fontsize=8.2,
                    color='#D4544A', fontweight='bold')
        elif metric == 'matte_Cu_pct':
            ax.axhline(MATTE_CU_TARGET, color='#D4544A', linestyle='--',
                       linewidth=1.0, alpha=0.9, zorder=1)
            ax.text(0.5, 0.985, f'Target = {MATTE_CU_TARGET:.0f} wt%',
                    transform=ax.transAxes, ha='center', va='top', fontsize=8.2,
                    color='#D4544A', fontweight='bold')
        else:
            ax.axhspan(FESIO2_LO, FESIO2_HI,
                       color='#7BAA72', alpha=0.13, zorder=0)
            ax.text(0.5, 0.94, 'Target range',
                    transform=ax.transAxes, ha='center', va='top', fontsize=8.2,
                    color='#3D7435', fontweight='bold')

        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Manual\nblend', 'Optimized\nblend'], fontsize=7.4)
        ax.set_ylabel(label, fontsize=9.5)
        ax.grid(axis='y', color='#DDDDDD', linewidth=0.5, alpha=0.72)
        ax.set_axisbelow(True)

    fig.suptitle('Manual and Optimized Blending: Key Metallurgical Targets',
                 fontsize=11.5)
    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f'fig_optimization_comparison.{ext}', dpi=300,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  → fig_optimization_comparison.png/pdf")


# ============================================================
#                    主流程
# ============================================================

def main():
    print("=" * 50)
    print("NSGA-II Ore Blending Optimization (2 objectives + 1 constraint)")
    print(f"  f1: min(slag_Cu_pct)")
    print(f"  f2: min(|matte_Cu - {MATTE_CU_TARGET}|)")
    print(f"  Constraint: FeO/SiO2 in [{FESIO2_LO}, {FESIO2_HI}]")
    print("=" * 50)

    # 1. 加载模型
    print("\n[1/5] Loading TabPFN models...")
    models = load_models()

    # 2. 运行优化
    print("\n[2/5] Running NSGA-II optimization...")
    res = run_optimization(models)

    # 3. 提取结果
    print("\n[3/5] Extracting results...")
    df_all = extract_results(res, models)
    print(f"  Solutions: {len(df_all)}")

    if len(df_all) == 0:
        print("  Error: no feasible solutions found!")
        return

    # 主结果文件保存完整 Pareto 解集。旧文件名继续写入兼容副本，
    # 避免已有分析流程立即失效。
    all_results_path = OPT_DIR / 'optimization_results.csv'
    df_all.to_csv(all_results_path, index=False, encoding='utf-8-sig')
    print(f"  Full results saved: {all_results_path}")
    legacy_all_path = OPT_DIR / 'optimization_pareto_all.csv'
    df_all.to_csv(legacy_all_path, index=False, encoding='utf-8-sig')
    print(f"  Compatibility copy saved: {legacy_all_path}")

    # 选取代表性方案
    df_rep = select_representative_solutions(df_all, n=5)
    rep_path = OPT_DIR / 'optimization_representative_solutions.csv'
    df_rep.to_csv(rep_path, index=False, encoding='utf-8-sig')
    print(f"  Representative solutions saved: {rep_path}")

    # 打印代表性方案
    print("\n  Representative solutions:")
    print("  " + "-" * 80)
    labels = ['A', 'B', 'C', 'D', 'E']
    for idx, (_, row) in enumerate(df_rep.iterrows()):
        label = labels[idx] if idx < len(labels) else str(idx)
        print(f"  {label}: slag_Cu={row['slag_Cu_pct']:.4f} wt%, "
              f"matte_Cu={row['matte_Cu_pct']:.2f} wt%, "
              f"Fe/SiO2={row['slag_FeO_SiO2_ratio']:.3f}")

    # 4. 生成对比数据
    print("\n[4/5] Generating comparison...")
    manual, opt = generate_comparison(df_rep)
    print(f"  Manual: slag_Cu={manual['slag_Cu_pct']}, "
          f"matte_Cu={manual['matte_Cu_pct']}, Fe/SiO2={manual['slag_FeO_SiO2_ratio']}")
    print(f"  Optimized: slag_Cu={opt['slag_Cu_pct']:.4f}, "
          f"matte_Cu={opt['matte_Cu_pct']:.2f}, Fe/SiO2={opt['slag_FeO_SiO2_ratio']:.3f}")

    # 5. 生成图表
    print("\n[5/5] Generating figures...")
    plot_pareto_front(df_all, df_rep, OUTPUT_DIR)
    plot_comparison_bar(df_rep, manual, opt, OUTPUT_DIR)

    # 汇总
    print("\n" + "=" * 50)
    slag_cu_reduction = (manual['slag_Cu_pct'] - opt['slag_Cu_pct']) / manual['slag_Cu_pct'] * 100
    print(f"Manual slag_Cu: {manual['slag_Cu_pct']:.3f} wt% → Optimized: {opt['slag_Cu_pct']:.3f} wt% (reduction {slag_cu_reduction:.1f}%)")
    print(f"Manual matte_Cu: {manual['matte_Cu_pct']:.2f} wt% → Optimized: {opt['matte_Cu_pct']:.2f} wt% (target 60 wt%)")
    print(f"Manual Fe/SiO2: {manual['slag_FeO_SiO2_ratio']:.3f} → Optimized: {opt['slag_FeO_SiO2_ratio']:.3f} (target [{FESIO2_LO}, {FESIO2_HI}])")
    print("=" * 50)


if __name__ == '__main__':
    main()
