"""补全优化结果中的13项组分及FactSage验证输入。

主结果 ``optimization_results.csv`` 保存全部Pareto方案；本脚本根据每个
方案的11种矿料配比和SiO2熔剂比例，补充最终100 g入炉料的完整输入质量。
5个代表方案另外保存在 ``optimization_representative_solutions.csv``。
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

import os
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OPT_DIR = PROJECT_ROOT / 'experiments' / '3000_fixed_temp' / 'optimization'

# ---- 11种矿料的化学组成（与 nsga2_optimization.py 一致）----
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

ELEMENT_COLS = ['Cu', 'S', 'Fe', 'SiO2', 'Pb', 'Zn', 'MgO', 'Al2O3',
                'CaO', 'Ni', 'Sb', 'Bi', 'As']
INPUT_COMPONENT_ORDER = [
    'Cu', 'S', 'Fe', 'SiO2', 'Pb', 'Zn', 'MgO', 'Al2O3',
    'Ni', 'Sb', 'Bi', 'CaO', 'As',
]
TOTAL_CHARGE = 100.0
T_DEFAULT = 1200.0
P_DEFAULT = 1.0
O2_DEFAULT = 21.0


def calc_blend(ratios_dict, sio2_flux):
    """根据矿料配比和SiO2熔剂量计算完整混合矿成分 (wt%)。"""
    ratios = np.array([ratios_dict[name] for name in ORE_COMPOSITION['name']])
    ore_comp = ORE_COMPOSITION[ELEMENT_COLS].values  # (11, 13)

    blend_pct = ratios @ ore_comp  # (13,)

    # 加入SiO2熔剂并归一化
    total_ratio = 1.0 + sio2_flux
    blend_pct[3] += sio2_flux * 100  # SiO2列（index=3）
    blend_pct = blend_pct / total_ratio

    return dict(zip(ELEMENT_COLS, blend_pct))


def enrich_results(df):
    """将矿料配比换算为最终100 g入炉料和16维模型输入。"""
    rows = []
    for _, row in df.iterrows():
        ratios = {name: row[f'ratio_{name}'] for name in ORE_COMPOSITION['name']}
        sio2_flux = float(row['SiO2_flux'])
        blend = calc_blend(ratios, sio2_flux)

        entry = row.to_dict()
        entry['ore_charge_g'] = TOTAL_CHARGE / (1.0 + sio2_flux)
        entry['pure_SiO2_flux_g'] = TOTAL_CHARGE * sio2_flux / (1.0 + sio2_flux)
        entry['T(℃)'] = T_DEFAULT
        entry['P(atm)'] = P_DEFAULT
        for component in INPUT_COMPONENT_ORDER:
            entry[f'{component}(g)'] = blend[component] * TOTAL_CHARGE / 100.0
        entry['O2(g)'] = O2_DEFAULT
        entry.update({f'blend_{component}': value for component, value in blend.items()})
        rows.append(entry)
    return pd.DataFrame(rows)


def main():
    # 补全63个Pareto方案，并同步旧文件名兼容副本。
    all_path = OPT_DIR / 'optimization_results.csv'
    all_results = enrich_results(pd.read_csv(all_path, encoding='utf-8-sig'))
    all_results.to_csv(all_path, index=False, encoding='utf-8-sig')

    # 补全5个代表方案。
    representative_path = OPT_DIR / 'optimization_representative_solutions.csv'
    representative = enrich_results(
        pd.read_csv(representative_path, encoding='utf-8-sig')
    )
    representative.to_csv(representative_path, index=False, encoding='utf-8-sig')

    # 另存便于论文表格使用的代表方案完整成分表。
    result_columns = [
        'solution_idx', 'slag_Cu_pct', 'matte_Cu_pct',
        'slag_FeO_SiO2_ratio', 'ore_charge_g', 'pure_SiO2_flux_g',
        *[f'blend_{component}' for component in ELEMENT_COLS],
    ]
    result = representative[result_columns].copy()
    out_path = OPT_DIR / 'blend_full_composition.csv'
    result.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"已补全全部Pareto方案: {all_path} ({len(all_results)}行)")
    print(f"已补全代表方案: {representative_path} ({len(representative)}行)")
    print(f"已保存代表方案完整成分: {out_path}\n")

    # 旧文件名仅作为兼容副本；被Excel等程序占用时不影响主结果生成。
    legacy_path = OPT_DIR / 'optimization_pareto_all.csv'
    try:
        all_results.to_csv(legacy_path, index=False, encoding='utf-8-sig')
        print(f"已同步兼容副本: {legacy_path}")
    except PermissionError:
        print(f"警告: {legacy_path} 正被占用，已跳过兼容副本同步")

    # 打印
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print("=" * 100)
    print("各优化方案的混合矿完整元素成分 (wt%)")
    print("=" * 100)
    print(result.to_string(index=False))

    # 同时打印人工配矿的年度平均成分作为对比
    print("\n" + "=" * 100)
    print("对比：人工配矿年度平均成分 (wt%) — 来自 data/人工配矿.xlsx")
    print("=" * 100)
    manual = {'Cu': 20.42, 'S': 28.50, 'Fe': 27.15, 'SiO2': 12.15,
              'Pb': 0.19, 'Zn': 1.04, 'MgO': 0.95, 'Al2O3': 3.33,
              'CaO': 1.10, 'Ni': 0.041, 'Sb': 0.024, 'Bi': 0.01, 'As': 0.53}
    for k, v in manual.items():
        print(f"  {k:>6s}: {v:.4f}")


if __name__ == '__main__':
    main()
