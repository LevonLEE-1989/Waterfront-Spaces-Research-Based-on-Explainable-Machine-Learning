# -*- coding: utf-8 -*-
"""
郑州东风渠滨水感知研究 —— 多感知维度随机森林 PDP 分析
作者：你
功能：对 Scenic Beauty / Safety / Recreational Value 分别建模，并绘制指定特征的 PDP 图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence

# 设置全局字体
plt.rcParams['font.family'] = 'Times New Roman'

# ----------------------------
# 读取原始数据
# ----------------------------
print("正在读取 perceptions.csv 和 features.csv...")
df_percep_raw = pd.read_csv('perceptions.csv')
df_feat_raw = pd.read_csv('features.csv')

# 目标感知变量（英文列名）
target_vars = ['Scenic Beauty', 'Safety', 'Recreational Value']

# 要绘制 PDP 的视觉特征（必须与 features.csv 列名一致！）
features_to_plot = [
    'Color Diversity', 'Color Uniformity', 'Openness',
    'Waterfront Accessibility', 'Walkability', 'Spatial Definition',
    'Greenness', 'Plant Diversity'
]

# 自动筛选 features.csv 中存在的数值特征
numeric_feat_cols = df_feat_raw.select_dtypes(include=[np.number]).columns.tolist()
if 'ID' in numeric_feat_cols:
    numeric_feat_cols.remove('ID')

# 确保 features_to_plot 中的列都存在
available_features = [f for f in features_to_plot if f in numeric_feat_cols]
missing_features = [f for f in features_to_plot if f not in numeric_feat_cols]
if missing_features:
    print(f"⚠️ 警告：以下特征在 features.csv 中不存在，将跳过：{missing_features}")

print(f"✅ 将分析以下感知变量：{target_vars}")
print(f"✅ 将绘制以下特征的 PDP：{available_features}")

# ----------------------------
# 对每个感知变量进行建模和 PDP 绘制
# ----------------------------
for target in target_vars:
    print(f"\n{'=' * 50}")
    print(f"正在处理感知变量: {target}")

    # --- 1. 提取 y ---
    if target not in df_percep_raw.columns:
        print(f"❌ 列 '{target}' 不存在于 perceptions.csv，跳过")
        continue

    y_series = pd.to_numeric(df_percep_raw[target], errors='coerce')

    # --- 2. 对齐 X 和 y（按行对齐，假设 ID 顺序一致）---
    min_len = min(len(df_percep_raw), len(df_feat_raw))
    df_percep = df_percep_raw.iloc[:min_len].reset_index(drop=True)
    df_feat = df_feat_raw.iloc[:min_len].reset_index(drop=True)
    y_aligned = y_series.iloc[:min_len].reset_index(drop=True)

    # --- 3. 提取 X（仅限 available_features）---
    X_df = df_feat[available_features].copy()
    X_numeric = X_df.apply(pd.to_numeric, errors='coerce')  # 确保数值化

    # --- 4. 合并并清洗 NaN ---
    data_clean = pd.concat([X_numeric, y_aligned.rename('y')], axis=1)
    data_clean = data_clean.dropna()  # 删除任何含 NaN 的行

    if len(data_clean) == 0:
        print(f"❌ 清洗后无有效数据，跳过 {target}")
        continue

    X = data_clean[available_features].values
    y = data_clean['y'].values

    print(f"📊 使用 {len(y)} 个样本训练模型")

    # --- 5. 训练随机森林 ---
    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        oob_score=True,
        max_features='sqrt',
        n_jobs=-1
    )
    rf.fit(X, y)
    print(f"✅ 模型训练完成，袋外 R²: {rf.oob_score_:.4f}")

    # --- 6. 为每个特征绘制 PDP ---
    feature_names = available_features
    for i, feat_name in enumerate(feature_names):
        feat_idx = i

        try:
            pdp_result = partial_dependence(
                estimator=rf,
                X=X,
                features=[feat_idx],
                grid_resolution=50
            )
            if isinstance(pdp_result, tuple):
                mean_pdp = pdp_result[0][0]
                grid_vals = pdp_result[1][0]
            else:
                mean_pdp = pdp_result['average'][0]
                grid_vals = pdp_result['grid_values'][0]
        except Exception as e:
            print(f"⚠️ 计算 {feat_name} 的 PDP 时出错: {e}")
            continue

        # 创建图形
        plt.figure(figsize=(7, 5))
        ax = plt.gca()

        # 绘制主曲线和置信区间
        std_val = np.std(mean_pdp)
        ax.plot(grid_vals, mean_pdp, linewidth=2.5, color='#D55276', zorder=3)
        ax.fill_between(grid_vals, mean_pdp - std_val, mean_pdp + std_val,
                        color='#F1A7B5', alpha=0.3, zorder=2)

        # --- 关键：让图形横向铺满（x 轴无空白）---
        x_min, x_max = grid_vals.min(), grid_vals.max()
        ax.set_xlim(x_min, x_max)
        ax.margins(x=0)  # 关闭 x 方向的自动边距（y 方向保留默认）

        # --- 频率条形码（紧贴 x 轴下方）---
        X_feature = X[:, feat_idx]

        # 只绘制在当前 x 范围内的点（理论上全部都在）
        mask = (X_feature >= x_min) & (X_feature <= x_max)
        X_visible = X_feature[mask]

        # 条形码位置：使用当前 y 轴最小值作为基准
        y_barcode = ax.get_ylim()[0]
        bar_height = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.015

        for x in X_visible:
            ax.plot([x, x], [y_barcode, y_barcode - bar_height],
                    color='black', linewidth=0.5, alpha=0.6, zorder=0)

        # 设置标签
        ax.set_xlabel(feat_name, fontsize=12)
        ax.set_ylabel('Partial Dependence', fontsize=12)
        ax.set_title(target, fontsize=13, pad=15)

        # 背景和网格
        ax.set_facecolor('#F8F9FA')
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.5, color='#CCCCCC')
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color('black')

        plt.tight_layout()

        # 保存文件
        safe_target = target.replace(' ', '_')
        safe_feat = feat_name.replace(' ', '_')
        filename = f'pdp_{safe_feat}_on_{safe_target}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
        plt.close()
        print(f"   ✅ 已保存: {filename}")

print("\n🎉 所有 PDP 图已生成完毕！")