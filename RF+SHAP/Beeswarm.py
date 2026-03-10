# -*- coding: utf-8 -*-
"""
郑州东风渠滨水感知研究 —— SHAP Beeswarm 图（自定义配色）
- 自定义 diverging colormap（深蓝→白→深红）
- 保留真实 SHAP 分布（不对称）
- 横向虚线网格 + 无边框
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import shap
from matplotlib.colors import LinearSegmentedColormap

# 设置字体和绘图风格
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 11

# =============================
# ✅ 定义自定义 colormap（您的配色）
# =============================
colors_smooth = [
    (0.0, (81 / 255, 132 / 255, 178 / 255)),  # 深蓝（最负）
    (0.4, (170 / 255, 212 / 255, 248 / 255)),  # 浅蓝
    (0.5, (242 / 255, 245 / 255, 250 / 255)),  # 白色（SHAP=0）
    (0.6, (241 / 255, 167 / 255, 181 / 255)),  # 浅红
    (1.0, (213 / 255, 82 / 255, 118 / 255))  # 深红（最正）
]
cmap_custom = LinearSegmentedColormap.from_list("custom_diverging", colors_smooth, N=256)

# ----------------------------
# 读取数据
# ----------------------------
print("正在读取 perceptions.csv 和 features.csv...")
df_percep = pd.read_csv('perceptions.csv')
df_feat = pd.read_csv('features.csv')

if 'ID' not in df_percep.columns or 'ID' not in df_feat.columns:
    raise ValueError("❌ 两个 CSV 文件都必须包含 'ID' 列用于对齐！")

df_percep = df_percep.set_index('ID')
df_feat = df_feat.set_index('ID')
df_merged = df_percep.join(df_feat, how='inner')
print(f"✅ 对齐后样本数: {len(df_merged)}")

target_vars = ['Scenic Beauty', 'Safety', 'Recreational Value']
features_to_plot = [
    'Color Diversity', 'Color Uniformity', 'Openness',
    'Waterfront Accessibility', 'Walkability', 'Spatial Definition',
    'Greenness', 'Plant Diversity'
]

available_features = [f for f in features_to_plot if f in df_merged.columns]
missing = [f for f in features_to_plot if f not in df_merged.columns]
if missing:
    print(f"⚠️ 警告：以下特征缺失：{missing}")
print(f"✅ 使用特征：{available_features}")

# ----------------------------
# 主循环
# ----------------------------
for target in target_vars:
    safe_target = target.replace(' ', '_')
    print(f"\n{'=' * 50}")
    print(f"处理目标: {target}")

    if target not in df_merged.columns:
        print(f"❌ 目标变量 '{target}' 不存在，跳过")
        continue

    y_raw = pd.to_numeric(df_merged[target], errors='coerce')
    X_raw = df_merged[available_features].apply(pd.to_numeric, errors='coerce')
    data_clean = pd.concat([X_raw, y_raw.rename('y')], axis=1).dropna()
    print(f"📊 清洗后样本数: {len(data_clean)}")

    if len(data_clean) < 10:
        print("❌ 样本太少，跳过")
        continue

    X = data_clean[available_features].values
    y = data_clean['y'].values

    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        oob_score=True,
        max_features='sqrt',
        n_jobs=-1
    )
    rf.fit(X, y)
    print(f"✅ 模型 R² (OOB): {rf.oob_score_:.4f}")

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    print(f"   SHAP 范围: {shap_values.min():.5f} ~ {shap_values.max():.5f}")

    # 手动绘制 Beeswarm
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    feature_order = np.argsort(mean_abs_shap)[::-1]

    y_pos = np.arange(len(available_features))
    all_shap_vals = []
    all_y_vals = []
    all_colors = []

    np.random.seed(42)
    for i, feat_idx in enumerate(feature_order):
        shap_feat = shap_values[:, feat_idx]
        x_feat = X[:, feat_idx]
        y_jitter = np.random.uniform(-0.35, 0.35, size=len(shap_feat))
        y_plot = np.full_like(shap_feat, y_pos[i]) + y_jitter

        all_shap_vals.append(shap_feat)
        all_y_vals.append(y_plot)
        all_colors.append(x_feat)

    all_shap_vals = np.concatenate(all_shap_vals)
    all_y_vals = np.concatenate(all_y_vals)
    all_colors = np.concatenate(all_colors)

    # ========================
    # ✅ 绘图（使用自定义配色）
    # ========================
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        all_shap_vals,
        all_y_vals,
        c=all_colors,  # 特征值决定颜色（不是 SHAP 值！）
        cmap=cmap_custom,  # ✅ 使用您的自定义配色
        s=12,
        alpha=0.7,
        edgecolors='none'
    )

    # 使用原始 SHAP 范围（不强制对称）
    ax.set_xlim(all_shap_vals.min() - 0.1, all_shap_vals.max() + 0.1)

    # x=0 参考线
    ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)

    # y 轴标签
    ordered_features = [available_features[i] for i in feature_order]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered_features)
    ax.invert_yaxis()
    ax.set_xlabel('SHAP value (impact on model output)', fontsize=12)

    # 横向虚线网格
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.3, color='gray')

    # 移除上、左、右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # 颜色条
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Feature value', rotation=270, labelpad=15, fontsize=11)

    plt.tight_layout()
    filename = f'shap_beeswarm_{safe_target}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✅ 已保存: {filename}")

print("\n🎉 所有自定义配色 SHAP 图已生成！")