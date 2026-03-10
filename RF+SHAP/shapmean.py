# -*- coding: utf-8 -*-
"""
郑州东风渠滨水感知研究 —— 随机森林分析
作者：你
功能：特征重要性 + 偏依赖图（PDP）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
import shap
plt.rcParams['font.family'] = 'Times New Roman'

# ----------------------------
# 1. 读取 perceptions.csv（感知得分）
# ----------------------------
print("正在读取 perceptions.csv...")
df_percep = pd.read_csv('perceptions.csv')

# 安全提取 y（支持中英文）
if 'Scenic Beauty' in df_percep.columns:
    y_series = df_percep['Scenic Beauty']
elif '美景度' in df_percep.columns:
    y_series = df_percep['美景度']
else:
    print("⚠️ 未找到 'Scenic Beauty' 或 '美景度' 列，将默认使用第2列作为美景度标签")
    y_series = df_percep.iloc[:, 1]

# 转为数值型，非数值设为 NaN
y = pd.to_numeric(y_series, errors='coerce').values

# ----------------------------
# 2. 读取 features.csv（视觉特征）
# ----------------------------
print("正在读取 features.csv...")
df_feat = pd.read_csv('features.csv')

# 自动选择数值列（跳过 ID、文本等）
numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
if 'ID' in numeric_cols:
    numeric_cols.remove('ID')

if not numeric_cols:
    raise ValueError("❌ features.csv 中未找到任何数值型特征列！")

# 提取纯数值特征
X_df = df_feat[numeric_cols]
X_raw = X_df.values
feature_names = numeric_cols

print(f"✅ 成功读取 {len(df_feat)} 条特征数据")

# ----------------------------
# 3. 对齐并彻底清洗数据
# ----------------------------
min_len = min(len(y), len(df_feat))
df_percep_aligned = df_percep.iloc[:min_len].reset_index(drop=True)
df_feat_aligned = df_feat.iloc[:min_len].reset_index(drop=True)

# 清理 y
if 'Scenic Beauty' in df_percep_aligned.columns:
    y_series = df_percep_aligned['Scenic Beauty']
elif '美景度' in df_percep_aligned.columns:
    y_series = df_percep_aligned['美景度']
else:
    y_series = df_percep_aligned.iloc[:, 1]

y_clean = pd.to_numeric(y_series, errors='coerce')

# 清理 X：只保留数值列，并逐列转为数值
numeric_cols = []
X_clean_list = []

for col in df_feat_aligned.columns:
    if col == 'ID':
        continue
    # 尝试转为数值
    col_series = pd.to_numeric(df_feat_aligned[col], errors='coerce')
    if col_series.isna().all():
        print(f"⚠️ 列 '{col}' 全为非数值，已跳过")
        continue
    numeric_cols.append(col)
    X_clean_list.append(col_series.values)

if not X_clean_list:
    raise ValueError("❌ 没有任何有效数值特征列！")

# 合并为 (n_samples, n_features) 数组
X_clean = np.column_stack(X_clean_list)  # 自动为 float64
y_clean = y_clean.values

# 移除 y 或 X 中任一为 NaN 的行
valid_mask = ~(np.isnan(y_clean) | np.isnan(X_clean).any(axis=1))
X = X_clean[valid_mask]
y = y_clean[valid_mask]
feature_names = numeric_cols

print(f"\n📊 最终使用 {len(y)} 张图像进行分析（已对齐并彻底清理无效值）")
print(f"   X shape: {X.shape}, dtype: {X.dtype}")

# ----------------------------
# 4. 训练随机森林模型
# ----------------------------
print("\n正在训练随机森林模型（美景度）...")
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    oob_score=True,
    max_features='sqrt',
    n_jobs=-1
)
rf.fit(X, y)

print(f"✅ 模型训练完成！")
print(f"   袋外 R²: {rf.oob_score_:.4f}")
print(f"   特征重要性排序：")

importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
for i, idx in enumerate(sorted_idx):
    print(f"     {i + 1}. {feature_names[idx]}: {importances[idx]:.4f}")

from matplotlib.colors import LinearSegmentedColormap

# 创建自定义 colormap：红 -> 白 -> 蓝
colors = ['#d62728', 'white', '#1f77b4']  # 红 -> 白 -> 蓝
cmap_custom = LinearSegmentedColormap.from_list("red_white_blue", colors, N=256)

# ----------------------------
# ----------------------------
# 5. 计算 SHAP 值并绘制 mean(|SHAP|) 图
# ----------------------------
print("\n正在计算 SHAP 值...")
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)

# 取第一个输出（如果是多输出）
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# 计算每个特征的平均绝对 SHAP 值
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

# 排序
sorted_idx = np.argsort(mean_abs_shap)[::-1]
ordered_features = [feature_names[i] for i in sorted_idx]
mean_abs_sorted = mean_abs_shap[sorted_idx]

# 创建条形图后，立即设置边框
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(mean_abs_sorted)), mean_abs_sorted, color='#97B5D1', edgecolor='none', alpha=0.9)

# 设置 y 轴标签、x 轴等...
ax.set_yticks(range(len(ordered_features)))
ax.set_yticklabels(ordered_features, fontsize=11)
ax.invert_yaxis()

# x 轴设置
ax.set_xlabel('mean(|SHAP value|) (average impact on model output magnitude)', fontsize=11)
ax.set_xlim(0, max(mean_abs_sorted) * 1.1)

# 网格线
ax.grid(True, axis='x', linestyle='--', linewidth=0.8, alpha=0.4, color='#ddd')
ax.grid(False, axis='y')

# 明确设置边框：仅底部可见
for spine in ax.spines.values():
    spine.set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(0.8)

# 去掉背景填充（可选）
ax.set_facecolor('white')

# 标题
ax.set_title('SHAP Mean Absolute Value (Scenic Beauty)', fontsize=12, pad=20)

# 保存
plt.tight_layout()
plt.savefig('shap_mean_abs_beauty.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("\n🎉 分析完成！图表已保存为 PNG 文件。")