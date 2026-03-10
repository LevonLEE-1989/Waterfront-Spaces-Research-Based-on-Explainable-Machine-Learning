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

# ----------------------------
# 5. 绘制特征重要性图
# ----------------------------
plt.figure(figsize=(9, 6))
plt.barh(range(len(importances)), importances[sorted_idx], color='#4a90e2')
plt.yticks(range(len(importances)), [feature_names[i] for i in sorted_idx], fontsize=11)
plt.xlabel('Feature Importance', fontsize=12)
plt.title('Random Forest Feature Importance (Scenic Beauty)', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_beauty.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------
# 6. 绘制偏依赖图（PDP）— 前3个最重要特征
# ----------------------------
top3_idx = sorted_idx[:3]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, idx in enumerate(top3_idx):
    # 调用 partial_dependence（新版 API）
    pdp = partial_dependence(
        estimator=rf,
        X=X,
        features=[idx],
        grid_resolution=50
    )

    # 提取结果
    mean_pdp = pdp['average'][0]  # shape: (n_grid_points,)
    grid_vals = pdp['grid_values'][0]  # 该特征的网格点（x 轴）

    std_val = np.std(mean_pdp)

    axes[i].plot(grid_vals, mean_pdp, 'k-', linewidth=2.5)
    axes[i].fill_between(grid_vals,
                         mean_pdp - std_val,
                         mean_pdp + std_val,
                         color='gray', alpha=0.2)
    axes[i].set_xlabel(feature_names[idx], fontsize=12)
    axes[i].set_ylabel('Partial Dependence', fontsize=12)
    axes[i].set_title(f'{feature_names[idx]}', fontsize=13)
    axes[i].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('pdp_top3_beauty.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n🎉 分析完成！图表已保存为 PNG 文件。")