import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# 假设已有数据集：X_4d (n_samples, 4), Y_2d (n_samples, 2)
# 目标参数：target_t_omg = [T, OMG]

def load_and_combine(mat_files):
    X_combined, Y_combined = [], []
    for file in mat_files:
        data = loadmat(file)
        X = data['X_4d']  # 确保变量名与.mat文件中的一致
        Y = data['Y_2d']  # 如果维度是(4, n_samples)，需转置为X.T
        X_combined.append(X)
        Y_combined.append(Y)
    return np.vstack(X_combined), np.vstack(Y_combined)

mat_files = ['/home/djx/Poincare/pyglow/Scikiti-learn/ErgodicData_k0001to00035_a0to80_Hz2_DeltE0005_New.mat', 
             '/home/djx/Poincare/pyglow/Scikiti-learn/ErgodicData_k00003to00035_a0to80_Hz23452_DeltE001_New.mat', 
             '/home/djx/Poincare/pyglow/Scikiti-learn/ErgodicData_k00003to00035_a0to80_Hz24495_DeltE0005_New.mat']
X_4d, Y_2d = load_and_combine(mat_files)

# 数据标准化
scaler_4d = StandardScaler()
scaler_2d = StandardScaler()
X_scaled = scaler_4d.fit_transform(X_4d)
Y_scaled = scaler_2d.fit_transform(Y_2d)

# 训练K近邻模型
nbrs = NearestNeighbors(n_neighbors=100)  # 根据数据密度调整n_neighbors
nbrs.fit(Y_scaled)

# 对目标参数标准化
target_t_omg = [57.67,-0.39]
target_scaled = scaler_2d.transform([target_t_omg])

# 查找最近的k个样本
distances, indices = nbrs.kneighbors(target_scaled)
candidate_X = X_scaled[indices[0]]

# 在候选样本上拟合GMM以生成更多解
gmm = GaussianMixture(n_components=3)  # 根据数据分布调整组件数
gmm.fit(candidate_X)

# 从GMM中采样生成多组4维参数
generated_samples_scaled = gmm.sample(n_samples=10)[0]  # 生成10个样本
generated_samples = scaler_4d.inverse_transform(generated_samples_scaled)

print("Generated 4D Parameters:")
print(generated_samples)