%% 引入缺失值矩阵，缺失值权重比例降低；引入stacking，元学习器是基础的线性回归

clc; 
clear all
close all

% 加载数据
filename = 'DataSet0.95.xlsx';
opts = detectImportOptions(filename);
opts.SelectedVariableNames = opts.SelectedVariableNames(2:end);
data = readtable(filename, opts);

% 分离特征和目标变量
X = data(:, 1:end-1);
y = data(:, end);

% 转换为矩阵
X_matrix = table2array(X);
y_matrix = table2array(y);

% 加载缺失值矩阵
filename = 'Missing0.95.xlsx';
opts = detectImportOptions(filename);
opts.SelectedVariableNames = opts.SelectedVariableNames(2:end);
missing_mask = readtable(filename, opts);

% **提前转换为数组**
missing_mask_matrix = table2array(missing_mask);

% 划分训练集和测试集
n_samples = height(data);
train_indices = 1:round(0.8 * n_samples);
test_indices = (round(0.8 * n_samples) + 1):n_samples;

X_train = X_matrix(train_indices, :);
y_train = y_matrix(train_indices, :);
X_test = X_matrix(test_indices, :);
y_test = y_matrix(test_indices, :);
missing_mask_train = missing_mask_matrix(train_indices, :);

% 训练 AdaBoost 模型
num_learners = 250;
model = Stacking_AdaBoostRegressor(X_train, y_train, missing_mask_train, num_learners);

% 预测
y_pred = predict_Stacking_AdaBoost(model, X_test);

% 计算 R² 评分
SS_res = sum((y_test - y_pred).^2);
SS_tot = sum((y_test - mean(y_test)).^2);
r2_score = 1 - (SS_res / SS_tot);

fprintf('R²(Missing) score: %.4f\n', r2_score);

% 可视化拟合效果
figure;
hold on;

% 绘制预测结果
plot(y_pred, 'b-', 'LineWidth', 2, 'DisplayName', 'PredictedMissing');
% 绘制实际值
plot(y_test, 'k--', 'LineWidth', 2, 'DisplayName', 'Actual');

% 添加图例、标签和标题
legend('show');
xlabel('Sample Index');
ylabel('Target Value');
title('Predictions vs Actual Values');
hold off;
