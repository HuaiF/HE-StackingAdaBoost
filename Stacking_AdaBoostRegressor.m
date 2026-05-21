function model = Stacking_AdaBoostRegressor(X_train, y_train, missing_mask, num_learners)
    % 训练一个基于 Stacking 的 AdaBoost 回归模型
    % X_train: 训练集特征 (N x D)
    % y_train: 训练集目标值 (N x 1)
    % missing_mask: 缺失值矩阵 (N x D)，1 表示存在，0 表示缺失
    % num_learners: 基学习器数量

    % 训练集样本数
    n_samples = size(X_train, 1);

    % 设置填充值样本的权重缩减比例（例如 0.5 表示每次迭代填充值样本的权重减半）
    lambda = 0.5;

    % **初始化样本权重（均匀分配）**
    sample_weights = ones(n_samples, 1) / n_samples;

    % 存储基学习器和权重
    learners = cell(num_learners, 1);
    alphas = zeros(num_learners, 1);

    % 存储每轮的预测结果，用于 Stacking
    stacking_features = zeros(n_samples, num_learners);

    for t = 1:num_learners
        % 训练基学习器（回归树）
        tree = fitrtree(X_train, y_train, 'Weights', sample_weights, 'MaxNumSplits', 4);

        % 预测训练集
        y_pred = predict(tree, X_train);

        % 记录每轮预测值，作为 Stacking 输入
        stacking_features(:, t) = y_pred;

        % 计算误差（所有样本都计算）
        error = abs(y_train - y_pred);

        % 计算加权误差，使用平滑因子
        beta = 1 + 0.2 * (t / num_learners);  % 逐渐增加 beta
        weighted_error = sum(sample_weights .* (error .^ beta)) / sum(sample_weights);

        % 修正 weighted_error，避免 alpha 变负
        weighted_error = min(weighted_error, 0.49);
        alpha_t = 0.5 * log((1 - weighted_error) / max(weighted_error, eps));

        % **更新样本权重**
        sample_weights = sample_weights .* exp(-alpha_t * (y_train .* y_pred));

        % **对填充值样本的权重进行衰减**
        sample_weights(missing_mask(:, 1) == 0) = sample_weights(missing_mask(:, 1) == 0) * lambda;

        % **归一化样本权重**
        sample_weights = sample_weights + eps;  % 避免出现 0
        sample_weights = sample_weights / sum(sample_weights);  

        % 存储基学习器和权重
        learners{t} = tree;
        alphas(t) = alpha_t;

        % 打印训练进度
        fprintf('Iteration %d: Weighted Error = %.4f, Alpha = %.4f\n', t, weighted_error, alpha_t);
    end

    % 训练元学习器（meta-learner），基于 stacking_features 进行最终预测
    meta_learner = fitrlinear(stacking_features, y_train);

    % 返回模型结构体
    model.learners = learners;
    model.alphas = alphas;
    model.meta_learner = meta_learner;
end