function y_pred = predict_Stacking_AdaBoost(model, X_test)
    % 使用 Stacking-AdaBoost 进行预测
    % X_test: 测试集特征 (N x D)
    % y_pred: 预测结果 (N x 1)

    num_learners = length(model.learners);
    n_samples = size(X_test, 1);

    % 1️⃣ 生成 Stacking 特征：基学习器的预测结果
    stacking_features = zeros(n_samples, num_learners);

    for t = 1:num_learners
        stacking_features(:, t) = predict(model.learners{t}, X_test);
    end

    % 2️⃣ 使用元学习器进行最终预测
    y_pred = predict(model.meta_learner, stacking_features);
end
