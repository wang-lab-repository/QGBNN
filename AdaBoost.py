from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer, mean_absolute_error
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score
from data import get_data, get_dataloader
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from joblib import dump, load
# import shap
x_train, x_val, x_test, y_train, y_val, y_test = get_data("pfas_nalv.xlsx", random=9204)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
y_val = np.ravel(y_val)
# 创建梯度增强回归模型
# 定义参数网格
param_grid = {
    'n_estimators': [20, 50, 70, 90, 120],
    'learning_rate':[0.1, 0.05, 0.01, 0.2],
    'loss': ['linear', 'square', 'exponential']
    
}

# 创建梯度增强回归模型
model = AdaBoostRegressor()

# 使用网格搜索进行参数调优
grid_search = GridSearchCV(model, param_grid, cv=5)
# gbr = GradientBoostingRegressor(learning_rate=0.2, max_depth=4, n_estimators=130, subsample=1, alpha=0.9)

grid_search.fit(x_train, y_train)


# 输出最佳参数组合和对应的模型性能
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
dump(grid_search,'checkpoint/bestAdaboost.joblib')
# 在验证集上评估最佳模型性能
# grid_search = load('checkpoint/bestAdaboost.joblib')

val_predictions = grid_search.predict(x_val)
val_mse = mean_squared_error(y_val, val_predictions)
print("Validation RMSE:", math.sqrt(val_mse))

# 在测试集上评估最佳模型性能
test_predictions = grid_search.predict(x_test)
test_mse = mean_squared_error(y_test, test_predictions)
print("Test RMSE:", math.sqrt(test_mse))


prediction_train = grid_search.predict(x_train)
prediction_val = grid_search.predict(x_val)
prediction_test = grid_search.predict(x_test)


y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()
y_val = torch.from_numpy(y_val).float()
R2_train = 1 - torch.mean((y_train - prediction_train) ** 2) / torch.mean((y_train - torch.mean(y_train)) ** 2)
R2_val = 1 - torch.mean((y_val - prediction_val) ** 2) / torch.mean((y_val - torch.mean(y_val)) ** 2)
R2_test = 1 - torch.mean((y_test - prediction_test) ** 2) / torch.mean((y_test - torch.mean(y_test)) ** 2)
print("------------------------结果------------------------")
print(f'train: R2：{R2_train.detach().numpy()}\n')
print(f'val: R2：{R2_val.detach().numpy()}\n')
print(f'test: R2：{R2_test.detach().numpy()}\n')
print(f'train: RMSE：{np.sqrt(mean_squared_error(y_train, prediction_train))}\n')
print(f'val: RMSE：{np.sqrt(mean_squared_error(y_val, prediction_val))}\n')
print(f'test: RMSE：{np.sqrt(mean_squared_error(y_test, prediction_test))}\n')


# best_model = grid_search.best_estimator_
# # 创建 SHAP 解释器
# explainer = shap.Explainer(best_model)
# # 计算训练集上的 SHAP 值
# x_train = x_train.numpy()
# shap_values = explainer.shap_values(x_train)
# # 绘制特征重要性图表
# shap.summary_plot(shap_values, x_train)
# plt.show()