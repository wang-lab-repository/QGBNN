from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from skorch import NeuralNetRegressor
from data import get_data, get_dataloader
from Model import Net
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from joblib import dump, load
import shap
from captum.attr import IntegratedGradients, DeepLift, DeepLiftShap, ShapleyValueSampling
x_train, x_val, x_test, y_train, y_val, y_test = get_data("pfas_nalv.xlsx", random=9204)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
y_val = np.ravel(y_val)
# 创建梯度增强回归模型
# 定义参数网格
param_grid = {
    'n_estimators': [100, 120, 150],
    'learning_rate': [0.1, 0.05, 0.01, 0.2],
    'max_depth': [3, 4, 5],
    'alpha': [0.8, 0.9]
}

# 创建梯度增强回归模型
mymodel = Net()
mymodel.load_state_dict(torch.load('checkpoint/qnn_8_4-1.pth'))
# mymodel.qnn.neural_network.circuit.decompose().draw(output='mpl',style='iqp').savefig('result/the_model_circuit_8_3-18.tif', dpi=600)
# print("画图完成")
nnr = NeuralNetRegressor(mymodel)
model = GradientBoostingRegressor(init=nnr)
# 使用网格搜索进行参数调优
grid_search = GridSearchCV(model, param_grid, cv=5)
# gbr = GradientBoostingRegressor(learning_rate=0.2, max_depth=5, n_estimators=160, subsample=1, alpha=0.9, min_samples_split=3, min_samples_leaf=3)

grid_search.fit(x_train, y_train)


# # 输出最佳参数组合和对应的模型性能
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
dump(grid_search,'checkpoint/bestMyGBM.joblib')
# 在验证集上评估最佳模型性能
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

# gbr.fit(x_train, y_train)



# dump(gbr,'checkpoint/bestMyGBM1.joblib')
# # 在验证集上评估最佳模型性能
# val_predictions = gbr.predict(x_val)
# val_mse = mean_squared_error(y_val, val_predictions)
# print("Validation RMSE:", math.sqrt(val_mse))

# # 在测试集上评估最佳模型性能
# test_predictions = gbr.predict(x_test)
# test_mse = mean_squared_error(y_test, test_predictions)
# print("Test RMSE:", math.sqrt(test_mse))


# prediction_train = gbr.predict(x_train)
# prediction_val = gbr.predict(x_val)
# prediction_test = gbr.predict(x_test)

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


features = ['Membrane type', 'Temperature (˚C)', 'PFOS con (ppb)', 'pH',
                    'Pressure (MPa)', 'Divalent cations (mmol/L)',
                    'Monovalent cations (mmol/L)', 'Trivalent cations (mmol/L)'
                    ]

def model_predict(data):
    return grid_search.predict(data)
explainer = shap.KernelExplainer(model_predict, x_train.numpy())
shap_values = explainer.shap_values(x_train.numpy())
print(shap_values.shape)

shap.summary_plot(shap_values, x_train.numpy(), max_display=8,feature_names = features, title = 'Feature Importance Map')
plt.show()