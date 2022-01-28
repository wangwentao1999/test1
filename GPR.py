import pandas as pd
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import sklearn.gaussian_process as gp
from sklearn.metrics import  r2_score,mean_squared_error,mean_absolute_error

data = pd.read_excel('标准化数据集.xls')

X = data.iloc[:,:-1]
Y = data.iloc[:,-1]
cols = list(X.columns)
kernel = C(0.1, (0.001, 0.1)) * RBF(0.5, (1e-4, 10))
model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-3)



model.fit(X, Y)
pred = model.predict(X)
print('R2_train =',r2_score(Y,pred))
print('MAE_train =',mean_absolute_error(Y,pred))
print('RMSE_train =', mean_squared_error(Y,pred)  ** (1/2))

for var in pred:
    print(var)