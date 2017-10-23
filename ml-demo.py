#coding=utf-8
'''
#我们的问题是得到一个线性的关系，对应PE是样本输出，而AT/V/AP/RH这4个是样本特征， 机器学习的目的就是得到一个线性回归模型，而需要学习的，就是θ0,θ1,θ2,θ3,θ4这5个参数。
#导入的库声明
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split
'''1. 获取数据，定义问题
　　　　没有数据，当然没法研究机器学习啦。:) 这里我们用UCI大学公开的机器学习数据来跑线性回归。

　　　　数据的介绍在这： http://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant

　　　　数据的下载地址在这： http://archive.ics.uci.edu/ml/machine-learning-databases/00294/

　　　　里面是一个循环发电场的数据，共有9568个样本数据，每个数据有5列，分别是:AT（温度）, V（压力）, AP（湿度）, RH（压强）, PE（输出电力)。我们不用纠结于每项具体的意思。
#2. 整理数据
#3. 用pandas来读取数据
#4. 准备运行算法的数据
#read_csv里面的参数是csv在你电脑上的路径，此处csv文件放在notebook运行目录下面的CCPP目录里,接着我们就可以用pandas读取数据了：
'''
data = pd.read_csv('.\CCPP\ccpp.csv')
#测试下读取数据是否成功：读取前五行数据，如果是最后五行，用data.tail()
data.head()
print data.shape
#开始准备样本特征X，我们用AT， V，AP和RH这4个列作为样本特征。
X = data[['AT', 'V', 'AP', 'RH']]
print X.head()
#准备样本输出y， 我们用PE作为样本输出。
y = data[['PE']]
print y.head()
#5. 划分训练集和测试集
#把X和y的样本组合划分成两部分，一部分是训练集，一部分是测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
#查看下训练集和测试集的维度：可以看到75%的样本数据被作为训练集，25%的样本被作为测试集。
print X_train.shape
print X_test.shape
print y_train.shape
print y_test.shape
#6. 运行scikit-learn的线性模型
#用scikit-learn的线性模型来拟合我们的问题了。scikit-learn的线性回归算法使用的是最小二乘法来实现的。
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
#拟合完毕后，我们看看我们的需要的模型系数结果：这样我们就得到了在步骤1里面需要求得的5个值。
print linreg.intercept_
print linreg.coef_
#7. 模型评价
#模型拟合测试集
y_pred = linreg.predict(X_test)
from sklearn import metrics
#我们需要评估我们的模型的好坏程度，对于线性回归来说，我们一般用均方差（Mean Squared Error, MSE）或者均方根差(Root Mean Squared Error, RMSE)在测试集上的表现来评价模型的好坏。
# 用scikit-learn计算MSE
print "MSE:",metrics.mean_squared_error(y_test, y_pred)
# 用scikit-learn计算RMSE
print "RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred))

#得到了MSE或者RMSE，如果我们用其他方法得到了不同的系数，需要选择模型时，就用MSE小的时候对应的参数。

# new 比如这次我们用AT， V，AP这3个列作为样本特征。不要RH， 输出仍然是PE。代码如下：
X = data[['AT', 'V', 'AP']]
y = data[['PE']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
#模型拟合测试集
y_pred = linreg.predict(X_test)
from sklearn import metrics
# 用scikit-learn计算MSE
print "MSE:",metrics.mean_squared_error(y_test, y_pred)
# 用scikit-learn计算RMSE
print "RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred))
'''
#8. 交叉验证
#我们可以通过交叉验证来持续优化模型，代码如下，我们采用10折交叉验证，即cross_val_predict中的cv参数为10：
#可以看出，采用交叉验证模型的MSE比第6节的大，主要原因是我们这里是对所有折的样本做测试集对应的预测值的MSE，而第6节仅仅对25%的测试集做了MSE。两者的先决条件并不同。
'''
X = data[['AT', 'V', 'AP', 'RH']]
y = data[['PE']]
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(linreg, X, y, cv=10)
# 用scikit-learn计算MSE
print "MSE:",metrics.mean_squared_error(y, predicted)
# 用scikit-learn计算RMSE
print "RMSE:",np.sqrt(metrics.mean_squared_error(y, predicted))

#9. 画图观察结果,这里画图真实值和预测值的变化关系，离中间的直线y=x直接越近的点代表预测损失越低。代码如下：
fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
#以上就是用scikit-learn和pandas学习线性回归的过程，希望可以对初学者有所帮助。