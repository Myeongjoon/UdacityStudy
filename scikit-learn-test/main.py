from sklearn import datasets # sklearn comes with a variety of preloaded datasets
from sklearn import metrics # calculate how well our model is doing
from sklearn.linear_model import LinearRegression
housing_data = datasets.load_boston()
linear_regression_model = LinearRegression()
linear_regression_model.fit(housing_data.data, housing_data.target)
predictions = linear_regression_model.predict(housing_data.data)
score = metrics.r2_score(housing_data.target, predictions)
print(score)