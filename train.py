# pn дл.раб.таб. num дл.числ.выч. mat дл.постр.гр      dfтаб.с.дан.     1 раз.дан.текст.2мод.лин.рег.3.средн.квадр.ош.
import pandas as pd                           
import numpy as np
import matplotlib.pyplot as plt
import pickle


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
  


data = pd.read_csv('multiple_linear_regression_dataset-2.csv')
df = pd.DataFrame(data)

X = df[['age', 'experience']]
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Модель сохранена как model.pkl")


y_pred = model.predict(X_test)



print("Коэффициенты:", model.coef_)
print("Перехват (intercept):", model.intercept_)
print("R2 score:", r2_score(y_test, y_pred))                                       # r2 sc наск.хор.мод.об.данн. mse.сред.кв.ош.
print("MSE:", mean_squared_error(y_test, y_pred))


# 1
plt.figure(figsize=(6, 5))

plt.scatter(y_test, y_pred, color='blue', label='Предсказания')


plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],                                                    #красная линия
         color='red', linewidth=2, label='Идеально')

plt.xlabel('Реальные значения income')
plt.ylabel('Предсказанные значения income')                             
plt.title('Реальные vs Предсказанные')
plt.legend()

plt.grid(True)

plt.show()



