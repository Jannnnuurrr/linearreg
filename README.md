import pandas as pd
import statsmodels.api as sm
# Загрузка вашего датасета
file_path = "Fashion_Retail_Sales (1).csv"
data = pd.read_csv(file_path)
# Выбор соответствующих столбцов
selected_columns = ["Review Rating", "Purchase Amount (USD)"]
df = data[selected_columns]
 
# Удаление пропущенных значений, если они есть
df = df.dropna()
 
# Выделение переменных
X = df["Review Rating"]
y = df["Purchase Amount (USD)"]
 
# Добавление константного члена к матрице независимых переменных
X = sm.add_constant(X)
 
# Построение модели регрессии
model = sm.OLS(y, X).fit()
 
# Создание таблицы коэффициентов
coefficients_table = model.summary().tables[1]
 
# Вывод таблицы коэффициентов
print(coefficients_table)
