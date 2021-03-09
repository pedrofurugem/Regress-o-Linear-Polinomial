import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


#Obtenha um novo modelo de regressão polinomial utilizando degree=4. E exiba o gráfico aseguir.

#importação da base de dados
dataset = pd.read_csv('HeightVsWeight.csv')

#selecionando as variáveis dependentes e independentes
x = dataset.iloc[:, 0].values.reshape(-1, 1) #x é independente
y = dataset.iloc[:, 1].values.reshape(-1, 1) #y é dependente
print(dataset.shape)

#Comparando os resultados
lr = LinearRegression()
lr.fit(x, y)

#AQUI ENTRA O MODELO DE REGRESSÃO LINEAR POLINOMIAL
poly_features = PolynomialFeatures (degree= 4)
x_poly = poly_features.fit_transform(x)
polyLinearRegression = LinearRegression()
polyLinearRegression.fit(x_poly, y)

#Vizualizar os resultados com regressão linear simples
plt.scatter(x, y, color="red")
plt.plot(x, lr.predict(x), color="blue")
plt.title("Dados por Regressão Linear ")
plt.xlabel("Age")
plt.ylabel("Height")
plt.show()

#Vizualizar os resultados com regressão polinomial 
plt.scatter(x, y, color="red")
plt.plot(x, polyLinearRegression.predict(x_poly),color="blue")
plt.title("Dados Regressão Linear Polinomial")
plt.xlabel("Age")
plt.ylabel("Height")
plt.show()
