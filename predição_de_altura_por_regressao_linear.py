import numpy as np
from sklearn.linear_model import LinearRegression

# Ler os dados de entrada (por exemplo, a altura e o peso de uma pessoa)
altura = np.array([1.60, 1.65, 1.70, 1.75, 1.80])
peso = np.array([60, 65, 70, 75, 80])

# Criar o modelo de regressão linear
modelo = LinearRegression()

# Treinar o modelo com os dados de entrada
modelo.fit(altura.reshape(-1,1), peso)

# Fazer a predição para um novo valor de altura (por exemplo, 1.85 metros)
nova_altura = np.array([1.60])
nova_predicao = modelo.predict(nova_altura.reshape(-1,1))

# Imprimir o resultado da predição
print("Para uma altura de {:.2f} metros, o peso estimado é de {:.2f} kg.".format(nova_altura[0], nova_predicao[0]))
