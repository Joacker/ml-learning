# Regresión Lineal Múltiple
# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('./50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Codificar datos categoricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x = LabelEncoder()
labelencoder_x.fit_transform(X[:, 3])
X[:, 3] = labelencoder_x.fit_transform(X[:, 3])
# Para ser traducida a variable dummy
column_transformer = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), [3])  # Aplicamos OneHotEncoder en la columna 0
    ], 
    remainder='passthrough'  # Mantenemos el resto de las columnas sin cambiar
)

X = column_transformer.fit_transform(X)
# labelencoder_y = LabelEncoder()
# y = labelencoder_y.fit_transform(y)

# Evitar la trampa de las variables ficticias
X = X[:, 1:]

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Escalado de variables
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Ajustar el modelo de Regresión Lineal Múltiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predicción de los resultados en el conjunto de testing
y_pred = regression.predict(X_test)


# Construir el modelo óptimo de RLM utilizando la Eliminación hacia atrás
import statsmodels.regression.linear_model as sm

X = np.append(arr = np.ones((50,1)).astype(int), values =X ,axis=1)

SL = 0.05

X_opt = X[:,[0,1,2,3,4,5]].tolist()

regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regression_OLS.summary()
