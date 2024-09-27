# Importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Tratamiento de ls NAs
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Codificar datos categoricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x = LabelEncoder()
labelencoder_x.fit_transform(x[:, 0])
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

column_transformer = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), [0])  # Aplicamos OneHotEncoder en la columna 0
    ], 
    remainder='passthrough'  # Mantenemos el resto de las columnas sin cambiar
)

x = column_transformer.fit_transform(x)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)