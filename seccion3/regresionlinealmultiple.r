# Regresion lineal multiple
# Plantilla para el Pre Procesado de Datos
# Importar el dataset
dataset = read.csv('./seccion3/50_Startups.csv')
#dataset = dataset[, 2:3]

# Codificar datos categoricos
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))


# Dividir los datos en conjunto de entrenamiento y conjunto de test
# install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Escalado de valores
# training_set[,2:3] = scale(training_set[,2:3])
# testing_set[,2:3] = scale(testing_set[,2:3])