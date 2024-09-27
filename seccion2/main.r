
dataset = read.csv("./seccion2/Salary_Data.csv")

# Dividir los datos en conjunto de entrenamiento y conjunto de test
# install.packages("caTools")
head(dataset)
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Ajustar el modelo de regresión lineal simple al conjunto de entrenamiento
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)
# Predecir el conjunto de test
y_pred = predict(regressor, newdata = testing_set)

print(summary(regressor))

# Visualizar los resultados de entrenamiento
library(ggplot2)

ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             color = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Years of Experience') +
  ylab('Salary')

# Visualizar los resultados de test
ggplot() +
  geom_point(aes(x = testing_set$YearsExperience, y = testing_set$Salary),
             color = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Salary vs Experience (Test set)') +
  xlab('Years of Experience') +
  ylab('Salary')

# install.packages('Metrics')
library(Metrics)
mae(testing_set$Salary, y_pred)
mse(testing_set$Salary, y_pred)
rmse(testing_set$Salary, y_pred)


# Escalado de valores
# training_set[,2:3] = scale(training_set[,2:3])
# testing_set[,2:3] = scale(testing_set[,2:3]) # nolint