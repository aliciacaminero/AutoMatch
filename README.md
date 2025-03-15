# AutoMatch - Predictor de Precio y Recomendador de Vehículos

AutoMatch es un proyecto que consta de dos modelos basados en Machine Learning para trabajar con un conjunto de datos de vehículos:

- Predictor de Precio de Vehículos: Un modelo de regresión que predice el precio de venta de un vehículo basado en sus características (como kilometraje, año, tipo de combustible, etc.).
- Recomendador de Vehículos: Un sistema recomendador basado en contenido que sugiere vehículos similares a un vehículo dado según sus características.

## Preprocesamiento de Datos:

Se limpian los datos, manejando valores nulos o atípicos.
Se crean nuevas características si es necesario (por ejemplo, la diferencia entre el año de fabricación y el año de publicación).
Se codifican las variables categóricas y se normalizan las variables numéricas.

## Entrenamiento del Modelo:

El conjunto de datos se divide en un conjunto de entrenamiento y uno de prueba utilizando train_test_split.
Se entrena el modelo de regresión y se ajustan los hiperparámetros si es necesario.
Evaluación del Modelo:

Se evalúa el modelo utilizando métricas como el Error Cuadrático Medio (MSE) o Raíz del Error Cuadrático Medio (RMSE).
