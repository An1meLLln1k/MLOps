import os
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Пути к файлам
test_path = "test/test_scaled.csv"
model_path = "model/calories_model.pkl"

# Проверяем существование файлов
if not os.path.exists(test_path):
    raise FileNotFoundError("❌ Файл test_scaled.csv не найден. Сначала запустите data_preprocessing.py!")

if not os.path.exists(model_path):
    raise FileNotFoundError("❌ Файл модели calories_model.pkl не найден. Сначала запустите model_preparation.py!")

# Загружаем тестовые данные
test_df = pd.read_csv(test_path)

# Определяем целевую переменную и признаки
target = "Calories"
if target not in test_df.columns:
    raise KeyError(f"❌ Целевая переменная '{target}' не найдена в данных!")

X_test = test_df.drop(columns=[target])  # Признаки
y_test = test_df[target]  # Целевая переменная

# Загружаем обученную модель
model = joblib.load(model_path)

# Делаем предсказания
y_pred = model.predict(X_test)

# Оценка модели
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Вывод метрик
print("✅ Тестирование модели завершено!")
print(f"📊 MAE: {mae:.2f}")
print(f"📊 MSE: {mse:.2f}")
print(f"📊 R² Score: {r2:.2f}")
