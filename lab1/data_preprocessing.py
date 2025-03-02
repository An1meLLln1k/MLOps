import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Пути к файлам
train_path = "train/train.csv"
test_path = "test/test.csv"

# Проверяем, существуют ли файлы
if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise FileNotFoundError("❌ Файлы train.csv и test.csv не найдены. Сначала запустите data_creation.py!")

# Загружаем данные
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Определяем числовые признаки (исключаем 'Calories', если он является целевой переменной)
numeric_features = train_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
if "Calories" in numeric_features:
    numeric_features.remove("Calories")  # Исключаем целевую переменную

# Инициализируем StandardScaler
scaler = StandardScaler()

# Обучаем scaler на train и трансформируем train/test
train_df[numeric_features] = scaler.fit_transform(train_df[numeric_features])
test_df[numeric_features] = scaler.transform(test_df[numeric_features])

# Сохраняем предобработанные файлы
train_df.to_csv("train/train_scaled.csv", index=False)
test_df.to_csv("test/test_scaled.csv", index=False)

print("✅ Предобработка завершена. Данные сохранены:")
print("📂 train/train_scaled.csv")
print("📂 test/test_scaled.csv")
