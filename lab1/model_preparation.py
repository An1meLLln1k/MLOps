import os
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Пути к файлам
train_path = "train/train_scaled.csv"

# Проверяем, существует ли train_scaled.csv
if not os.path.exists(train_path):
    raise FileNotFoundError("❌ Файл train_scaled.csv не найден. Сначала запустите data_preprocessing.py!")

# Загружаем данные
df = pd.read_csv(train_path)

# Определяем целевую переменную и признаки
target = "Calories"
if target not in df.columns:
    raise KeyError(f"❌ Целевая переменная '{target}' не найдена в данных!")

X = df.drop(columns=[target])  # Признаки
y = df[target]  # Целевая переменная

# Определяем числовые и категориальные признаки
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# Разделение на train и validation (80% - обучение, 20% - валидация)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем трансформер для обработки категориальных и числовых данных
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Создаем пайплайн: трансформация + модель
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Обучение модели
model.fit(X_train, y_train)

# Предсказание на validation
y_pred = model.predict(X_val)

# Оценка модели
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

# Вывод метрик
print("✅ Модель обучена!")
print(f"📊 MAE: {mae:.2f}")
print(f"📊 MSE: {mse:.2f}")
print(f"📊 R² Score: {r2:.2f}")

# Сохранение модели
model_path = "model/calories_model.pkl"
os.makedirs("model", exist_ok=True)
joblib.dump(model, model_path)

print(f"✅ Модель сохранена в: {model_path}")
