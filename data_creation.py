import os
import pandas as pd
from sklearn.model_selection import train_test_split
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

# Установка переменных окружения (небезопасно, лучше использовать ~/.kaggle/kaggle.json)
os.environ['KAGGLE_USERNAME'] = "lexlukyanov"
os.environ['KAGGLE_KEY'] = "78527335507f1559616af53559eef769"

# Инициализация API
api = KaggleApi()
api.authenticate()

# Папка для загрузки
dataset_path = "calories_dataset"
os.makedirs(dataset_path, exist_ok=True)

# Название датасета
dataset_name = "ruchikakumbhar/calories-burnt-prediction"

# Скачивание и разархивирование
api.dataset_download_files(dataset_name, path=dataset_path, unzip=True)

# Определяем скачанный CSV-файл (проверьте его имя в датасете)
csv_file = os.path.join("calories_dataset/calories.csv")

# Чтение данных
df = pd.read_csv(csv_file)

# Вывод первых 10 строк

os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)

# Разделение данных (80% - train, 20% - test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Сохранение файлов
train_df.to_csv("train/train.csv", index=False)
test_df.to_csv("test/test.csv", index=False)

print("✅ Данные успешно загружены, разделены и сохранены.")
print("📂 train/train.csv")
print("📂 test/test.csv")
