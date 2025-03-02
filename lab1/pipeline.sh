#!/bin/bash

echo "🚀 Запуск полного пайплайна машинного обучения..."

# Создание виртуального окружения, если его нет
if [ ! -d "venv" ]; then
    echo "🔧 Создаём виртуальное окружение..."
    python3 -m venv venv
fi

# Активация виртуального окружения
source venv/bin/activate

# Установка зависимостей
echo "📦 Устанавливаем зависимости..."
pip install --upgrade pip
pip install pandas numpy scikit-learn joblib kaggle

# Запуск Python-скриптов последовательно
echo "📂 Запускаем data_creation.py..."
python3 data_creation.py || { echo "❌ Ошибка в data_creation.py"; exit 1; }

echo "🔄 Запускаем data_preprocessing.py..."
python3 data_preprocessing.py || { echo "❌ Ошибка в data_preprocessing.py"; exit 1; }

echo "🧠 Обучение модели в model_preparation.py..."
python3 model_preparation.py || { echo "❌ Ошибка в model_preparation.py"; exit 1; }

echo "📝 Тестирование модели в model_testing.py..."
TEST_OUTPUT=$(python3 model_testing.py)

# Извлекаем R² Score из вывода Python-скрипта
R2_SCORE=$(echo "$TEST_OUTPUT" | grep "R² Score" | awk '{print $NF}')

# Вывод финального результата
echo "✅ Model test R² Score is: $R2_SCORE"

# Деактивация виртуального окружения
deactivate
