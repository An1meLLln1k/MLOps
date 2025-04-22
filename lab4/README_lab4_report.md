# Лабораторная работа 4: Работа с DVC

## 📌 Цель
Изучение утилиты DVC (Data Version Control) на примере версионирования датасета Titanic. Демонстрация всех основных операций: добавление данных, создание версий, настройка удалённого хранилища и переключение между версиями.

---

## 📁 Структура проекта
```
lab4/
├── dataset.py           # Исходный датасет Titanic
├── dataset_v2.py        # Версия 2: только Pclass, Sex, Age
├── dataset_v3.py        # Версия 3: заполнение NaN в Age
├── dataset_v4.py        # Версия 4: one-hot encoding для Sex
├── titanic.csv          # Сам файл данных (переопределяется скриптами)
├── titanic.csv.dvc      # Файл DVC-трекинга
├── .dvcignore
└── .gitignore
```

---

## ⚙️ Шаги по выполнению

### 1. Инициализация репозитория
```bash
git init
dvc init
```

### 2. Настройка удалённого хранилища
#### 🔸 Вариант 1: Google Drive
```Не получилось. DVC использует неофициальное приложение для авторизации в Google Drive, и Google блокирует его по умолчанию.
![image](https://github.com/user-attachments/assets/955ac8a6-9cfd-420c-b98a-f3625a1e0b22)

```

#### 🔸 Вариант 2: Локально
```bash
mkdir dvc_storage
dvc remote add -d localremote ./dvc_storage
```

---

### 3. Версионирование данных

#### 🟢 Версия 1: исходный датасет
```bash
python3 dataset.py
dvc add titanic.csv
git add titanic.csv.dvc .gitignore
git commit -m "v1: Исходный датасет Titanic"
dvc push
```

#### 🔵 Версия 2: фильтрация признаков
```bash
python3 dataset_v2.py
dvc add titanic.csv
git commit -am "v2: Только Pclass, Sex, Age"
dvc push
```

#### 🟡 Версия 3: заполнение NaN
```bash
python3 dataset_v3.py
dvc add titanic.csv
git commit -am "v3: Заполнены пропущенные значения Age средним"
dvc push
```

#### 🟣 Версия 4: one-hot encoding
```bash
python3 dataset_v4.py
dvc add titanic.csv
git commit -am "v4: One-hot encoding для Sex"
dvc push
```

---

## 🔄 Переключение между версиями
```bash
git log --oneline          # Узнать ID нужного коммита
git checkout <commit_id>
dvc checkout
```

---

## ✅ Заключительный коммит
```bash
cd ..
git add lab4
git commit -m "Лабораторная 4: Работа с DVC"
git push -u origin main
```

---

## 🔁 Pull Request
Создан pull request в (https://github.com/kimigara1337/MLOps)
