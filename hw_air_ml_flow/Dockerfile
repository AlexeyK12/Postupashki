# Используем базовый образ Python
FROM python:3.10

# Устанавливаем рабочую директорию
WORKDIR /airflow

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Создаем папку для DAG-файлов
RUN mkdir -p dags

# Копируем исходный код
COPY . .

# Копируем DAG-файл в папку dags
COPY dag_v2.py dags/

# Устанавливаем переменные окружения Airflow
ENV AIRFLOW_HOME=/airflow
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__CORE__DAGS_FOLDER=/airflow/dags

# Инициализируем базу данных Airflow
RUN airflow db init

# Создаем пользователя Airflow
RUN airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname Admin \
    --role Admin \
    --email alkud12@yandex.ru

# Открываем порты
EXPOSE 8080
EXPOSE 5000

# Запускаем Airflow и MLflow
CMD ["sh", "-c", "airflow webserver & mlflow ui --host 0.0.0.0"]