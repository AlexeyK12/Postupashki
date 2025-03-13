FROM python:3.12.7

WORKDIR /hw_kuber

# копируем все файлы в контейнер
COPY . /hw_kuber

# устанавливаем зависимости из requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# открываем порт 5000
EXPOSE 5000

# запускаем Flask-приложение
CMD ["python", "app.py"]