# DZ 1

Задание: Создание REST API на Flask для управления списком задач (To-Do List)

# Цель

Разработать REST API на Flask, которое поддерживает CRUD операции (Create, Read, Update, Delete) для управления списком задач. API должно быть протестировано с использованием модуля unittest.

# Требования

Используйте Flask для создания REST API.  
Поддержите следующие ручки:  
```
GET /tasks — возвращает список всех задач.  
GET /tasks/<int:task_id> — возвращает задачу по её ID.  
POST /tasks — создаёт новую задачу (тело запроса должно содержать title и description).  
PUT /tasks/<int:task_id> — обновляет задачу по её ID.  
DELETE /tasks/<int:task_id> — удаляет задачу по её ID.  
```
Пример структуры данных для задачи:  

```python
tasks = [
    {"id": 1, "title": "Learn Flask", "description": "Study Flask framework"},
    {"id": 2, "title": "Build a REST API", "description": "Create a simple REST API"}
]
```

# Тестирование 

Запустите команду
`python -m unittest tests.py`
