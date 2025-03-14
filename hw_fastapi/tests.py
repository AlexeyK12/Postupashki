import unittest
import requests

BASE_URL = "http://localhost:5000"

class TestTodoAPI(unittest.TestCase):
    def setUp(self):
        # Восстановление начального состояния списка задач перед каждым тестом
        requests.post(f"{BASE_URL}/tasks", json={"title": "Initial Task", "description": "This is an initial task"})

    def tearDown(self):
        # Очистка списка задач после каждого теста
        response = requests.get(f"{BASE_URL}/tasks")
        for task in response.json():
            requests.delete(f"{BASE_URL}/tasks/{task['id']}")

    def test_get_all_tasks(self):
        response = requests.get(f"{BASE_URL}/tasks")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)

    def test_get_task_by_id(self):
        # Создаем задачу и получаем её ID
        new_task = {"title": "Test Task", "description": "This is a test task"}
        create_response = requests.post(f"{BASE_URL}/tasks", json=new_task)
        task_id = create_response.json()["id"]

        # Получаем задачу по ID
        response = requests.get(f"{BASE_URL}/tasks/{task_id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], task_id)

    def test_create_task(self):
        new_task = {"title": "New Task", "description": "This is a new task"}
        response = requests.post(f"{BASE_URL}/tasks", json=new_task)
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["title"], "New Task")

    def test_update_task(self):
        # Создаем задачу и получаем её ID
        new_task = {"title": "Test Task", "description": "This is a test task"}
        create_response = requests.post(f"{BASE_URL}/tasks", json=new_task)
        task_id = create_response.json()["id"]

        # Обновляем задачу
        updated_task = {"title": "Updated Task", "description": "This task has been updated"}
        response = requests.put(f"{BASE_URL}/tasks/{task_id}", json=updated_task)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["title"], "Updated Task")

    def test_delete_task(self):
        # Создаем задачу и получаем её ID
        new_task = {"title": "Test Task", "description": "This is a test task"}
        create_response = requests.post(f"{BASE_URL}/tasks", json=new_task)
        task_id = create_response.json()["id"]

        # Удаляем задачу
        response = requests.delete(f"{BASE_URL}/tasks/{task_id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["result"], True)

if __name__ == "__main__":
    unittest.main()
