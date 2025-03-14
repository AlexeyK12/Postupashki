from flask import Flask, jsonify, request, abort

app = Flask(__name__)

tasks = [
    {"id": 1, "title": "Learn Flask", "description": "Study Flask framework"},
    {"id": 2, "title": "Build a REST API", "description": "Create a simple REST API"}
]

# генератор ID новых задач
def generate_id():
    return max(task['id'] for task in tasks) + 1 if tasks else 1

# ручка для получения всех задач
@app.route('/tasks', methods=['GET'])
def get_tasks():
    return jsonify(tasks)

# ручка для получения задачи по ID
@app.route('/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    task = next((task for task in tasks if task['id'] == task_id), None)
    if task is None:
        abort(404)
    return jsonify(task)

# ручка для создания новой задачи
@app.route('/tasks', methods=['POST'])
def create_task():
    if not request.json or not 'title' in request.json:
        abort(400)
    new_task = {
        'id': generate_id(),
        'title': request.json['title'],
        'description': request.json.get('description', "")
    }
    tasks.append(new_task)
    return jsonify(new_task), 201

# ручка для обновления задачи по ID
@app.route('/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    task = next((task for task in tasks if task['id'] == task_id), None)
    if task is None:
        abort(404)
    if not request.json:
        abort(400)
    task['title'] = request.json.get('title', task['title'])
    task['description'] = request.json.get('description', task['description'])
    return jsonify(task)

# ручка для удаления задачи по ID
@app.route('/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    global tasks
    task = next((task for task in tasks if task['id'] == task_id), None)
    if task is None:
        abort(404)
    tasks = [task for task in tasks if task['id'] != task_id]
    return jsonify({'result': True})

if __name__ == '__main__':
    app.run(debug=True)