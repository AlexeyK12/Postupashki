from airflow import DAG
from datetime import datetime, timedelta

import mlflow

from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator


def train_and_log_model(n_estimators, max_depth):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
    with mlflow.start_run():
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('max_depth', max_depth)

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric('accuracy', accuracy)

        run_id = mlflow.active_run().info.run_id

        model_uri = f"runs:/{run_id}/iris-app"
        registered_model = mlflow.register_model(model_uri, "IrisApp")

        client = MlflowClient()
        client.set_model_version_tag(
            name="IrisApp",
            version=registered_model.version,
            key='accuracy',
            value=str(round(accuracy, 3))
        )


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 10 ,1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    "ml_pipeline",
    default_args=default_args,
    schedule_interval="@weekly"
) as dag:
    train_task = PythonOperator(
        task_id='train_and_log_model',
        python_callable=train_and_log_model,
    ),

    build_image_task = BashOperator(
        task_id='build_docker_image',
        bash_command="docker build -t iris-model-app:latest . && docker tag artmakar/iris-model-app:latest iris-model-app:latest && docker push artmakar/iris-model-app:latest"
    )

    deploy_to_k8s_task = BashOperator(
        task_id = 'deploy_to_k8s',
        bash_command='kubectl apply -f deployment.yaml && kubectl apply -f service.yaml && kubecttl apply -f ingress.yaml'
    )

    train_task >> build_image_task >> deploy_to_k8s_task