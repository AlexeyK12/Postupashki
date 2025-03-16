from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from airflow.utils.log.logging_mixin import LoggingMixin

# MLflow
mlflow.set_tracking_uri('http://localhost:5000')

def train_and_log_model(iterations, depth):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        mlflow.log_param('iterations', iterations)
        mlflow.log_param('depth', depth)

        model = CatBoostClassifier(iterations=iterations, depth=depth, verbose=0)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1', f1)

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
        client.set_model_version_tag(
            name="IrisApp",
            version=registered_model.version,
            key='precision',
            value=str(round(precision, 3))
        )
        client.set_model_version_tag(
            name="IrisApp",
            version=registered_model.version,
            key='recall',
            value=str(round(recall, 3))
        )
        client.set_model_version_tag(
            name="IrisApp",
            version=registered_model.version,
            key='f1',
            value=str(round(f1, 3))
        )

def promote_best_model(model_name):
    client = MlflowClient()
    best_accuracy = 0
    best_precision = 0
    best_recall = 0
    best_f1 = 0
    best_version = None

    for version in client.search_model_versions(f"name='{model_name}'"):
        tmp_accuracy = version.tags.get("accuracy")
        tmp_precision = version.tags.get("precision")
        tmp_recall = version.tags.get("recall")
        tmp_f1 = version.tags.get("f1")

        if tmp_accuracy:
            tmp_accuracy = float(tmp_accuracy)
            tmp_precision = float(tmp_precision)
            tmp_recall = float(tmp_recall)
            tmp_f1 = float(tmp_f1)

            if tmp_accuracy > best_accuracy:
                best_accuracy = tmp_accuracy
                best_precision = tmp_precision
                best_recall = tmp_recall
                best_f1 = tmp_f1
                best_version = version
            elif tmp_accuracy == best_accuracy:
                if tmp_precision > best_precision:
                    best_accuracy = tmp_accuracy
                    best_precision = tmp_precision
                    best_recall = tmp_recall
                    best_f1 = tmp_f1
                    best_version = version
                elif tmp_precision == best_precision:
                    if tmp_recall > best_recall:
                        best_accuracy = tmp_accuracy
                        best_precision = tmp_precision
                        best_recall = tmp_recall
                        best_f1 = tmp_f1
                        best_version = version
                    elif tmp_recall == best_recall:
                        if tmp_f1 > best_f1:
                            best_accuracy = tmp_accuracy
                            best_precision = tmp_precision
                            best_recall = tmp_recall
                            best_f1 = tmp_f1
                            best_version = version

    if best_version:
        client.transition_model_version_stage(
            name=best_version.name,
            version=best_version.version,
            stage="Production"
        )
        LoggingMixin().log.info(f'Version {best_version.version} promoted acc_{best_accuracy}, prec_{best_precision}, rec_{best_recall}, f1_{best_f1}')
    else:
        LoggingMixin().log.info('No model')

# DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 10, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    "best_model_pipeline_ak12",
    default_args=default_args,
    schedule="@weekly",
) as dag:
    train_task_1 = PythonOperator(
        task_id='train_model_1',
        python_callable=train_and_log_model,
        op_kwargs={'iterations': 100, 'depth': 5},
    )

    train_task_2 = PythonOperator(
        task_id='train_model_2',
        python_callable=train_and_log_model,
        op_kwargs={'iterations': 200, 'depth': 12},
    )

    train_task_3 = PythonOperator(
        task_id='train_model_3',
        python_callable=train_and_log_model,
        op_kwargs={'iterations': 500, 'depth': None},
    )

    promote_task = PythonOperator(
        task_id='promote_best_model',
        python_callable=promote_best_model,
        op_kwargs={'model_name': 'IrisApp'},
    )

    [train_task_1, train_task_2, train_task_3] >> promote_task