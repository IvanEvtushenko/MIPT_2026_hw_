"""DAG обучения Olist: extract -> validate -> prepare -> train.

Через XCom идут только путь к снимку, словарь с путями и SHA256 и MLflow run_id.
Тяжёлые импорты (pandas, torch, mlflow) делаются внутри задач, чтобы разбор DAG был быстрым.
"""
from datetime import datetime, timedelta, timezone

from airflow.sdk import dag, get_current_context, task


@dag(
    dag_id='olist_training',
    schedule='@weekly',
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    default_args={'retries': 0, 'execution_timeout': timedelta(minutes=10)},
    tags=['olist'],
)
def olist_training():
    @task(retries=3, retry_delay=timedelta(seconds=30), execution_timeout=timedelta(minutes=5))
    def extract():
        from olist_pipeline import extract as run
        return run(get_current_context()['run_id'])

    @task(retries=0, trigger_rule='all_success', execution_timeout=timedelta(minutes=5))
    def validate(snapshot):
        from olist_pipeline import validate as run
        return run(snapshot)

    @task(trigger_rule='all_success', execution_timeout=timedelta(minutes=5))
    def prepare(snapshot):
        from olist_pipeline import prepare as run
        return run(snapshot)

    @task(trigger_rule='all_success', execution_timeout=timedelta(minutes=20))
    def train(prepared):
        from olist_pipeline import train as run
        return run(prepared, dag_run_id=get_current_context()['run_id'])

    train(prepare(validate(extract())))


olist_training()
