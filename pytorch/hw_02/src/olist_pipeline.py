"""Этапы пайплайна Olist: extract -> validate -> prepare -> train.

Адреса и параметры берутся из окружения: DATABASE_URL, MLFLOW_TRACKING_URI,
OLIST_WORKDIR, OLIST_CUTOFF, OLIST_EPOCHS, OLIST_SEED.
Каждый запуск DAG пишет файлы в свой каталог OLIST_WORKDIR/runs/<run_key>,
поэтому повторный запуск не дописывает строки к старому снимку.
Через XCom идут только пути, идентификаторы и короткие метаданные.
"""
import hashlib
import os
import re
from pathlib import Path

REQUIRED_COLUMNS = [
    'order_id', 'order_purchase_timestamp', 'price', 'freight', 'weight', 'n_items', 'installments',
    'est_days', 'customer_state', 'seller_state', 'category', 'payment_type', 'late',
]
NUMERIC_COLUMNS = ['price', 'freight', 'weight', 'n_items', 'installments', 'est_days', 'late']
TEXT_COLUMNS = ['order_id', 'customer_state', 'seller_state', 'category', 'payment_type']
LR = 0.003
EXPERIMENT = 'olist_late_delivery'


def _run_dir(run_key):
    """Отдельный каталог запуска; run_id Airflow содержит ':' и '+', заменяем их."""
    safe_key = re.sub(r'[^A-Za-z0-9._-]', '_', str(run_key))
    return Path(os.environ.get('OLIST_WORKDIR', '/opt/olist/work')) / 'runs' / safe_key


def _write_csv(df, path):
    """Запись через временный файл: при повторе задачи снимок заменяется целиком."""
    tmp = path.with_suffix('.tmp')
    df.to_csv(tmp, index=False, encoding='utf-8')
    os.replace(tmp, path)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def extract(run_key):
    """Прочитать public.orders, сохранить обычный UTF-8 CSV; вернуть абсолютный путь строкой."""
    import pandas as pd
    from sqlalchemy import create_engine

    run_dir = _run_dir(run_key)
    run_dir.mkdir(parents=True, exist_ok=True)
    engine = create_engine(os.environ['DATABASE_URL'])
    query = f"SELECT {', '.join(REQUIRED_COLUMNS)} FROM public.orders ORDER BY order_purchase_timestamp, order_id"
    try:
        with engine.connect() as connection:
            df = pd.read_sql_query(query, connection)
    finally:
        engine.dispose()
    snapshot = run_dir / 'snapshot.csv'
    _write_csv(df, snapshot)
    return str(snapshot.resolve())


def validate(snapshot):
    """Проверить файл snapshot до обучения; вернуть путь или вызвать исключение.

    Только чтение CSV: без записей, mkdir/chdir, процессов, сети и БД.
    """
    import numpy as np
    import pandas as pd

    path = Path(snapshot)
    if path.suffix != '.csv' or not path.is_file():
        raise ValueError(f'Ожидается существующий файл .csv: {snapshot}')
    df = pd.read_csv(path, dtype={column: str for column in TEXT_COLUMNS})
    if df.empty:
        raise ValueError('Снимок пустой')
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f'Нет обязательных столбцов: {missing}')
    if df['order_id'].isna().any() or not df['order_id'].is_unique:
        raise ValueError(f"order_id пустой или повторяется: дублей {int(df['order_id'].duplicated().sum())}")
    numbers = df[NUMERIC_COLUMNS].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
    if not np.isfinite(numbers).all():
        bad = [c for c, ok in zip(NUMERIC_COLUMNS, np.isfinite(numbers).all(axis=0)) if not ok]
        raise ValueError(f'Нечисловые или бесконечные значения в столбцах: {bad}')
    if not df['late'].isin([0, 1]).all():
        raise ValueError('Метка late должна быть 0 или 1')
    dates = pd.to_datetime(df['order_purchase_timestamp'], format='ISO8601', errors='coerce')
    if dates.isna().any():
        raise ValueError(f'Неразборчивая дата покупки в {int(dates.isna().sum())} строках')
    if df[TEXT_COLUMNS[1:]].isna().any().any():
        raise ValueError('Пропуски в категориальных столбцах')
    return str(snapshot)


def prepare(snapshot):
    """Разделить данные по времени, как в ноутбуке; вернуть пути и метаданные для train."""
    import pandas as pd
    from olist_model import split_by_time

    cutoff = os.environ.get('OLIST_CUTOFF', '2017-07-31')
    path = Path(snapshot)
    snapshot_sha256 = _sha256(path)       # байты исходного CSV из extract, до подготовки выборок
    train_df, valid_df = split_by_time(pd.read_csv(path), cutoff=cutoff)
    train_path, valid_path = path.parent / 'train.csv', path.parent / 'validation.csv'
    _write_csv(train_df, train_path)
    _write_csv(valid_df, valid_path)
    return {
        'snapshot': str(path), 'snapshot_sha256': snapshot_sha256, 'cutoff': cutoff,
        'train': str(train_path), 'validation': str(valid_path),
        'train_rows': len(train_df), 'validation_rows': len(valid_df),
    }


def train(prepared, *, dag_run_id, epochs=None, seed=None):
    """Обучить модель, записать параметры, метрики по эпохам и полную модель в MLflow; вернуть run_id."""
    import mlflow
    import pandas as pd
    from olist_model import fit_model, save_predictor

    epochs = int(os.environ.get('OLIST_EPOCHS', '5')) if epochs is None else int(epochs)
    seed = int(os.environ.get('OLIST_SEED', '42')) if seed is None else int(seed)
    train_df, valid_df = pd.read_csv(prepared['train']), pd.read_csv(prepared['validation'])
    predictor, history = fit_model(train_df, valid_df, epochs=epochs, seed=seed, lr=LR)
    model_dir = Path(prepared['train']).parent / 'model'
    save_predictor(predictor, model_dir)

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f'olist {dag_run_id}') as run:
        mlflow.set_tags({'dag_run_id': dag_run_id, 'snapshot_path': prepared['snapshot']})
        mlflow.log_params({
            'epochs': epochs, 'seed': seed, 'lr': LR, 'cutoff': prepared['cutoff'],
            'train_rows': prepared['train_rows'], 'validation_rows': prepared['validation_rows'],
            'snapshot_sha256': prepared['snapshot_sha256'],
        })
        for row in history:
            mlflow.log_metrics({'train_loss': row['train_loss'], 'validation_pr_auc': row['validation_pr_auc']},
                               step=row['epoch'])
        mlflow.log_artifacts(str(model_dir), artifact_path='model')
        return run.info.run_id
