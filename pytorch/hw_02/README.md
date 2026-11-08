# ДЗ 2: стартовый набор

Задание находится в [ASSIGNMENT.md](ASSIGNMENT.md). Здесь описаны запуск и настройки сервисов.

В наборе есть данные, модель, PostgreSQL и Airflow. Ваша часть: функции в `src/olist_pipeline.py`, DAG обучения и сервис MLflow. Заготовки функций пока вызывают `NotImplementedError`.

## Запустите окружение

Понадобятся Git, Docker с работающим Linux engine и Docker Compose v2. На Windows используйте Docker Desktop с WSL2 и терминал WSL; на macOS — Docker Desktop или совместимую VM. Python-зависимости работают внутри контейнера, обучение выполняется на CPU.

Для планирования: 8 ГБ RAM на компьютере, около 15 ГБ диска и 4 ГБ RAM с 2 CPU для Docker. Расход ресурсов зависит от платформы и сборки; минимальные требования не измерялись. Набор использует учебный Airflow standalone.

Распакуйте выданный `olist-homework.zip`, откройте каталог `olist-homework/` и создайте свой репозиторий:

```bash
cd olist-homework
cp .env.example .env
docker compose up -d --build
```

Все следующие команды выполняйте из этого каталога. В README своего решения укажите запуск от клонирования, подставив реальный адрес репозитория:

```bash
git clone YOUR_REPOSITORY_URL olist-homework
cd olist-homework
cp .env.example .env
docker compose up -d --build
```

Откройте Airflow: `http://127.0.0.1:8080`. Включите DAG `example_one_task` и нажмите Trigger. Интерфейс работает без пароля; сохраняйте доступ только через `127.0.0.1`.

Если порт занят, измените `AIRFLOW_PORT` в `.env` и пересоздайте сервис. Порты из `.env` относятся к компьютеру; внутри Docker используются порты сервисов.

Проверьте состояние, логи и число заказов:

```bash
docker compose ps
docker compose logs --tail=100 airflow
docker compose exec -T source psql -U olist -d olist -c 'SELECT count(*), count(DISTINCT order_id) FROM orders;'
```

PostgreSQL загружает данные при создании нового volume. При обычном перезапуске данные сохраняются. Пароль из `.env.example` предназначен для локальной учебной БД.

## Запустите готовое обучение

Ноутбук читает `data/olist_orders.csv.gz`. Выполните его код в контейнере:

```bash
docker compose exec -T airflow python /opt/olist/src/run_notebook.py
```

Результат появится в `/opt/olist/work/notebook/`. Для интерактивной работы откройте `prototype.ipynb` в Jupyter с окружением из Dockerfile.

Описание столбцов находится в `data/README.md`, версия набора — в `data/metadata.json`. Используйте идентификатор заказа только для проверки уникальности. Модель и преобразования оставьте выданными.

## Добавьте MLflow

Добавьте сервис `mlflow` в `compose.yaml`:

- образ `ghcr.io/mlflow/mlflow:v3.16.1`;
- команда `mlflow server --host 0.0.0.0 --port 5000 --workers 1 --backend-store-uri sqlite:////mlflow/mlflow.db --artifacts-destination /mlflow/artifacts --serve-artifacts --allowed-hosts mlflow:5000,localhost:*,127.0.0.1:*`;
- переменная `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`;
- именованный Docker volume на `/mlflow` для SQLite и артефактов;
- лимит памяти 768 МБ;
- публикация `127.0.0.1:${MLFLOW_PORT:-5050}:5000`.

Для healthcheck используйте `python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=5).read()"`. Задайте интервал, timeout и число попыток с запасом на первый старт.

У Airflow сохраните зависимость от `source` и добавьте зависимость от MLflow:

```yaml
depends_on:
  # Здесь также остаётся существующая зависимость source.
  mlflow:
    condition: service_healthy
```

Добавляйте сервис и зависимость вместе. Airflow дождётся успешного healthcheck MLflow. После запуска проверьте `docker compose ps`. Из контейнера Airflow выполните ту же Python-команду с адресом `http://mlflow:5000/health`.

Адреса MLflow:

| Откуда подключаться | Адрес |
|---|---|
| Браузер на компьютере | `http://127.0.0.1:5050` |
| Код внутри Airflow | `http://mlflow:5000` |

В интерфейсе откройте Model training слева сверху. Внутри контейнера `localhost` указывает на этот контейнер. Для соседнего сервиса используйте его имя в Docker.

Код Airflow получает адрес PostgreSQL из `DATABASE_URL`, адрес MLflow — из `MLFLOW_TRACKING_URI`. Доступ к PostgreSQL с компьютера выполняется через `docker compose exec source psql`.

## Сохраните и загрузите модель

`save_predictor` сохраняет веса и параметры преобразования признаков в одной директории. Загрузите всю директорию в артефакты MLflow run под именем `model`.

Для загрузки используйте `mlflow.artifacts.download_artifacts` с URI `runs:/RUN_ID/model`, затем `load_predictor`. Готовая проверка выполняет эти действия и выводит пять прогнозов в JSON:

```bash
docker compose exec -T airflow python -m check_model --run-id RUN_ID
```

Подставьте MLflow `run_id` и сохраните точную команду в своём README.

Имена параметров и метрик перечислены в части 4 задания. Тег `dag_run_id` берите из контекста текущего запуска Airflow. Идентификатор MLflow run возвращает этап `train`.

`extract` возвращает строку с абсолютным путём к обычному UTF-8 CSV (`.csv`). Вычисляйте `snapshot_sha256` по исходным байтам этого файла через `hashlib.sha256` до подготовки train/validation. Дальше путь и хеш можно передать небольшим словарём, например из `prepare`.

## Где писать решение

| Файл | Что добавить |
|---|---|
| `src/olist_pipeline.py` | Четыре этапа |
| `dags/olist_training.py` | DAG обучения |
| `compose.yaml` | MLflow и зависимость Airflow |
| README | Команды запуска, проверки модели и результат Clear |

Каталоги `src/` и `dags/` подключены в контейнер: правки Python применяются без пересборки образа. После изменения зависимостей пересоберите образ.

Храните рабочие файлы в постоянном `/opt/olist/work`. Передавайте через XCom только пути, идентификаторы и хеши. Формат результата `prepare` выберите сами.

`validate(snapshot)` только читает CSV, проверяет все 13 столбцов и данные, возвращает путь либо вызывает исключение. Запись файлов, создание каталогов, смена рабочего каталога, сеть, БД и запуск процессов запрещены. Выданные `data/`, `src/olist_model.py` и `src/check_model.py` оставьте без изменений.

У `validate`, `prepare`, `train` сохраните `trigger_rule='all_success'`. Для `validate` задайте `retries=0`.

Проверьте два случая:

- Испорченная копия снимка останавливает DAG на `validate`. Временно верните путь к ней из `extract`, запустите DAG, затем восстановите код.
- Исключение в `train` исправляется через Clear только этой задачи. Повторно запускается только `train`.

По желанию добавьте настройку `epochs` и `seed` через Airflow Params или `dag_run.conf`. Условия бонуса указаны в задании.

## Перезапуск и остановка

```bash
docker compose restart       # результаты и данные сохраняются
docker compose up -d --force-recreate --no-deps mlflow  # новый контейнер, прежний именованный volume
docker compose down          # останавливает и удаляет контейнеры, но не volumes
```

После добавления MLflow проверьте сохранность модели: пересоздайте его контейнер, дождитесь healthy и выполните прогноз по прежнему `run_id`.

`docker compose down -v` удаляет локальную БД, историю Airflow и модели. Для проверки сохранности используйте команды выше.

## Лицензия данных

Условия находятся в `data/LICENSE.md`. Ограничение NonCommercial требует от организатора коммерческого курса отдельно проверить право использования данных.
