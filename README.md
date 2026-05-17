# Лабораторные по методам оптимизации

## Где что лежит

```text
optimization_methods/  # код методов
labs/                  # отдельные лабораторные работы
app.py                 # приложение на Streamlit
tests/                 # тесты
outputs/               # графики после запуска labs
```

Основные файлы с методами:

- `optimization_methods/one_dimensional.py` — одномерные методы;
- `optimization_methods/multidimensional.py` — многомерные методы;
- `optimization_methods/constrained.py` — методы штрафов;
- `optimization_methods/parsing.py` — разбор формул и производные через `SymPy`.

Лабораторные лежат в папке `labs/`.

## Запуск через uv

Установка:

```bash
uv sync
```

Запуск приложения:

```bash
uv run streamlit run app.py
```

Пример запуска лабораторной:

```bash
uv run python labs/lab01_passive_search.py
```

## Запуск через pip

Установка:

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

Запуск приложения:

```bash
streamlit run app.py
```

Пример запуска лабораторной:

```bash
python labs/lab01_passive_search.py
```

После запуска приложения Streamlit покажет локальный адрес, обычно:

```text
http://localhost:8501
```
