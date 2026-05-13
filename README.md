# Лабораторные по методам оптимизации

Проект с реализациями методов оптимизации на Python, отдельными файлами-лабораторными и минимальным приложением для запуска методов через интерфейс.

## Установка

В проекте используется `uv`. Из корня проекта:

```bash
uv sync
```

После этого `uv` сам создаст локальное окружение `.venv` и поставит зависимости из `pyproject.toml` / `uv.lock`.

## Структура проекта

```text
optimization_methods/  # чистые реализации методов
labs/                  # отдельные лабораторные запуски
tests/                 # тесты корректности
app.py                 # Streamlit-приложение
outputs/               # сюда сохраняются графики после запуска labs
```

Главная идея такая:

- `optimization_methods/` — код самих алгоритмов;
- `labs/` — готовые демонстрации для запуска из терминала;
- `app.py` — обертка, где функцию и метод можно выбрать руками;
- `tests/` — проверки, что методы сходятся на простых примерах.

## Основные 13 методов для сдачи

После уточнения про "13 методов строго по методичке" основной список для сдачи лучше держать таким:

1. метод пассивного поиска;
2. метод дихотомии;
3. метод золотого сечения;
4. метод Фибоначчи;
5. метод покоординатного спуска;
6. градиентный метод с постоянным шагом;
7. градиентный метод с дроблением шага;
8. метод наискорейшего градиентного спуска;
9. метод Флетчера-Ривза;
10. модифицированный метод Ньютона;
11. метод внешних штрафов;
12. метод внутренних штрафов;
13. метод условного градиента.

Дополнительно в коде есть градиентный метод с заранее заданным шагом и метод Полака-Рибьера. Они тоже описаны в методичке, но если нужен именно лимит в 13 методов, их можно не включать в основной список сдачи.

## Запуск отдельных лабораторных

Одномерная минимизация:

```bash
uv run python labs/lab01_passive_search.py
uv run python labs/lab02_dichotomy.py
uv run python labs/lab03_golden_section.py
uv run python labs/lab04_fibonacci.py
```

Многомерная минимизация:

```bash
uv run python labs/lab05_coordinate_descent.py
uv run python labs/lab06_gradient_fixed_step.py
uv run python labs/lab07_gradient_backtracking.py
uv run python labs/lab09_steepest_descent.py
uv run python labs/lab10_conjugate_gradient.py
uv run python labs/lab11_newton.py
```

Условная оптимизация:

```bash
uv run python labs/lab14_external_penalty.py
uv run python labs/lab15_internal_penalty.py
uv run python labs/lab16_conditional_gradient.py
```

Дополнительные запуски:

```bash
uv run python labs/lab08_gradient_scheduled_step.py
uv run python labs/lab12_compare_multidimensional.py
uv run python labs/lab13_polak_ribiere.py
```

После запуска лабораторных графики сохраняются в папку `outputs/`.

## Запуск приложения

```bash
uv run streamlit run app.py
```

После запуска Streamlit покажет ссылку вида:

```text
http://localhost:8501
```

В приложении можно:

- ввести функцию;
- выбрать метод;
- задать интервал или начальную точку;
- для методов условной оптимизации задать ограничения;
- посмотреть результат;
- для двумерных функций увидеть траекторию минимизации и направления антиградиента.

## Как задавать функции

Функции пишутся в синтаксисе Python/SymPy:

```text
x + 2/x
x**2 + sin(x)
4*x**2 + y**2 + x*y + 0.25*sin(x + y)**2
```

Доступны стандартные функции:

```text
sin, cos, tan, exp, log, sqrt, Abs
```

## Запуск метода без редактирования labs

Одномерный пример прямо из терминала:

```bash
uv run python -c 'from optimization_methods import build_scalar_function, golden_section_search; p = build_scalar_function("x**2 + sin(x)"); r = golden_section_search(p.f, -3, 3, eps=1e-5); print(r.x_min, r.f_min)'
```

Многомерный пример:

```bash
uv run python -c 'from optimization_methods import build_multivariate_function, gradient_descent; p = build_multivariate_function("4*x**2 + y**2 + x*y + 0.25*sin(x+y)**2", ["x", "y"]); r = gradient_descent(p.f, p.gradient, [1.5, -1.0], strategy="steepest"); print(r.x_min, r.f_min)'
```

## Проверка проекта

Тесты:

```bash
uv run pytest
```

Линтер:

```bash
uv run ruff check .
```

Полезный быстрый прогон:

```bash
uv run pytest
uv run ruff check .
```
