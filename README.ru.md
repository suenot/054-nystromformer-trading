# Глава 56: Nyströmformer для трейдинга

Эта глава исследует **Nyströmformer** — эффективную архитектуру трансформера, которая использует метод Нистрёма для аппроксимации self-attention с линейной сложностью O(n) вместо стандартной квадратичной O(n²). Это делает её идеальной для обработки длинных финансовых временных рядов: тиковых данных, снимков книги ордеров и расширенных исторических последовательностей.

<p align="center">
<img src="https://i.imgur.com/8KqPL4v.png" width="70%">
</p>

## Содержание

1. [Введение в Nyströmformer](#введение-в-nyströmformer)
    * [Проблема квадратичного внимания](#проблема-квадратичного-внимания)
    * [Обзор метода Нистрёма](#обзор-метода-нистрёма)
    * [Ключевые преимущества для трейдинга](#ключевые-преимущества-для-трейдинга)
2. [Математические основы](#математические-основы)
    * [Стандартное self-attention](#стандартное-self-attention)
    * [Аппроксимация Нистрёма](#аппроксимация-нистрёма)
    * [Выбор опорных точек](#выбор-опорных-точек)
    * [Итеративный псевдообратный](#итеративный-псевдообратный)
3. [Архитектура Nyströmformer](#архитектура-nyströmformer)
    * [Segment-Means для опорных точек](#segment-means-для-опорных-точек)
    * [Трёхматричная декомпозиция](#трёхматричная-декомпозиция)
    * [Анализ сложности](#анализ-сложности)
4. [Применение в трейдинге](#применение-в-трейдинге)
    * [Предсказание цен на длинных последовательностях](#предсказание-цен-на-длинных-последовательностях)
    * [Анализ книги ордеров](#анализ-книги-ордеров)
    * [Объединение нескольких таймфреймов](#объединение-нескольких-таймфреймов)
5. [Практические примеры](#практические-примеры)
    * [01: Подготовка данных](#01-подготовка-данных)
    * [02: Модель Nyströmformer](#02-модель-nyströmformer)
    * [03: Обучение](#03-обучение)
    * [04: Бэктестинг стратегии](#04-бэктестинг-стратегии)
6. [Реализация на Rust](#реализация-на-rust)
7. [Реализация на Python](#реализация-на-python)
8. [Лучшие практики](#лучшие-практики)
9. [Ресурсы](#ресурсы)

## Введение в Nyströmformer

### Проблема квадратичного внимания

Стандартный механизм self-attention в трансформерах вычисляет оценки внимания между всеми парами токенов, что приводит к сложности O(n²) по времени и памяти. Для последовательности длиной n:

```
Стоимость стандартного Attention:
- Длина 512:    262,144 операций
- Длина 1024:   1,048,576 операций
- Длина 4096:   16,777,216 операций
- Длина 8192:   67,108,864 операций

Стоимость растёт квадратично!
```

В трейдинге часто нужно обрабатывать:
- **Тиковые данные**: Тысячи обновлений цен в минуту
- **Снимки книги ордеров**: Глубокая история уровней bid/ask
- **Корреляции между активами**: Длинные окна просмотра по многим инструментам
- **Высокочастотные признаки**: Потоки данных микросекундного уровня

Стандартные трансформеры становятся непомерно дорогими для этих задач.

### Обзор метода Нистрёма

**Метод Нистрёма** — это классическая техника из численной линейной алгебры для аппроксимации больших матриц с использованием подмножества их столбцов и строк. Изначально разработанный для ядерных методов, он работает следующим образом:

1. **Выбор опорных точек (landmarks)**: Выбираем m представительных точек из n общих (m << n)
2. **Вычисление взаимодействий**: Рассчитываем полное внимание только для связей опорных точек со всеми
3. **Восстановление матрицы**: Аппроксимируем полную матрицу n×n из меньших вычислений

```
┌─────────────────────────────────────────────────────────────────┐
│                    АППРОКСИМАЦИЯ НИСТРЁМА                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Стандартное Attention (n×n):   Аппроксимация Нистрёма:         │
│                                                                  │
│  ┌─────────────────┐           ┌───┬───────────┐               │
│  │█████████████████│           │ A │     B     │  A: m×m       │
│  │█████████████████│    →      ├───┼───────────┤  B: m×(n-m)   │
│  │█████████████████│           │ C │  B·A⁺·C   │  C: (n-m)×m   │
│  │█████████████████│           └───┴───────────┘               │
│  └─────────────────┘                                            │
│                                                                  │
│  Стоимость: O(n²)              Стоимость: O(n·m)                │
│                                                                  │
│  При m=64 и n=4096:  ускорение в 64 раза!                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Ключевые преимущества для трейдинга

| Характеристика | Стандартное Attention | Nyströmformer | Польза для трейдинга |
|----------------|----------------------|---------------|---------------------|
| Сложность | O(n²) | O(n) | Обработка более длинной истории |
| Память | O(n²) | O(n) | Большие размеры батчей |
| Длина последовательности | ~512-2048 | 4096-8192+ | Больше контекста для предсказаний |
| Скорость инференса | Медленно для длинных | Быстро | Обработка в реальном времени |
| Точность | Точная | Почти точная | Минимальная потеря качества |

## Математические основы

### Стандартное self-attention

Стандартный механизм self-attention вычисляет:

```
Attention(Q, K, V) = softmax(QK^T / √d) · V
```

Где:
- **Q** (Query): матрица n × d
- **K** (Key): матрица n × d
- **V** (Value): матрица n × d
- **d**: размерность головы
- **n**: длина последовательности

Softmax применяется построчно к матрице n×n QK^T, что является вычислительным узким местом.

### Аппроксимация Нистрёма

Nyströmformer аппроксимирует матрицу softmax attention **S** = softmax(QK^T/√d) как:

```
Ŝ = F̃ · Ã⁺ · B̃

Где:
- F̃ = softmax(Q · K̃^T / √d)     ∈ ℝ^(n×m)  — Полные запросы к опорным ключам
- Ã = softmax(Q̃ · K̃^T / √d)     ∈ ℝ^(m×m)  — Опорные к опорным
- B̃ = softmax(Q̃ · K^T / √d)     ∈ ℝ^(m×n)  — Опорные запросы к полным ключам
- Ã⁺ — псевдообратная матрица Мура-Пенроуза для Ã
```

Итоговый выход attention:
```
Output = Ŝ · V = F̃ · Ã⁺ · B̃ · V
```

### Выбор опорных точек

**Метод Segment-Means** (используется в Nyströmformer):

Для n токенов делим их на m сегментов размера l = n/m:

```python
# Для запросов Q ∈ ℝ^(n×d)
Q̃[i] = mean(Q[i*l : (i+1)*l])  для i = 0, 1, ..., m-1

# Для ключей K ∈ ℝ^(n×d)
K̃[i] = mean(K[i*l : (i+1)*l])  для i = 0, 1, ..., m-1
```

Это создаёт m опорных запросов Q̃ и m опорных ключей K̃, каждый представляющий сегмент входной последовательности.

```
Входная последовательность (n=8 токенов):
[t₁, t₂, t₃, t₄, t₅, t₆, t₇, t₈]

С m=2 опорными точками:
Сегмент 1: [t₁, t₂, t₃, t₄] → Опорная L₁ = mean([t₁, t₂, t₃, t₄])
Сегмент 2: [t₅, t₆, t₇, t₈] → Опорная L₂ = mean([t₅, t₆, t₇, t₈])
```

### Итеративный псевдообратный

Вычисление точной псевдообратной через SVD дорого на GPU. Nyströmformer использует итеративную аппроксимацию, сходящуюся за ~6 итераций:

```python
def iterative_pinv(A, num_iter=6):
    """
    Итеративная аппроксимация псевдообратной Мура-Пенроуза.
    Основана на итерации Ньютона-Шульца.
    """
    # Инициализация
    Z = A.transpose(-1, -2) / (torch.norm(A) ** 2)
    I = torch.eye(A.shape[-1], device=A.device)

    for _ in range(num_iter):
        Z = 2 * Z - Z @ A @ Z

    return Z
```

### Анализ сложности

| Компонент | Стандартный | Nyströmformer |
|-----------|------------|---------------|
| Вычисление QK^T | O(n²d) | O(nmd) |
| Softmax | O(n²) | O(nm + m²) |
| Псевдообратная | N/A | O(m³) |
| Attention × V | O(n²d) | O(nmd) |
| **Итого** | **O(n²d)** | **O(nmd + m³)** |

При фиксированном m (например, m=64), Nyströmformer достигает **линейной сложности O(n)** по длине последовательности.

## Архитектура Nyströmformer

### Segment-Means для опорных точек

```python
class NystromAttention(nn.Module):
    def __init__(self, d_model, n_heads, num_landmarks=64, seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_landmarks = num_landmarks
        self.seq_len = seq_len
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Размер сегмента для вычисления опорных точек
        self.segment_size = seq_len // num_landmarks

    def compute_landmarks(self, x):
        """
        Вычисление опорных точек методом segment-means.
        x: [batch, n_heads, seq_len, head_dim]
        возвращает: [batch, n_heads, num_landmarks, head_dim]
        """
        batch, n_heads, seq_len, head_dim = x.shape

        # Преобразование в сегменты
        x = x.reshape(
            batch, n_heads,
            self.num_landmarks,
            self.segment_size,
            head_dim
        )

        # Среднее каждого сегмента
        landmarks = x.mean(dim=3)

        return landmarks
```

### Трёхматричная декомпозиция

```python
def nystrom_attention(self, Q, K, V):
    """
    Вычисление внимания с аппроксимацией Нистрёма.

    Q, K, V: [batch, n_heads, seq_len, head_dim]
    """
    # Вычисление опорных запросов и ключей
    Q_landmarks = self.compute_landmarks(Q)  # [batch, n_heads, m, head_dim]
    K_landmarks = self.compute_landmarks(K)  # [batch, n_heads, m, head_dim]

    scale = 1.0 / math.sqrt(self.head_dim)

    # Матрица F̃: Полные запросы к опорным ключам
    # [batch, n_heads, n, m]
    kernel_1 = F.softmax(
        torch.matmul(Q, K_landmarks.transpose(-1, -2)) * scale,
        dim=-1
    )

    # Матрица Ã: Опорные к опорным
    # [batch, n_heads, m, m]
    kernel_2 = F.softmax(
        torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)) * scale,
        dim=-1
    )

    # Матрица B̃: Опорные запросы к полным ключам
    # [batch, n_heads, m, n]
    kernel_3 = F.softmax(
        torch.matmul(Q_landmarks, K.transpose(-1, -2)) * scale,
        dim=-1
    )

    # Вычисление псевдообратной kernel_2
    kernel_2_inv = self.iterative_pinv(kernel_2)

    # Эффективное вычисление: Никогда не материализуем матрицу n×n
    # Шаг 1: B̃ @ V → [batch, n_heads, m, head_dim]
    context_1 = torch.matmul(kernel_3, V)

    # Шаг 2: Ã⁺ @ (B̃ @ V) → [batch, n_heads, m, head_dim]
    context_2 = torch.matmul(kernel_2_inv, context_1)

    # Шаг 3: F̃ @ (Ã⁺ @ B̃ @ V) → [batch, n_heads, n, head_dim]
    output = torch.matmul(kernel_1, context_2)

    return output
```

### Анализ сложности

```
Сравнение памяти и вычислений для seq_len=4096, num_landmarks=64:

Стандартное Attention:
- QK^T: 4096 × 4096 = 16,777,216 элементов
- Память: ~64MB на голову внимания (fp32)
- Вычисления: O(n²) = O(16.7M)

Nyströmformer:
- F̃: 4096 × 64 = 262,144 элементов
- Ã: 64 × 64 = 4,096 элементов
- B̃: 64 × 4096 = 262,144 элементов
- Итого: ~528,384 элементов
- Память: ~2MB на голову внимания (fp32)
- Вычисления: O(n·m) = O(262K)

Снижение: ~32× меньше памяти, ~64× меньше вычислений
```

## Применение в трейдинге

### Предсказание цен на длинных последовательностях

Nyströmformer отлично справляется с обработкой расширенных историй цен:

```python
class NystromPricePredictor(nn.Module):
    """
    Предсказание будущих движений цен с использованием
    длинных исторических последовательностей.

    Применения:
    - Внутридневная торговля с минутными данными (4096 мин = ~68 часов)
    - Потиковый анализ (8192 тиков = расширенная глубина рынка)
    - Многодневные паттерны с часовыми данными
    """
    def __init__(
        self,
        input_dim: int = 6,        # OHLCV + признаки
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        num_landmarks: int = 64,
        seq_len: int = 4096,
        pred_horizon: int = 24,    # Предсказание на 24 шага вперёд
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, seq_len)

        self.layers = nn.ModuleList([
            NystromEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                num_landmarks=num_landmarks,
                seq_len=seq_len,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        self.pred_head = nn.Linear(d_model, pred_horizon)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x = self.input_proj(x)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x)

        # Используем последнюю позицию для предсказания
        output = self.pred_head(x[:, -1, :])

        return output
```

### Анализ книги ордеров

Эффективная обработка глубокой истории книги ордеров:

```python
class NystromLOBModel(nn.Module):
    """
    Анализ Limit Order Book (LOB) с использованием Nyströmformer.

    Обрабатывает длинные последовательности снимков книги ордеров:
    - Множество ценовых уровней (10-50 уровней на сторону)
    - Временные зависимости через тысячи снимков
    - Паттерны внимания между уровнями
    """
    def __init__(
        self,
        n_levels: int = 20,        # Ценовых уровней на сторону
        d_model: int = 128,
        n_heads: int = 4,
        num_landmarks: int = 32,
        seq_len: int = 2048,       # Снимков книги ордеров
    ):
        super().__init__()

        # Каждый снимок: [bid_prices, bid_volumes, ask_prices, ask_volumes]
        input_dim = n_levels * 4

        self.input_proj = nn.Linear(input_dim, d_model)

        self.nystrom_attention = NystromAttention(
            d_model=d_model,
            n_heads=n_heads,
            num_landmarks=num_landmarks,
            seq_len=seq_len
        )

        # Предсказание: направление mid-price, изменение спреда, дисбаланс объёма
        self.output_head = nn.Linear(d_model, 3)
```

### Объединение нескольких таймфреймов

Эффективное объединение сигналов с разных таймфреймов:

```python
class MultiTimeframeNystrom(nn.Module):
    """
    Объединение информации с разных таймфреймов через Nyströmformer.

    Пример конфигурации:
    - 1-минутные бары: 1440 сэмплов (1 день)
    - 5-минутные бары: 288 сэмплов (1 день)
    - 1-часовые бары: 168 сэмплов (1 неделя)

    Итого: 1896 токенов эффективно обрабатываются с Nyström attention.
    """
    def __init__(self, d_model=256, num_landmarks=64):
        super().__init__()

        self.timeframe_embeds = nn.ModuleDict({
            '1m': nn.Linear(5, d_model),   # OHLCV
            '5m': nn.Linear(5, d_model),
            '1h': nn.Linear(5, d_model),
        })

        self.timeframe_tokens = nn.ParameterDict({
            '1m': nn.Parameter(torch.randn(1, 1, d_model)),
            '5m': nn.Parameter(torch.randn(1, 1, d_model)),
            '1h': nn.Parameter(torch.randn(1, 1, d_model)),
        })

        # Единый Nyströmformer обрабатывает все таймфреймы
        self.nystrom_encoder = NystromEncoder(
            d_model=d_model,
            n_heads=8,
            n_layers=4,
            num_landmarks=num_landmarks,
            seq_len=2048  # Вмещает все таймфреймы
        )

    def forward(self, data_1m, data_5m, data_1h):
        # Эмбеддинг каждого таймфрейма
        x_1m = self.timeframe_embeds['1m'](data_1m) + self.timeframe_tokens['1m']
        x_5m = self.timeframe_embeds['5m'](data_5m) + self.timeframe_tokens['5m']
        x_1h = self.timeframe_embeds['1h'](data_1h) + self.timeframe_tokens['1h']

        # Конкатенация по измерению последовательности
        x = torch.cat([x_1m, x_5m, x_1h], dim=1)

        # Обработка Nyström attention
        output = self.nystrom_encoder(x)

        return output
```

## Практические примеры

### 01: Подготовка данных

```python
# python/01_data_preparation.py

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

def prepare_long_sequence_data(
    symbols: List[str],
    lookback: int = 4096,      # Расширенный lookback для Nyströmformer
    horizon: int = 24,
    source: str = 'bybit'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Подготовка данных длинных последовательностей для Nyströmformer.

    Аргументы:
        symbols: Торговые пары (например, ['BTCUSDT', 'ETHUSDT'])
        lookback: Длина исторической последовательности (может быть очень длинной!)
        horizon: Горизонт предсказания
        source: Источник данных ('bybit', 'binance', 'yahoo')

    Возвращает:
        X: Признаки [n_samples, lookback, n_features]
        y: Целевые значения [n_samples, horizon]
    """
    all_features = []

    for symbol in symbols:
        # Загрузка данных из источника
        if source == 'bybit':
            df = load_bybit_data(symbol, interval='1m')
        elif source == 'binance':
            df = load_binance_data(symbol, interval='1m')
        else:
            df = load_yahoo_data(symbol)

        # Вычисление признаков
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(20).std()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(50).mean()
        df['price_ma_ratio'] = df['close'] / df['close'].rolling(200).mean()
        df['rsi'] = calculate_rsi(df['close'], period=14)
        df['atr'] = calculate_atr(df, period=14)

        all_features.append(df)

    # Объединение признаков
    features = pd.concat(all_features, axis=1, keys=symbols)
    features = features.dropna()

    # Создание последовательностей
    X, y = [], []
    feature_cols = ['log_return', 'volatility', 'volume_ma_ratio',
                    'price_ma_ratio', 'rsi', 'atr']

    for i in range(lookback, len(features) - horizon):
        # Вход: [lookback, n_features * n_symbols]
        x_seq = features.iloc[i-lookback:i][
            [(s, f) for s in symbols for f in feature_cols]
        ].values
        X.append(x_seq)

        # Цель: будущие доходности основного символа
        y_seq = features.iloc[i:i+horizon][(symbols[0], 'log_return')].values
        y.append(y_seq)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def load_bybit_data(symbol: str, interval: str = '1m') -> pd.DataFrame:
    """Загрузка исторических данных с Bybit."""
    import requests

    url = f"https://api.bybit.com/v5/market/kline"
    params = {
        'category': 'spot',
        'symbol': symbol,
        'interval': interval_to_bybit(interval),
        'limit': 10000
    }

    response = requests.get(url, params=params)
    data = response.json()['result']['list']

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])

    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = pd.to_numeric(df[col])

    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df
```

### 02: Модель Nyströmformer

Полная реализация модели находится в файле [python/model.py](python/model.py).

### 03: Обучение

```python
# python/03_training.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NystromTrainer:
    """Пайплайн обучения для торговой модели Nyströmformer."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )

        # Функция потерь в зависимости от типа выхода
        if model.output_type == 'regression':
            self.criterion = nn.MSELoss()
        elif model.output_type == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif model.output_type == 'allocation':
            self.criterion = self._sharpe_loss

    def _sharpe_loss(self, allocations: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """
        Дифференцируемая функция потерь на основе коэффициента Шарпа.

        Аргументы:
            allocations: Предсказанные аллокации модели [batch, horizon]
            returns: Фактические доходности [batch, horizon]

        Возвращает:
            Отрицательный коэффициент Шарпа (для минимизации)
        """
        # Доходности портфеля
        portfolio_returns = allocations * returns

        # Коэффициент Шарпа (отрицательный для минимизации)
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std() + 1e-8
        sharpe = mean_return / std_return

        return -sharpe

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Обучение одной эпохи."""
        self.model.train()
        total_loss = 0.0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)

            loss.backward()

            # Клиппинг градиентов для стабильности
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()

        self.scheduler.step()

        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Оценка производительности модели."""
        self.model.eval()

        all_preds = []
        all_targets = []
        total_loss = 0.0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)

            total_loss += loss.item()
            all_preds.append(predictions.cpu())
            all_targets.append(batch_y.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        metrics = {
            'loss': total_loss / len(dataloader),
        }

        if self.model.output_type == 'regression':
            mse = ((all_preds - all_targets) ** 2).mean().item()
            mae = (all_preds - all_targets).abs().mean().item()
            metrics['mse'] = mse
            metrics['mae'] = mae

            # Точность направления
            pred_direction = (all_preds[:, 0] > 0).float()
            true_direction = (all_targets[:, 0] > 0).float()
            metrics['direction_accuracy'] = (pred_direction == true_direction).float().mean().item()

        elif self.model.output_type == 'classification':
            pred_classes = all_preds.argmax(dim=1)
            metrics['accuracy'] = (pred_classes == all_targets).float().mean().item()

        return metrics
```

### 04: Бэктестинг стратегии

```python
# python/04_backtest.py

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class BacktestConfig:
    """Конфигурация для бэктестинга."""
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001  # 0.1% за сделку
    slippage: float = 0.0005  # 0.05% проскальзывание
    max_position_size: float = 1.0  # Максимальная позиция как доля капитала
    risk_per_trade: float = 0.02  # 2% риска на сделку


class NystromBacktester:
    """Движок бэктестинга для торговой стратегии Nyströmformer."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: BacktestConfig = BacktestConfig()
    ):
        self.model = model
        self.config = config
        self.model.eval()

    @torch.no_grad()
    def generate_signals(
        self,
        data: np.ndarray,
        threshold: float = 0.001
    ) -> np.ndarray:
        """
        Генерация торговых сигналов из предсказаний модели.

        Аргументы:
            data: [n_samples, seq_len, n_features]
            threshold: Минимальная предсказанная доходность для сигнала

        Возвращает:
            signals: [n_samples] со значениями {-1, 0, 1}
        """
        self.model.eval()

        # Получение предсказаний
        x = torch.tensor(data, dtype=torch.float32)
        predictions = self.model(x).numpy()

        # Используем первый шаг горизонта предсказания
        pred_returns = predictions[:, 0] if len(predictions.shape) > 1 else predictions

        # Генерация сигналов
        signals = np.zeros_like(pred_returns)
        signals[pred_returns > threshold] = 1   # Сигнал на покупку
        signals[pred_returns < -threshold] = -1  # Сигнал на продажу

        return signals

    def _calculate_metrics(
        self,
        returns: np.ndarray,
        equity_curve: np.ndarray,
        trades: List[Dict]
    ) -> Dict:
        """Расчёт метрик производительности."""

        # Базовые метрики
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]

        # Коэффициент Шарпа (при дневных доходностях, 252 торговых дня)
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe_ratio = 0.0

        # Коэффициент Сортино (отклонение по убыткам)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            sortino_ratio = np.sqrt(252) * returns.mean() / (downside_std + 1e-8)
        else:
            sortino_ratio = sharpe_ratio

        # Максимальная просадка
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()

        # Процент выигрышных сделок
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0.0

        # Коэффициент Кальмара
        if max_drawdown < 0:
            calmar_ratio = total_return / abs(max_drawdown)
        else:
            calmar_ratio = float('inf')

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'calmar_ratio': calmar_ratio,
            'num_trades': len(trades),
            'final_capital': equity_curve[-1]
        }
```

## Реализация на Rust

См. [rust/](rust/) для полной реализации на Rust с использованием данных Bybit.

```
rust/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Основные экспорты библиотеки
│   ├── api/                # Клиент API Bybit
│   │   ├── mod.rs
│   │   ├── client.rs       # HTTP клиент для Bybit
│   │   └── types.rs        # Типы ответов API
│   ├── data/               # Обработка данных
│   │   ├── mod.rs
│   │   ├── loader.rs       # Утилиты загрузки данных
│   │   ├── features.rs     # Инженерия признаков
│   │   └── dataset.rs      # Датасет для обучения
│   ├── model/              # Архитектура Nyströmformer
│   │   ├── mod.rs
│   │   ├── attention.rs    # Реализация Nyström attention
│   │   ├── encoder.rs      # Слои энкодера
│   │   └── nystromformer.rs # Полная модель
│   └── strategy/           # Торговая стратегия
│       ├── mod.rs
│       ├── signals.rs      # Генерация сигналов
│       └── backtest.rs     # Движок бэктестинга
└── examples/
    ├── fetch_data.rs       # Загрузка данных Bybit
    ├── train.rs            # Обучение модели
    └── backtest.rs         # Запуск бэктеста
```

### Быстрый старт (Rust)

```bash
# Перейти в проект Rust
cd rust

# Загрузить данные с Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT,SOLUSDT

# Обучить модель
cargo run --example train -- --epochs 100 --batch-size 32

# Запустить бэктест
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Реализация на Python

См. [python/](python/) для реализации на Python.

```
python/
├── __init__.py
├── model.py              # Реализация Nyströmformer
├── data.py               # Загрузка и предобработка данных
├── features.py           # Инженерия признаков
├── train.py              # Пайплайн обучения
├── backtest.py           # Утилиты бэктестинга
├── requirements.txt      # Зависимости
└── examples/
    ├── 01_data_preparation.ipynb
    ├── 02_model_training.ipynb
    ├── 03_backtesting.ipynb
    └── 04_visualization.ipynb
```

### Быстрый старт (Python)

```bash
# Установить зависимости
cd python
pip install -r requirements.txt

# Загрузить данные
python data.py --symbols BTCUSDT,ETHUSDT,SOLUSDT

# Обучить модель
python train.py --config configs/default.yaml

# Запустить бэктест
python backtest.py --model checkpoints/nystromformer_best.pt
```

## Лучшие практики

### Когда использовать Nyströmformer

**Идеальные случаи:**
- Обработка последовательностей >2048 токенов (тиковые данные, снимки LOB)
- Инференс в реальном времени с ограниченными ресурсами
- Многотаймфреймный анализ, требующий длинного контекста
- Развёртывание с ограниченными ресурсами

**Рассмотрите альтернативы когда:**
- Длина последовательности <512 (стандартное attention достаточно)
- Критически важна точное attention (некоторая потеря точности)
- Очень разреженные паттерны внимания (используйте sparse attention)

### Рекомендации по гиперпараметрам

| Параметр | Рекомендация | Примечания |
|----------|--------------|------------|
| `num_landmarks` | 32-64 | Больше опорных точек = лучшая аппроксимация, больше вычислений |
| `seq_len` | 2048-8192 | Степень 2, делится на количество опорных точек |
| `d_model` | 128-512 | Соответствует сложности задачи |
| `n_layers` | 3-6 | Больше слоёв для сложных паттернов |
| `pinv_iterations` | 6 | Достаточно для сходимости |

### Частые ошибки

1. **Длина последовательности не делится на количество опорных точек**: Убедитесь, что seq_len % num_landmarks == 0
2. **Недостаточно опорных точек**: Используйте минимум 32 для длинных последовательностей
3. **Численная нестабильность**: Используйте правильную нормализацию в псевдообратной
4. **Проблемы с памятью**: Уменьшите размер батча или количество опорных точек при OOM

## Ресурсы

### Научные работы

- [Nyströmformer: A Nyström-based Algorithm for Approximating Self-Attention](https://arxiv.org/abs/2102.03902) — Оригинальная статья (AAAI 2021)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Оригинальный Transformer
- [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732) — Сравнение эффективных методов внимания
- [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436) — Связанная архитектура

### Реализации

- [HuggingFace Transformers](https://huggingface.co/docs/transformers/model_doc/nystromformer) — Официальная реализация HF
- [Официальный репозиторий](https://github.com/mlpen/Nystromformer) — Код авторов
- [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/) — Библиотека временных рядов

### Связанные главы

- [Глава 44: ProbSparse Attention](../44_probsparse_attention) — Другой эффективный механизм внимания
- [Глава 51: Linformer Long Sequences](../51_linformer_long_sequences) — Подход линейного внимания
- [Глава 52: Performer Efficient Attention](../52_performer_efficient_attention) — Внимание со случайными признаками
- [Глава 57: Longformer Financial](../57_longformer_financial) — Скользящее окно + глобальное внимание

---

## Уровень сложности

**Средний - Продвинутый**

Необходимые знания:
- Понимание архитектуры трансформеров и self-attention
- Знакомство с методами аппроксимации матриц
- Базовые знания прогнозирования временных рядов
- Опыт работы с PyTorch или библиотеками ML на Rust
