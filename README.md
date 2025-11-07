# Outlier Detection System

Automated price outlier detection for shipment data using EWM-MAD (Exponentially Weighted Moving - Median Absolute Deviation).

## Features

- **Adaptive Detection**: EWM-MAD algorithm with log-transformed prices
- **Auto-Segmentation**: Automatically detects best category attribute per product
- **BigQuery Integration**: Direct updates to `shipment_master_taxonomy`
- **Configurable**: Product-specific parameters via `config.py`

## Installation

```bash
pip install -r requirements.txt
gcloud auth application-default login
```

## Usage

### Command Line

```bash
# Analyze all products
python run_analysis.py --all

# Analyze specific product
python run_analysis.py --product "Black Pepper"

# Test mode (no BigQuery updates)
python run_analysis.py --all --test --limit 5
```

### Python API

```python
from outlier_analyzer import OutlierAnalyzer

analyzer = OutlierAnalyzer(
    span=600,
    mad_multiplier=3.0,
    median_pct_window=0.30
)

# Auto-detect best category attribute
results = analyzer.analyze_all_products(
    category_col='auto',
    update_bq=True
)
```

## Algorithm

1. **Filter**: `value >= $2500`, `weight >= 500kg`, `unit_price > 0`
2. **Auto-Segment**: Detect best category attribute (2-20 unique values, highest non-null %)
3. **Log Transform**: `log_price = log(unit_price)`
4. **EWM Statistics**: 
   - `ewm_mean = ewm(log_price, span=600)`
   - `ewm_mad = ewm(|log_price - ewm_mean|, span=600)`
5. **Calculate Bounds**:
   - MAD bounds: `exp(ewm_mean ± 3.0 × ewm_mad)`
   - PCT bounds: `median ± 30%`
   - Final: `min(mad_lower, pct_lower)`, `max(mad_upper, pct_upper)`
6. **Flag**: `LOWER_PRICE` if `price < lower_bound`, `UPPER_PRICE` if `price > upper_bound`

## Configuration

Edit `config.py`:

```python
OUTLIER_PARAMS = {
    'span': 600,                 # EWM smoothness (larger = smoother)
    'mad_multiplier': 3.0,       # Bounds width (larger = fewer outliers)
    'median_pct_window': 0.30,   # ±30% from median
    'value_threshold': 2500.0,   # Min shipment value USD
    'weight_threshold': 500.0,   # Min weight kg
    'max_unit_price': None,      # Optional max price filter
}
```

## Output

Updates BigQuery `shipment_master_taxonomy`:
- `is_outlier` (BOOL): `TRUE` if outlier
- `outlier_type` (STRING): 
  - `'LOWER_PRICE'`: Price below EWM-MAD lower bound
  - `'UPPER_PRICE'`: Price above EWM-MAD upper bound
  - `'LOW_VALUE'`: Value < $2,500 threshold
  - `'LOW_WEIGHT'`: Weight < 500kg threshold
  - `'LOW_VALUE_LOW_WEIGHT'`: Both value and weight below thresholds
  - `'HIGH_PRICE'`: Unit price exceeds max threshold (if configured)
  - `NULL`: Not an outlier

## Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `span` | 600 | Smoothness (100-300: fast, 400-600: balanced, 800+: very smooth) |
| `mad_multiplier` | 3.0 | Width (2.0: tight, 3.0: moderate, 4.0: loose) |
| `median_pct_window` | 0.30 | Percentage range (0.20: ±20%, 0.30: ±30%, 0.50: ±50%) |

## Project Structure

```
outlier_management/
├── outlier_analyzer.py    # Core implementation
├── config.py              # Configuration
├── run_analysis.py        # CLI interface
├── requirements.txt       # Dependencies
└── README.md             # This file
```
