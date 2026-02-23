# 🛒 Sales Forecasting Research

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| RAM | 8GB minimum |
| Disk Space | 500MB |

## Step 1: Clone

```bash
git clone https://github.com/Nevvyboi/SalesForecastingResearch.git
cd SalesForecastingResearch
```

## Step 2: Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Download Dataset

```bash
python downloadData.py
```

## Step 5: Run Experiment

**Quick (~2 min):**
```bash
python main.py --mode quick --sample 50000
```

**Full (~30 min):**
```bash
python main.py --mode full
```

## Options

| Flag | Description |
|------|-------------|
| `--mode quick` | 10 epochs, faster |
| `--mode full` | 30 epochs, complete |
| `--sample N` | Use N rows only |

## Output

```
experiments/
├── figures/
│   ├── model_comparison.png
│   ├── feature_importance.png
│   ├── ablation_study.png
│   └── learning_curves.png
└── results/
    └── results.json
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError | `pip install -r requirements.txt` |
| Dataset not found | `python downloadData.py` |
| Out of memory | Use `--sample 50000` |
| Slow training | Use `--mode quick` |

## Licence

MIT
