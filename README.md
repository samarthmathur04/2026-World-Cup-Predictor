# 2026 World Cup Predictor

A time-decayed Dixon-Coles match predictor for the 2026 FIFA World Cup, with a Kalshi market edge finder.

## Pipeline

1. Load historical international results from the [martj42 results dataset](https://github.com/martj42/international_results), plus any manual recent results that haven't synced to the CSV yet.
2. Fit a time-decayed Dixon-Coles model — recent form is weighted more heavily, with an extra boost for current-tournament games.
3. Predict `P(home win) / P(draw) / P(away win)` and expected goals for any fixture.
4. Pull Kalshi market prices (live via API if you have credentials, or paste manually), convert to implied probabilities, and compute your edge against the model.

Nothing here places trades — it only reads market data. Treat outputs as estimates, not certainty.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy requests cryptography
```

## Usage

```bash
python wc_predictor.py
```

This runs a demo: loads history, fits the model, predicts an example fixture, and compares it against sample Kalshi prices.

To use it for your own fixture:

```python
from wc_predictor import load_history, DixonColes, compute_edge, flag_edges

df = load_history()
model = DixonColes(half_life_days=180, wc_boost=2.0).fit(df)

pred = model.predict("Morocco", "Scotland", neutral=True)

kalshi = {"home": 62, "draw": 22, "away": 18}  # YES prices in cents
edges = compute_edge(pred, kalshi)
print(flag_edges(edges, threshold=0.05))
```

### Kalshi live fetch (optional)

`fetch_kalshi_market(ticker, key_id=..., private_key_pem=...)` pulls live market prices. A verified Kalshi account and API key pair are only needed for authenticated endpoints — single-market reads are often public. Leave the credential args as `None` to skip live fetching and enter prices manually instead.
