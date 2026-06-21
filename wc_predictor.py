"""
2026 FIFA World Cup match predictor + Kalshi edge finder.

Pipeline:
  1. Load historical international results (martj42 Kaggle CSV) + any manual
     recent results you punch in for games that haven't synced yet.
  2. Fit a time-decayed Dixon-Coles model (recent form dominates).
  3. Predict P(home win) / P(draw) / P(away win) for any fixture.
  4. Pull Kalshi market prices (live API if you have creds, else paste manually),
     convert to implied probabilities, and compute your edge.

Nothing here places trades. It only reads. Treat outputs as estimates, not certainty.
"""

import math
import time
import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


# ----------------------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------------------

# martj42 dataset, raw CSV on GitHub (mirror of the Kaggle one, no key needed).
# Columns: date, home_team, away_team, home_score, away_score, tournament,
#          city, country, neutral
RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/"
    "international_results/master/results.csv"
)


def load_history(url=RESULTS_URL, manual_recent=None):
    """Load historical results. `manual_recent` is a list of dicts for games
    that finished in the last few hours and haven't synced to the CSV yet."""
    df = pd.read_csv(url, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    if manual_recent:
        extra = pd.DataFrame(manual_recent)
        extra["date"] = pd.to_datetime(extra["date"])
        # default the optional columns if you didn't supply them
        for col, default in [
            ("tournament", "FIFA World Cup"),
            ("city", ""),
            ("country", ""),
            ("neutral", True),
        ]:
            if col not in extra.columns:
                extra[col] = default
        df = pd.concat([df, extra], ignore_index=True)

    df = df.sort_values("date").reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# 2. TIME-DECAYED DIXON-COLES MODEL
# ----------------------------------------------------------------------------

@dataclass
class DixonColes:
    half_life_days: float = 180.0      # recency: how fast old games fade
    wc_boost: float = 2.0              # extra weight on current-tournament games
    max_goals: int = 10               # truncation for the score matrix

    attack: dict = field(default_factory=dict)
    defense: dict = field(default_factory=dict)
    home_adv: float = 0.0
    rho: float = 0.0                   # low-score correlation correction
    teams: list = field(default_factory=list)

    def _weights(self, dates, tournaments, as_of):
        """Exponential time-decay weight per match, with a WC multiplier."""
        age_days = (as_of - dates).dt.total_seconds().values / 86400.0
        decay = 0.5 ** (age_days / self.half_life_days)
        is_wc = tournaments.str.contains("World Cup", case=False, na=False).values
        return decay * np.where(is_wc, self.wc_boost, 1.0)

    @staticmethod
    def _tau(hg, ag, lam, mu, rho):
        """Dixon-Coles correction for 0-0/1-0/0-1/1-1 dependence."""
        tau = np.ones_like(lam, dtype=float)
        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)
        tau[m00] = 1 - lam[m00] * mu[m00] * rho
        tau[m01] = 1 + lam[m01] * rho
        tau[m10] = 1 + mu[m10] * rho
        tau[m11] = 1 - rho
        return np.clip(tau, 1e-10, None)

    def fit(self, df, as_of=None, recent_years=12):
        """Estimate team strengths by weighted maximum likelihood."""
        as_of = pd.Timestamp(as_of) if as_of else df["date"].max()
        # only keep recent-ish history; old decayed weights ~0 anyway
        cutoff = as_of - pd.Timedelta(days=365 * recent_years)
        d = df[(df["date"] >= cutoff) & (df["date"] <= as_of)].copy()

        teams = sorted(set(d["home_team"]) | set(d["away_team"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        hg = d["home_score"].values
        ag = d["away_score"].values
        hi = d["home_team"].map(idx).values
        ai = d["away_team"].map(idx).values
        # neutral-venue games get no home advantage (most WC games)
        neutral = d.get("neutral", pd.Series(False, index=d.index)).fillna(False).values
        w = self._weights(d["date"], d["tournament"], as_of)

        # params: [attack(n), defense(n), home_adv, rho]
        # identifiability: mean attack fixed at 0 via a soft penalty
        def neg_log_lik(p):
            atk = p[:n]
            dfn = p[n:2 * n]
            home_adv = p[2 * n]
            rho = p[2 * n + 1]

            lam = np.exp(atk[hi] + dfn[ai] + home_adv * (~neutral))
            mu = np.exp(atk[ai] + dfn[hi])
            lam = np.clip(lam, 1e-8, 30)
            mu = np.clip(mu, 1e-8, 30)

            ll = (poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)
                  + np.log(self._tau(hg, ag, lam, mu, rho)))
            penalty = 100.0 * (atk.mean() ** 2)   # anchor mean attack to 0
            return -(w * ll).sum() + penalty

        x0 = np.concatenate([
            np.zeros(n),          # attack
            np.zeros(n),          # defense
            [0.25],               # home advantage
            [-0.1],               # rho
        ])
        res = minimize(neg_log_lik, x0, method="L-BFGS-B",
                       options={"maxiter": 500})

        p = res.x
        self.teams = teams
        self.attack = {t: p[idx[t]] for t in teams}
        self.defense = {t: p[n + idx[t]] for t in teams}
        self.home_adv = p[2 * n]
        self.rho = p[2 * n + 1]
        return self

    def predict(self, home, away, neutral=True):
        """Return (P_home, P_draw, P_away) plus expected goals."""
        for t in (home, away):
            if t not in self.attack:
                raise ValueError(f"No history/rating for team: {t!r}")

        lam = math.exp(self.attack[home] + self.defense[away]
                       + self.home_adv * (0 if neutral else 1))
        mu = math.exp(self.attack[away] + self.defense[home])

        mg = self.max_goals
        hg = np.arange(mg + 1)
        # outer product of the two independent Poisson pmfs
        ph = poisson.pmf(hg, lam)
        pa = poisson.pmf(hg, mu)
        mat = np.outer(ph, pa)

        # apply Dixon-Coles correction to the four low-score cells
        for (i, j, factor) in [
            (0, 0, 1 - lam * mu * self.rho),
            (0, 1, 1 + lam * self.rho),
            (1, 0, 1 + mu * self.rho),
            (1, 1, 1 - self.rho),
        ]:
            mat[i, j] *= max(factor, 1e-10)
        mat /= mat.sum()

        p_home = np.tril(mat, -1).sum()   # home goals > away goals
        p_away = np.triu(mat, 1).sum()    # away goals > home goals
        p_draw = np.trace(mat)
        return {
            "home": home, "away": away,
            "exp_home_goals": round(lam, 2), "exp_away_goals": round(mu, 2),
            "p_home": round(float(p_home), 4),
            "p_draw": round(float(p_draw), 4),
            "p_away": round(float(p_away), 4),
        }


# ----------------------------------------------------------------------------
# 3. KALSHI PRICES -> IMPLIED PROBABILITY -> EDGE
# ----------------------------------------------------------------------------

def price_to_prob(cents):
    """Kalshi YES price in cents (1-99) -> implied probability."""
    return cents / 100.0


def compute_edge(model_probs, kalshi_yes_cents):
    """
    model_probs: dict with p_home/p_draw/p_away from .predict()
    kalshi_yes_cents: dict mapping outcome -> YES price in cents, e.g.
        {"home": 58, "draw": 24, "away": 22}
    Edge = your model prob - Kalshi implied prob. Positive = market underpricing
    that outcome relative to your model.
    """
    key = {"home": "p_home", "draw": "p_draw", "away": "p_away"}
    rows = []
    for outcome, cents in kalshi_yes_cents.items():
        implied = price_to_prob(cents)
        model_p = model_probs[key[outcome]]
        rows.append({
            "outcome": outcome,
            "model_prob": round(model_p, 4),
            "kalshi_implied": round(implied, 4),
            "edge": round(model_p - implied, 4),
        })
    out = pd.DataFrame(rows)
    # Kalshi's implied probs sum to >1 (the vig). Show how much.
    out.attrs["overround"] = round(sum(price_to_prob(c) for c in
                                       kalshi_yes_cents.values()) - 1, 4)
    return out


def flag_edges(edge_df, threshold=0.05):
    """Return only rows where your edge clears the threshold (covers the vig)."""
    return edge_df[edge_df["edge"] >= threshold].sort_values("edge",
                                                             ascending=False)


# ----------------------------------------------------------------------------
# 4. KALSHI LIVE FETCH (optional - needs a free verified account + API key)
# ----------------------------------------------------------------------------

def fetch_kalshi_market(ticker, key_id=None, private_key_pem=None):
    """
    Read-only market fetch. Free, no account funding required, but you need a
    verified Kalshi account and an API key pair (Settings -> API).

    Leave creds as None to skip live fetching and enter prices manually instead.
    Market data endpoints are public, so for a single market you often don't
    even need auth - but signing is included for when you do.
    """
    import requests

    base = "https://api.elections.kalshi.com/trade-api/v2"
    url = f"{base}/markets/{ticker}"
    headers = {}

    if key_id and private_key_pem:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
        import base64

        ts = str(int(time.time() * 1000))
        msg = f"{ts}GET/trade-api/v2/markets/{ticker}".encode()
        pk = serialization.load_pem_private_key(private_key_pem.encode(), password=None)
        sig = pk.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        headers = {
            "KALSHI-ACCESS-KEY": key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }

    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    m = r.json()["market"]
    # Kalshi quotes in cents; yes_bid/yes_ask are the live book
    return {
        "ticker": ticker,
        "yes_bid": m.get("yes_bid"),
        "yes_ask": m.get("yes_ask"),
        "last_price": m.get("last_price"),
    }


# ----------------------------------------------------------------------------
# DEMO
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # Games that finished hours ago but may not be in the CSV yet:
    manual_recent = [
        # {"date": "2026-06-18", "home_team": "Brazil", "away_team": "Serbia",
        #  "home_score": 2, "away_score": 0},
    ]

    print("Loading history...")
    df = load_history(manual_recent=manual_recent)
    print(f"  {len(df):,} matches, latest {df['date'].max().date()}")

    print("Fitting time-decayed Dixon-Coles (recent form weighted heavily)...")
    model = DixonColes(half_life_days=180, wc_boost=2.0).fit(df)
    print(f"  home_adv={model.home_adv:.3f}  rho={model.rho:.3f}  "
          f"teams rated={len(model.teams)}")

    # Example fixture (neutral venue -> no home advantage)
    home, away = "Morocco", "Scotland"
    pred = model.predict(home, away, neutral=True)
    print(f"\n{home} vs {away} (neutral):")
    print(f"  xG: {pred['exp_home_goals']} - {pred['exp_away_goals']}")
    print(f"  P({home})={pred['p_home']:.1%}  "
          f"P(draw)={pred['p_draw']:.1%}  P({away})={pred['p_away']:.1%}")

    # Paste the YES prices you see on Kalshi (in cents):
    kalshi = {"home": 62, "draw": 22, "away": 18}
    edges = compute_edge(pred, kalshi)
    print(f"\nEdge table (Kalshi overround = {edges.attrs['overround']:.1%}):")
    print(edges.to_string(index=False))

    bets = flag_edges(edges, threshold=0.05)
    if len(bets):
        print("\nFlagged (edge >= 5%):")
        print(bets.to_string(index=False))
    else:
        print("\nNo outcome clears the 5% edge threshold.")
