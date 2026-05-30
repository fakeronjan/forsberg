# ============================================================
# FORSBERG - International Men's Ice Hockey Power Ratings
# fakeronjan WLS weighted-least-squares rolling rating engine.
#
# Cloned from CARMELO (intl basketball), which itself cloned the DUNCAN solver:
# same WLS fakeronjan WLS solver, linear recency decay across a FIXED CALENDAR-TIME
# window, sign-preserving margin-cap transform, zero-sum constraint via a
# high-weight extra row.
#
# Hockey-specific adaptations (user-locked 2026-05-29):
#   - Era anchor: 1976+ (Canada Cup era).
#   - 4-year fixed calendar window (Olympic-cycle natural; annual IIHF WC keeps
#     density adequate without friendlies, so friendlies are filtered upstream).
#   - MARGIN_CAP = 5 goals (from SAKIC; hockey's low-scoring scale).
#   - HCA = 0.15 goals, applied ONLY to non-neutral games (Canada Cup / WCoH
#     host-side games; Olympics + IIHF WC are neutral host venues).
#   - Tier weights set per-tournament-family in scrape_wiki.py: Olympics +
#     WCoH + Canada Cup = 1.5 (pinnacle / best-on-best), IIHF Worlds = 1.0.
#     NHL participation variation in the Olympics is intentionally absorbed
#     rather than encoded as a separate tier ("part of the story").
#
# Composite teams (2016 WCoH Team Europe / Team North America) are skipped at
# the scraper, so the rating system is national-teams-only.
# ============================================================
import os
import numpy as np
import pandas as pd
from datetime import datetime

# IOC code -> display country name. Includes era-specific entities kept
# distinct (Soviet Union != Russia, East/West Germany != Germany).
CODE_TO_COUNTRY = {
    "USA": "United States", "CAN": "Canada", "MEX": "Mexico",
    "RUS": "Russia", "URS": "Soviet Union",
    "SWE": "Sweden", "FIN": "Finland",
    "CZE": "Czech Republic", "TCH": "Czechoslovakia", "SVK": "Slovakia",
    "GER": "Germany", "FRG": "West Germany", "GDR": "East Germany",
    "SUI": "Switzerland", "NOR": "Norway", "DEN": "Denmark",
    "AUT": "Austria", "LAT": "Latvia", "BLR": "Belarus", "KAZ": "Kazakhstan",
    "FRA": "France", "ITA": "Italy", "GBR": "Great Britain",
    "NED": "Netherlands", "POL": "Poland", "HUN": "Hungary",
    "ROU": "Romania", "UKR": "Ukraine", "SLO": "Slovenia", "CRO": "Croatia",
    "YUG": "Yugoslavia", "SCG": "Serbia and Montenegro", "SRB": "Serbia",
    "BUL": "Bulgaria", "EST": "Estonia", "LTU": "Lithuania",
    "BEL": "Belgium", "ESP": "Spain", "ISL": "Iceland",
    "JPN": "Japan", "KOR": "South Korea", "CHN": "China", "ISR": "Israel",
    "AUS": "Australia", "NZL": "New Zealand", "PRK": "North Korea",
    "GRE": "Greece", "TUR": "Turkey", "IRL": "Ireland", "LUX": "Luxembourg",
    "ARM": "Armenia", "BIH": "Bosnia and Herzegovina", "MGL": "Mongolia",
    "HKG": "Hong Kong", "RSA": "South Africa",
    # Historical Unified Team (1992 Albertville Olympics): post-USSR dissolution,
    # pre-independent national federations. Distinct entity, kept separate.
    "EUN": "Unified Team",
}


# Codes that refer to the SAME nation in different sources -- canonicalize at
# load time so a team's history isn't split across two records.
#   * SVN (IOC) and SLO (IIHF) both = Slovenia
#   * "FR YUGOSLAVIA" leaks from the name fallback for Serbia and Montenegro
#     era pages (1992-2006) -- same entity as SCG
#   * "RUSSIA" (all caps) is a name-fallback leak from 2021 IIHF WC = RUS
CANON_CODE = {
    "SVN": "SLO",
    "FR YUGOSLAVIA": "SCG",
    "RUSSIA": "RUS",
    # Continuity merges per user-locked policy 2026-05-29 (DILLON pattern):
    # the team's history is unbroken across the rename, with era-aware
    # display_name applied in generate_data.py (see NAME_HISTORY below).
    "URS": "RUS",   # Soviet Union  -> Russia (1991 dissolution)
    "USSR": "RUS",  # Alt-form leaked from infobox auto-curation
    "EUN": "RUS",   # Unified Team  -> Russia (1992 Albertville Olympics only)
    "FRG": "GER",   # West Germany  -> Germany (1990 reunification, the DEB
                    # federation continued as unified Germany)
    # NOT merged (per Q2/Q3 policy): Czechoslovakia (TCH), East Germany (GDR),
    # Yugoslavia (YUG), Serbia-Montenegro variants -- splits/non-continuations
    # stay as distinct defunct entities.
}


# Date-based name history for the DILLON era-aware display pattern.
# Maps canonical code -> ordered list of (historical_name, start_iso, end_iso).
# generate_data.py uses this to set per-row display_name so a 1985 Russia
# snapshot reads "Soviet Union" inline, but the team page itself is unified.
NAME_HISTORY = {
    "RUS": [
        # USSR dissolution: December 26, 1991.
        ("Soviet Union", "1900-01-01", "1991-12-25"),
        # Unified Team competed at 1992 Albertville Olympics (closed Feb 23).
        # 1992 IIHF WC in May was already Russia.
        ("Unified Team", "1991-12-26", "1992-02-23"),
    ],
    "GER": [
        # German reunification: October 3, 1990.
        ("West Germany", "1900-01-01", "1990-10-03"),
    ],
}

# Display grouping. IIHF assigns Israel + Kazakhstan to the European pool,
# but for fan-facing geography we keep them in Asia / Europe respectively.
CONFEDERATION = {
    "USA": "Americas", "CAN": "Americas", "MEX": "Americas",
    "RUS": "Europe", "URS": "Europe",
    "SWE": "Europe", "FIN": "Europe", "CZE": "Europe", "TCH": "Europe",
    "SVK": "Europe", "GER": "Europe", "FRG": "Europe", "GDR": "Europe",
    "SUI": "Europe", "NOR": "Europe", "DEN": "Europe", "AUT": "Europe",
    "LAT": "Europe", "BLR": "Europe", "KAZ": "Europe", "FRA": "Europe",
    "ITA": "Europe", "GBR": "Europe", "NED": "Europe", "POL": "Europe",
    "HUN": "Europe", "ROU": "Europe", "UKR": "Europe", "SLO": "Europe",
    "CRO": "Europe", "YUG": "Europe", "SCG": "Europe", "SRB": "Europe",
    "BUL": "Europe", "EST": "Europe", "LTU": "Europe", "BEL": "Europe",
    "ESP": "Europe", "ISL": "Europe",
    "JPN": "Asia", "KOR": "Asia", "CHN": "Asia", "ISR": "Asia",
    "PRK": "Asia", "MGL": "Asia", "HKG": "Asia",
    "GRE": "Europe", "TUR": "Europe", "IRL": "Europe", "LUX": "Europe",
    "ARM": "Europe", "BIH": "Europe",
    "RSA": "Africa", "EUN": "Europe",  # Unified Team (1992) was mostly Russia
    "AUS": "Oceania", "NZL": "Oceania",
}


# ============================================================
# PARAMETERS  (user-locked 2026-05-29; flag in build report when tuning)
# ============================================================
WINDOW_YEARS = 4
WINDOW_DAYS = int(WINDOW_YEARS * 365.25)
RECENCY_FLOOR = 0.15

MARGIN_TRANSFORM = "cap"
# SAKIC's NHL cap=5 was set for tight pro games where 5-goal wins are rare.
# International hockey has much bigger blowouts (USA / CAN routinely beat
# tier-3 nations 10+); cap=5 was capping ~20% of games. Raised to 10 per user
# 2026-05-30 -- caps only the genuine routs (~5%).
MARGIN_CAP = 10

HOME_COURT_ADJUSTMENT = 0.15  # goals; applied only to non-neutral games
WEIGHTING_MODE = "wls"
MIN_GAMES = 4

GAMES_CSV = "all_games.csv"
RATINGS_CSV = "forsberg_ratings.csv"


# ============================================================
# FAKERONJAN WLS SOLVER  (same math as CARMELO/DUNCAN)
# ============================================================

def _apply_margin_transform(margin, transform, cap):
    m = np.asarray(margin, dtype=float)
    if transform == "raw":
        return m
    if transform == "cap":
        return np.clip(m, -cap, cap)
    if transform == "tanh":
        return cap * np.tanh(m / cap)
    raise ValueError(f"Unknown MARGIN_TRANSFORM: {transform}")


def _solve_wls(window_df):
    """WLS fakeronjan WLS solve on one rolling window. X has +1 home / -1 road, y is
    transformed (HCA-adjusted) home margin, W = recency x tier. Zero-sum
    constraint via a high-weight extra row. WLS via sqrt(w) row-scaling ->
    ordinary lstsq."""
    teams = sorted(set(window_df["home_team"]) | set(window_df["road_team"]))
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    n_games = len(window_df)

    X = np.zeros((n_games + 1, n_teams))
    y = np.zeros(n_games + 1)
    w = np.zeros(n_games + 1)

    home_runs = window_df["home_runs"].to_numpy(dtype=float)
    road_runs = window_df["road_runs"].to_numpy(dtype=float)
    hca = window_df["hca"].to_numpy(dtype=float)
    weights = window_df["weight"].to_numpy(dtype=float)
    home_names = window_df["home_team"].to_numpy()
    road_names = window_df["road_team"].to_numpy()

    raw_margin = home_runs - road_runs - hca
    transformed = _apply_margin_transform(raw_margin, MARGIN_TRANSFORM, MARGIN_CAP)

    for i in range(n_games):
        X[i, team_idx[home_names[i]]] = 1.0
        X[i, team_idx[road_names[i]]] = -1.0

    y[:n_games] = transformed
    w[:n_games] = weights

    X[-1, :] = 1.0
    y[-1] = 0.0
    w[-1] = 1.0e8

    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    r, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

    out = pd.DataFrame({"code": teams, "rating": r})
    out["rank"] = out["rating"].rank(ascending=False, method="min").astype(int)
    return out


# ============================================================
# DATA PREP
# ============================================================

def load_games():
    df = pd.read_csv(GAMES_CSV)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.dropna(subset=["home_team", "road_team", "home_runs", "road_runs", "date"])
    # Canonicalize alternate codes so a team's history isn't split.
    df["home_team"] = df["home_team"].replace(CANON_CODE)
    df["road_team"] = df["road_team"].replace(CANON_CODE)
    df["home_runs"] = pd.to_numeric(df["home_runs"], errors="coerce")
    df["road_runs"] = pd.to_numeric(df["road_runs"], errors="coerce")
    df = df.dropna(subset=["home_runs", "road_runs"])
    df = df[df["home_team"] != df["road_team"]]

    if "neutral" not in df.columns:
        df["neutral"] = True
    df["neutral"] = df["neutral"].fillna(True).astype(bool)
    if "tier" not in df.columns:
        df["tier"] = 1.0
    df["tier"] = pd.to_numeric(df["tier"], errors="coerce").fillna(1.0)

    df["hca"] = np.where(df["neutral"], 0.0, HOME_COURT_ADJUSTMENT)

    df["home_win"] = (df["home_runs"] > df["road_runs"]).astype(int)
    df["road_win"] = 1 - df["home_win"]

    df = df.sort_values("date").reset_index(drop=True)
    df["grouped_date_id"] = df.groupby("date").ngroup() + 1
    return df


# ============================================================
# ROLLING RATINGS
# ============================================================

def compute_ratings(df):
    max_id = int(df["grouped_date_id"].max())
    frames = []
    last_year = None

    df = df.copy()
    df["_ord"] = df["date"].map(lambda d: d.toordinal())

    for i in range(1, max_id + 1):
        current_date = df.loc[df["grouped_date_id"] == i, "date"].max()
        if pd.isnull(current_date):
            continue

        cur_ord = current_date.toordinal()
        window = df[(df["_ord"] <= cur_ord) & (df["_ord"] > cur_ord - WINDOW_DAYS)].copy()
        if len(window) < 10:
            continue

        window["days_ago"] = cur_ord - window["_ord"]
        window["date_weight"] = (1 - window["days_ago"] / WINDOW_DAYS).clip(lower=RECENCY_FLOOR)
        window["weight"] = window["date_weight"] * window["tier"]
        window = window[window["weight"] > 0]
        if len(window) < 10:
            continue

        ranked = _solve_wls(window)
        if ranked["rating"].isna().any() or np.isinf(ranked["rating"]).any():
            continue

        ga = window.groupby("home_team").size().rename("ga")
        gb = window.groupby("road_team").size().rename("gb")
        gp = pd.concat([ga, gb], axis=1).fillna(0)
        gp["games_in_window"] = (gp["ga"] + gp["gb"]).astype(int)
        ranked = ranked.merge(
            gp[["games_in_window"]], left_on="code", right_index=True, how="left"
        )
        ranked["games_in_window"] = ranked["games_in_window"].fillna(0).astype(int)

        ranked["ranking_id"] = i
        ranked["date"] = current_date
        ranked["season"] = current_date.year
        frames.append(ranked)

        if current_date.year != last_year:
            pct = round(100 * i / max_id)
            print(f"  Ratings: {current_date.year} ({pct}%)")
            last_year = current_date.year

    ratings = pd.concat(frames, ignore_index=True)
    ratings["country"] = ratings["code"].map(CODE_TO_COUNTRY).fillna(ratings["code"])
    ratings["confederation"] = ratings["code"].map(CONFEDERATION).fillna("Other")
    return ratings


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print(f"FORSBERG rating engine -- window={WINDOW_YEARS}yr calendar ({WINDOW_DAYS}d), "
          f"floor={RECENCY_FLOOR}, cap={MARGIN_CAP}, HCA={HOME_COURT_ADJUSTMENT} (non-neutral only)")
    games = load_games()
    print(f"Loaded {len(games):,} games over "
          f"{games['date'].min()} .. {games['date'].max()} "
          f"({games['grouped_date_id'].max()} game-days)")

    ratings = compute_ratings(games)

    ratings = ratings.sort_values(["ranking_id", "rank"]).reset_index(drop=True)
    ratings.to_csv(RATINGS_CSV, index=False)
    print(f"\n{RATINGS_CSV} saved ({len(ratings):,} rows)")

    # Face-validity: most-recent snapshot top 15.
    latest_id = ratings["ranking_id"].max()
    latest_date = ratings.loc[ratings['ranking_id']==latest_id,'date'].iloc[0]
    latest = ratings[(ratings["ranking_id"] == latest_id) &
                     (ratings["games_in_window"] >= MIN_GAMES)].copy()
    latest = latest.sort_values("rating", ascending=False).head(15)
    print(f"\n=== FACE-VALIDITY: top 15 as of {latest_date} ===")
    print(latest[["rank", "country", "confederation", "rating", "games_in_window"]]
          .to_string(index=False))
