# ============================================================
# FORSBERG - International men's ice hockey scraper
# Parses {{Ice hockey box}} / {{hockeybox2}} / {{IceHockeybox}} game templates
# from Wikipedia tournament pages.
#
# All three variants share the same fields (date, team1, team2, score, ...);
# team codes come from {{ih|CODE}} / {{ih-rt|CODE}}. Winner is bolded with '''.
#
# DATA-INTEGRITY (fleet lesson): fetch_wikitext NEVER raises (returns '' on
# failure) so a flaky scrape degrades to "no new games" rather than crashing.
# union_with_existing treats all_games.csv as the database -- a short re-scrape
# can never delete stored games.
# ============================================================
import re
import sys
import time
import os
import requests
import pandas as pd
from datetime import datetime

WIKI_RAW = "https://en.wikipedia.org/w/index.php?title={title}&action=raw"
HEADERS = {"User-Agent": "forsberg-ratings/1.0 (international hockey ratings; contact via github.com/fakeronjan)"}

ALL_GAMES_CSV = "all_games.csv"

# Tournament tier weights (passed through to engine as a column; engine
# multiplies into the WLS observation weight). User-locked 2026-05-29:
# Olympics + WCoH + Canada Cup = 1.5x (pinnacle / best-on-best), IIHF Worlds =
# 1.0x. NHL-participation variation in the Olympics is intentionally absorbed
# rather than encoded as a separate tier ("part of the story").
TIER_WEIGHTS = {
    "Olympics":              1.5,
    "World Cup of Hockey":   1.5,
    "Canada Cup":            1.5,
    "IIHF World Championship": 1.0,
}

# Most games are at neutral host venues -> neutral=True. Canada Cup and World
# Cup of Hockey had home-venue games for the host nation; those games get
# neutral=False so the engine's HCA kicks in only when the host is playing.
# Default per family below; per-game neutrality is overridden in scrape_event
# for Canada Cup / WCoH host-side games.
NEUTRAL_DEFAULT = {
    "Olympics":                True,
    "IIHF World Championship": True,
    "World Cup of Hockey":     False,  # mixed; host plays at home, refined later
    "Canada Cup":              False,
}

_SKIP_SUBPAGE_RE = re.compile(r"qualif|roster|division", re.IGNORECASE)


# ------------------------------------------------------------
# Event manifest 1976+ (era anchor: 1976 Canada Cup, user-locked).
# ------------------------------------------------------------
def _iihf_wc_title(y):
    if y == 1990:
        return "1990 Men's World Ice Hockey Championships"
    if y <= 1989:
        return f"{y} World Ice Hockey Championships"  # plural pre-1991
    return f"{y} IIHF World Championship"             # canonical from 1991


def _olympics_title(y):
    # Pre-1998 Olympics had no women's ice hockey, so no "Men's tournament"
    # disambiguator in the title.
    if y <= 1994:
        return f"Ice hockey at the {y} Winter Olympics"
    return f"Ice hockey at the {y} Winter Olympics – Men's tournament"


# 1980 / 1984 / 1988 IIHF WC: Olympics doubled as the championship those years
# (no separate WC was held). 2020 IIHF WC was cancelled due to COVID -- left in
# the manifest so a re-scrape doesn't drop it if Wikipedia later backfills.
_IIHF_WC_YEARS = sorted(set(range(1976, 2027)) - {1980, 1984, 1988})
_OLYMPIC_YEARS = [1976, 1980, 1984, 1988, 1992, 1994,
                  1998, 2002, 2006, 2010, 2014, 2018, 2022, 2026]
_CANADA_CUP_YEARS = [1976, 1981, 1984, 1987, 1991]
_WCOH_YEARS = [1996, 2004, 2016]

EVENTS = (
    [(_iihf_wc_title(y), "IIHF World Championship", y) for y in _IIHF_WC_YEARS]
    + [(_olympics_title(y), "Olympics", y) for y in _OLYMPIC_YEARS]
    + [(f"{y} Canada Cup", "Canada Cup", y) for y in _CANADA_CUP_YEARS]
    + [(f"{y} World Cup of Hockey", "World Cup of Hockey", y) for y in _WCOH_YEARS]
)


def fetch_wikitext(title, max_retries=3, _redirect_depth=0):
    """Fetch raw wikitext. Returns '' on failure (NEVER raises). Follows
    #REDIRECT up to 3 hops."""
    url = WIKI_RAW.format(title=requests.utils.quote(title.replace(" ", "_")))
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 404:
                return ""
            r.raise_for_status()
            time.sleep(0.3)
            text = r.text
            m = re.match(r"\s*#REDIRECT\s*\[\[([^\]|#]+)", text, re.IGNORECASE)
            if m and _redirect_depth < 3:
                return fetch_wikitext(m.group(1).strip(), max_retries, _redirect_depth + 1)
            return text
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [warn] fetch failed for {title!r}: {e}")
                return ""
            time.sleep(2 ** attempt)
    return ""


# ------------------------------------------------------------
# Hockey-box parsing (handles 3 template-name variants)
# ------------------------------------------------------------
# {{Ice hockey box}}, {{hockeybox2}}, {{IceHockeybox}} all share field names.
# team1 / team2 hold {{ih|CODE}} or {{ih-rt|CODE}} (winner bolded with ''').
# score is "A–B" (en-dash). date is free-text; handled by _parse_date.

_IH_CODE_RE = re.compile(r"\{\{\s*ih(?:-rt|-rb)?\s*\|\s*([^}|]+)", re.IGNORECASE)

# Full country name -> IOC code, used when a game has a team encoded as a
# plain wikilink like [[Russia men's national ice hockey team|Olympic
# Athletes from Russia]] rather than the standard {{ih|RUS}} template.
# (Hits the 2018 OAR + 2022 ROC special designations, and any older edition
# that spells the country out.)
_NAME_TO_CODE = {
    # Major hockey nations
    "United States": "USA", "Canada": "CAN", "Russia": "RUS", "Sweden": "SWE",
    "Finland": "FIN", "Czechoslovakia": "TCH", "Czech Republic": "CZE",
    "Czechia": "CZE", "Slovakia": "SVK", "Germany": "GER", "Switzerland": "SUI",
    "Norway": "NOR", "Denmark": "DEN", "Austria": "AUT", "Latvia": "LAT",
    "Belarus": "BLR", "Kazakhstan": "KAZ", "France": "FRA", "Japan": "JPN",
    "South Korea": "KOR", "Korea": "KOR", "Italy": "ITA", "Great Britain": "GBR",
    "Netherlands": "NED", "Poland": "POL", "Hungary": "HUN", "Romania": "ROU",
    "Ukraine": "UKR", "Slovenia": "SLO", "Croatia": "CRO",
    # Era-specific entities (kept distinct: Soviet Union != Russia)
    "Soviet Union": "URS", "East Germany": "GDR", "West Germany": "FRG",
    "Yugoslavia": "YUG", "Serbia and Montenegro": "SCG", "Serbia": "SRB",
    # Misc
    "Bulgaria": "BUL", "Estonia": "EST", "Lithuania": "LTU",
    "China": "CHN", "Australia": "AUS", "Spain": "ESP", "Belgium": "BEL",
    "Iceland": "ISL", "Israel": "ISR", "Mexico": "MEX", "Greece": "GRE",
    # Special-designation aliases -> the same nation
    "Olympic Athletes from Russia": "RUS",  # 2018 PyeongChang
    "Russian Olympic Committee": "RUS",     # 2022 Beijing
    "ROC": "RUS",
}
# A name like "ice hockey box" / "icehockeybox" / "hockeybox2" / "Hockeybox" --
# normalize by removing internal spaces / case.
_HOCKEYBOX_NAME_RE = re.compile(r"icehockeybox|hockeybox2?", re.IGNORECASE)


def _split_top_pipes(s):
    """Split a template body on top-level '|' (ignoring nested {{}}, [[]])."""
    parts, depth_c, depth_b, cur = [], 0, 0, []
    i = 0
    while i < len(s):
        two = s[i:i + 2]
        if two == "{{":
            depth_c += 1; cur.append(two); i += 2
        elif two == "}}":
            depth_c -= 1; cur.append(two); i += 2
        elif two == "[[":
            depth_b += 1; cur.append(two); i += 2
        elif two == "]]":
            depth_b -= 1; cur.append(two); i += 2
        elif s[i] == "|" and depth_c == 0 and depth_b == 0:
            parts.append("".join(cur)); cur = []; i += 1
        else:
            cur.append(s[i]); i += 1
    parts.append("".join(cur))
    return parts


_HEADER_RE = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$", re.MULTILINE)


def _round_for_position(text, pos):
    """Section-header classifier: 'final' / 'bronze' / 'semifinal' / ''.
    Used so the engine can tell the gold-medal game from the bronze game on
    the same closing date."""
    label = ""
    for m in _HEADER_RE.finditer(text):
        if m.start() > pos:
            break
        h = m.group(2).lower()
        if "championship final" in h or h.strip() in ("final", "finals", "gold medal game", "gold medal match"):
            label = "final"
        elif "bronze" in h or "third place" in h or "3rd place" in h:
            label = "bronze"
        elif "semifinal" in h or "semi-final" in h or "semi final" in h:
            label = "semifinal"
        elif "final" in h and "quarterfinal" not in h and "qualif" not in h:
            label = "final"
        else:
            label = ""
    return label


def _section_at(wikitext, pos):
    """Lowercase title of the deepest section header containing `pos`."""
    title = ""
    for m in _HEADER_RE.finditer(wikitext):
        if m.start() > pos:
            break
        title = m.group(2).lower()
    return title


def _section_path_at(wikitext, pos):
    """Concatenated lowercase titles of EVERY section header enclosing `pos`
    (outermost h2 down to deepest), joined with ' / '. Lets the friendly
    filter catch boxes nested inside e.g. 'Pre-tournament games / In North
    America' where the deepest header alone ('In North America') would slip
    through."""
    stack = []  # (level, title)
    for m in _HEADER_RE.finditer(wikitext):
        if m.start() > pos:
            break
        level = len(m.group(1))
        title = m.group(2).lower()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
    return " / ".join(t for _, t in stack)


def _hockeybox_blocks(text):
    """Yield (block_body, start_offset) for each {{... hockey box ...}} block."""
    i = 0
    while True:
        m = re.search(r"\{\{\s*(IceHockeybox|Ice\s*hockey\s*box|hockeybox2?)\b",
                      text[i:], re.IGNORECASE)
        if not m:
            return
        start = i + m.start()
        depth, j = 0, start
        while j < len(text):
            two = text[j:j + 2]
            if two == "{{":
                depth += 1; j += 2
            elif two == "}}":
                depth -= 1; j += 2
                if depth == 0:
                    yield text[start:j], start
                    break
            else:
                j += 1
        else:
            return
        i = j


def _extract_code(raw):
    """Pull the IOC team code. Tries {{ih|CODE}} / {{ih-rt|CODE}} first, then
    a wikilink fallback for cases like
    [[Russia men's national ice hockey team|Olympic Athletes from Russia]]
    where no team template is used (2018 OAR, 2022 ROC, some older editions)."""
    m = _IH_CODE_RE.search(raw)
    if m:
        return m.group(1).strip().upper()
    m = re.search(r"\[\[([^|\]]+?)\s+(?:men's\s+)?national\s+ice\s+hockey\s+team",
                  raw, re.IGNORECASE)
    if m:
        country = m.group(1).strip()
        if country in _NAME_TO_CODE:
            return _NAME_TO_CODE[country]
    # Display-name fallback: "Olympic Athletes from Russia" / "ROC" appears
    # without a wikilink wrapper in some boxes.
    for name, code in _NAME_TO_CODE.items():
        if name in raw and len(name) > 3:  # skip short ambiguous tokens
            return code
    return None


def _parse_score(raw):
    """Parse 'A–B' (en-dash or hyphen)."""
    clean = raw.replace("'''", "")
    clean = re.sub(r"\[\[[^\]]*\|", "", clean).replace("[[", "").replace("]]", "")
    m = re.search(r"(\d+)\s*[–-\-]\s*(\d+)", clean)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _parse_date(raw, default_year):
    """Parse many date formats: {{dts}}/{{Start date}} templates, DMY, MDY,
    Weekday-prefixed, and no-year fallbacks."""
    s = raw

    # {{dts|format=dmy|YYYY|M|D}} or {{Start date|YYYY|M|D|...}}
    m = re.search(r"\{\{\s*(?:dts|start\s*date)\b([^}]*)\}\}", s, re.IGNORECASE)
    if m:
        nums = [int(x) for x in re.findall(r"\b\d{1,4}\b", m.group(1))]
        if len(nums) >= 3:
            yr, mo, dy = nums[0], nums[1], nums[2]
            if yr < 100:
                yr += 2000
            try:
                from datetime import date
                return date(yr, mo, dy)
            except Exception:
                pass

    s = re.sub(r"\[\[[^\]]*\|", "", s).replace("[[", "").replace("]]", "")
    s = re.sub(r"\{\{[^}]*\}\}", "", s)
    s = s.strip().rstrip(",").strip()
    s = re.sub(r"^[A-Za-z]+,\s*", "", s)
    fmts = ["%b %d, %Y", "%B %d, %Y", "%d %B %Y", "%d %b %Y",
            "%B %d %Y", "%Y-%m-%d"]
    for f in fmts:
        try:
            return datetime.strptime(s, f).date()
        except ValueError:
            pass
    for f in ["%d %B", "%d %b", "%B %d", "%b %d"]:
        try:
            return datetime.strptime(s, f).replace(year=default_year).date()
        except ValueError:
            pass
    return None


def _field(parts, key):
    """Find a named field (|key=value) in a list of pipe-split body parts."""
    pat = re.compile(rf"^\s*{re.escape(key)}\s*=\s*(.*)\s*$", re.IGNORECASE | re.DOTALL)
    for p in parts:
        m = pat.match(p)
        if m:
            return m.group(1).strip()
    return ""


# Sections / contexts known to contain non-tournament games (filtered out so
# pre-tournament friendlies on the same page don't pollute the ratings).
_FRIENDLY_SECTION_RE = re.compile(
    r"pre[\s\-]?tournament|exhibition|friendly|warm[\s\-]?up|prelim(?:inary)?\s*friendly",
    re.IGNORECASE,
)


def parse_hockey_boxes(wikitext, tournament, season):
    """Yield game dicts from {{Ice hockey box}} / {{hockeybox2}} /
    {{IceHockeybox}} blocks. Skips boxes inside pre-tournament / exhibition
    sections."""
    rows = []
    for block, pos in _hockeybox_blocks(wikitext):
        if _FRIENDLY_SECTION_RE.search(_section_path_at(wikitext, pos)):
            continue
        body = block[block.find("|") + 1: block.rfind("}}")]
        parts = _split_top_pipes(body)
        date_raw = _field(parts, "date")
        t1_raw = _field(parts, "team1")
        t2_raw = _field(parts, "team2")
        score_raw = _field(parts, "score")
        c1 = _extract_code(t1_raw)
        c2 = _extract_code(t2_raw)
        sa, sb = _parse_score(score_raw)
        if not (c1 and c2 and sa is not None and sb is not None):
            continue
        gdate = _parse_date(date_raw, season)
        if gdate is None:
            continue
        rows.append({
            "date": gdate, "tournament": tournament, "season": season,
            "road_team": c1, "road_runs": sa,        # team1 = road
            "home_team": c2, "home_runs": sb,        # team2 = home
            "round": _round_for_position(wikitext, pos),
        })
    return rows


# Legacy "Scores" bullet format used by the 1996 WCoH page (no box templates;
# games encoded as one line each):  *August 29, Vancouver: Russia 3–5 Canada
_BULLET_GAME_RE = re.compile(
    r"^\*\s*([A-Z][a-z]+\s+\d{1,2})\s*,\s*[^:]+:\s*"  # *Month DD, City:
    r"(.+?)\s+(\d+)\s*[–-\-]\s*(\d+)\s+(.+?)\s*$",     # team1 A–B team2
    re.MULTILINE,
)


def parse_score_bullets(wikitext, tournament, season):
    """Older 'Scores' subsections list games as bulleted text rather than box
    templates. Match those lines and map full country names -> IOC codes."""
    rows = []
    for m in _BULLET_GAME_RE.finditer(wikitext):
        date_raw = m.group(1)
        team1 = m.group(2).strip()
        sa = int(m.group(3)); sb = int(m.group(4))
        # Strip a trailing parenthetical (handles "(OT)", "(2OT)", "(SO)" etc.
        # on game-result lines so "Sweden (2OT)" -> "Sweden").
        team2 = re.sub(r"\s*\([^)]*\)\s*$", "", m.group(5)).strip()
        c1 = _bullet_team_code(team1)
        c2 = _bullet_team_code(team2)
        if not (c1 and c2):
            continue
        gdate = _parse_date(date_raw, season)
        if not gdate:
            continue
        rows.append({
            "date": gdate, "tournament": tournament, "season": season,
            "road_team": c1, "road_runs": sa,
            "home_team": c2, "home_runs": sb,
            "round": _round_for_position(wikitext, m.start()),
        })
    return rows


# Old-style bullets use IOC abbreviations rather than spelled-out names
# ("USSR" / "USA" on 1976 Olympics, etc.). Kept separate from _NAME_TO_CODE so
# the hockey-box substring fallback can't false-match on short tokens.
_BULLET_ABBREV = {
    "USSR": "URS", "USA": "USA",
    "GDR": "GDR", "FRG": "FRG",
    "TCH": "TCH", "FIN": "FIN", "SWE": "SWE", "CAN": "CAN",
}


def _bullet_team_code(name):
    return _NAME_TO_CODE.get(name) or _BULLET_ABBREV.get(name)


def _clean_team_name(s):
    """Strip bold wrappers, embedded templates, and trailing parenthetical
    notes (e.g. '(5th)', '(OT)') from a team-name token in a bullet line."""
    s = re.sub(r"\{\{[^{}]+\}\}", "", s)        # drop {{efn|...}} etc.
    s = re.sub(r"'''(.+?)'''", r"\1", s)        # unwrap bold
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s)      # trailing (5th) / (OT)
    return s.strip()


_INDENTED_DATE_RE = re.compile(r"^\*\s*([A-Z][a-z]+\s+\d{1,2})\s*$")
_INDENTED_GAME_RE = re.compile(
    r"^\*\*\s*(.+?)\s+(\d+)\s*[–-\-]\s*(\d+)\s+(.+?)\s*$"
)


def parse_indented_bullets(wikitext, tournament, season):
    """Pre-1980 Olympic event pages list results as
       *Month DD
       **Team1 A-B Team2
       **Team1 A-B Team2
    Each ** line is a game whose date inherits from the preceding * header."""
    rows = []
    cur_date = None
    pos = 0
    for line in wikitext.split("\n"):
        line_pos = pos
        pos += len(line) + 1
        cleaned = re.sub(r"\{\{[^{}]+\}\}", "", line)  # drop inline templates
        m = _INDENTED_DATE_RE.match(cleaned)
        if m:
            d = _parse_date(m.group(1), season)
            if d:
                cur_date = d
            continue
        m = _INDENTED_GAME_RE.match(cleaned)
        if not (m and cur_date):
            continue
        t1 = _clean_team_name(m.group(1))
        t2 = _clean_team_name(m.group(4))
        c1 = _bullet_team_code(t1)
        c2 = _bullet_team_code(t2)
        if not (c1 and c2):
            continue
        rows.append({
            "date": cur_date, "tournament": tournament, "season": season,
            "road_team": c1, "road_runs": int(m.group(2)),
            "home_team": c2, "home_runs": int(m.group(3)),
            "round": _round_for_position(wikitext, line_pos),
        })
    return rows


# ------------------------------------------------------------
# Sub-page discovery + per-event scrape
# ------------------------------------------------------------
def discover_subpages(main_title, wikitext):
    """Find sub-pages via [[wikilinks]] AND {{main|}}/{{#lst:}} transclusions.
    Qualifier/roster feeders are filtered out."""
    subs = set()
    base = re.escape(main_title)
    for m in re.finditer(rf"\[\[({base}[^\]|#]+)", wikitext):
        page = m.group(1).strip()
        if not _SKIP_SUBPAGE_RE.search(page):
            subs.add(page)
    for m in re.finditer(rf"\{{\{{\s*(?:main|#lst|#lstx)\s*[:|]\s*({base}[^|}}#\n]+)",
                         wikitext, re.IGNORECASE):
        page = m.group(1).strip()
        if not _SKIP_SUBPAGE_RE.search(page):
            subs.add(page)
    return sorted(subs)


def scrape_event(main_title, tournament, season):
    """Scrape one event: main page + auto-discovered sub-pages."""
    main_wt = fetch_wikitext(main_title)
    if not main_wt:
        print(f"  [warn] no wikitext for event {main_title!r}")
        return pd.DataFrame()
    pages = [main_title] + discover_subpages(main_title, main_wt)
    all_rows, cache = [], {main_title: main_wt}
    for p in pages:
        wt = cache.get(p) or fetch_wikitext(p)
        rows = parse_hockey_boxes(wt, tournament, season)
        # Legacy fallbacks for older pages with no box templates:
        #   - parse_score_bullets: 1996 WCoH "*Date, City: Team A-B Team" lines
        #   - parse_indented_bullets: 1976 Olympics "*Date\n**Team A-B Team"
        # Modern pages don't use these formats, so no double-counting risk.
        rows += parse_score_bullets(wt, tournament, season)
        rows += parse_indented_bullets(wt, tournament, season)
        if rows:
            print(f"    {p}: {len(rows)} games")
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    if len(df):
        if "round" not in df.columns:
            df["round"] = ""
        df["round"] = df["round"].fillna("")
        # Strict (date, teams, scores) dedup -- NOT the team-set-only key used
        # for ICHIRO. Hockey has best-of-3 series (1987 Canada Cup final, 2016
        # WCoH final) where the same two teams play multiple games with
        # identical scores on different dates; a set-only key wrongly merges
        # them. Cross-source date conflicts are rare on hockey pages, so the
        # strict date-inclusive key is the right call here.
        df["_has_round"] = (df["round"] != "").astype(int)
        df = (df.sort_values("_has_round", ascending=False)
                .drop_duplicates(subset=["date", "road_team", "home_team",
                                         "road_runs", "home_runs"], keep="first")
                .drop(columns="_has_round"))
        df["tier"] = TIER_WEIGHTS.get(tournament, 1.0)
        df["neutral"] = NEUTRAL_DEFAULT.get(tournament, True)
    return df


# ------------------------------------------------------------
# Append-only union (data-integrity guard, fleet pattern)
# ------------------------------------------------------------
def union_with_existing(fresh_df, path=ALL_GAMES_CSV):
    if not os.path.exists(path):
        return fresh_df
    prev = pd.read_csv(path)
    prev["date"] = pd.to_datetime(prev["date"], errors="coerce").dt.date
    fresh = fresh_df.copy()
    fresh["date"] = pd.to_datetime(fresh["date"], errors="coerce").dt.date
    key = ["date", "road_team", "home_team", "road_runs", "home_runs"]
    f = fresh.copy(); f["_pri"] = 0
    p = prev.copy(); p["_pri"] = 1
    combined = pd.concat([f, p], ignore_index=True, sort=False)
    combined = combined.sort_values("_pri").drop_duplicates(subset=key, keep="first")
    fk = set(map(tuple, fresh[key].astype(str).values))
    preserved = sum(1 for k in map(tuple, prev[key].astype(str).values) if k not in fk)
    if preserved:
        print(f"[db-union] preserved {preserved:,} stored games this run's fetch "
              f"did not return (flaky source -- not deleting history)")
    return combined.drop(columns=["_pri"]).reset_index(drop=True)


def build_dataset(events, write=True):
    frames = []
    for main_title, tournament, season in events:
        print(f"== {main_title} ({tournament} {season}) ==")
        df = scrape_event(main_title, tournament, season)
        if len(df):
            print(f"   -> {len(df)} games")
            frames.append(df)
    fresh = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not len(fresh):
        print("No games scraped this run.")
        return fresh
    fresh = fresh.sort_values(["date", "road_team", "home_team"]).reset_index(drop=True)
    merged = union_with_existing(fresh)
    merged = merged.sort_values(["date", "road_team", "home_team"]).reset_index(drop=True)
    if write:
        merged.to_csv(ALL_GAMES_CSV, index=False)
        print(f"\nWrote {ALL_GAMES_CSV}: {len(merged)} games total.")
    return merged


if __name__ == "__main__":
    df = build_dataset(EVENTS)
    if len(df):
        print(f"\n=== Totals by tournament ===")
        print(df.groupby("tournament").size().to_string())
