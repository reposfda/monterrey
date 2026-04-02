# core/scrap_transfers_from_tmkt.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path

import config


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-AR,es;q=0.9,en;q=0.8",
}

CLUBS_FILE = config.DATA_DIR / "maestros" / "club_names.csv"
TRANSFERS_DIR = config.DATA_DIR / "transfers"

# =============================================================================
# HELPERS
# =============================================================================

def money_to_eur_float(txt: str) -> float | None:
    """
    Convert strings like:
      '4,00 mill. €', '500 mil €', '1,25 mill. €', '750 mil €', '-',
      'Libre', 'Cesión', 'Fin de cesión 30/06/2014'
    into a numeric euro value (float). Returns None if not a numeric fee.
    """
    if not txt:
        return None

    t = txt.lower().strip()

    non_numeric = ["-", "libre", "cesión", "cesion"]
    if any(x in t for x in non_numeric) or "fin de cesión" in t or "fin de cesion" in t:
        return None

    t = t.replace("€", "").replace("eur", "").strip()

    is_million = "mill" in t
    is_thousand = re.search(r"\bmil\b", t) is not None

    m = re.search(r"([\d.,]+)", t)
    if not m:
        return None

    num = m.group(1)
    num = num.replace(".", "").replace(",", ".")

    try:
        val = float(num)
    except ValueError:
        return None

    if is_million:
        return val * 1_000_000.0
    if is_thousand:
        return val * 1_000.0

    return val


def position_short(pos: str) -> str:
    """
    Simple, editable mapping to short codes.
    """
    if not pos:
        return ""

    p = pos.strip().lower()
    mapping = {
        "portero": "GK",
        "defensa central": "CB",
        "libero": "SW",
        "lateral derecho": "RB",
        "lateral izquierdo": "LB",
        "pivote": "DM",
        "mediocentro defensivo": "DM",
        "mediocentro": "CM",
        "interior derecho": "RM",
        "interior izquierdo": "LM",
        "mediocentro ofensivo": "AM",
        "mediapunta": "AM",
        "extremo derecho": "RW",
        "extremo izquierdo": "LW",
        "delantero centro": "CF",
    }
    return mapping.get(p, "".join(w[0].upper() for w in pos.split()))


def get_id_from_href(href: str, key: str) -> str | None:
    """
    key='spieler' or 'verein' to pull IDs from URLs like:
      /perfil/jugador/manuel-lanzini/spieler/167012
      /al-jazira-club/startseite/verein/631
    """
    if not href:
        return None

    m = re.search(rf"/{re.escape(key)}/(\d+)", href)
    return m.group(1) if m else None


def clean_text(x: str) -> str:
    return re.sub(r"\s+", " ", x).strip() if x else ""


def fetch_soup(url: str) -> BeautifulSoup:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "lxml")


def find_td_idx_for_href(tr, pattern: str) -> int | None:
    """Return the TD index within this TR that contains an <a href=...> matching pattern."""
    tds = tr.find_all("td")
    for i, td in enumerate(tds):
        if td.find("a", href=re.compile(pattern)):
            return i
    return None


def get_country_flags(td) -> list[str]:
    """
    Return country names from flag IMG(s) inside this TD.
    Robust to different TM paths/classes.
    """
    names = []
    for img in td.find_all("img"):
        alt = (img.get("alt") or img.get("title") or "").strip()
        src = (img.get("src") or "").lower()
        cls = " ".join(img.get("class", [])).lower()

        if alt and ("flagge" in src or "flaggen" in src or "flaggenrahmen" in cls):
            names.append(alt)

    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)

    return out


def extract_season(soup: BeautifulSoup) -> tuple[str | None, str | None]:
    """
    Find the season <select name="saison_id">, get the <option selected>,
    and return:
      periodo -> option value (e.g., '2014')
      season  -> option text (e.g., '14/15')
    If no selected option is present, falls back to the first option.
    """
    selected_season = soup.select_one('select[name="saison_id"]')
    if not selected_season:
        return None, None

    opt = selected_season.select_one('option[selected]') or selected_season.find('option', selected=True)
    if not opt:
        opt = selected_season.find('option')

    if not opt:
        return None, None

    periodo = (opt.get('value') or '').strip()
    season = opt.get_text(strip=True)

    return periodo, season


def detect_transfer_window(soup: BeautifulSoup) -> str:
    """
    Detect current window filter from the page.
    """
    href = None

    tabs = soup.find('div', class_='tm-tabs')
    if tabs:
        a_active = tabs.find('a', class_=lambda c: c and 'tm-tab__active--parent' in c)
        if a_active and a_active.has_attr('href'):
            href = a_active['href']
        else:
            for a in tabs.find_all('a', class_='tm-tab'):
                if a.find(class_='tm-tab__active') and a.has_attr('href'):
                    href = a['href']
                    break

    if not href:
        a_any = soup.find('a', href=lambda h: h and '/w_s/' in h)
        href = a_any['href'] if a_any else None

    if not href:
        return 'Desconocido'

    if re.search(r'/w_s/w(?:/|$)', href):
        return 'Invierno europeo'
    if re.search(r'/w_s/s(?:/|$)', href):
        return 'Verano europeo'
    if re.search(r'/w_s/(?:/|$)', href):
        return 'Sin especificar'

    return 'unknown'


def extract_transfer_id_from_href(href: str) -> str | None:
    m = re.search(r"/transfer_id/(\d+)", href or "")
    return m.group(1) if m else None


def extract_transfer_type(coste_txt: str) -> str:
    s = (coste_txt or "").strip()
    has_eur = "€" in s

    pat_loan_word = r'\b(loan|pr[eé]stamo|cesi[óo]n)\b'
    pat_free = r'\b(free\s*transfer|libre)\b'
    pat_end_loan_es = r'\bfin\s*de\s*cesi[óo]n(?:\s*\d{1,2}/\d{1,2}/\d{2,4})?\s*$'
    pat_end_loan_en = r'\bend\s*of\s*loan\b'
    pat_exact_loan = r'^\s*(cesi[óo]n|loan\s+transfer)\s*$'
    pat_loan_es = r'\bcesi[óo]n\b'

    if re.search(pat_end_loan_es, s, flags=re.I) or re.search(pat_end_loan_en, s, flags=re.I):
        return "Fin de prestamo"

    if re.search(pat_free, s, flags=re.I):
        return "Libre"

    if has_eur and re.search(pat_loan_es, s, flags=re.I):
        return "Prestamo pago"

    if re.fullmatch(pat_exact_loan, s, flags=re.I):
        return "Prestamo"

    if has_eur and not re.search(pat_loan_word, s, flags=re.I):
        return "Transferencia"

    return s.title()


def _parse_amount_eur(text: str) -> float | None:
    """Return the fee in euros as a float, or None if not numeric."""
    if not text:
        return None

    try:
        return money_to_eur_float(text)
    except NameError:
        pass

    t = str(text).strip()

    m = re.search(r'€\s*([\d\.,]+)\s*([mk])', t, flags=re.I)
    if m:
        num = m.group(1).replace(',', '')
        try:
            val = float(num)
        except ValueError:
            return None
        unit = m.group(2).lower()
        return val * (1_000_000 if unit == 'm' else 1_000)

    m2 = re.search(r'([\d\.,]+)\s*(mill\.?|mil)\s*€', t, flags=re.I)
    if m2:
        num = m2.group(1).replace('.', '').replace(',', '.')
        try:
            val = float(num)
        except ValueError:
            return None
        unit = m2.group(2).lower()
        return val * (1_000_000 if 'mill' in unit else 1_000)

    return None


def extract_transfer_price(coste_txt: str, transfer_type: str):
    """
    Return rounded EUR amount (int) if:
      - transfer_type == 'Transferencia', or
      - transfer_type == 'Prestamo pago', or
      - transfer_type == 'Prestamo' AND coste_txt has a numeric € amount.
    Else return None.
    """
    t = (transfer_type or '').strip().lower()
    t = (
        t.replace('ó', 'o')
        .replace('á', 'a')
        .replace('é', 'e')
        .replace('í', 'i')
        .replace('ú', 'u')
    )

    amt = _parse_amount_eur(coste_txt)
    if amt is None:
        return None

    if t in ('transferencia', 'prestamo pago'):
        return int(round(amt))

    if t == 'prestamo':
        return int(round(amt))

    return None


def extract_transfer_date(coste_txt: str, transfer_type: str):
    """
    Return 'dd/mm/yyyy' when transfer_type == 'Fin de prestamo' and a date
    of the form dd.mm.yyyy or dd/mm/yyyy is present in coste_txt.
    Otherwise return None.
    """
    t = (transfer_type or '').strip().lower()
    t = (
        t.replace('ó', 'o')
        .replace('á', 'a')
        .replace('é', 'e')
        .replace('í', 'i')
        .replace('ú', 'u')
    )
    if t != 'fin de prestamo':
        return None

    s = (coste_txt or '').strip()
    m = re.search(r'(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})', s)
    if not m:
        return None

    raw = m.group(1)
    std = re.sub(r'[.-]', '/', raw)
    parts = std.split('/')

    if len(parts) != 3:
        return None

    dd, mm, yy = parts[0].zfill(2), parts[1].zfill(2), parts[2]

    if len(yy) == 2:
        yy = '20' + yy

    try:
        dt = datetime.strptime(f'{dd}/{mm}/{yy}', '%d/%m/%Y')
        return dt.strftime('%d/%m/%Y')
    except ValueError:
        return None

# =============================================================================
# CORE PARSERS
# =============================================================================

def parse_balance_table(soup: BeautifulSoup, club_id, club_name) -> pd.DataFrame:
    """
    Extract the little 'Balance de fichajes' box.
    Returns a tidy 1-row DataFrame with numeric euros and counts where available.
    """
    h2 = None
    for tag in soup.find_all(["h2", "h3"]):
        title = tag.get_text(" ", strip=True)
        if re.search(r"\bbalance\s+de\s+fichajes\b", title, flags=re.IGNORECASE):
            h2 = tag
            break

    if not h2:
        return pd.DataFrame(columns=[
            "club_id", "club_name", "periodo", "season",
            "altas_cant", "bajas_cant", "ingresos_eur",
            "gastos_eur", "balance_eur"
        ])

    box = h2.find_next()
    text = box.get_text(" ", strip=True) if box else ""

    def find_count_and_money(label):
        cnt = None
        eur = None
        m = re.search(rf"{label}\s+(\d+)\s+([^\s].*?€)", text, flags=re.I)
        if m:
            cnt = int(m.group(1))
            eur = money_to_eur_float(m.group(2))
        return cnt, eur

    bajas_cnt, ingresos_eur = find_count_and_money("Ingresos")
    altas_cnt, gastos_eur = find_count_and_money("Gastos")

    periodo, season = extract_season(soup)

    bal = None
    m_bal = re.search(r"Balance total\s+([^\s].*?€)", text, flags=re.I)
    if m_bal:
        bal = money_to_eur_float(m_bal.group(1))

    return pd.DataFrame([{
        "club_id": club_id,
        "club_name": club_name,
        "periodo": periodo,
        "season": season,
        "altas_cant": altas_cnt,
        "bajas_cant": bajas_cnt,
        "ingresos_eur": ingresos_eur,
        "gastos_eur": gastos_eur,
        "balance_eur": bal
    }])


def parse_transfers_table(
    soup: BeautifulSoup,
    header_text: str,
    direction_value: str,
    club_id: str,
    club_name: str
) -> pd.DataFrame:
    h2 = None
    for tag in soup.find_all(["h2", "h3"]):
        if tag.get_text(strip=True).lower() == header_text.lower():
            h2 = tag
            break

    if not h2:
        return pd.DataFrame(columns=[
            "periodo", "season", "transfer_window",
            "club_id", "club_name", "direction", "player_id", "player_name", "player_age",
            "player_nationality", "player_position", "player_position_short",
            "market_value", "counterparty_club_id", "counterparty_club_name",
            "counterparty_club_country", "Fee", "transfer_id", "transfer_type",
            "transfer_price", "transfer_date",
        ])

    table = h2.find_next("table")
    if not table:
        wrap = h2.find_next("div")
        if wrap:
            table = wrap.find("table")
    if not table:
        return pd.DataFrame()

    rows = []

    for tr in table.select("tbody tr"):
        cls = " ".join(tr.get("class", []))
        if "bg_blau_20" in cls or "thead" in cls:
            continue

        tds = tr.find_all("td")
        if len(tds) < 5:
            continue

        idx_player = find_td_idx_for_href(tr, r"/spieler/\d+")
        idx_club = find_td_idx_for_href(tr, r"/verein/\d+")
        if idx_player is None:
            continue

        player_a = tds[idx_player].find("a", href=re.compile(r"/spieler/\d+"))
        player_name = clean_text(player_a.get_text()) if player_a else ""
        player_id = get_id_from_href(player_a["href"], "spieler") if player_a else None

        player_pos = ""
        name_td = tds[idx_player]
        td_text = " ".join(name_td.stripped_strings)
        if player_name and td_text:
            player_pos = clean_text(td_text.replace(player_name, "").strip())
        if not player_pos:
            small = name_td.find("span") or name_td.find("div")
            if small:
                player_pos = clean_text(small.get_text())

        player_age = None
        for td in tds:
            txt = clean_text(td.get_text())
            if txt.isdigit():
                player_age = int(txt)
                break

        player_nationality = ""
        search_end = len(tds) if idx_club is None else idx_club

        for i in range(idx_player, search_end):
            if idx_club is not None and i >= idx_club:
                break
            if tds[i].find("a", href=re.compile(r"/verein/\d+")):
                continue
            flags = get_country_flags(tds[i])
            if flags:
                player_nationality = "|".join(flags[:3])
                break

        if not player_nationality:
            flags = get_country_flags(name_td)
            if flags:
                player_nationality = "|".join(flags[:3])

        if not player_nationality:
            for i in range(0, search_end):
                if tds[i].find("a", href=re.compile(r"/verein/\d+")):
                    continue
                flags = get_country_flags(tds[i])
                if flags:
                    player_nationality = "|".join(flags[:3])
                    break

        counterparty_club_id = None
        counterparty_club_name = ""
        counterparty_club_country = ""

        club_td = None
        if idx_club is not None:
            club_td = tds[idx_club]
            club_a = club_td.find("a", href=re.compile(r"/verein/\d+"))
        else:
            club_a = tr.find("a", href=re.compile(r"/verein/\d+"))
            if club_a:
                club_td = club_a.find_parent("td")

        if club_a:
            counterparty_club_id = get_id_from_href(club_a["href"], "verein")
            counterparty_club_name = clean_text(club_a.get_text()) or clean_text(club_a.get("title") or "")
            if not counterparty_club_name and club_td:
                crest = club_td.find("img", alt=True)
                if crest and "/wappen/" in (crest.get("src") or "").lower():
                    counterparty_club_name = clean_text(crest.get("alt") or "")

        if club_td:
            flags = get_country_flags(club_td)
            if flags:
                counterparty_club_country = flags[0]
            else:
                try:
                    club_idx = tds.index(club_td)
                except ValueError:
                    club_idx = None

                if club_idx is not None and club_idx + 1 < len(tds):
                    flags = get_country_flags(tds[club_idx + 1])
                    if flags:
                        counterparty_club_country = flags[0]

        market_value = None
        for td in tds:
            txt = clean_text(td.get_text())
            if "€" in txt or "mill" in txt or re.search(r"\bmil\b", txt):
                val = money_to_eur_float(txt)
                if val is not None:
                    market_value = val
                    break

        coste_txt = ""
        transfer_id = None
        for td in reversed(tds):
            txt = clean_text(td.get_text())
            if txt and (
                txt == "-" or "€" in txt or "mil" in txt or "mill" in txt
                or "libre" in txt.lower() or "cesi" in txt.lower()
            ):
                coste_txt = txt
                a = td.find("a", href=re.compile(r"/transfer_id/"))
                if a and a.has_attr("href"):
                    transfer_id = extract_transfer_id_from_href(a["href"])
                break

        if transfer_id is None:
            a_any = tr.find("a", href=re.compile(r"/transfer_id/"))
            if a_any and a_any.has_attr("href"):
                transfer_id = extract_transfer_id_from_href(a_any["href"])

        transfer_type = extract_transfer_type(coste_txt)
        transfer_price = extract_transfer_price(coste_txt, transfer_type)
        transfer_date = extract_transfer_date(coste_txt, transfer_type)

        periodo, season = extract_season(soup)
        transfer_window = detect_transfer_window(soup)

        rows.append({
            "periodo": periodo,
            "season": season,
            "transfer_window": transfer_window,
            "club_id": str(club_id),
            "club_name": club_name,
            "direction": direction_value,
            "player_id": player_id,
            "player_name": player_name,
            "player_age": player_age,
            "player_nationality": player_nationality,
            "player_position": player_pos,
            "player_position_short": position_short(player_pos),
            "market_value": market_value,
            "counterparty_club_id": counterparty_club_id,
            "counterparty_club_name": counterparty_club_name,
            "counterparty_club_country": counterparty_club_country,
            "Fee": coste_txt,
            "transfer_id": transfer_id,
            "transfer_type": transfer_type,
            "transfer_price": transfer_price,
            "transfer_date": transfer_date
        })

    df = pd.DataFrame(rows, columns=[
        "periodo", "season", "transfer_window",
        "club_id", "club_name", "direction", "player_id", "player_name", "player_age",
        "player_nationality", "player_position", "player_position_short",
        "market_value", "counterparty_club_id", "counterparty_club_name",
        "counterparty_club_country", "Fee", "transfer_id", "transfer_type",
        "transfer_price", "transfer_date",
    ])

    return df

# =============================================================================
# RUNNER
# =============================================================================

def scrape_transfers(base_url: str, club_info: dict):
    soup = fetch_soup(base_url)

    periodo, season = extract_season(soup)
    transfer_window = detect_transfer_window(soup)

    club_id = club_info["club_id"]
    club_name = club_info["club_name"]

    altas_df = parse_transfers_table(soup, "Altas", "In", club_id, club_name)
    bajas_df = parse_transfers_table(soup, "Bajas", "Out", club_id, club_name)
    balance_df = parse_balance_table(soup, club_id, club_name)

    if not balance_df.empty:
        balance_df["neto"] = balance_df["ingresos_eur"] - balance_df["gastos_eur"]

    return club_name, periodo, transfer_window, altas_df, bajas_df, balance_df


def build_transfer_url(club_info: dict, season: int, window: str) -> str:
    return (
        f"https://www.transfermarkt.com.ar/"
        f"{club_info['club_url_name']}/transfers/verein/{club_info['club_id']}/"
        f"saison_id/{season}/pos//detailpos/0/w_s/{window}/plus/1#zugaenge"
    )


def combine_transfers(altas: pd.DataFrame, bajas: pd.DataFrame) -> pd.DataFrame:
    dfs = [df for df in [altas, bajas] if not df.empty]
    dfs = [df.dropna(axis=1, how="all") for df in dfs]
    dfs = [df for df in dfs if not df.empty]

    return pd.concat(dfs, axis=0, ignore_index=True) if dfs else pd.DataFrame()


def sanitize_filename(text: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", str(text)).strip()


def transfer_window_to_filename(transfer_window: str) -> str:
    if transfer_window == "Invierno europeo":
        return "inv_europeo"
    if transfer_window == "Verano europeo":
        return "ver_europeo"
    if transfer_window in ["Desconocido", "Sin especificar", "unknown"]:
        return "ambas_desconocida"
    return "ambas_desconocida"


def load_clubs() -> list[dict]:
    clubes_mx = pd.read_csv(CLUBS_FILE)

    clubes_list = [
        {
            "club_id": str(cid).strip(),
            "club_name": str(name),
            "club_url_name": str(url_name).strip(),
        }
        for cid, name, url_name in clubes_mx[
            ["club_id", "club_name", "club_url_name"]
        ].dropna().itertuples(index=False, name=None)
    ]

    return clubes_list


def run_scraping() -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    TRANSFERS_DIR.mkdir(parents=True, exist_ok=True)

    clubes_list = load_clubs()
    transfers_total = []
    balance_total = []

    for element in clubes_list:
        print(f"Extrayendo transferencias de {element['club_name']}...")

        for season in range(config.TMKT_START_YEAR, config.TMKT_END_YEAR):
            print(f"Para el periodo {season}...")

            for window in ["s", "w"]:
                print(f"Para la ventana del {window} europeo...")

                url = build_transfer_url(element, season, window)

                club_name, periodo, transfer_window, altas, bajas, balance = scrape_transfers(url, element)

                transfers = combine_transfers(altas, bajas)

                ventana = transfer_window_to_filename(transfer_window)
                safe_club_name = sanitize_filename(club_name)

                output_file = TRANSFERS_DIR / f"transfers_{safe_club_name}_{periodo}_{ventana}.csv"
                transfers.to_csv(output_file, index=False)

                transfers_total.append(transfers)
                balance_total.append(balance)

                time.sleep(5)

    return transfers_total, balance_total


if __name__ == "__main__":
    transfers_total, balance_total = run_scraping()

    print("\nResumen final:")
    print(f"CSV de transferencias generados: {len(transfers_total)}")
    print(f"DataFrames de balance generados: {len(balance_total)}")