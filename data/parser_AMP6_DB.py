import re
import csv
import time
import random
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import urllib3
import certifi
import requests
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE = "https://aps.unmc.edu"
URL_SEARCH = f"{BASE}/database"
URL_RESULT = f"{BASE}/database/result"
URL_PEPTIDE = f"{BASE}/database/peptide"

CSV_FIELDS = [
    "APD ID", "Name/Class", "Source", "Sequence", "Length", "Net charge",
    "Hydrophobic residue%", "Boman Index", "3D Structure", "Method",
    "Activity", "Crucial residues", "Additional info",
    "History and discovery", "Sequence analysis",
    "Title", "Author", "Reference",
]

def norm_ws(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def get_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")

def request_with_retry(session: requests.Session, method: str, url: str, **kwargs) -> requests.Response:
    last_exc = None
    for attempt in range(1, 6):
        try:
            kwargs.setdefault("timeout", (15, 120))
            r = session.request(method, url, **kwargs)
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            time.sleep(min(30, 2 * attempt) + random.random())
    raise RuntimeError(f"Request failed after retries: {method} {url}") from last_exc

def build_form_payload(form) -> List[Tuple[str, str]]:
    data: List[Tuple[str, str]] = []

    # inputs
    for inp in form.select("input"):
        name = inp.get("name")
        if not name:
            continue
        itype = (inp.get("type") or "text").lower()

        if itype in {"submit", "button", "image"}:
            val = inp.get("value")
            if val is not None:
                data.append((name, val))
            continue

        if itype in {"checkbox", "radio"}:
            if inp.has_attr("checked"):
                data.append((name, inp.get("value", "on")))
            continue

        # text/hidden/etc
        data.append((name, inp.get("value", "")))

    # selects
    for sel in form.select("select"):
        name = sel.get("name")
        if not name:
            continue
        opt = sel.select_one("option[selected]") or sel.select_one("option")
        if opt is None:
            data.append((name, ""))
        else:
            data.append((name, opt.get("value") or opt.get_text(strip=True)))

    # textareas
    for ta in form.select("textarea"):
        name = ta.get("name")
        if not name:
            continue
        data.append((name, ta.get_text() or ""))

    return data

def submit_empty_search(session: requests.Session) -> str:
    # 1) GET search page
    r0 = request_with_retry(session, "GET", URL_SEARCH)
    soup = get_soup(r0.text)

    # 2) find form that goes to result
    forms = soup.select("form")
    target = None
    for f in forms:
        action = (f.get("action") or "").strip()
        if "database/result" in action or action.endswith("/result"):
            target = f
            break
    if target is None:
        open("debug_database.html", "w", encoding="utf-8").write(r0.text)
        raise RuntimeError("Не нашёл форму поиска на /database (сохранил debug_database.html)")

    action = target.get("action") or "/database/result"
    post_url = urljoin(URL_SEARCH, action)

    payload = build_form_payload(target)

    # 3) POST as browser would
    r1 = request_with_retry(session, "POST", post_url, data=payload)
    return r1.text

def extract_ids_from_result(html: str) -> List[str]:
    ids = re.findall(r'name=["\']ID["\']\s+value=["\'](\d{5})["\']', html)
    if ids:
        # unique keep order
        seen, out = set(), []
        for x in ids:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    # fallback через bs4
    soup = get_soup(html)
    ids2 = []
    for inp in soup.select("input[name='ID']"):
        v = (inp.get("value") or "").strip()
        if re.fullmatch(r"\d{5}", v):
            ids2.append(v)
    seen, out = set(), []
    for x in ids2:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def parse_peptide_page(html: str) -> Dict[str, str]:
    soup = get_soup(html)
    table = soup.select_one("table.peptide")
    if not table:
        raise ValueError("No peptide table found (table.peptide)")

    rows = table.select("tr")
    kv: Dict[str, str] = {}
    for tr in rows:
        tds = tr.find_all("td", recursive=False)
        if len(tds) != 2:
            continue
        key = norm_ws(tds[0].get_text(" ", strip=True)).rstrip(":")
        val = norm_ws(tds[1].get_text("\n", strip=True))
        if key:
            kv[key] = val

    seq_tag = soup.select_one("p.peptide_sequence")
    if seq_tag:
        kv["Sequence"] = norm_ws(seq_tag.get_text(" ", strip=True))

    add_cell = None
    for tr in rows:
        tds = tr.find_all("td", recursive=False)
        if len(tds) == 2:
            k = norm_ws(tds[0].get_text(" ", strip=True)).rstrip(":")
            if k.lower() == "additional info":
                add_cell = tds[1]
                break

    additional_full = history = seq_analysis = ""
    if add_cell is not None:
        additional_full = norm_ws(add_cell.get_text("\n", strip=True))
        sections: Dict[str, str] = {}
        for b in add_cell.find_all("b"):
            title = norm_ws(b.get_text(" ", strip=True)).rstrip(":")
            if not title:
                continue
            parts = []
            for sib in b.next_siblings:
                if getattr(sib, "name", None) == "b":
                    break
                txt = sib if isinstance(sib, str) else sib.get_text(" ", strip=True)
                txt = norm_ws(txt)
                if txt:
                    parts.append(txt)
            body = norm_ws(" ".join(parts))
            if body.startswith(":"):
                body = norm_ws(body[1:])
            if body:
                sections[title.lower()] = body
        history = sections.get("history and discovery", "")
        seq_analysis = sections.get("sequence analysis", "")

    out = {k: "" for k in CSV_FIELDS}
    out["APD ID"] = kv.get("APD ID", "")
    out["Name/Class"] = kv.get("Name/Class", "")
    out["Source"] = kv.get("Source", "")
    out["Sequence"] = kv.get("Sequence", "")
    out["Length"] = kv.get("Length", "")
    out["Net charge"] = kv.get("Net charge", "")
    out["Hydrophobic residue%"] = kv.get("Hydrophobic residue%", "")
    out["Boman Index"] = kv.get("Boman Index", "")
    out["3D Structure"] = kv.get("3D Structure", "")
    out["Method"] = kv.get("Method", "")
    out["Activity"] = kv.get("Activity", "")
    out["Crucial residues"] = kv.get("Crucial residues", "")
    out["Additional info"] = additional_full
    out["History and discovery"] = history
    out["Sequence analysis"] = seq_analysis
    out["Title"] = kv.get("Title", "")
    out["Author"] = kv.get("Author", "")
    out["Reference"] = kv.get("Reference", "")
    return out

def scrape_all(csv_path: str, limit: Optional[int] = None, sleep_range=(0.2, 0.6)) -> None:
    session = requests.Session()
    session.verify = certifi.where()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; apd6-scraper/1.0)",
        "Accept-Language": "en-US,en;q=0.9",
    })

    result_html = submit_empty_search(session)
    ids = extract_ids_from_result(result_html)

    if not ids:
        open("debug_result.html", "w", encoding="utf-8").write(result_html)
        raise RuntimeError("No peptide IDs found (сохранил debug_result.html)")

    if limit is not None:
        ids = ids[:limit]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()

        for i, short_id in enumerate(ids, 1):
            time.sleep(random.uniform(*sleep_range))
            rp = request_with_retry(session, "POST", URL_PEPTIDE, data={"ID": short_id})
            row = parse_peptide_page(rp.text)
            w.writerow(row)
            print(f"{i}/{len(ids)}: {row['APD ID']}")

    print(f"Done. Wrote: {csv_path} (rows: {len(ids)})")

if __name__ == "__main__":
    # scrape_all("apd6_peptides.csv", limit=5) # line for test runtime
    scrape_all("apd6_peptides_raw_data.csv", limit=None)