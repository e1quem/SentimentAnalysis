import csv
import time
import random
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException


########## CONFIG ##########

KEYWORD_TO_TICKER = {
    "Apple":  "AAPL",
    "iPhone": "AAPL",
    "Steve Jobs": "AAPL",
    "Tim Cook": "AAPL",
    "iPad": "AAPL",
    "AirPods": "AAPL",
    "MacBook": "AAPL",

    "Amazon":     "AMZN",
    "Jeff Bezos": "AMZN",
    "AWS": "AMZN",

    "Google":     "GOOG",
    "Sundar Pichai": "GOOG",
    "Alphabet":   "GOOGL",

    "Satya Nadella": "MSFT",
    "Microsoft":  "MSFT",

    "Tesla":      "TSLA",
    "Model 3": "TSLA",
    "Elon Musk": "TSLA",
}
KEYWORDS = list(KEYWORD_TO_TICKER.keys())

SITES = [
    "reuters.com",
    #"theguardian.com",
    #"bloomberg.com",
    #"nytimes.com",
    #"wsj.com",
    #"cnbc.com",
    #"bbc.com",
]

DATE_AFTER  = "2014-12-31"
DATE_BEFORE = "2020-01-01"
OUTPUT_CSV  = "results_scraping.csv"

DELAY_BETWEEN_PAGES   = (0, 0.2)
DELAY_BETWEEN_QUERIES = (0, 0.2)
MAX_PAGES_PER_QUERY   = 100


##############################

def build_query(keyword, site):
    return f'intitle:"{keyword}" site:www.{site} after:{DATE_AFTER} before:{DATE_BEFORE}'

def human_delay(bounds):
    time.sleep(random.uniform(*bounds))

def wait_for_human(msg="Résous le CAPTCHA puis appuie sur Entrée..."):
    input(f"\n  {msg}\n")



def is_captcha_page(driver):
    """Détecte UNIQUEMENT les vraies pages de blocage Google (/sorry/)."""
    url = driver.current_url.lower()
    if "/sorry/" in url or "sorry/index" in url:
        return True
    # Texte caractéristique des pages de blocage (pas juste "recaptcha" dans le JS)
    title = driver.title.lower()
    if "avant de continuer" in title or "before you continue" in title:
        return True
    # Vérifier si la div#search est absente ET qu'il y a un formulaire captcha visible
    try:
        driver.find_element(By.CSS_SELECTOR, "div#search")
        return False  # La page de résultats est là, pas de captcha
    except NoSuchElementException:
        # Pas de résultats ET pas sur google.com normal = probablement bloqué
        if "google.com/search" in url:
            return True
    return False



#def extract_date(block):
#    try:
#        snippet_div = block.find_element(By.CSS_SELECTOR, "div[data-snf='nke7rc']")
#        date_span = snippet_div.find_element(By.CSS_SELECTOR, "span.YrbPuc span")  # pas > span
#        t = date_span.text.strip()
#        if t and re.search(r'20\d{2}', t):
#            return t
#    except NoSuchElementException:
#        pass
#
#    # Fallback regex sur le texte du bloc snippet uniquement (pas tout le container)
#    try:
#        snippet_div = block.find_element(By.CSS_SELECTOR, "div[data-snf='nke7rc']")
#        full_text = snippet_div.text
#    except NoSuchElementException:
#        full_text = block.text
#
#    m = re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+20\d{2}', full_text, re.I)
#    if m:
#        return m.group(0)
#    m = re.search(r'\d{1,2}\s+(?:jan|fév|mar|avr|mai|juin|juil|ao[uû]t|sep|oct|nov|déc)[a-zéûôà]*\.?\s+20\d{2}', full_text, re.I)
#    if m:
#        return m.group(0)
#    m = re.search(r'20\d{2}-\d{2}-\d{2}', full_text)
#    if m:
#        return m.group(0)
#    return ""


def extract_date(block):
    # Méthode 1: Chercher dans le span spécifique (le plus fiable)
    try:
        snippet_div = block.find_element(By.CSS_SELECTOR, "div[data-snf='nke7rc']")
        date_span = snippet_div.find_element(By.CSS_SELECTOR, "span.YrbPuc span")
        t = date_span.text.strip()
        if t and re.search(r'\d{1,2}\s+\w+\s+20\d{2}|\w+\s+\d{1,2},?\s+20\d{2}', t, re.I):
            return t
    except NoSuchElementException:
        pass
    
    # Méthode 2: Chercher dans tout le bloc snippet
    try:
        snippet_div = block.find_element(By.CSS_SELECTOR, "div[data-snf='nke7rc']")
        full_text = snippet_div.text
    except NoSuchElementException:
        full_text = block.text
    
    # Priorité: dates complètes (Dec 27, 2018 ou 27 Dec 2018)
    patterns = [
        # Format: "Dec 27, 2018" ou "December 27, 2018"
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)[a-z]*\.?\s+\d{1,2},?\s+20\d{2}',
        # Format: "27 Dec 2018" ou "27 December 2018"
        r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)[a-z]*\.?\s+20\d{2}',
        # Format français: "27 décembre 2018"
        r'\d{1,2}\s+(?:jan|fév|mar|avr|mai|juin|juil|août|sep|oct|nov|déc)[a-zéûôà]*\.?\s+20\d{2}',
    ]
    
    for pattern in patterns:
        m = re.search(pattern, full_text, re.I)
        if m:
            return m.group(0)
    
    # Fallback: juste l'année
    m = re.search(r'20\d{2}', full_text)
    if m:
        return m.group(0)
    
    return ""


MONTH_MAP = {
    # Français
    "janvier": "01", "jan": "01", "janv.": "01",
    "février": "02", "fév": "02", "fevrier": "02", "fev": "02", "févr.": "02",
    "mars": "03", "mar": "03",
    "avril": "04", "avr": "04",
    "mai": "05",
    "juin": "06",
    "juillet": "07", "juil": "07",
    "août": "08", "aout": "08",
    "septembre": "09", "sep": "09", "sept": "09",
    "octobre": "10", "oct": "10",
    "novembre": "11", "nov": "11",
    "décembre": "12", "dec": "12", "décembre": "12", "déc": "12",
    # Anglais
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
    "feb": "02", "apr": "04", "jun": "06", "jul": "07",
    "aug": "08", "nov": "11", "Dec": "12", "Jan": "01", "Feb": "02", "Jun": "06",
}



def normalize_date(date_str):
    """Convertit n'importe quel format de date en YYYY-MM-DD."""
    if not date_str:
        return ""
    s = date_str.strip()
    
    # Déjà ISO
    m = re.match(r'^(20\d{2})[-/](\d{2})[-/](\d{2})$', s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    
    # Format: "Dec 27, 2018" ou "December 27, 2018"
    m = re.match(r'^([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(20\d{2})$', s, re.I)
    if m:
        mon, day, year = m.group(1).lower().rstrip('.'), m.group(2).zfill(2), m.group(3)
        month_num = MONTH_MAP.get(mon)
        if month_num:
            return f"{year}-{month_num}-{day}"
    
    # Format: "27 Dec 2018" ou "27 December 2018"
    m = re.match(r'^(\d{1,2})\s+([A-Za-z]+)\.?\s+(20\d{2})$', s, re.I)
    if m:
        day, mon, year = m.group(1).zfill(2), m.group(2).lower().rstrip('.'), m.group(3)
        month_num = MONTH_MAP.get(mon)
        if month_num:
            return f"{year}-{month_num}-{day}"
    
    # Format français: "27 décembre 2018"
    m = re.match(r'^(\d{1,2})\s+([a-zéûôà]+)\.?\s+(20\d{2})$', s, re.I)
    if m:
        day, mon, year = m.group(1).zfill(2), m.group(2).lower().rstrip('.'), m.group(3)
        month_num = MONTH_MAP.get(mon)
        if month_num:
            return f"{year}-{month_num}-{day}"
    
    # Année seule
    m = re.search(r'(20\d{2})', s)
    if m:
        return m.group(1)
    
    return date_str



def scrape_page(driver):
    results = []
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div#search"))
        )
    except TimeoutException:
        print("  [timeout en attendant div#search]")
        return results

    containers = driver.find_elements(By.CSS_SELECTOR, "div.N54PNb, div.BToiNc")
    if not containers:
        containers = driver.find_elements(By.CSS_SELECTOR, "div.g")

    for container in containers:
        try:
            # --- TITRE : depuis l'attribut aria-label du lien, ou le href en dernier recours ---
            try:
                link = container.find_element(By.CSS_SELECTOR, "a[jsname='UWckNb'], h3 ~ a, a:has(h3), a[href]")
            except NoSuchElementException:
                continue

            # 1) Essayer aria-label sur le lien (parfois le titre complet y est)
            title = link.get_attribute("aria-label") or ""

            # 2) Sinon, textContent du h3 via JS
            if not title:
                try:
                    h3 = container.find_element(By.CSS_SELECTOR, "h3.LC20lb, h3.DKV0Md, h3")
                    title = driver.execute_script("return arguments[0].textContent;", h3) or h3.text
                except NoSuchElementException:
                    continue

            # 3) Nettoyer les "..." trailing (Google tronque dans le DOM)
            title = title.strip().rstrip(".")
            # Supprimer le "..." ou "…" final proprement
            title = re.sub(r'\s*\.{3}$', '', title).strip()
            title = re.sub(r'\s*…$', '', title).strip()

            if not title:
                continue

            # --- DATE ---
            date_raw = ""
            try:
                snippet_div = container.find_element(By.CSS_SELECTOR, "div[data-snf='nke7rc']")
                # Cherche le span de date n'importe où dans le snippet, pas juste enfant direct
                date_span = snippet_div.find_element(By.CSS_SELECTOR, "span.YrbPuc span")
                date_raw = date_span.text.strip()
            except NoSuchElementException:
                pass

            if not date_raw:
                date_raw = extract_date(container)

            date_text = normalize_date(date_raw)
            results.append({"title": title, "date": date_text})

        except Exception:
            continue

    return results



def get_next_page_button(driver):
    """Retourne le bouton page suivante ou None."""
    for selector in ["a#pnnext", "a[aria-label='Page suivante']", "a[aria-label='Next']"]:
        try:
            return driver.find_element(By.CSS_SELECTOR, selector)
        except NoSuchElementException:
            pass
    try:
        return driver.find_element(By.XPATH, "//a[contains(@aria-label,'uivante') or contains(@aria-label,'Next')]")
    except NoSuchElementException:
        return None



def run_query(driver, keyword, site, writer):
    query = build_query(keyword, site)
    # &num=10 pour 10 résultats par page (standard)
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"

    print(f" {query}")

    driver.get(url)
    human_delay((0, 0.2))

    if is_captcha_page(driver):
        wait_for_human("CAPTCHA détecté ! Résous-le dans Chrome, puis appuie sur Entrée ici.")

    page_num = 1
    total_found = 0

    while page_num <= MAX_PAGES_PER_QUERY:
        print(f"  p{page_num}", end=" ", flush=True)

        if is_captcha_page(driver):
            wait_for_human("CAPTCHA détecté ! Résous-le puis appuie sur Entrée.")

        results = scrape_page(driver)
        print(f"{len(results)} résultats")

        ticker = KEYWORD_TO_TICKER.get(keyword, keyword)
        site_clean = site.replace(".com", "").replace(".org", "").replace(".net", "")
        for r in results:
            writer.writerow({
                "ticker":   ticker,
                "site":     site_clean,
                "headline": r["title"],
                "date":     r["date"],
            })
        total_found += len(results)

        next_btn = get_next_page_button(driver)
        if not next_btn:
            print(f"   Fin ({total_found} résultats au total)")
            break

        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", next_btn)
        human_delay((0, 0.2))
        next_btn.click()
        human_delay(DELAY_BETWEEN_PAGES)
        page_num += 1

    return total_found



def main():
    opts = Options()
    opts.add_argument("--start-maximized")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument("--lang=fr-FR")

    driver = webdriver.Chrome(options=opts)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    })

    combos = [(kw, site) for kw in KEYWORDS for site in SITES]
    print(f" {len(combos)} combinaisons ({len(KEYWORDS)} keywords × {len(SITES)} sites)")
    print(f" Output : {OUTPUT_CSV}\n")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "site", "headline", "date"])
        writer.writeheader()

        for i, (keyword, site) in enumerate(combos, 1):
            print(f"\n[{i}/{len(combos)}]", end="")
            run_query(driver, keyword, site, writer)
            f.flush()

            if i < len(combos):
                wait_sec = random.uniform(*DELAY_BETWEEN_QUERIES)
                print(f"   Pause {wait_sec:.0f}s...")
                time.sleep(wait_sec)

    driver.quit()
    df = pd.read_csv(OUTPUT_CSV)
    df.drop_duplicates(subset=['ticker', 'date', 'headline'], keep='first', inplace=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print(f"\n Terminé ! CSV : {OUTPUT_CSV}")

if __name__ == "__main__":
    main()