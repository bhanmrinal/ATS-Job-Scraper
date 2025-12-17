import csv
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup


DATA_DIR = Path("data")
MASTER_CSV_PATH = DATA_DIR / "master_jobs.csv"
DELTA_CSV_PATH = DATA_DIR / "delta_jobs.csv"

KEYWORDS = [
    "ai",
    "artificial intelligence",
    "ml",
    "machine learning",
    "engineer",
    "developer",
]

EUROPE_REMOTE_HINTS = [
    "remote",
    "europe",
    "eu",
    "germany",
    "france",
    "uk",
    "united kingdom",
    "england",
    "spain",
    "italy",
    "netherlands",
    "belgium",
    "sweden",
    "norway",
    "finland",
    "denmark",
    "switzerland",
    "austria",
    "ireland",
    "portugal",
    "poland",
    "czech",
    "estonia",
    "lithuania",
    "latvia",
]

LEVER_COMPANIES = [
    "openai",
    "huggingface",
    "deepmind",
    "stabilityai",
]

ASHBY_COMPANIES = [
    "perplexity",
    "mistral",
    "anthropic",
    "cohere",
]

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ATSJobScraper/1.0; +https://example.com)",
    "Accept-Language": "en-US,en;q=0.9",
}


def generate_job_id(title: str, company: str, location: str, url: str) -> str:
    """Generate a deterministic job ID based on title, company, location, and URL."""
    base = "||".join(
        [
            (title or "").strip().lower(),
            (company or "").strip().lower(),
            (location or "").strip().lower(),
            (url or "").strip().lower(),
        ]
    )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def load_master_csv(path: Path = MASTER_CSV_PATH) -> pd.DataFrame:
    """Load the master CSV if it exists, otherwise return an empty DataFrame."""
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "job_id",
                "title",
                "company",
                "location",
                "description",
                "url",
                "source",
                "scraped_at",
            ]
        )
    return pd.read_csv(path, dtype=str)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to CSV, ensuring the parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL)


def fetch_url(url: str) -> Optional[str]:
    """Fetch URL content with basic error handling."""
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
        if response.status_code != 200:
            return None
        return response.text
    except requests.RequestException:
        return None


def passes_keyword_filter(title: str, description: str) -> bool:
    """Return True if the job matches any of the keyword filters."""
    combined = f"{title} {description}".lower()
    return any(keyword in combined for keyword in KEYWORDS)


def passes_location_filter(location: str, description: str) -> bool:
    """
    Simple Europe / Remote heuristic.

    - If location is empty, fall back to description text.
    - Keep jobs mentioning 'remote', 'europe', or common European country names.
    """
    text = f"{location} {description}".lower()
    return any(hint in text for hint in EUROPE_REMOTE_HINTS)


def scrape_lever_company(company: str) -> List[Dict[str, str]]:
    """
    Scrape jobs from a single Lever company page.
    Returns a list of normalized job dicts without job_id/scraped_at.
    """
    base_url = f"https://jobs.lever.co/{company}"
    html = fetch_url(base_url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    postings = soup.find_all("div", class_="posting")

    jobs: List[Dict[str, str]] = []

    for posting in postings:
        link = posting.find("a", href=True)
        if not link:
            continue

        title_el = posting.find("h5") or linking_text_element(link=link)
        title = title_el.get_text(strip=True) if title_el else link.get_text(strip=True)
        location_el = posting.find("span", class_="sort-by-location")
        location = location_el.get_text(strip=True) if location_el else ""

        job_url = link["href"]
        if not job_url.startswith("http"):
            job_url = f"https://jobs.lever.co{job_url}"

        description = fetch_lever_job_description(job_url)

        if not passes_keyword_filter(title, description):
            continue
        if not passes_location_filter(location, description):
            continue

        jobs.append(
            {
                "title": title,
                "company": company,
                "location": location or "Unknown",
                "description": description,
                "url": job_url,
                "source": "lever",
            }
        )

    return jobs


def linking_text_element(link) -> Optional[BeautifulSoup]:
    """
    Fallback helper to get a title element from a link if structured tags are missing.
    """
    return link


def fetch_lever_job_description(job_url: str) -> str:
    """Fetch basic job description text from Lever job detail page."""
    html = fetch_url(job_url)
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    description_container = soup.find("div", class_="section-wrapper")
    if not description_container:
        description_container = soup.find("div", class_="posting")
    if not description_container:
        return ""
    return " ".join(description_container.get_text(separator=" ", strip=True).split())


def scrape_lever() -> List[Dict[str, str]]:
    """Scrape Lever for all configured companies."""
    all_jobs: List[Dict[str, str]] = []
    for company in LEVER_COMPANIES:
        all_jobs.extend(scrape_lever_company(company))
    return all_jobs


def scrape_ashby_company(company: str) -> List[Dict[str, str]]:
    """
    Scrape jobs from a single Ashby company page.
    Returns a list of normalized job dicts without job_id/scraped_at.
    """
    base_url = f"https://jobs.ashbyhq.com/{company}"
    html = fetch_url(base_url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    jobs: List[Dict[str, str]] = []
    seen_links = set()

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/job/" not in href:
            continue
        if href.startswith("mailto:") or href.startswith("tel:"):
            continue
        if href in seen_links:
            continue
        seen_links.add(href)

        job_url = href
        if not job_url.startswith("http"):
            if job_url.startswith("/"):
                job_url = f"https://jobs.ashbyhq.com{job_url}"
            else:
                job_url = f"https://jobs.ashbyhq.com/{href}"

        title = link.get_text(strip=True)
        if not title:
            continue

        # Default Ashby location assumption for this simple implementation
        location = "Remote / Europe"
        description = fetch_ashby_job_description(job_url)

        if not passes_keyword_filter(title, description):
            continue
        if not passes_location_filter(location, description):
            continue

        jobs.append(
            {
                "title": title,
                "company": company,
                "location": location,
                "description": description,
                "url": job_url,
                "source": "ashby",
            }
        )

    return jobs


def fetch_ashby_job_description(job_url: str) -> str:
    """Fetch basic job description text from Ashby job detail page."""
    html = fetch_url(job_url)
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    # Try common Ashby containers; fall back to full page text if needed.
    description_container = (
        soup.find("section", attrs={"data-testid": "job-description"})
        or soup.find("div", attrs={"data-testid": "job-description"})
        or soup.find("main")
    )
    if not description_container:
        return ""

    return " ".join(description_container.get_text(separator=" ", strip=True).split())


def scrape_ashby() -> List[Dict[str, str]]:
    """Scrape Ashby for all configured companies."""
    all_jobs: List[Dict[str, str]] = []
    for company in ASHBY_COMPANIES:
        all_jobs.extend(scrape_ashby_company(company))
    return all_jobs


def normalize_jobs(raw_jobs: List[Dict[str, str]]) -> pd.DataFrame:
    """Normalize raw job dicts into a DataFrame with required schema."""
    now_iso = datetime.utcnow().isoformat() + "Z"

    normalized: List[Dict[str, str]] = []
    for job in raw_jobs:
        job_id = generate_job_id(
            title=job.get("title", ""),
            company=job.get("company", ""),
            location=job.get("location", ""),
            url=job.get("url", ""),
        )
        normalized.append(
            {
                "job_id": job_id,
                "title": job.get("title", ""),
                "company": job.get("company", ""),
                "location": job.get("location", ""),
                "description": job.get("description", ""),
                "url": job.get("url", ""),
                "source": job.get("source", ""),
                "scraped_at": now_iso,
            }
        )

    if not normalized:
        return pd.DataFrame(
            columns=[
                "job_id",
                "title",
                "company",
                "location",
                "description",
                "url",
                "source",
                "scraped_at",
            ]
        )

    return pd.DataFrame(normalized)


def compute_delta(master_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    """Compute new jobs by job_id compared to master."""
    if master_df.empty:
        return current_df.copy()

    master_ids = set(master_df["job_id"].astype(str))
    mask = ~current_df["job_id"].astype(str).isin(master_ids)
    return current_df[mask].copy()


def main() -> None:
    """Main entry point for CLI execution."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    lever_jobs = scrape_lever()
    ashby_jobs = scrape_ashby()

    all_raw_jobs = lever_jobs + ashby_jobs
    all_jobs_df = normalize_jobs(all_raw_jobs)

    master_df = load_master_csv()
    delta_df = compute_delta(master_df, all_jobs_df)

    # Append new jobs to master and save both CSVs
    if not delta_df.empty:
        updated_master = pd.concat([master_df, delta_df], ignore_index=True)
    else:
        updated_master = master_df

    save_csv(updated_master, MASTER_CSV_PATH)
    save_csv(delta_df, DELTA_CSV_PATH)

    total_scraped = len(all_jobs_df)
    new_jobs_count = len(delta_df)

    print(f"Total jobs scraped this run: {total_scraped}")
    print(f"New jobs found: {new_jobs_count}")
    print(f"Master CSV path: {MASTER_CSV_PATH}")
    print(f"Delta CSV path: {DELTA_CSV_PATH}")


if __name__ == "__main__":
    main()

"""
Design notes:

- ATS sources like Lever and Ashby provide relatively stable, structured job data compared to LinkedIn,
  which actively blocks scraping and has complex, dynamic pages. Using ATS endpoints keeps the scraper
  simpler, more reliable, and less likely to violate platform terms.

- Delta detection works by assigning each job a deterministic job_id based on its title, company,
  location, and URL, then comparing against the job_id values stored in the master CSV. Any job_id
  that does not already exist in master_jobs.csv is treated as a new job for the current run and is
  written to delta_jobs.csv while also being appended to the master.

- This approach can be extended to more ATS platforms by adding additional scraper functions that
  return normalized job dicts in the same schema (title, company, location, description, url, source).
  As long as new scrapers generate consistent job_id values and feed into the same normalization and
  delta logic, the rest of the pipeline and CSV storage will continue to work unchanged.
"""
