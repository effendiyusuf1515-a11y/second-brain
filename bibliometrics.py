"""Lightweight bibliometric signals derived from the local document library."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from research_core import ExtractedDocument, SourceSection


COUNTRY_LOCATIONS: dict[str, tuple[float, float]] = {
    "Argentina": (-38.4161, -63.6167),
    "Australia": (-25.2744, 133.7751),
    "Austria": (47.5162, 14.5501),
    "Bangladesh": (23.685, 90.3563),
    "Belgium": (50.5039, 4.4699),
    "Brazil": (-14.235, -51.9253),
    "Canada": (56.1304, -106.3468),
    "Chile": (-35.6751, -71.543),
    "China": (35.8617, 104.1954),
    "Colombia": (4.5709, -74.2973),
    "Czechia": (49.8175, 15.473),
    "Denmark": (56.2639, 9.5018),
    "Egypt": (26.8206, 30.8028),
    "Finland": (61.9241, 25.7482),
    "France": (46.2276, 2.2137),
    "Germany": (51.1657, 10.4515),
    "Greece": (39.0742, 21.8243),
    "Hong Kong": (22.3193, 114.1694),
    "India": (20.5937, 78.9629),
    "Indonesia": (-0.7893, 113.9213),
    "Iran": (32.4279, 53.6880),
    "Iraq": (33.2232, 43.6793),
    "Ireland": (53.1424, -7.6921),
    "Israel": (31.0461, 34.8516),
    "Italy": (41.8719, 12.5674),
    "Japan": (36.2048, 138.2529),
    "Malaysia": (4.2105, 101.9758),
    "Mexico": (23.6345, -102.5528),
    "Nepal": (28.3949, 84.1240),
    "Netherlands": (52.1326, 5.2913),
    "New Zealand": (-40.9006, 174.8860),
    "Nigeria": (9.0820, 8.6753),
    "Norway": (60.4720, 8.4689),
    "Pakistan": (30.3753, 69.3451),
    "Philippines": (12.8797, 121.7740),
    "Poland": (51.9194, 19.1451),
    "Portugal": (39.3999, -8.2245),
    "Qatar": (25.3548, 51.1839),
    "Romania": (45.9432, 24.9668),
    "Russia": (61.5240, 105.3188),
    "Saudi Arabia": (23.8859, 45.0792),
    "Singapore": (1.3521, 103.8198),
    "South Africa": (-30.5595, 22.9375),
    "South Korea": (35.9078, 127.7669),
    "Spain": (40.4637, -3.7492),
    "Sweden": (60.1282, 18.6435),
    "Switzerland": (46.8182, 8.2275),
    "Taiwan": (23.6978, 120.9605),
    "Thailand": (15.8700, 100.9925),
    "Turkey": (38.9637, 35.2433),
    "Ukraine": (48.3794, 31.1656),
    "United Arab Emirates": (23.4241, 53.8478),
    "United Kingdom": (55.3781, -3.4360),
    "United States": (39.8283, -98.5795),
    "Vietnam": (14.0583, 108.2772),
}

COUNTRY_ALIASES = {
    "Argentina": ("argentina",),
    "Australia": ("australia",),
    "Austria": ("austria",),
    "Bangladesh": ("bangladesh",),
    "Belgium": ("belgium",),
    "Brazil": ("brazil",),
    "Canada": ("canada",),
    "Chile": ("chile",),
    "China": ("people's republic of china", "pr china", "p.r. china", "china"),
    "Colombia": ("colombia",),
    "Czechia": ("czech republic", "czechia"),
    "Denmark": ("denmark",),
    "Egypt": ("egypt",),
    "Finland": ("finland",),
    "France": ("france",),
    "Germany": ("germany",),
    "Greece": ("greece",),
    "Hong Kong": ("hong kong",),
    "India": ("india",),
    "Indonesia": ("indonesia",),
    "Iran": ("iran",),
    "Iraq": ("iraq",),
    "Ireland": ("ireland",),
    "Israel": ("israel",),
    "Italy": ("italy",),
    "Japan": ("japan",),
    "Malaysia": ("malaysia",),
    "Mexico": ("mexico",),
    "Nepal": ("nepal",),
    "Netherlands": ("the netherlands", "netherlands", "holland"),
    "New Zealand": ("new zealand",),
    "Nigeria": ("nigeria",),
    "Norway": ("norway",),
    "Pakistan": ("pakistan",),
    "Philippines": ("philippines",),
    "Poland": ("poland",),
    "Portugal": ("portugal",),
    "Qatar": ("qatar",),
    "Romania": ("romania",),
    "Russia": ("russian federation", "russia"),
    "Saudi Arabia": ("saudi arabia",),
    "Singapore": ("singapore",),
    "South Africa": ("south africa",),
    "South Korea": ("republic of korea", "south korea", "korea"),
    "Spain": ("spain",),
    "Sweden": ("sweden",),
    "Switzerland": ("switzerland",),
    "Taiwan": ("taiwan",),
    "Thailand": ("thailand",),
    "Turkey": ("turkey", "türkiye"),
    "Ukraine": ("ukraine",),
    "United Arab Emirates": ("united arab emirates", "uae"),
    "United Kingdom": ("united kingdom", "great britain", "england", "scotland", "wales", "uk"),
    "United States": ("united states of america", "united states", "u.s.a.", "usa"),
    "Vietnam": ("vietnam", "viet nam"),
}

STOP_WORDS = {
    "about", "after", "analysis", "based", "bridge", "bridges", "concrete",
    "dynamic", "evaluation", "experimental", "finite", "from", "into", "load",
    "model", "models", "paper", "performance", "research", "results", "static",
    "study", "testing", "using", "with", "yang", "dan", "dengan", "dari",
    "untuk", "pada", "dalam", "jembatan", "beban", "uji", "analisis", "hasil",
    "abstract",
}
NON_KEYWORD_HEADERS = {"abstract", "keywords", "author keywords", "introduction", "article info"}


def _first_page_text(sections: tuple[SourceSection, ...]) -> str:
    page_one = next(
        (section.text for section in sections if section.locator.casefold() == "page 1"),
        "",
    )
    return page_one or (sections[0].text if sections else "")


def _country_from_affiliation(text: str) -> tuple[str, str]:
    front_matter = text[:7000]
    boundary = re.search(
        r"\b(?:abstract|article info|keywords?|1\s*\.\s*introduction|introduction)\b",
        front_matter,
        re.IGNORECASE,
    )
    affiliation_text = front_matter[: boundary.start()] if boundary else front_matter
    candidates = []
    for country, aliases in COUNTRY_ALIASES.items():
        for alias in aliases:
            pattern = rf"(?<![\w]){re.escape(alias)}(?![\w])"
            match = re.search(pattern, affiliation_text, re.IGNORECASE)
            if match:
                candidates.append((match.start(), -len(alias), country))
    if not candidates:
        return "", "unknown"
    candidates.sort(reverse=True)
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2], "detected from first-page affiliation text"


def _title_from_text(text: str, filename: str) -> str:
    ignored = re.compile(
        r"^(?:contents lists|available online|received|accepted|revised|article info|"
        r"abstract|keywords?|issn|doi|\d{4}\s*[-–]|copyright|©|engineering structures\b)",
        re.IGNORECASE,
    )
    for line in (part.strip() for part in text.splitlines()):
        if 18 <= len(line) <= 220 and len(line.split()) >= 4 and not ignored.search(line):
            if not re.search(r"@|https?://|www\.", line, re.IGNORECASE):
                return line
    return re.sub(r"^\d+[\s._-]*", "", Path(filename).stem).strip()


def _publication_year(filename: str, text: str) -> int | None:
    first_page = text[:2500]
    publication_hint = re.search(
        r"(?:available online|published|publication date|accepted|copyright|©)[^\n]{0,100}"
        r"\b(19\d{2}|20\d{2})\b",
        first_page,
        re.IGNORECASE,
    )
    if publication_hint:
        return int(publication_hint.group(1))
    page_year = re.search(r"\b(19\d{2}|20\d{2})\b", first_page)
    if page_year:
        return int(page_year.group(1))
    filename_year = re.search(r"\b(19\d{2}|20\d{2})\b", filename)
    return int(filename_year.group(1)) if filename_year else None


def _author_keywords(text: str) -> str:
    match = re.search(
        r"\b(?:author\s+)?keywords?\b\s*[:\-]?\s*(.+?)"
        r"(?=\n\s*\n|\babstract\b|\bintroduction\b|$)",
        text[:9000],
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ""
    phrases = []
    for phrase in re.split(r"[,;\n|]+", match.group(1)):
        cleaned = re.sub(r"\s+", " ", phrase).strip(" .:-")
        if 2 <= len(cleaned) <= 80 and cleaned.casefold() not in {
            "abstract", "author", "author keywords", "index terms", "introduction", "keywords",
        }:
            phrases.append(cleaned)
    return "; ".join(dict.fromkeys(phrases[:12]))


def infer_document_metadata(document: ExtractedDocument) -> dict[str, Any]:
    first_page = _first_page_text(document.sections)
    country, country_source = _country_from_affiliation(first_page)
    return {
        "title": _title_from_text(first_page, document.filename),
        "year": _publication_year(document.filename, first_page),
        "country": country,
        "country_source": country_source,
        "keywords": _author_keywords(first_page),
    }


def bibliometric_summary(
    documents: list[dict[str, Any]], chunks: list[dict[str, str]], top_n: int = 10
) -> dict[str, Any]:
    years = Counter(
        int(document["year"])
        for document in documents
        if document.get("year") is not None
    )
    countries = Counter(
        document["country"]
        for document in documents
        if document.get("country") in COUNTRY_LOCATIONS
    )
    author_keywords = Counter()
    inferred_terms = Counter()
    for document in documents:
        keywords = []
        for item in str(document.get("keywords") or "").split(";"):
            normalized = re.sub(r"\s+", " ", item.strip().casefold())
            compact = normalized.replace(" ", "")
            if normalized and normalized not in NON_KEYWORD_HEADERS and compact not in NON_KEYWORD_HEADERS:
                keywords.append(normalized)
        if keywords:
            author_keywords.update(set(keywords))
            continue
        title_terms = {
            term.casefold()
            for term in re.findall(r"[\w-]{3,}", str(document.get("title") or ""))
            if term.casefold() not in STOP_WORDS
        }
        inferred_terms.update(title_terms)

    keyword_counts = author_keywords if author_keywords else inferred_terms
    keyword_source = "author keywords" if author_keywords else "terms inferred from titles"
    points = [
        {
            "country": country,
            "latitude": COUNTRY_LOCATIONS[country][0],
            "longitude": COUNTRY_LOCATIONS[country][1],
            "publications": count,
        }
        for country, count in countries.most_common()
    ]
    return {
        "year_counts": sorted(years.items()),
        "country_counts": countries.most_common(top_n),
        "map_points": points,
        "keyword_counts": keyword_counts.most_common(top_n),
        "keyword_source": keyword_source,
        "known_country_count": sum(countries.values()),
        "unknown_country_count": len(documents) - sum(countries.values()),
        "chunk_count": len(chunks),
    }


def filter_documents(
    documents: list[dict[str, Any]],
    chunks: list[dict[str, str]],
    *,
    query: str = "",
    countries: list[str] | None = None,
    year_range: tuple[int, int] | None = None,
    include_unknown_country: bool = True,
    include_unknown_year: bool = True,
) -> list[dict[str, Any]]:
    text_by_source: dict[str, list[str]] = {}
    for chunk in chunks:
        text_by_source.setdefault(chunk["source"], []).append(chunk["text"].casefold())
    terms = [term.casefold() for term in re.findall(r"[\w-]{2,}", query)]
    selected_countries = set(countries or [])
    filtered = []
    for document in documents:
        country = str(document.get("country") or "")
        year = document.get("year")
        if selected_countries:
            is_unknown = not country
            if is_unknown:
                if not include_unknown_country or "Belum diketahui" not in selected_countries:
                    continue
            elif country not in selected_countries:
                continue
        if year_range:
            if year is None:
                if not include_unknown_year:
                    continue
            elif not year_range[0] <= int(year) <= year_range[1]:
                continue
        if terms:
            searchable = " ".join(
                [
                    str(document.get("filename") or "").casefold(),
                    str(document.get("title") or "").casefold(),
                    str(document.get("keywords") or "").casefold(),
                    *text_by_source.get(str(document.get("filename") or ""), []),
                ]
            )
            if not all(term in searchable for term in terms):
                continue
        filtered.append(document)
    return filtered
