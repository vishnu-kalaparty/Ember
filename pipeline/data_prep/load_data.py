"""
Load breach_report_202509.csv, keep non-empty Web Description rows, sort
globally by submission date, and assign a 1-indexed breach_id. Cached to
disk (config.REPORTS_CACHE) so every downstream script sees identical IDs.
"""

import csv
import datetime as dt
import json

from .. import config


def _date_key(entry):
    raw = (entry.get(config.DATE_FIELD) or "").strip()
    try:
        return dt.datetime.strptime(raw, config.DATE_FORMAT)
    except (ValueError, TypeError):
        return dt.datetime.max


def load_reports(csv_path=None, use_cache=True):
    """Return a list of report dicts: breach_id, text, type, date, metadata."""
    if use_cache and config.REPORTS_CACHE.exists():
        with open(config.REPORTS_CACHE, encoding="utf-8") as fh:
            return json.load(fh)

    csv_path = str(csv_path or config.DATA_CSV)
    header = None
    rows = []
    with open(csv_path, encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if header is None:
                header = row
                continue
            if len(row) < len(header):
                continue
            entry = {header[i]: row[i] for i in range(len(header))}
            if (entry.get(config.TEXT_FIELD) or "").strip():
                rows.append(entry)

    rows.sort(key=_date_key)

    reports = []
    for i, entry in enumerate(rows):
        reports.append(
            {
                "breach_id": i + 1,
                "text": entry.get(config.TEXT_FIELD, "").strip(),
                "type": entry.get(config.TYPE_FIELD, "").strip(),
                "date": entry.get(config.DATE_FIELD, "").strip(),
                "metadata": {
                    f: (entry.get(f) or "").strip() for f in config.METADATA_FIELDS
                },
            }
        )

    with open(config.REPORTS_CACHE, "w", encoding="utf-8") as fh:
        json.dump(reports, fh, indent=2)

    return reports


if __name__ == "__main__":
    reps = load_reports(use_cache=False)
    print(f"Loaded {len(reps)} non-empty reports -> cached at {config.REPORTS_CACHE}")
    print("First:", reps[0]["breach_id"], reps[0]["date"], "-", reps[0]["text"][:100])
    print("Last :", reps[-1]["breach_id"], reps[-1]["date"], "-", reps[-1]["text"][:100])
