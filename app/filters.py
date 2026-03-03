# app/filters.py

import re
import calendar
from datetime import datetime, timedelta

# ============================================================
# Date Utilities
# ============================================================

def extract_date_filter(query: str):
    """last X days / last month"""
    q = query.lower()
    today = datetime.today()

    match = re.search(r"last (\d+) days", q)
    if match:
        days = int(match.group(1))
        start = today - timedelta(days=days)
        start_ts = int(start.timestamp())
        end_ts = int(today.timestamp())
        return {
            "$and": [
                {"date": {"$gte": start_ts}},
                {"date": {"$lte": end_ts}}
            ]
        }

    if "last month" in q:
        first_this_month = today.replace(day=1)
        last_month_end = first_this_month - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        return {
            "$and": [
                {"date": {"$gte": int(last_month_start.timestamp())}},
                {"date": {"$lte": int(last_month_end.timestamp())}}
            ]
        }
    return None

def extract_month_filter(query: str):
    """in december / in january 2025 kind of queries"""
    q = query.lower()
    months = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}

    year_match = re.search(r"(20\d{2})", q)
    year = int(year_match.group(1)) if year_match else datetime.today().year

    for month_name, month_num in months.items():
        if month_name in q:
            start = datetime(year, month_num, 1)
            if month_num == 12:
                end = datetime(year + 1, 1, 1) - timedelta(days=1)
            else:
                end = datetime(year, month_num + 1, 1) - timedelta(days=1)
            return {
                "$and": [
                    {"date": {"$gte": int(start.timestamp())}},
                    {"date": {"$lte": int(end.timestamp())}}
                ]
            }
    return None

def extract_between_months(query: str):
    """between january and march / between jan 2025 and march 2025"""
    q = query.lower()
    months = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}

    match = re.search(
        r"between (\w+)\s*(\d{4})?\s+and\s+(\w+)\s*(\d{4})?",
        q
    )

    if match:
        m1, y1, m2, y2 = match.groups()
        if m1 in months and m2 in months:
            year1 = int(y1) if y1 else datetime.today().year
            year2 = int(y2) if y2 else year1

            start = datetime(year1, months[m1], 1)
            if months[m2] == 12:
                end = datetime(year2 + 1, 1, 1) - timedelta(days=1)
            else:
                end = datetime(year2, months[m2] + 1, 1) - timedelta(days=1)

            return {
                "$and": [
                    {"date": {"$gte": int(start.timestamp())}},
                    {"date": {"$lte": int(end.timestamp())}}
                ]
            }
    return None

def extract_exact_date_range(query: str):
    """between 1 January 2025 and 15 February 2025"""
    q = query.lower()

    match = re.search(
        r"between (\d{1,2} \w+ \d{4}) and (\d{1,2} \w+ \d{4})",
        q
    )

    if match:
        start_str, end_str = match.groups()
        start = datetime.strptime(start_str, "%d %B %Y")
        end = datetime.strptime(end_str, "%d %B %Y")
        return {
            "$and": [
                {"date": {"$gte": int(start.timestamp())}},
                {"date": {"$lte": int(end.timestamp())}}
            ]
        }
    return None

def extract_exact_date(query: str):
    """on 5 March 2025"""
    q = query.lower()

    match = re.search(
        r"on (\d{1,2}) (\w+) (\d{4})",
        q
    )

    if match:
        day, month_name, year = match.groups()
        months = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}

        if month_name in months:
            date_obj = datetime(int(year), months[month_name], int(day))
            start = int(date_obj.timestamp())
            end = int((date_obj + timedelta(days=1)).timestamp())
            return {
                "$and": [
                    {"date": {"$gte": start}},
                    {"date": {"$lte": end}}
                ]
            }
    return None

# ============================================================
# Amount Utility (unchanged - already correct)
# ============================================================

def extract_amount_filter(query: str):
    q = query.lower()

    match = re.search(r"(above|greater than|more than) (\d+)", q)
    if match:
        return {"amount": {"$gt": float(match.group(2))}}

    match = re.search(r"(below|less than) (\d+)", q)
    if match:
        return {"amount": {"$lt": float(match.group(2))}}

    return None

# ============================================================
# MASTER FILTER EXTRACTOR (unchanged - already perfect)
# ============================================================

def extract_filters(query: str):
    """
    Extracts all metadata filters from a natural language query.
    Returns a Chroma-compatible where clause.
    """
    q = query.lower()
    conditions = []

    # Payment Rail
    for rail in ["SWIFT", "RTGS", "RTP"]:
        if rail.lower() in q:
            conditions.append({"payment_rail": rail})

    # Status
    for status in ["COMPLETED", "FAILED", "PENDING", "RETURNED"]:
        if status.lower() in q:
            conditions.append({"status": status})

    # Customer Segment
    for segment in ["RETAIL", "SME", "CORPORATE", "FI"]:
        if re.search(rf"\b{segment.lower()}\b", q):
            conditions.append({"customer_segment": segment})

    # Transaction Type
    for t in ["SALARY", "TRADE", "TRANSFER", "FEE"]:
        if re.search(rf"\b{t.lower()}\b", q):
            conditions.append({"transaction_type": t})

    # Value Category
    if "high value" in q:
        conditions.append({"value_category": "high value"})
    if "low value" in q:
        conditions.append({"value_category": "low value"})

    # Scope
    if "domestic" in q:
        conditions.append({"transaction_scope": "domestic"})
    if "cross border" in q or "cross-border" in q:
        conditions.append({"transaction_scope": "cross-border"})

    # Country
    country_map = {
        "india": "IN",
        "united states": "US",
        "united kingdom": "UK",
        "germany": "DE",
        "uae": "AE",
        "bahrain": "BH",
        "qatar": "QA"
    }
    for word, code in country_map.items():
        if word in q:
            conditions.append({"receiver_country": code})

    # Amount
    amount_filter = extract_amount_filter(query)
    if amount_filter:
        conditions.append(amount_filter)

    # Date (priority: exact date > exact range > between months > month > last N days)
    date_filter = (
        extract_exact_date(query)
        or extract_exact_date_range(query)
        or extract_between_months(query)
        or extract_month_filter(query)
        or extract_date_filter(query)
    )
    if date_filter:
        conditions.append(date_filter)

    # Final Chroma Format
    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}
