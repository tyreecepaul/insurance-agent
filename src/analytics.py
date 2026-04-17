"""
analytics.py
- SQL analytics over claims data using DuckDB
- Complements the vector retrieval pipeline with structured aggregate queries

The claims JSON is loaded into an in-memory DuckDB table so you can run
standard SQL without a database server. Results are returned as pandas
DataFrames for easy display and downstream use.

Usage:
    python src/analytics.py                    # run all four queries
    python src/analytics.py --query approval   # approval rate by type
    python src/analytics.py --query lodge      # days-to-lodge by incident
    python src/analytics.py --query excess     # excess recovery ratio
    python src/analytics.py --query pipeline   # full pipeline summary
"""

import argparse
from pathlib import Path

import duckdb
import pandas as pd

CLAIMS_FILE = Path("data/claims_data/claims.json")


# ── Schema helper ──────────────────────────────────────────────────────────────

def load_claims_table(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Load claims.json into an in-memory DuckDB table called 'claims'.

    DuckDB's read_json_auto infers column types from the JSON structure, so
    settlement_amount (nullable) becomes DOUBLE and dates become VARCHAR that
    we CAST inside queries.
    """
    claims_path = str(CLAIMS_FILE.resolve())
    conn.execute(f"""
        CREATE OR REPLACE TABLE claims AS
        SELECT * FROM read_json_auto('{claims_path}')
    """)


# ── Analytic queries ───────────────────────────────────────────────────────────

def approval_rate_by_type(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Q1: Approval rate and average settlement by insurance type.

    Business framing: identifies which lines of business have high rejection
    rates — a proxy for either unclear policy wording or mis-lodged claims.
    """
    return conn.execute("""
        SELECT
            insurance_type,
            COUNT(*)                                                       AS total_claims,
            SUM(CASE WHEN claim_status = 'approved' THEN 1 ELSE 0 END)    AS approved,
            SUM(CASE WHEN claim_status = 'rejected' THEN 1 ELSE 0 END)    AS rejected,
            ROUND(
                100.0
                * SUM(CASE WHEN claim_status = 'approved' THEN 1 ELSE 0 END)
                / COUNT(*),
                1
            )                                                              AS approval_pct,
            ROUND(
                AVG(CASE WHEN claim_status = 'approved'
                         THEN settlement_amount END),
                2
            )                                                              AS avg_settlement_aud
        FROM claims
        GROUP BY insurance_type
        ORDER BY approval_pct DESC
    """).fetchdf()


def avg_days_to_lodge(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Q2: Average days between incident and claim lodgement, by incident type.

    Business framing: late lodgement is a leading indicator of claim leakage
    and adjuster workload spikes. Surfacing it per incident type tells ops
    where to focus first-notice-of-loss (FNOL) automation.
    """
    return conn.execute("""
        SELECT
            incident_type,
            COUNT(*)                                                        AS n_claims,
            ROUND(
                AVG(
                    DATEDIFF(
                        'day',
                        CAST(incident_date AS DATE),
                        CAST(lodged_date   AS DATE)
                    )
                ),
                1
            )                                                               AS avg_days_to_lodge,
            MIN(
                DATEDIFF('day',
                    CAST(incident_date AS DATE),
                    CAST(lodged_date   AS DATE))
            )                                                               AS min_days,
            MAX(
                DATEDIFF('day',
                    CAST(incident_date AS DATE),
                    CAST(lodged_date   AS DATE))
            )                                                               AS max_days
        FROM claims
        WHERE incident_date IS NOT NULL
          AND lodged_date   IS NOT NULL
        GROUP BY incident_type
        ORDER BY avg_days_to_lodge
    """).fetchdf()


def excess_recovery_ratio(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Q3: Excess-to-claim ratio for approved claims by insurance type.

    Business framing: a high excess percentage signals that customers are
    lodging small claims where the excess nearly offsets the settlement —
    a pattern that drives up cost-per-claim without meaningful customer benefit.
    """
    return conn.execute("""
        SELECT
            insurance_type,
            COUNT(*)                                                        AS n_approved,
            ROUND(AVG(claim_amount),    2)                                  AS avg_claim_aud,
            ROUND(AVG(excess_paid),     2)                                  AS avg_excess_aud,
            ROUND(AVG(settlement_amount), 2)                                AS avg_settlement_aud,
            ROUND(
                100.0 * AVG(excess_paid) / NULLIF(AVG(claim_amount), 0),
                1
            )                                                               AS excess_pct
        FROM claims
        WHERE claim_status = 'approved'
          AND claim_amount  > 0
        GROUP BY insurance_type
        ORDER BY excess_pct DESC
    """).fetchdf()


def claims_pipeline_summary(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Q4: Pipeline summary — status distribution, total exposure, and settled value.

    Business framing: total_claimed minus total_settled in the 'under_review'
    bucket is the current unresolved liability — the number a CFO watches for
    reserving purposes.
    """
    return conn.execute("""
        SELECT
            claim_status,
            COUNT(*)                                                        AS n_claims,
            ROUND(SUM(claim_amount),        2)                             AS total_claimed_aud,
            ROUND(SUM(settlement_amount),   2)                             AS total_settled_aud,
            ROUND(
                100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
                1
            )                                                               AS pct_of_total
        FROM claims
        GROUP BY claim_status
        ORDER BY n_claims DESC
    """).fetchdf()


# ── Registry + runner ──────────────────────────────────────────────────────────

QUERIES: dict[str, tuple[str, callable]] = {
    "approval": (
        "Q1 — Approval rate by insurance type",
        approval_rate_by_type,
    ),
    "lodge": (
        "Q2 — Avg days to lodge by incident type",
        avg_days_to_lodge,
    ),
    "excess": (
        "Q3 — Excess recovery ratio (approved claims)",
        excess_recovery_ratio,
    ),
    "pipeline": (
        "Q4 — Claims pipeline summary",
        claims_pipeline_summary,
    ),
}


def run_analytics(query: str = "all") -> None:
    """Run one or all analytic queries and print results."""
    if not CLAIMS_FILE.exists():
        print(f"Claims file not found: {CLAIMS_FILE}")
        print("Run: python src/ingest.py --only claims")
        return

    conn = duckdb.connect(":memory:")
    load_claims_table(conn)

    row_count = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    print(f"\nClaims Analytics  |  source: {CLAIMS_FILE}  |  {row_count} records")
    print("=" * 65)

    targets = list(QUERIES.items()) if query == "all" else [(query, QUERIES[query])]

    for key, (title, fn) in targets:
        print(f"\n── {title} {'─' * max(0, 50 - len(title))}")
        df = fn(conn)
        print(df.to_string(index=False))

    conn.close()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SQL analytics over insurance claims data via DuckDB"
    )
    parser.add_argument(
        "--query",
        choices=["all"] + list(QUERIES.keys()),
        default="all",
        help="Which analytic query to run (default: all)",
    )
    args = parser.parse_args()
    run_analytics(args.query)
