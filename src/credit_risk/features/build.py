"""Stage 2.3 - build the modelling table: one row per applicant.

Everything here runs in DuckDB so the 58M history rows never land in pandas.

Three rules, taken from the availability matrix and the data quality notes:

* Only pre-application information. The matrix classifies every column; the
  bureau row filter (rows updated after the application date) comes from it too.
* "No history" is not zero. An applicant without a bureau record gets NULL
  aggregates plus an explicit HAS_* flag, so binning can give them their own bin.
* Sentinels are not numbers. 365243 means "not applicable"; it becomes NULL, and
  how often it occurs becomes a feature of its own.

Aggregates are deliberately a few dozen readable features, not a Kaggle-sized
feature factory: each one has to survive the IV / stability review in 2.5-2.6.
"""

from __future__ import annotations

import logging
from typing import Any

import duckdb

from credit_risk.data.contracts import SourceContract
from credit_risk.data.db import sql_str
from credit_risk.data.ingest import processed_path
from credit_risk.features.availability import AvailabilityMatrix
from credit_risk.settings import Settings

log = logging.getLogger(__name__)

FEATURES_FILE = "features.parquet"
SENTINEL = 365243

# Recent-history window, in days / months before the application date.
RECENT_DAYS = 365
RECENT_MONTHS = 12


def _rel(contract: SourceContract, table: str, settings: Settings) -> str:
    return f"read_parquet({sql_str(processed_path(contract, contract.tables[table], settings))})"


def _application_sql(contract: SourceContract, matrix: AvailabilityMatrix, settings: Settings,
                     con: duckdb.DuckDBPyConnection) -> str:
    """Application columns the matrix allows, plus a few standard ratios."""
    described = con.execute(f"DESCRIBE SELECT * FROM {_rel(contract, 'application_train', settings)}").fetchall()
    features = matrix.feature_columns("application_train", [row[0] for row in described])
    # DAYS_EMPLOYED is replaced by a cleaned version below.
    kept = ", ".join(f'"{c}"' for c in features if c != "DAYS_EMPLOYED")
    return f"""
    WITH app_all AS (
        SELECT 'train' AS population, * FROM {_rel(contract, 'application_train', settings)}
        UNION ALL BY NAME
        SELECT 'test' AS population, * FROM {_rel(contract, 'application_test', settings)}
    )
    SELECT
        SK_ID_CURR, population, TARGET, {kept},
        -- 365243 = retired or unemployed: a state of its own, not a missing value
        CASE WHEN DAYS_EMPLOYED = {SENTINEL} THEN NULL ELSE DAYS_EMPLOYED END AS DAYS_EMPLOYED,
        (DAYS_EMPLOYED = {SENTINEL})::INT AS APP_NOT_EMPLOYED_FLAG,
        -DAYS_BIRTH / 365.25 AS APP_AGE_YEARS,
        CASE WHEN DAYS_EMPLOYED = {SENTINEL} THEN NULL
             ELSE DAYS_EMPLOYED::DOUBLE / NULLIF(DAYS_BIRTH, 0) END AS APP_EMPLOYED_TO_AGE,
        AMT_CREDIT / NULLIF(AMT_INCOME_TOTAL, 0) AS APP_CREDIT_TO_INCOME,
        AMT_ANNUITY / NULLIF(AMT_INCOME_TOTAL, 0) AS APP_ANNUITY_TO_INCOME,
        AMT_ANNUITY / NULLIF(AMT_CREDIT, 0) AS APP_ANNUITY_TO_CREDIT,
        AMT_GOODS_PRICE / NULLIF(AMT_CREDIT, 0) AS APP_GOODS_TO_CREDIT,
        AMT_INCOME_TOTAL / NULLIF(CNT_FAM_MEMBERS, 0) AS APP_INCOME_PER_FAMILY_MEMBER
    FROM app_all
    """


def _bureau_sql(contract: SourceContract, matrix: AvailabilityMatrix, settings: Settings) -> str:
    """Credit bureau loans. Impossible dates (end before start) become NULL."""
    row_filter = matrix.table("bureau").row_filter or "TRUE"
    return f"""
    WITH b AS (
        SELECT
            SK_ID_CURR, SK_ID_BUREAU, CREDIT_ACTIVE, CREDIT_TYPE, DAYS_CREDIT,
            CREDIT_DAY_OVERDUE, CNT_CREDIT_PROLONG, AMT_CREDIT_SUM, AMT_CREDIT_SUM_DEBT,
            AMT_CREDIT_SUM_OVERDUE, AMT_CREDIT_MAX_OVERDUE,
            CASE WHEN DAYS_ENDDATE_FACT < DAYS_CREDIT THEN NULL ELSE DAYS_ENDDATE_FACT END AS DAYS_ENDDATE_FACT,
            CASE WHEN DAYS_CREDIT_ENDDATE < DAYS_CREDIT THEN NULL ELSE DAYS_CREDIT_ENDDATE END AS DAYS_CREDIT_ENDDATE
        FROM {_rel(contract, 'bureau', settings)}
        WHERE {row_filter}
    )
    SELECT
        SK_ID_CURR,
        COUNT(*) AS BUR_COUNT,
        COUNT(*) FILTER (WHERE CREDIT_ACTIVE = 'Active') AS BUR_ACTIVE_COUNT,
        AVG((CREDIT_ACTIVE = 'Active')::INT) AS BUR_ACTIVE_RATE,
        COUNT(*) FILTER (WHERE DAYS_CREDIT >= -{RECENT_DAYS}) AS BUR_COUNT_12M,
        COUNT(DISTINCT CREDIT_TYPE) AS BUR_CREDIT_TYPES,
        MAX(DAYS_CREDIT) AS BUR_DAYS_CREDIT_MAX,          -- most recent bureau loan
        MIN(DAYS_CREDIT) AS BUR_DAYS_CREDIT_MIN,          -- oldest: length of credit history
        AVG(DAYS_CREDIT) AS BUR_DAYS_CREDIT_MEAN,
        MAX(DAYS_ENDDATE_FACT) AS BUR_DAYS_ENDDATE_FACT_MAX,
        SUM(AMT_CREDIT_SUM) AS BUR_AMT_SUM,
        MAX(AMT_CREDIT_SUM) AS BUR_AMT_MAX,
        SUM(AMT_CREDIT_SUM_DEBT) AS BUR_DEBT_SUM,
        SUM(AMT_CREDIT_SUM_DEBT) / NULLIF(SUM(AMT_CREDIT_SUM), 0) AS BUR_DEBT_TO_CREDIT,
        SUM(AMT_CREDIT_SUM_OVERDUE) AS BUR_OVERDUE_SUM,
        MAX(AMT_CREDIT_MAX_OVERDUE) AS BUR_MAX_OVERDUE,
        MAX(CREDIT_DAY_OVERDUE) AS BUR_DAY_OVERDUE_MAX,
        AVG((CREDIT_DAY_OVERDUE > 0)::INT) AS BUR_OVERDUE_RATE,
        SUM(CNT_CREDIT_PROLONG) AS BUR_PROLONG_SUM
    FROM b GROUP BY SK_ID_CURR
    """


def _bureau_balance_sql(contract: SourceContract, matrix: AvailabilityMatrix, settings: Settings) -> str:
    """Monthly bureau statuses, joined to the applicant through bureau.

    STATUS: C closed, X unknown, 0 no DPD, 1-5 increasing delinquency.
    Rows whose SK_ID_BUREAU is missing from bureau cannot be attributed and drop out.
    """
    row_filter = matrix.table("bureau").row_filter or "TRUE"
    return f"""
    WITH bb AS (
        SELECT b.SK_ID_CURR, bb.MONTHS_BALANCE, TRY_CAST(bb.STATUS AS INT) AS DPD_LEVEL, bb.STATUS
        FROM {_rel(contract, 'bureau_balance', settings)} bb
        JOIN (SELECT SK_ID_BUREAU, SK_ID_CURR FROM {_rel(contract, 'bureau', settings)} WHERE {row_filter}) b
          USING (SK_ID_BUREAU)
    )
    SELECT
        SK_ID_CURR,
        COUNT(*) AS BB_MONTHS,
        MAX(DPD_LEVEL) AS BB_DPD_LEVEL_MAX,
        AVG((DPD_LEVEL > 0)::INT) AS BB_DPD_MONTH_RATE,
        COUNT(*) FILTER (WHERE DPD_LEVEL > 0) AS BB_DPD_MONTHS,
        COUNT(*) FILTER (WHERE DPD_LEVEL > 0 AND MONTHS_BALANCE >= -{RECENT_MONTHS}) AS BB_DPD_MONTHS_12M,
        AVG((STATUS = 'C')::INT) AS BB_CLOSED_RATE,
        AVG((STATUS = 'X')::INT) AS BB_UNKNOWN_RATE
    FROM bb GROUP BY SK_ID_CURR
    """


def _previous_application_sql(contract: SourceContract, settings: Settings) -> str:
    """Earlier applications at Home Credit. 365243 in the schedule dates means
    "no such date", so it is nulled out and its frequency becomes a feature."""
    return f"""
    WITH p AS (
        SELECT SK_ID_CURR, NAME_CONTRACT_STATUS, AMT_APPLICATION, AMT_CREDIT, AMT_ANNUITY,
               AMT_DOWN_PAYMENT, RATE_DOWN_PAYMENT, CNT_PAYMENT, DAYS_DECISION,
               NULLIF(DAYS_TERMINATION, {SENTINEL}) AS DAYS_TERMINATION,
               NULLIF(DAYS_FIRST_DRAWING, {SENTINEL}) AS DAYS_FIRST_DRAWING,
               (DAYS_TERMINATION = {SENTINEL})::INT AS TERMINATION_UNKNOWN
        FROM {_rel(contract, 'previous_application', settings)}
    )
    SELECT
        SK_ID_CURR,
        COUNT(*) AS PREV_COUNT,
        COUNT(*) FILTER (WHERE DAYS_DECISION >= -{RECENT_DAYS}) AS PREV_COUNT_12M,
        AVG((NAME_CONTRACT_STATUS = 'Approved')::INT) AS PREV_APPROVED_RATE,
        AVG((NAME_CONTRACT_STATUS = 'Refused')::INT) AS PREV_REFUSED_RATE,
        AVG((NAME_CONTRACT_STATUS = 'Canceled')::INT) AS PREV_CANCELED_RATE,
        MAX(DAYS_DECISION) AS PREV_DAYS_DECISION_MAX,       -- most recent application
        MIN(DAYS_DECISION) AS PREV_DAYS_DECISION_MIN,
        AVG(AMT_CREDIT) AS PREV_AMT_CREDIT_MEAN,
        MAX(AMT_CREDIT) AS PREV_AMT_CREDIT_MAX,
        SUM(AMT_CREDIT) AS PREV_AMT_CREDIT_SUM,
        AVG(AMT_CREDIT / NULLIF(AMT_APPLICATION, 0)) AS PREV_CREDIT_TO_APPLICATION,
        AVG(AMT_ANNUITY) AS PREV_AMT_ANNUITY_MEAN,
        AVG(RATE_DOWN_PAYMENT) AS PREV_DOWN_PAYMENT_RATE_MEAN,
        AVG(CNT_PAYMENT) AS PREV_CNT_PAYMENT_MEAN,
        MAX(DAYS_TERMINATION) AS PREV_DAYS_TERMINATION_MAX,
        AVG(TERMINATION_UNKNOWN) AS PREV_TERMINATION_UNKNOWN_RATE
    FROM p GROUP BY SK_ID_CURR
    """


def _pos_cash_sql(contract: SourceContract, settings: Settings) -> str:
    """Monthly status of earlier POS / cash loans. Aggregated straight to
    SK_ID_CURR: 3.4% of rows have an SK_ID_PREV that previous_application lacks."""
    return f"""
    SELECT
        SK_ID_CURR,
        COUNT(*) AS POS_MONTHS,
        COUNT(DISTINCT SK_ID_PREV) AS POS_CONTRACTS,
        MAX(SK_DPD) AS POS_SK_DPD_MAX,
        AVG((SK_DPD > 0)::INT) AS POS_DPD_MONTH_RATE,
        COUNT(*) FILTER (WHERE SK_DPD > 0 AND MONTHS_BALANCE >= -{RECENT_MONTHS}) AS POS_DPD_MONTHS_12M,
        AVG(CNT_INSTALMENT_FUTURE) AS POS_INSTALMENTS_LEFT_MEAN,
        AVG((NAME_CONTRACT_STATUS = 'Active')::INT) AS POS_ACTIVE_MONTH_RATE
    FROM {_rel(contract, 'pos_cash_balance', settings)} GROUP BY SK_ID_CURR
    """


def _credit_card_sql(contract: SourceContract, settings: Settings) -> str:
    """Monthly credit card behaviour. Utilisation is undefined when the limit is
    0 (19.6% of rows), so it stays NULL instead of becoming infinity."""
    return f"""
    WITH c AS (
        SELECT SK_ID_CURR, SK_ID_PREV, MONTHS_BALANCE, SK_DPD, NAME_CONTRACT_STATUS,
               AMT_BALANCE, AMT_DRAWINGS_ATM_CURRENT, AMT_DRAWINGS_CURRENT,
               AMT_PAYMENT_CURRENT, AMT_INST_MIN_REGULARITY,
               AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0) AS UTILISATION
        FROM {_rel(contract, 'credit_card_balance', settings)}
    )
    SELECT
        SK_ID_CURR,
        COUNT(*) AS CC_MONTHS,
        COUNT(DISTINCT SK_ID_PREV) AS CC_CARDS,
        AVG(UTILISATION) AS CC_UTILISATION_MEAN,
        MAX(UTILISATION) AS CC_UTILISATION_MAX,
        AVG(UTILISATION) FILTER (WHERE MONTHS_BALANCE >= -{RECENT_MONTHS}) AS CC_UTILISATION_MEAN_12M,
        AVG(AMT_DRAWINGS_ATM_CURRENT) AS CC_ATM_DRAWINGS_MEAN,
        AVG(AMT_PAYMENT_CURRENT / NULLIF(AMT_INST_MIN_REGULARITY, 0)) AS CC_PAYMENT_TO_MINIMUM_MEAN,
        MAX(SK_DPD) AS CC_SK_DPD_MAX,
        AVG((SK_DPD > 0)::INT) AS CC_DPD_MONTH_RATE,
        COUNT(*) FILTER (WHERE SK_DPD > 0 AND MONTHS_BALANCE >= -{RECENT_MONTHS}) AS CC_DPD_MONTHS_12M
    FROM c GROUP BY SK_ID_CURR
    """


def _installments_sql(contract: SourceContract, settings: Settings) -> str:
    """Scheduled versus actual repayment - the sharpest behavioural signal.

    A missing DAYS_ENTRY_PAYMENT / AMT_PAYMENT pair (2,905 rows) means no payment
    was recorded: it is counted, never read as "paid 0".
    """
    return f"""
    WITH i AS (
        SELECT SK_ID_CURR, DAYS_INSTALMENT, AMT_INSTALMENT, AMT_PAYMENT,
               DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT AS DAYS_LATE,
               AMT_INSTALMENT - AMT_PAYMENT AS AMT_SHORT,
               (AMT_PAYMENT IS NULL)::INT AS NO_PAYMENT_RECORD
        FROM {_rel(contract, 'installments_payments', settings)}
    )
    SELECT
        SK_ID_CURR,
        COUNT(*) AS INS_COUNT,
        SUM(NO_PAYMENT_RECORD) AS INS_NO_PAYMENT_RECORD,
        AVG(DAYS_LATE) AS INS_DAYS_LATE_MEAN,
        MAX(DAYS_LATE) AS INS_DAYS_LATE_MAX,
        AVG((DAYS_LATE > 0)::INT) AS INS_LATE_RATE,
        AVG((DAYS_LATE > 0)::INT) FILTER (WHERE DAYS_INSTALMENT >= -{RECENT_DAYS}) AS INS_LATE_RATE_12M,
        AVG((AMT_SHORT > 0.01)::INT) AS INS_UNDERPAY_RATE,
        SUM(AMT_SHORT) AS INS_AMT_SHORT_SUM,
        AVG(AMT_PAYMENT / NULLIF(AMT_INSTALMENT, 0)) AS INS_PAYMENT_RATIO_MEAN,
        SUM(AMT_PAYMENT) AS INS_AMT_PAID_SUM
    FROM i GROUP BY SK_ID_CURR
    """


#: History groups: name -> (sql builder, flag column, count column used for the flag)
HISTORY_GROUPS = {
    "bureau": ("BUR", "BUR_COUNT"),
    "bureau_balance": ("BB", "BB_MONTHS"),
    "previous_application": ("PREV", "PREV_COUNT"),
    "pos_cash_balance": ("POS", "POS_MONTHS"),
    "credit_card_balance": ("CC", "CC_MONTHS"),
    "installments_payments": ("INS", "INS_COUNT"),
}


def build_features(
    contract: SourceContract,
    matrix: AvailabilityMatrix,
    settings: Settings,
    con: duckdb.DuckDBPyConnection,
) -> dict[str, Any]:
    """Write one row per applicant to data/processed/home_credit/features.parquet."""
    groups = {
        "bureau": _bureau_sql(contract, matrix, settings),
        "bureau_balance": _bureau_balance_sql(contract, matrix, settings),
        "previous_application": _previous_application_sql(contract, settings),
        "pos_cash_balance": _pos_cash_sql(contract, settings),
        "credit_card_balance": _credit_card_sql(contract, settings),
        "installments_payments": _installments_sql(contract, settings),
    }

    ctes = [f"app AS ({_application_sql(contract, matrix, settings, con)})"]
    ctes += [f"{name} AS ({sql})" for name, sql in groups.items()]
    flags = ", ".join(
        f"({count} IS NOT NULL)::INT AS HAS_{prefix}_HISTORY"
        for name, (prefix, count) in HISTORY_GROUPS.items()
    )
    joins = " ".join(f"LEFT JOIN {name} USING (SK_ID_CURR)" for name in groups)
    selected = ["app.*"] + [f"{name}.* EXCLUDE (SK_ID_CURR)" for name in groups] + [flags]
    query = f"WITH {', '.join(ctes)} SELECT {', '.join(selected)} FROM app {joins}"

    out_path = processed_path(contract, contract.tables["application_train"], settings).with_name(FEATURES_FILE)
    tmp_path = out_path.with_suffix(".parquet.tmp")
    con.execute(f"COPY ({query}) TO {sql_str(tmp_path)} (FORMAT parquet, COMPRESSION zstd)")
    tmp_path.replace(out_path)

    rel = f"read_parquet({sql_str(out_path)})"
    n_rows, n_train, n_test = con.execute(
        f"SELECT COUNT(*), COUNT(*) FILTER (WHERE population = 'train'), "
        f"COUNT(*) FILTER (WHERE population = 'test') FROM {rel}"
    ).fetchone()
    columns = [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {rel}").fetchall()]
    coverage = {
        prefix: con.execute(f"SELECT AVG(HAS_{prefix}_HISTORY) FROM {rel}").fetchone()[0]
        for prefix, _ in HISTORY_GROUPS.values()
    }
    log.info("  features: %d rows (%d train, %d test), %d columns", n_rows, n_train, n_test, len(columns))
    log.info("  history coverage: %s", ", ".join(f"{k} {v:.1%}" for k, v in coverage.items()))

    return {
        "rows": n_rows,
        "rows_train": n_train,
        "rows_test": n_test,
        "columns": len(columns),
        "feature_columns": [c for c in columns if c not in {"SK_ID_CURR", "population", "TARGET"}],
        "history_coverage": {k: float(v) for k, v in coverage.items()},
        "output": out_path.relative_to(settings.data_dir).as_posix(),
    }
