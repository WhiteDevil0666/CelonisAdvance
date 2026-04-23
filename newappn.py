# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PQL Query Assistant  ·  Celonis-Grade  ·  5-Layer Context-Aware Validation ║
# ║  Groq + LLaMA  ·  Streamlit Cloud  ·  250+ PQL Functions                  ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  WHAT'S NEW (v2):                                                           ║
# ║  + Context Engine: table-level, function-interaction, execution awareness   ║
# ║  + Relationship Rule Engine: 20+ context-aware rules (not just regex)       ║
# ║  + Safe Auto-Corrector: non-destructive, targeted fixes with explanations   ║
# ║  + WHY Explanation Engine: every error explains root cause + fix            ║
# ║  + LLM = final validator only (not primary fixer)                           ║
# ║  + MATCH_* / conformance query protection                                   ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  LOCAL RUN                                                                  ║
# ║    pip install streamlit groq                                               ║
# ║    export GROQ_API_KEY=gsk_...                                              ║
# ║    streamlit run app.py                                                     ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  STREAMLIT CLOUD DEPLOY                                                     ║
# ║    1. Push this file + requirements.txt to GitHub                           ║
# ║    2. go to share.streamlit.io → New app → your repo                       ║
# ║    3. App Settings → Secrets → paste: GROQ_API_KEY = "gsk_..."             ║
# ║    4. Deploy ✓                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import os
import re
from dataclasses import dataclass, field
from typing import List, Set, Optional
import streamlit as st
from groq import Groq

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 · KNOWLEDGE BASE  (250+ PQL functions — sourced from official docs)
# ─────────────────────────────────────────────────────────────────────────────

COMPACT_REFS = {
    # ── Standard Aggregation ──────────────────────────────────────────────────
    'COUNT': '''[OFFICIAL] Counts non-NULL rows in the specified column.
Syntax: COUNT( table.column )
- NULL values are ignored; use COUNT_TABLE to include NULLs
- Returns INT
- Wrap with GLOBAL() when mixing case-level count with activity-level columns
- Example: COUNT("CASES"."CASE_ID") → case count
- With GLOBAL: GLOBAL(COUNT("CASES"."CASE_ID"))''',

    'COUNT_DISTINCT': '''[OFFICIAL] Counts distinct non-NULL values per group. (Alias: COUNT DISTINCT)
Syntax: COUNT( DISTINCT table.column )
- Significantly more expensive than COUNT — use COUNT when column is already a key
- NULL values are not counted
- Example: COUNT(DISTINCT "ACTIVITIES"."USER")''',

    'COUNT_TABLE': '''[OFFICIAL] Counts ALL rows in a table, including NULLs.
Syntax: COUNT_TABLE( table )
- Returns the original row count even when the common table has shifted due to joins
- Unlike COUNT which ignores NULLs, COUNT_TABLE includes them
- Use when you need a stable denominator unaffected by join multiplication
- Example: COUNT_TABLE("CASES")''',

    'SUM': '''[OFFICIAL] Sums values per group. Respects global filters (unlike PU_SUM).
Syntax: SUM( table.column )
- NULL values are ignored
- Returns same data type as input (INT→INT, FLOAT→FLOAT)
- Wrap with GLOBAL() when mixing table levels
- Example: SUM("ORDERS"."AMOUNT")''',

    'AVG': '''[OFFICIAL] Average per group. Respects global filters (unlike PU_AVG).
Syntax: AVG( table.column )
- Always returns FLOAT; NULL values ignored
- Much cheaper than MEDIAN — use AVG unless true median is required
- Wrap with GLOBAL() when mixing table levels
- Example: AVG("ORDERS"."LEAD_TIME_DAYS")''',

    'MAX': '''[OFFICIAL] Maximum value per group.
Syntax: MAX( table.column )
- NULL values ignored; returns NULL if all values are NULL
- Works with INT, FLOAT, DATE, STRING
- Example: MAX("ACTIVITIES"."TIMESTAMP")''',

    'MIN': '''[OFFICIAL] Minimum value per group.
Syntax: MIN( table.column )
- NULL values ignored; returns NULL if all values are NULL
- Works with INT, FLOAT, DATE, STRING
- Example: MIN("ACTIVITIES"."TIMESTAMP")''',

    'MEDIAN': '''[OFFICIAL] Median per group.
Syntax: MEDIAN( table.column )
- Applies to INT, FLOAT, DATE; NULL values ignored
- SIGNIFICANTLY more expensive than AVG — use only when true median is required
- Example: MEDIAN("ORDERS"."PROCESSING_DAYS")''',

    'STDEV': '''[OFFICIAL] Standard deviation (n-1 method / sample stdev) per group.
Syntax: STDEV( table.column )
- Returns FLOAT; NULL values ignored
- Example: STDEV("ORDERS"."AMOUNT")''',

    'VAR': '''[OFFICIAL] Variance (n-1 method / sample variance) per group.
Syntax: VAR( table.column )
- Returns FLOAT; NULL values ignored
- Example: VAR("ORDERS"."LEAD_TIME")''',

    'QUANTILE': '''[OFFICIAL] Quantile value per group.
Syntax: QUANTILE( table.column, quantile )
- quantile: FLOAT between 0.0 and 1.0
- Example: QUANTILE("ORDERS"."AMOUNT", 0.9) → 90th percentile''',

    'TRIMMED_MEAN': '''[OFFICIAL] Mean excluding outliers per group.
Syntax: TRIMMED_MEAN( table.column [, lower_cutoff [, upper_cutoff]] )
- lower_cutoff / upper_cutoff: fraction (0.0–1.0) of rows to trim from each end
- Useful to exclude extreme outliers without full median calculation
- Example: TRIMMED_MEAN("ORDERS"."AMOUNT", 0.05, 0.95)''',

    'MODE': '''[OFFICIAL] Most frequently occurring value per group.
Syntax: MODE( table.column )
- Applies to STRING, INT, FLOAT
- Returns NULL if all values are NULL; if tied, returns one of the tied values
- Example: MODE("ACTIVITIES"."ACTIVITY") → most common activity''',

    'PRODUCT': '''[OFFICIAL] Product (multiplication) of all values per group.
Syntax: PRODUCT( table.column )
- Multiplies all non-NULL values together
- Returns FLOAT; returns NULL if all values are NULL
- Example: PRODUCT("RATES"."FACTOR")''',

    'FIRST': '''[OFFICIAL] First element per group (standard aggregation, not pull-up).
Syntax: FIRST( table.column [, ORDER BY table.column [ASC|DESC]] )
- Returns NULL if no rows; applies to any data type
- Without ORDER BY: result depends on physical order (non-deterministic)
- ALWAYS specify ORDER BY for deterministic results
- Example: FIRST("ACTIVITIES"."ACTIVITY", ORDER BY "ACTIVITIES"."TIMESTAMP" ASC)''',

    'LAST': '''[OFFICIAL] Last element per group (standard aggregation, not pull-up).
Syntax: LAST( table.column [, ORDER BY table.column [ASC|DESC]] )
- Returns NULL if no rows; applies to any data type
- ALWAYS specify ORDER BY for deterministic results
- Example: LAST("ACTIVITIES"."ACTIVITY", ORDER BY "ACTIVITIES"."TIMESTAMP" ASC)''',

    'STRING_AGG': '''[OFFICIAL] Concatenates string values with a delimiter.
Syntax: STRING_AGG( table.column, "delimiter" [, ORDER BY table.column] [, PARTITION BY table.column] )
- NULL values are skipped
- Example: STRING_AGG("ACTIVITIES"."ACTIVITY", " → ", ORDER BY "ACTIVITIES"."TIMESTAMP" ASC)''',

    'GLOBAL': '''[OFFICIAL DOCS] Isolates aggregation from common table — prevents join multiplication.
Syntax: GLOBAL( aggregation_expression )
- WHEN TO USE: when a query mixes columns from different table levels (e.g., case + activity),
  Celonis shifts the common table to the lowest level (e.g., activity), causing case-level
  aggregations to be multiplied by the number of activities per case.
- GLOBAL() anchors the aggregation back to the original table, ignoring the join shift.
- Official FAQ example: CASE WHEN AVG("Companies"."Value") > GLOBAL(AVG("Companies"."Value")) THEN 'larger' ELSE 'smaller' END
- ALWAYS wrap CALC_THROUGHPUT with other aggregations:
  GLOBAL(AVG(CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", DAYS))))
- ALWAYS wrap case-level aggregations mixed with activity columns:
  GLOBAL(COUNT("CASES"."CASE_ID")) / GLOBAL(COUNT("ACTIVITIES"."ACTIVITY"))
- GLOBAL cannot be used inside FILTER statements
- GLOBAL result cannot be a grouper column''',

    # ── Window / Running Aggregation ───────────────────────────────────────────
    'RUNNING_TOTAL': '''[OFFICIAL] Cumulative running total of values in ordered rows. (Replaces RUNNING_SUM)
Syntax: RUNNING_TOTAL( table.column [, ORDER BY (...)] [, PARTITION BY (...)] )
- Computes the cumulative sum up to and including the current row
- ORDER BY is required for meaningful results
- PARTITION BY groups before computing running total
- Example: RUNNING_TOTAL("ORDERS"."AMOUNT", ORDER BY ("ORDERS"."ORDER_DATE" ASC))
- With partition: RUNNING_TOTAL("ORDERS"."AMOUNT", ORDER BY ("ORDERS"."DATE" ASC), PARTITION BY ("ORDERS"."VENDOR"))''',

    'RUNNING_SUM': 'Alias of RUNNING_TOTAL (older PQL). Prefer RUNNING_TOTAL in new queries. Syntax: RUNNING_SUM( column [, ORDER BY (...)] [, PARTITION BY (...)] )',

    'WINDOW_AVG': '''[OFFICIAL] Average over a sliding window of rows.
Syntax: WINDOW_AVG( table.column, lower_bound, upper_bound [, ORDER BY (...)] [, PARTITION BY (...)] )
- lower_bound / upper_bound: integer row offsets from current row (0 = current, -1 = previous, 1 = next)
- Example: WINDOW_AVG("ORDERS"."AMOUNT", -2, 2, ORDER BY ("ORDERS"."DATE" ASC)) → 5-row centered avg''',

    'INDEX_ORDER': '''[OFFICIAL] Integer row indices starting from 1.
Syntax: INDEX_ORDER( table.column [, ORDER BY (...)] [, PARTITION BY (...)] )
- Returns INT: 1 for first row, 2 for second, etc.
- ORDER BY required for deterministic numbering
- Use for ranking, pagination, or selecting Nth rows
- Example: INDEX_ORDER("CASES"."CASE_ID", ORDER BY ("CASES"."CREATE_DATE" ASC))''',

    'ZSCORE': '''[OFFICIAL] Z-score normalization (standard deviations from mean).
Syntax: ZSCORE( table.column [, PARTITION BY (...)] )
- Returns FLOAT: (value - mean) / stdev
- With PARTITION BY: z-score computed within each partition
- Example: ZSCORE("ORDERS"."AMOUNT") → outlier detection
- With partition: ZSCORE("ORDERS"."AMOUNT", PARTITION BY ("ORDERS"."VENDOR"))''',

    'INTERPOLATE': '''[OFFICIAL] Fills NULL values in a column via interpolation.
Syntax: INTERPOLATE( column, CONSTANT | LINEAR [, ORDER BY (...)] [, PARTITION BY (...)] )
- CONSTANT: fills NULLs with the last non-NULL value (forward-fill)
- LINEAR: fills NULLs with linearly interpolated values between neighbors
- ORDER BY required for meaningful interpolation
- Example: INTERPOLATE("PRICES"."AMOUNT", LINEAR, ORDER BY ("PRICES"."DATE" ASC))''',

    # ── Moving Window Aggregation ──────────────────────────────────────────────
    'MOVING_AVG': 'Moving average over a window. Syntax: MOVING_AVG( table.col, lower_bound, upper_bound [, ORDER BY ...] [, PARTITION BY ...] )',
    'MOVING_SUM': 'Moving sum over a window. Syntax: MOVING_SUM( table.col, lower_bound, upper_bound [, ORDER BY ...] [, PARTITION BY ...] )',
    'MOVING_COUNT': 'Moving count over a window. Syntax: MOVING_COUNT( table.col, lower_bound, upper_bound [, ORDER BY ...] [, PARTITION BY ...] )',
    'MOVING_COUNT_DISTINCT': 'Moving distinct count. Syntax: MOVING_COUNT_DISTINCT( table.col, lower_bound, upper_bound [, ORDER BY ...] [, PARTITION BY ...] )',
    'MOVING_MAX': 'Moving maximum. Syntax: MOVING_MAX( table.col, lower_bound, upper_bound [, ORDER BY ...] [, PARTITION BY ...] )',
    'MOVING_MIN': 'Moving minimum. Syntax: MOVING_MIN( table.col, lower_bound, upper_bound [, ORDER BY ...] [, PARTITION BY ...] )',
    'MOVING_MEDIAN': 'Moving median (expensive — requires sorting). Syntax: MOVING_MEDIAN( table.col, lower_bound, upper_bound [, ORDER BY ...] [, PARTITION BY ...] )',
    'MOVING_STDEV': 'Moving standard deviation. Syntax: MOVING_STDEV( table.col, lower_bound, upper_bound [, ORDER BY ...] [, PARTITION BY ...] )',
    'MOVING_TRIMMED_MEAN': 'Moving trimmed mean. Syntax: MOVING_TRIMMED_MEAN( table.col, lower_bound, upper_bound [, lower_cutoff, upper_cutoff] [, ORDER BY ...] [, PARTITION BY ...] )',
    'MOVING_VAR': 'Moving variance. Syntax: MOVING_VAR( table.col, lower_bound, upper_bound [, ORDER BY ...] [, PARTITION BY ...] )',

    # ── Pull-Up (PU) Aggregation ───────────────────────────────────────────────
    'PU_COUNT': '''[OFFICIAL DOCS] Counts non-NULL rows in source per target row.
Syntax: PU_COUNT( target_table, source_table.column [, filter_expression] )
- Returns 0 (not NULL) when no matching rows exist — UNIQUE among PU functions
- Requires 1:N relationship: target_table is parent (1-side), source is child (N-side)
- target_table can also be DOMAIN_TABLE(...) or CONSTANT()
- PU_COUNT IGNORES global filters — use filter_expression arg for filter-aware counts
- PREFER over PU_COUNT_DISTINCT when column is already a key (much faster)
- PREFER over PU_SUM for counting; PU_COUNT is less expensive than PU_SUM
- Example: PU_COUNT("CASES", "ACTIVITIES"."CASE_ID", "ACTIVITIES"."ACTIVITY" = 'Approve')
- Count all: PU_COUNT("CASES", "ACTIVITIES"."CASE_ID")''',

    'PU_SUM': '''[OFFICIAL DOCS] Sums source column per target row.
Syntax: PU_SUM( target_table, source_table.column [, filter_expression] )
- Returns NULL (not 0) when no matching rows exist
- Requires 1:N relationship between target_table and source table
- PU_SUM IGNORES global filters — filter via filter_expression argument
- PU_COUNT is less expensive — prefer PU_COUNT when counting (not summing)
- Example: PU_SUM("VENDORS", "ORDERS"."AMOUNT")
- Filtered: PU_SUM("CASES", "ACTIVITIES"."AMOUNT", "ACTIVITIES"."TYPE" = 'Invoice')''',

    'PU_AVG': '''[OFFICIAL DOCS] Average of source column per target row. Always returns FLOAT.
Syntax: PU_AVG( target_table, source_table.column [, filter_expression] )
- Returns NULL when no matching rows exist
- Input column must be INT or FLOAT; result is always FLOAT
- MUCH cheaper than PU_MEDIAN — prefer PU_AVG unless true median required
- PU_AVG IGNORES global filters — filter via filter_expression argument
- Example: PU_AVG("VENDORS", "ORDERS"."LEAD_TIME_DAYS")''',

    'PU_MAX': '''[OFFICIAL DOCS] Maximum of source column per target row.
Syntax: PU_MAX( target_table, source_table.column [, filter_expression] )
- Returns NULL when no matching rows exist
- Used for throughput over multiple grouped cases (grouped by a parent entity):
  DATEDIFF(\'dd\', PU_MIN("VENDORS","ACTIVITIES"."TIMESTAMP"), PU_MAX("VENDORS","ACTIVITIES"."TIMESTAMP"))
- FAQ example: PU_MAX("_CEL_CASES", SECONDS_BETWEEN(TARGET("_CEL_ACTIVITIES"."EVENTTIME"), SOURCE("_CEL_ACTIVITIES"."EVENTTIME")))''',

    'PU_MIN': '''[OFFICIAL DOCS] Minimum of source column per target row.
Syntax: PU_MIN( target_table, source_table.column [, filter_expression] )
- Returns NULL when no matching rows exist
- Combined with PU_MAX for throughput over multiple grouped cases''',

    'PU_FIRST': '''[OFFICIAL DOCS] Returns first element of source column for each target row.
Syntax: PU_FIRST( target_table, source_table.column [, filter_expression] [, ORDER BY source_table.column [ASC|DESC]] )
- Returns NULL when no matching rows exist (not 0)
- ALWAYS use explicit ORDER BY for deterministic results
- PU_FIRST(…, ORDER BY col DESC) == PU_LAST(…, ORDER BY col ASC)
- Result is scalar at target_table level — DO NOT wrap in another PU function with same target
- Example (first activity timestamp per case):
  PU_FIRST("CASES", "ACTIVITIES"."TIMESTAMP", ORDER BY "ACTIVITIES"."TIMESTAMP" ASC)
- Filtered example (first activity of specific type):
  PU_FIRST("CASES", "ACTIVITIES"."ACTIVITY", "ACTIVITIES"."TYPE" = \'System\', ORDER BY "ACTIVITIES"."TIMESTAMP" ASC)
- BIND example (1:N:1 relationship):
  PU_FIRST("VBAK", BIND("VBPA", "KNKK"."KKBER"), "VBPA"."PARVW" = \'RE\')''',

    'PU_LAST': '''[OFFICIAL DOCS] Returns last element of source column for each target row.
Syntax: PU_LAST( target_table, source_table.column [, filter_expression] [, ORDER BY source_table.column [ASC|DESC]] )
- Returns NULL when no matching rows exist
- ALWAYS use explicit ORDER BY for deterministic results
- PU_LAST(…, ORDER BY col DESC) == PU_FIRST(…, ORDER BY col ASC)
- Example (last status per order):
  PU_LAST("ORDERS", "STATUS_TABLE"."STATUS", ORDER BY "STATUS_TABLE"."CHANGE_DATE" ASC)''',

    'PU_MEDIAN': '''[OFFICIAL DOCS] Median of source column per target row.
Syntax: PU_MEDIAN( target_table, source_table.column [, filter_expression] )
- SIGNIFICANTLY more expensive than PU_AVG (requires full sort)
- Only use when true median is required — otherwise use PU_AVG
- Returns NULL when no matching rows exist''',

    'PU_COUNT_DISTINCT': '''[OFFICIAL DOCS] Distinct count of source column values per target row.
Syntax: PU_COUNT_DISTINCT( target_table, source_table.column [, filter_expression] )
- Returns 0 (not NULL) when no matching rows exist
- USE PU_COUNT instead when column is already a key (PU_COUNT is less expensive)
- Example: PU_COUNT_DISTINCT("VENDORS", "ORDERS"."PRODUCT_ID")''',

    'PU_MODE': 'Most frequent value per target row. Syntax: PU_MODE( target_table, source_table.column [, filter_expression] ) Returns the most common value; NULL if all values are NULL.',
    'PU_PRODUCT': 'Product of source column per target row. Syntax: PU_PRODUCT( target_table, source_table.column [, filter_expression] ) Returns NULL when no matching rows.',
    'PU_QUANTILE': 'Quantile of source column per target row. Syntax: PU_QUANTILE( target_table, source_table.column, quantile [, filter_expression] ) quantile: 0.0–1.0.',
    'PU_TRIMMED_MEAN': 'Trimmed mean (excludes outliers) per target row. Syntax: PU_TRIMMED_MEAN( target_table, source_table.column [, lower_cutoff [, upper_cutoff]] [, filter_expression] )',
    'PU_STRING_AGG': 'Concatenates strings from source per target row. Syntax: PU_STRING_AGG( target_table, source_table.column, delimiter [, filter_expression] [, ORDER BY col] )',
    'PU_STDEV': 'Standard deviation (n-1 method) per target row. Syntax: PU_STDEV( target_table, source_table.column [, filter_expression] )',

    # ── Pull-Up Table Options ──────────────────────────────────────────────────
    'DOMAIN_TABLE': '''[OFFICIAL] Creates a table with all distinct value combinations of the given columns.
Syntax: DOMAIN_TABLE( table.col1, table.col2, ... )
- Used as target_table in PU-functions when you need a cross-product domain
- Useful for building dimension tables on-the-fly
- Example: PU_SUM(DOMAIN_TABLE("ORDERS"."YEAR"), "ORDERS"."AMOUNT")''',

    'CONSTANT': '''[OFFICIAL] Used as target table in PU-functions to produce a single constant result.
Syntax: CONSTANT()
- When used as target_table in PU_*, produces one result row regardless of any table
- Useful for global totals in complex multi-level queries
- Example: PU_SUM(CONSTANT(), "ORDERS"."AMOUNT") → grand total''',

    'COMMON_TABLE': '''[OFFICIAL] References the common table of multiple expressions in a query.
Syntax: COMMON_TABLE( expr1, expr2 )
- Determines what the common (shared) table is for a set of expressions
- Advanced use: controlling join behavior for complex multi-table KPIs''',

    # ── Process Mining Operators ───────────────────────────────────────────────
    'CALC_THROUGHPUT': '''[OFFICIAL DOCS] Calculates throughput time per case between two event range specifiers.
Syntax: CALC_THROUGHPUT( begin_specifier TO end_specifier, REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", unit) [, activity_table.string_col] )
begin_specifier: CASE_START | FIRST_OCCURRENCE[\'activity\'] | LAST_OCCURRENCE[\'activity\']
end_specifier:   CASE_END   | FIRST_OCCURRENCE[\'activity\'] | LAST_OCCURRENCE[\'activity\']
unit: DAYS | HOURS | MINUTES | SECONDS | MILLISECONDS
Returns NULL if start > end, case has only one activity, or activity name not found.
NOTE: ALL_OCCURRENCE[\'\'] is DEPRECATED since 4.6 — use CASE_START instead.
IMPORTANT: Preferred over DATEDIFF(PU_MIN, PU_MAX) for case-level throughput.
Wrap with GLOBAL() when combined with activity-level columns.

Official patterns:
  -- Basic case throughput:
  CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", DAYS))
  -- Average throughput:
  AVG(CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", DAYS)))
  -- Between activities:
  AVG(CALC_THROUGHPUT(FIRST_OCCURRENCE[\'Create\'] TO LAST_OCCURRENCE[\'Approve\'], REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", HOURS)))
  -- Conforming cases only:
  AVG(CASE WHEN PU_SUM("CASES", ABS(conformance)) = 0
      THEN CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", HOURS)) / 24
      ELSE NULL END)''',

    'REMAP_TIMESTAMPS': '''[OFFICIAL DOCS] Converts DATE column to integer count of time units since epoch.
Syntax: REMAP_TIMESTAMPS( activity_table.timestamp_col, unit [, calendar_specification] )
Units: DAYS | HOURS | MINUTES | SECONDS | MILLISECONDS
- Primary use: provides the timestamps argument to CALC_THROUGHPUT
- Also used in SOURCE/TARGET edge throughput calculations
- Returns INT (epoch offset in specified unit); NULL input → NULL output
- Supports 3 calendar types: WEEKDAY_CALENDAR, FACTORY_CALENDAR, WORKDAY_CALENDAR (can be combined with INTERSECT)
- Official examples:
  REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", DAYS)
  REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", HOURS, WEEKDAY_CALENDAR(MON,TUE,WED,THU,FRI))
  REMAP_TIMESTAMPS("_CEL_ACTIVITIES"."EVENTTIME", SECONDS) → for edge KPIs in Process Explorer''',

    'CALC_REWORK': '''[OFFICIAL DOCS] Counts number of activities per case. Result temporarily added to case table.
Syntax: CALC_REWORK() | CALC_REWORK( filter_expression ) | CALC_REWORK( activity_table.column )
- Returns INT column on CASE table (not activity table)
- NULL case IDs → result is 0; cases without join partner in case table are ignored
- filter_expression: restricts which activities are counted
- Rework detection (repeated activity): FILTER CALC_REWORK("ACTIVITIES"."ACTIVITY" = \'Review\') > 1
- Total step count: CALC_REWORK() counts ALL activities per case
- Automation rate: (PU_COUNT("CASES","ACTIVITIES"."CASE_ID","ACTIVITIES"."TYPE"=\'System\') / CALC_REWORK()) * 100''',

    'CALC_CROP': '''[OFFICIAL] Crops cases to specified event range. Returns 1 inside range, NULL outside.
Syntax: CALC_CROP( begin_specifier TO end_specifier, activity_table.column )
- begin_specifier / end_specifier: CASE_START, CASE_END, FIRST_OCCURRENCE[\'act\'], LAST_OCCURRENCE[\'act\']
- Used to restrict analysis to a subset of the process lifecycle
- Example: FILTER CALC_CROP(FIRST_OCCURRENCE[\'Create\'] TO LAST_OCCURRENCE[\'Ship\'], "ACTIVITIES"."ACTIVITY") = 1''',

    'CALC_CROP_TO_NULL': '''[OFFICIAL] Crops cases to event range. Keeps column values inside range, NULL outside.
Syntax: CALC_CROP_TO_NULL( begin_specifier TO end_specifier, activity_table.column )
- Returns column value if activity is in range, NULL otherwise
- Useful for computing KPIs only within specific process segments''',

    'MATCH_ACTIVITIES': '''[OFFICIAL DOCS] Flags cases containing specified activities. Order-INDEPENDENT.
Syntax: MATCH_ACTIVITIES( [STARTING node_list] [NODE node_list] [ENDING node_list] [EXCLUDING node_list] )
- Returns 1 matching / 0 non-matching — use with FILTER or CASE WHEN
- STARTING: activity must be the first event; ENDING: must be the last event
- NODE: appears anywhere; EXCLUDING: must NOT appear anywhere
- Multiple activities in a clause = OR logic: NODE(\'A\',\'B\') = A or B
- Use MATCH_PROCESS for order-SENSITIVE matching
- Examples:
  FILTER MATCH_ACTIVITIES(NODE(\'Approve\'), NODE(\'Pay\'), EXCLUDING(\'Cancel\')) = 1
  FILTER MATCH_ACTIVITIES(STARTING(\'Create\'), ENDING(\'Close\')) = 1
  CASE WHEN MATCH_ACTIVITIES(NODE(\'Blocked\')) = 1 THEN \'Blocked\' ELSE \'Clean\' END''',

    'MATCH_PROCESS': '''[OFFICIAL DOCS] Matches cases against an ordered node/edge pattern. Order-SENSITIVE.
Syntax: MATCH_PROCESS( [activity_table.string_col,] node(, node)* CONNECTED BY edge(, edge)* )
- Returns INT: 1 matching, 0 non-matching
- Node types:
  NODE [act1, act2]: one of the listed activities must appear (OR logic within brackets)
  OPTIONAL [act]: activity appears 0 or 1 times
  LOOP [act]: appears 1+ times (repetition)
  OPTIONAL_LOOP [act]: appears 0+ times
  STARTING [act]: must be the FIRST activity of the case
  ENDING [act]: must be the LAST activity of the case
- Edge types:
  DIRECT [nodeA, nodeB] = B directly follows A (no gap allowed)
  EVENTUALLY [nodeA, nodeB] = B eventually follows A (gaps allowed)
- LIKE supports wildcards: NODE [LIKE \'Approve%\']
- Example:
  FILTER MATCH_PROCESS(
    STARTING ["Create Order"] AS n1,
    NODE ["Approve"] AS n2,
    ENDING ["Close"] AS n3
    CONNECTED BY EVENTUALLY[n1, n2], EVENTUALLY[n2, n3]
  ) = 1''',

    'MATCH_PROCESS_REGEX': '''[OFFICIAL] Filters variants using regex over the sequence of activity names.
Syntax: MATCH_PROCESS_REGEX( [table.col,] "regex_pattern" )
- Returns 1 matching, 0 non-matching
- Regex applied to the full variant string (activity names joined with separators)
- Example: FILTER MATCH_PROCESS_REGEX("(Create.*Approve.*Pay)") = 1''',

    'ACTIVITY_LAG': '''[OFFICIAL DOCS] Returns value from preceding row by offset within same case.
Syntax: ACTIVITY_LAG( activity_table.column [, offset] )  Default offset: 1
- Returns NULL if no preceding row at that offset
- Used for transition time: SECONDS_BETWEEN(ACTIVITY_LAG("ACTIVITIES"."TIMESTAMP"), "ACTIVITIES"."TIMESTAMP")
- Example: ACTIVITY_LAG("ACTIVITIES"."ACTIVITY") → previous activity name''',

    'ACTIVITY_LEAD': '''[OFFICIAL DOCS] Returns value from following row by offset within same case.
Syntax: ACTIVITY_LEAD( activity_table.column [, offset] )  Default offset: 1
- Returns NULL if no following row at that offset
- Example: ACTIVITY_LEAD("ACTIVITIES"."ACTIVITY") → next activity name''',

    'INDEX_ACTIVITY_ORDER': '''[OFFICIAL DOCS] Returns 1-based position of each activity within its case.
Syntax: INDEX_ACTIVITY_ORDER( activity_table.column )
- Returns INT; only non-NULL activities counted
- Replaces deprecated PROCESS_ORDER
- Example (identify first/last): CASE WHEN INDEX_ACTIVITY_ORDER("ACTIVITIES"."ACTIVITY") = 1 THEN \'First\' ELSE \'Other\' END''',

    'INDEX_ACTIVITY_LOOP': '''[OFFICIAL DOCS] Returns how many times an activity has already occurred at that point in the case.
Syntax: INDEX_ACTIVITY_LOOP( activity_table.column )
- Returns INT: 0 = first occurrence, 1 = second, 2 = third occurrence, etc.
- Parallel activities ordered by absolute timestamp
- Used for rework analysis: FILTER INDEX_ACTIVITY_LOOP("ACTIVITIES"."ACTIVITY") > 0 → all rework rows''',

    'INDEX_ACTIVITY_TYPE': '''[OFFICIAL DOCS] Returns how many times a specific activity TYPE has occurred at that point in the case.
Syntax: INDEX_ACTIVITY_TYPE( activity_table.column )
- Returns INT — type-specific loop counter per case
- Used for Rework per Activity analysis
- Example: CASE WHEN INDEX_ACTIVITY_TYPE("ACTIVITIES"."ACTIVITY") > 0 THEN \'Rework\' ELSE \'Normal\' END''',

    'PROCESS_ORDER': 'DEPRECATED — use INDEX_ACTIVITY_ORDER instead. Returns position of each activity within a case.',

    'VARIANT': '''[OFFICIAL] Returns the process variant string per case.
Syntax: VARIANT( activity_table.string_column )
- Returns a string encoding the sequence of activity names for each case
- Used for variant analysis and grouping
- Example: VARIANT("ACTIVITIES"."ACTIVITY")''',

    'ACTIVATION_COUNT': '''[OFFICIAL] Returns number of times an edge (transition between activities) was activated.
Syntax: ACTIVATION_COUNT( SOURCE["A"] TARGET["B"] )
- SOURCE/TARGET define the edge; returns INT count
- Used in edge-level KPI analysis for the Process Explorer''',

    'SOURCE_TARGET': '''[OFFICIAL] Computes values for process edges (transitions between activities).
- SOURCE("ACTIVITIES"."TIMESTAMP") → timestamp of the source (preceding) activity on the edge
- TARGET("ACTIVITIES"."TIMESTAMP") → timestamp of the target (following) activity on the edge
- Used inside PU_MAX/PU_MIN for edge-level throughput:
  PU_MAX("_CEL_CASES", SECONDS_BETWEEN(TARGET("_CEL_ACTIVITIES"."EVENTTIME"), SOURCE("_CEL_ACTIVITIES"."EVENTTIME")))''',

    # ── Conformance ───────────────────────────────────────────────────────────
    'BPMN_CONFORMS': '''[OFFICIAL] Binary BPMN conformance check (1=conforming, 0=not conforming).
Syntax: BPMN_CONFORMS( event_table.col, bpmn_model [, ALLOW(...)] )
- bpmn_model defined with SEQUENCE(), PARALLEL(), EXCLUSIVE_CHOICE()
- ALLOW() tolerates specific deviation types
- Example: BPMN_CONFORMS("ACTIVITIES"."ACTIVITY", SEQUENCE("Create","Approve","Pay"), ALLOW(BPMN_MATCH_UNDESIRED(ANY)))''',

    'CONFORMANCE': 'Petri net conformance checking. Returns INT flags. Use with READABLE() for human-readable violation descriptions.',
    'READABLE': 'Human-readable violation descriptions from CONFORMANCE. Syntax: READABLE( conformance_query )',
    'SEQUENCE': 'BPMN_CONFORMS helper: models sequential flow. Syntax: SEQUENCE("A", "B", "C")',
    'PARALLEL': 'BPMN_CONFORMS helper: models parallel paths. Syntax: PARALLEL("A", "B")',
    'EXCLUSIVE_CHOICE': 'BPMN_CONFORMS helper: models XOR gateway. Syntax: EXCLUSIVE_CHOICE("A", "B")',
    'ALLOW': 'Allows specific deviations in BPMN_CONFORMS. Syntax: ALLOW( BPMN_MATCH_UNDESIRED(ANY) )',
    'BPMN_MATCH_EXCESSIVE': 'Activity occurs at right place but too often — used in BPMN_CONFORMS ALLOW list.',
    'BPMN_MATCH_MISSING': 'Required activity missing from trace — BPMN_CONFORMS shorthand.',
    'BPMN_MATCH_OUT_OF_SEQUENCE': 'Activity at wrong position — BPMN_CONFORMS shorthand.',
    'BPMN_MATCH_UNDESIRED': 'Activity present that should not be — BPMN_CONFORMS shorthand.',
    'BPMN_MATCH_UNMAPPED': 'Activity with no model mapping — BPMN_CONFORMS shorthand.',

    'PROCESS_EQUALS': '''[OFFICIAL] Checks if a case follows an exact process sequence.
Syntax: PROCESS_EQUALS( "A" > "B" > "C" )
- Returns 1 if the case has exactly those activities in that order, 0 otherwise
- Stricter than MATCH_PROCESS — no extra activities allowed''',

    # ── DateTime ───────────────────────────────────────────────────────────────
    'DATEDIFF': '''[OFFICIAL DOCS] Computes difference between two dates in specified unit. Returns FLOAT.
Syntax: DATEDIFF( unit, table.date1, table.date2 )
Units: ms (milliseconds) | ss (seconds) | mi (minutes) | hh (hours) | dd (days) | mm (months) | yy (years)
- Supported input: DATE column type
- NULL in any parameter → NULL result
- For sub-day precision with calendar support use SECONDS_BETWEEN / HOURS_BETWEEN
- Example: DATEDIFF(\'dd\', "ORDERS"."CREATE_DATE", "ORDERS"."CLOSE_DATE")
- Cycle time pattern (correct):
  DATEDIFF(\'dd\', PU_FIRST("CASES","ACTIVITIES"."TIMESTAMP", ORDER BY "ACTIVITIES"."TIMESTAMP" ASC),
                   PU_LAST("CASES","ACTIVITIES"."TIMESTAMP", ORDER BY "ACTIVITIES"."TIMESTAMP" ASC))''',

    'HOURS_BETWEEN': 'Difference in hours. Supports optional calendar. Syntax: HOURS_BETWEEN( table.date1, table.date2 [, calendar] )',
    'MINUTES_BETWEEN': 'Difference in minutes. Syntax: MINUTES_BETWEEN( table.date1, table.date2 [, calendar] )',
    'SECONDS_BETWEEN': 'Difference in seconds. Syntax: SECONDS_BETWEEN( table.date1, table.date2 [, calendar] )',
    'MILLIS_BETWEEN': 'Difference in milliseconds. Syntax: MILLIS_BETWEEN( table.date1, table.date2 [, calendar] )',
    'DAYS_BETWEEN': 'Difference in days (FLOAT). Syntax: DAYS_BETWEEN( table.date1, table.date2 [, calendar] )',
    'WORKDAYS_BETWEEN': 'Number of workdays between two dates. Syntax: WORKDAYS_BETWEEN( calendar, table.date1, table.date2 )',
    'DATE_BETWEEN': 'Difference in days (INT). Syntax: DATE_BETWEEN( table.date1, table.date2 )',
    'MONTHS_BETWEEN': 'Difference in months. Syntax: MONTHS_BETWEEN( table.date1, table.date2 )',
    'YEARS_BETWEEN': 'Difference in years. Syntax: YEARS_BETWEEN( table.date1, table.date2 )',

    'ADD_DAYS': 'Adds days to a date. Syntax: ADD_DAYS( table.base_col, table.days_col ) base: DATE, days: INT. Output: DATE.',
    'ADD_HOURS': 'Adds hours. Syntax: ADD_HOURS( table.start_col, table.hours_col [, calendar] )',
    'ADD_MINUTES': 'Adds minutes. Syntax: ADD_MINUTES( table.start_col, table.minutes_col [, calendar] )',
    'ADD_SECONDS': 'Adds seconds. Syntax: ADD_SECONDS( table.start_col, table.seconds_col [, calendar] )',
    'ADD_MILLIS': 'Adds milliseconds. Syntax: ADD_MILLIS( table.start_col, table.ms_col [, calendar] )',
    'ADD_WORKDAYS': 'Adds workdays using a calendar. Syntax: ADD_WORKDAYS( calendar, date, number_of_days )',
    'ADD_MONTHS': 'Adds months. Syntax: ADD_MONTHS( table.date_col, table.months_col )',
    'ADD_YEARS': 'Adds years. Syntax: ADD_YEARS( table.date_col, table.years_col )',

    'TODAY': 'Current date in specified timezone. Syntax: TODAY( [timezone_id] ) Default: UTC.',
    'HOUR_NOW': 'Current hour. Syntax: HOUR_NOW( [timezone_id] )',
    'MINUTE_NOW': 'Current minute. Syntax: MINUTE_NOW( [timezone_id] )',

    'ROUND_DAY': 'Rounds timestamp down to day. Syntax: ROUND_DAY( table.date_col )',
    'ROUND_HOUR': 'Rounds timestamp down to nearest hour. Syntax: ROUND_HOUR( table.timestamp_col )',
    'ROUND_MINUTE': 'Rounds timestamp down to nearest minute. Syntax: ROUND_MINUTE( table.timestamp_col )',
    'ROUND_SECOND': 'Rounds timestamp down to nearest second. Syntax: ROUND_SECOND( table.timestamp_col )',
    'ROUND_WEEK': 'Rounds date down to Monday of the week. Syntax: ROUND_WEEK( table.date_col )',
    'ROUND_MONTH': 'Rounds date down to first day of month. Syntax: ROUND_MONTH( table.date_col )',
    'ROUND_QUARTER': 'Rounds date down to beginning of quarter. Syntax: ROUND_QUARTER( col )',
    'ROUND_YEAR': 'Rounds date down to start of year. Syntax: ROUND_YEAR( table.date_col )',

    'CONVERT_TIMEZONE': 'Converts date between timezones. Syntax: CONVERT_TIMEZONE( table.date_col [, from_tz], to_tz )',
    'DATE_MATCH': 'Returns 1 if date matches filter lists. Syntax: DATE_MATCH( col, [YEARS], [QUARTERS], [MONTHS], [WEEKS], [DAYS] )',
    'DAYS_IN_MONTH': 'Returns number of days in the month of the given date. Syntax: DAYS_IN_MONTH( table.col )',
    'IN_CALENDAR': 'Checks if date is within a calendar period. Returns 1 or NULL. Syntax: IN_CALENDAR( ts_col, calendar )',

    'CALENDAR_WEEK': 'Returns calendar week number (1-53). Syntax: CALENDAR_WEEK( table.date_col )',
    'DAY': 'Day of month (1-31). Syntax: DAY( table.date_col )',
    'DAY_OF_WEEK': 'Day of week (1=Monday…7=Sunday). Syntax: DAY_OF_WEEK( table.date_col )',
    'MONTH': 'Month number (1-12). Syntax: MONTH( table.date_col )',
    'QUARTER': 'Quarter (1-4). Syntax: QUARTER( table.date_col )',
    'YEAR': '4-digit year. Syntax: YEAR( table.date_col )',
    'HOURS': 'Hour component (0-23). Syntax: HOURS( table.timestamp_col )',
    'MINUTES': 'Minute component (0-59). Syntax: MINUTES( table.timestamp_col )',
    'SECONDS': 'Seconds component (0-59). Syntax: SECONDS( table.timestamp_col )',
    'MILLIS': 'Milliseconds component. Syntax: MILLIS( table.timestamp_col )',

    'FACTORY_CALENDAR': 'Defines factory calendar with specific work intervals. Used with REMAP_TIMESTAMPS.',
    'WORKDAY_CALENDAR': 'Defines work days from a table. Used with ADD_WORKDAYS and date diff functions.',
    'WEEKDAY_CALENDAR': '''[OFFICIAL] Defines which weekdays count as work days.
Syntax: WEEKDAY_CALENDAR( MON, TUE, WED, THU, FRI )
- Used inside REMAP_TIMESTAMPS for working-hours throughput
- Example: REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", HOURS, WEEKDAY_CALENDAR(MON,TUE,WED,THU,FRI))
- Combined: INTERSECT( WEEKDAY_CALENDAR(MON,TUE,WED,THU,FRI), FACTORY_CALENDAR(...) )''',

    'TO_TIMESTAMP': 'Deprecated. Use TO_DATE. Converts STRING to DATE with format.',

    # ── String ────────────────────────────────────────────────────────────────
    'UPPER': 'Uppercase. Syntax: UPPER( table.column )',
    'LOWER': 'Lowercase. Syntax: LOWER( table.column )',
    'CONCAT': 'Concatenates strings. Syntax: CONCAT( col1, ..., colN ) or col1 || col2. NULL in any arg = NULL result.',
    'STRING_SPLIT': 'Splits string by pattern. Zero-based index. Syntax: STRING_SPLIT( table.col, pattern, index ) Returns NULL if index out of bounds.',
    'TO_STRING': 'Converts INT or DATE to STRING. Syntax: TO_STRING( table.col [, FORMAT("%Y-%m-%d")] )',
    'FORMAT': 'Specifies date/string format. Used in TO_DATE and TO_STRING. Syntax: FORMAT( "%Y-%m-%d" )',
    'IN_LIKE': 'Pattern matching with wildcards % and _. Syntax: table.col IN_LIKE( "pattern%" ) or IN_LIKE( table2.col )',
    'LIKE': 'Pattern matching with wildcards. Syntax: table.col LIKE "pattern%" where % = any chars, _ = one char.',
    'MATCH_STRINGS': 'Finds top-k matching strings by edit distance. Syntax: MATCH_STRINGS( table1.col, table2.col [, TOP_K(k)] [, SEPARATOR(sep)] )',
    'REMAP_VALUES': 'Maps STRING column values. Syntax: REMAP_VALUES( table.col, [old1, new1], ..., [default] )',
    'REMAP_INTS': 'Maps INT column values. Syntax: REMAP_INTS( table.col, [old1, new1], ..., [default] )',
    'LEN': 'Returns string length. Syntax: LEN( table.string_col ) Returns INT.',
    'SUBSTRING': 'Extracts substring. Syntax: SUBSTRING( table.string_col, start_pos [, length] ) 1-based indexing.',
    'LTRIM': 'Removes leading whitespace. Syntax: LTRIM( table.string_col )',
    'RTRIM': 'Removes trailing whitespace. Syntax: RTRIM( table.string_col )',
    'REVERSE': 'Reverses a string. Syntax: REVERSE( table.string_col )',
    'STRINGHASH': 'Returns hash of string as INT. Syntax: STRINGHASH( table.string_col )',
    'STR_TO_INT': 'Converts string to integer. Syntax: STR_TO_INT( table.string_col ) Returns INT or NULL if not numeric.',
    'EDIT_THRESHOLD': 'Edit distance threshold for CLUSTER_STRINGS. Syntax: EDIT_THRESHOLD( distance )',
    'TOP_K': 'Number of matches in MATCH_STRINGS. Syntax: TOP_K( k ) where k <= 100.',
    'SEPARATOR': 'Separator between results in MATCH_STRINGS. Syntax: SEPARATOR( "," )',

    # ── Math & Logic ──────────────────────────────────────────────────────────
    'ABS': 'Absolute value. Syntax: ABS( table.column )',
    'POWER': 'Value raised to a power. Syntax: POWER( table.col, exponent ) Output: FLOAT.',
    'MODULO': 'Remainder of division. Syntax: MODULO( dividend, divisor ) or dividend % divisor.',
    'GREATEST': 'Maximum value across multiple columns. Syntax: GREATEST( col1, col2, ..., colN ) Good CASE WHEN alternative.',
    'LEAST': 'Minimum value across multiple columns. Syntax: LEAST( col1, col2, ..., colN )',
    'COALESCE': 'First non-NULL value. Syntax: COALESCE( col1, col2, ..., colN )',
    'ISNULL': 'Returns 1 if NULL, 0 otherwise. Syntax: ISNULL( table.column )',
    'CEIL': 'Rounds up to nearest integer. Syntax: CEIL( table.column ) Returns INT.',
    'FLOOR': 'Rounds down to nearest integer. Syntax: FLOOR( table.column ) Returns INT.',
    'ROUND': 'Rounds to specified decimal places. Syntax: ROUND( table.column, decimal_places ) Returns FLOAT.',
    'SQRT': 'Square root. Syntax: SQRT( table.column ) Returns FLOAT.',
    'SQUARE': 'Squares a value. Syntax: SQUARE( table.column ) Returns FLOAT. Equivalent to POWER(col, 2).',
    'LOG': 'Natural logarithm. Syntax: LOG( table.column ) Returns FLOAT. Column must be > 0.',
    'QNORM': 'Quantile of normal distribution. Syntax: QNORM( probability ) Returns FLOAT. probability: 0.0-1.0.',

    # ── Conditional / Boolean ─────────────────────────────────────────────────
    'CASE': 'Conditional expression. Syntax: CASE WHEN cond THEN val [WHEN ...] ELSE default END',
    'IN': 'Checks membership in a list. Syntax: table.col IN( "val1", "val2" )',
    'MULTI_IN': 'Multi-column tuple membership. Syntax: MULTI_IN( (col,...), (val1,...), (val2,...) )',
    'BETWEEN': 'Checks if value is within range (inclusive). Syntax: table.col BETWEEN lower AND upper Returns 1/0.',
    'AND': 'Logical AND.',
    'OR': 'Logical OR.',
    'NOT': 'Logical NOT. Used in NOT IN, NOT IN_LIKE, IS NOT NULL.',

    # ── Filter ────────────────────────────────────────────────────────────────
    'FILTER': '''[OFFICIAL] Filters result set. Syntax: FILTER table.col = "value"
- Multiple FILTER statements merge by logical AND
- Cannot be used inside GLOBAL()
- FILTER cannot be used inside PU_* functions — use filter_expression argument instead
- Example: FILTER "CASES"."STATUS" = \'Open\'
- Multiple: FILTER "CASES"."STATUS" = \'Open\' FILTER "CASES"."YEAR" = 2024''',

    'FILTER_TO_NULL': '''[OFFICIAL] Makes a column filter-aware by returning NULL for filtered-out rows.
Syntax: FILTER_TO_NULL( table.col )
- Prefer PU-function filter_expression argument over FILTER_TO_NULL when inside PU functions
- NEVER use FILTER_TO_NULL inside PU functions (it will fail or give wrong results)
- Example: SUM(FILTER_TO_NULL("ORDERS"."AMOUNT"))''',

    'BIND_FILTERS': 'Pulls filter to specified table. Syntax: BIND_FILTERS( target_table, condition [, condition]* )',
    'BIND': 'Pulls a value to a target table. Used in PU-functions for 1:N:1 relationships. Syntax: BIND( target_table, value )',
    'LOOKUP': 'Left outer join ignoring predefined joins. Syntax: LOOKUP( target_table, source_col, (join_cond) )',

    # ── Histogram ─────────────────────────────────────────────────────────────
    'BUCKET_UPPER_BOUND': 'Histogram bucket upper bounds. Syntax: BUCKET_UPPER_BOUND( table.col [, SUGGESTED_COUNT(n)] )',
    'SUGGESTED_COUNT': 'Suggests number of buckets in BUCKET functions. Syntax: SUGGESTED_COUNT( count )',
    'SUGGESTED_WIDTH': 'Suggests bucket width. Syntax: SUGGESTED_WIDTH( width )',
    'MAX_COUNT': 'Limits number of buckets in BUCKET functions. Syntax: MAX_COUNT( max )',

    # ── Data Generation ───────────────────────────────────────────────────────
    'GENERATE_RANGE': 'Creates a value range. Syntax: GENERATE_RANGE( step_size, range_start, range_end ) Max 10,000 elements.',
    'RANGE_APPEND': 'Creates a range and appends to a column. Syntax: RANGE_APPEND( table.col, step_size, range_end )',
    'UNIQUE_ID': 'Unique INT for each unique tuple of input columns. Syntax: UNIQUE_ID( table.col1, ..., table.colN )',

    # ── OCPM / Event Log ──────────────────────────────────────────────────────
    'CREATE_EVENTLOG': '''[OFFICIAL] Returns an activity table based on a given lead object and included event types.
Syntax: CREATE_EVENTLOG( lead_object, event_type_list )
- Used in OCPM to generate event logs from an object perspective
- lead_object: the primary business object (e.g., "Order", "Invoice")''',

    'MERGE_EVENTLOG': 'Merges columns from two activity tables into one. Syntax: MERGE_EVENTLOG( target_table.col, [FILTER ...] )',
    'MERGE_EVENTLOG_DISTINCT': 'Like MERGE_EVENTLOG but removes duplicate activities.',
    'EVENTLOG_SOURCE_TABLE': 'Returns source table name for each row in a dynamic event log. Syntax: EVENTLOG_SOURCE_TABLE( eventlog.col )',

    'LINK_PATH': '''[OFFICIAL] Traverses object links in OCPM data models.
Syntax: LINK_PATH( table.col [, CONSTRAINED BY (START(...), END(...))] )
- Used to follow relationships between business objects across the object graph
- Supports CONSTRAINED BY to restrict the traversal path''',

    'LINK_SOURCE': 'Source objects of an Object Link. Syntax: LINK_SOURCE( link_name, table.col )',
    'LINK_TARGET': 'Target objects of an Object Link. Syntax: LINK_TARGET( link_name, table.col )',
    'LINK_FILTER': 'Filters by link traversal. Syntax: LINK_FILTER( filter_expr, ANCESTORS|DESCENDANTS [, hops] )',
    'LINK_FILTER_ORDERED': 'Order-aware LINK_FILTER (only for Signal Link). Considers timestamp order.',
    'LINK_ATTRIBUTES': 'Returns link attribute values. Syntax: LINK_ATTRIBUTES( link_name, attr_col )',
    'LINK_OBJECTS': 'Creates table of all objects in the Object Link graph.',

    'UNION_ALL': 'Vertical concatenation of columns. Use with UNION_ALL_PULLBACK.',
    'UNION_ALL_TABLE': 'Vertical concatenation of tables (2-16). Syntax: UNION_ALL_TABLE( table1, ..., tableN )',
    'UNION_ALL_PULLBACK': 'Projects UNION_ALL section back to source table. Syntax: UNION_ALL_PULLBACK( union_col, index )',

    'CASE_ID_COLUMN': 'References case ID column without exact name. Syntax: CASE_ID_COLUMN( [expr] )',
    'CASE_TABLE': 'References the case table. Syntax: CASE_TABLE( [expr] )',
    'ACTIVITY_TABLE': 'References the activity table in OCPM. Syntax: ACTIVITY_TABLE( LINK_PATH(...) )',
    'ACTIVITY_COLUMN': 'References the activity column. Syntax: ACTIVITY_COLUMN( [expr] )',
    'TIMESTAMP_COLUMN': 'References the timestamp column. Syntax: TIMESTAMP_COLUMN( [expr] )',
    'END_TIMESTAMP_COLUMN': 'References the end timestamp column. Syntax: END_TIMESTAMP_COLUMN( [expr] )',

    # ── Currency & Quantity ────────────────────────────────────────────────────
    'CURRENCY_CONVERT': '''[OFFICIAL] Converts currency using a rates table.
Syntax: CURRENCY_CONVERT( amount, FROM("USD"), TO("EUR"), date, "RATES_TABLE" )
- date: the exchange rate date to use
- RATES_TABLE: table containing currency rates''',

    'CURRENCY_CONVERT_SAP': 'Converts SAP currency using TCURR/TCURF/TCURX internal tables.',
    'CURRENCY_SAP': 'Adjusts SAP amounts for decimal places. Syntax: CURRENCY_SAP( table.amount_col, table.currency_col )',
    'QUANTITY_CONVERT': 'Converts quantity units. Syntax: QUANTITY_CONVERT( amount, FROM("unit1"), TO("unit2"), id_col, "RATES" )',

    # ── Static / Meta Functions ────────────────────────────────────────────────
    'COLUMN_TYPE': '''[OFFICIAL STATIC] Returns data type of a column as STRING before executing the query.
Syntax: COLUMN_TYPE( table.col )
Returns: "INT", "FLOAT", "STRING", or "DATE"
- Static function — evaluated at query-build time, not at runtime
- Use to conditionally change query logic based on column type''',

    'ARGUMENT_COUNT': '''[OFFICIAL STATIC] Counts number of arguments passed at query-build time.
Syntax: ARGUMENT_COUNT( arg1, arg2, ... )
- Static function — useful for dynamic query generation with variables''',

    'USER_NAME': 'Returns the currently logged-in username. Syntax: USER_NAME() Returns STRING.',
    'VARIABLE': 'Dynamic variable in PQL. Use <% if(VAR != "") { %> FILTER ... <% } %> to guard empty variables.',

    # ── ML & Clustering ────────────────────────────────────────────────────────
    'KMEANS': 'K-means++ clustering. Syntax: KMEANS( k, col1, col2 ) or KMEANS( TRAIN_KM(...), CLUSTER(...) )',
    'TRAIN_KM': 'Trains a KMeans model. Syntax: TRAIN_KM( k, INPUT( table.col1, ... ) )',
    'CLUSTER': 'Assigns rows to clusters. Syntax: CLUSTER( TRAIN_KM(...), table.col, ... )',
    'LINEAR_REGRESSION': 'Linear regression. Syntax: LINEAR_REGRESSION( TRAIN_LM( INPUT(...), OUTPUT(...) ), PREDICT( col ) )',
    'TRAIN_LM': 'Trains a Linear Regression model. Syntax: TRAIN_LM( INPUT( table.col, ... ), OUTPUT( table.col ) )',
    'PREDICT': 'Specifies prediction columns. Syntax: PREDICT( table.col, ... )',
    'CLUSTER_VARIANTS': 'Clusters process variants. Syntax: CLUSTER_VARIANTS( k [, ESTIMATE_CLUSTER_PARAMS(...)] )',
    'ESTIMATE_CLUSTER_PARAMS': 'Estimates optimal cluster parameters for CLUSTER_VARIANTS. Syntax: ESTIMATE_CLUSTER_PARAMS( max_k )',
    'DECISION_TREE': 'Decision tree classification. Syntax: DECISION_TREE( TRAIN_DT( INPUT(...), OUTPUT(...) ), PREDICT(...) )',
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1B · PANEL DATA (Sidebar reference panels)
# ─────────────────────────────────────────────────────────────────────────────

PANEL_DATA = {
    'Pull-Up (PU) Aggregation': [
        {'name': 'PU_COUNT',          'doc': 'Count rows in source per target row. Returns 0 (not NULL). Prefer over PU_COUNT_DISTINCT for key cols.'},
        {'name': 'PU_SUM',            'doc': 'Sum source column per target row. Returns NULL when no match.'},
        {'name': 'PU_AVG',            'doc': 'Average of source column per target row. Always FLOAT. Much cheaper than PU_MEDIAN.'},
        {'name': 'PU_MAX',            'doc': 'Maximum of source column per target row.'},
        {'name': 'PU_MIN',            'doc': 'Minimum of source column per target row.'},
        {'name': 'PU_FIRST',          'doc': 'First element of source column per target row. Always use ORDER BY.'},
        {'name': 'PU_LAST',           'doc': 'Last element of source column per target row. Always use ORDER BY.'},
        {'name': 'PU_MEDIAN',         'doc': 'Median per target row. Very expensive — use PU_AVG when possible.'},
        {'name': 'PU_COUNT_DISTINCT', 'doc': 'Distinct count per target row. Returns 0. Use PU_COUNT for key columns.'},
        {'name': 'PU_MODE',           'doc': 'Most frequent value per target row.'},
        {'name': 'PU_PRODUCT',        'doc': 'Product of source column per target row. Returns NULL on no match.'},
        {'name': 'PU_QUANTILE',       'doc': 'Quantile (0.0–1.0) per target row.'},
        {'name': 'PU_TRIMMED_MEAN',   'doc': 'Trimmed mean (excludes outliers) per target row.'},
        {'name': 'PU_STRING_AGG',     'doc': 'Concatenates strings from source per target row.'},
        {'name': 'PU_STDEV',          'doc': 'Standard deviation per target row.'},
        {'name': 'CONSTANT',          'doc': 'Used as target_table for a single global result.'},
        {'name': 'DOMAIN_TABLE',      'doc': 'All distinct combinations of columns — use as PU target.'},
    ],
    'Standard Aggregation': [
        {'name': 'COUNT',       'doc': 'Count non-NULL rows. Wrap with GLOBAL() when mixing table levels.'},
        {'name': 'COUNT_TABLE', 'doc': 'Counts rows including NULLs. Returns original count regardless of join shift.'},
        {'name': 'SUM',         'doc': 'Sum per group. Respects global filters.'},
        {'name': 'AVG',         'doc': 'Average per group. Returns FLOAT. Respects global filters.'},
        {'name': 'MAX',         'doc': 'Maximum per group. Works with INT, FLOAT, DATE, STRING.'},
        {'name': 'MIN',         'doc': 'Minimum per group.'},
        {'name': 'MEDIAN',      'doc': 'Median per group. Expensive — use AVG unless median required.'},
        {'name': 'STDEV',       'doc': 'Standard deviation (n-1) per group.'},
        {'name': 'VAR',         'doc': 'Variance (n-1) per group.'},
        {'name': 'MODE',        'doc': 'Most frequent value per group.'},
        {'name': 'PRODUCT',     'doc': 'Product of all values per group.'},
        {'name': 'QUANTILE',    'doc': 'Quantile per group. Syntax: QUANTILE( table.col, quantile )'},
        {'name': 'TRIMMED_MEAN','doc': 'Mean excluding outliers per group.'},
        {'name': 'FIRST',       'doc': 'First element per group. Always use ORDER BY.'},
        {'name': 'LAST',        'doc': 'Last element per group. Always use ORDER BY.'},
        {'name': 'STRING_AGG',  'doc': 'Concatenates strings with a delimiter.'},
        {'name': 'GLOBAL',      'doc': 'Isolates aggregation to prevent join multiplication. Use when mixing table levels.'},
    ],
    'Window Aggregation': [
        {'name': 'RUNNING_TOTAL',     'doc': 'Cumulative running total (preferred over RUNNING_SUM). Needs ORDER BY.'},
        {'name': 'WINDOW_AVG',        'doc': 'Average over a sliding window of rows.'},
        {'name': 'INDEX_ORDER',       'doc': 'Integer indices from 1. Supports ORDER BY and PARTITION BY.'},
        {'name': 'ZSCORE',            'doc': 'Z-score normalization. Supports PARTITION BY.'},
        {'name': 'INTERPOLATE',       'doc': 'Interpolates NULL values (CONSTANT or LINEAR).'},
        {'name': 'MOVING_AVG',        'doc': 'Moving average over a row window.'},
        {'name': 'MOVING_SUM',        'doc': 'Moving sum over a row window.'},
        {'name': 'MOVING_COUNT',      'doc': 'Moving count over a row window.'},
        {'name': 'MOVING_MAX',        'doc': 'Moving maximum over a row window.'},
        {'name': 'MOVING_MIN',        'doc': 'Moving minimum over a row window.'},
        {'name': 'MOVING_MEDIAN',     'doc': 'Moving median (expensive). Needs ORDER BY.'},
        {'name': 'MOVING_STDEV',      'doc': 'Moving standard deviation.'},
    ],
    'Process & Conformance': [
        {'name': 'CALC_THROUGHPUT',    'doc': 'Throughput time per case. Wrap with GLOBAL() when mixing with activity KPIs.'},
        {'name': 'REMAP_TIMESTAMPS',   'doc': 'Converts timestamp to units for CALC_THROUGHPUT. Supports calendars.'},
        {'name': 'CALC_REWORK',        'doc': 'Counts activities per case. Returns INT on case table.'},
        {'name': 'CALC_CROP',          'doc': 'Crops cases to event range. Returns 1 in range, NULL outside.'},
        {'name': 'CALC_CROP_TO_NULL',  'doc': 'Crops cases to event range. Keeps values in range, NULL outside.'},
        {'name': 'MATCH_ACTIVITIES',   'doc': 'Flags cases with activities (order-independent). Supports EXCLUDING.'},
        {'name': 'MATCH_PROCESS',      'doc': 'Matches variants against node/edge pattern (order-sensitive).'},
        {'name': 'MATCH_PROCESS_REGEX','doc': 'Filters variants using regex over activity names.'},
        {'name': 'ACTIVITY_LAG',       'doc': 'Previous row by offset within a case. Default offset: 1.'},
        {'name': 'ACTIVITY_LEAD',      'doc': 'Next row by offset within a case. Default offset: 1.'},
        {'name': 'INDEX_ACTIVITY_ORDER','doc': '1-based position of each activity within its case.'},
        {'name': 'INDEX_ACTIVITY_LOOP','doc': 'Number of prior occurrences of this activity in the case (0 = first).'},
        {'name': 'INDEX_ACTIVITY_TYPE','doc': 'Type-specific loop counter per case (for rework per activity).'},
        {'name': 'VARIANT',            'doc': 'Process variant string per case.'},
        {'name': 'PROCESS_EQUALS',     'doc': 'Checks if case follows an exact sequence. Strict — no extra activities.'},
        {'name': 'ACTIVATION_COUNT',   'doc': 'Count of edge activations between two activities.'},
        {'name': 'BPMN_CONFORMS',      'doc': 'Binary BPMN conformance (1/0). Supports ALLOW() for tolerances.'},
        {'name': 'CONFORMANCE',        'doc': 'Petri net conformance. Use with READABLE() for descriptions.'},
    ],
    'DateTime': [
        {'name': 'ADD_DAYS',          'doc': 'Adds days to a date.'},
        {'name': 'DATEDIFF',          'doc': 'Date difference. Units: ms|ss|mi|hh|dd|mm|yy. Returns FLOAT.'},
        {'name': 'HOURS_BETWEEN',     'doc': 'Difference in hours. Supports calendar.'},
        {'name': 'SECONDS_BETWEEN',   'doc': 'Difference in seconds. Supports calendar.'},
        {'name': 'WORKDAYS_BETWEEN',  'doc': 'Number of workdays between dates.'},
        {'name': 'DAYS_BETWEEN',      'doc': 'Difference in days (FLOAT). Supports calendar.'},
        {'name': 'MONTHS_BETWEEN',    'doc': 'Difference in months.'},
        {'name': 'ADD_HOURS',         'doc': 'Adds hours. Supports calendar.'},
        {'name': 'ADD_WORKDAYS',      'doc': 'Adds workdays using a calendar.'},
        {'name': 'ROUND_DAY',         'doc': 'Rounds down to day.'},
        {'name': 'ROUND_WEEK',        'doc': 'Rounds down to Monday of the week.'},
        {'name': 'ROUND_MONTH',       'doc': 'Rounds down to first day of month.'},
        {'name': 'ROUND_QUARTER',     'doc': 'Rounds down to start of quarter.'},
        {'name': 'TODAY',             'doc': 'Current date. Syntax: TODAY([timezone])'},
        {'name': 'CONVERT_TIMEZONE',  'doc': 'Converts date between timezones.'},
        {'name': 'DATE_MATCH',        'doc': 'Returns 1 if date matches filter lists.'},
        {'name': 'WEEKDAY_CALENDAR',  'doc': 'Calendar specifying work weekdays. Used in REMAP_TIMESTAMPS.'},
        {'name': 'FACTORY_CALENDAR',  'doc': 'Factory calendar with work intervals.'},
        {'name': 'DAY', 'doc': 'Day of month (1-31).'},
        {'name': 'MONTH', 'doc': 'Month number (1-12).'},
        {'name': 'YEAR', 'doc': '4-digit year.'},
        {'name': 'QUARTER', 'doc': 'Quarter (1-4).'},
        {'name': 'CALENDAR_WEEK', 'doc': 'Calendar week number (1-53).'},
    ],
    'String': [
        {'name': 'UPPER',        'doc': 'Uppercase.'},
        {'name': 'LOWER',        'doc': 'Lowercase.'},
        {'name': 'CONCAT',       'doc': 'Concatenates strings. NULL in any arg = NULL result.'},
        {'name': 'STRING_SPLIT', 'doc': 'Splits string by pattern. Zero-based index.'},
        {'name': 'TO_STRING',    'doc': 'Converts INT or DATE to STRING.'},
        {'name': 'SUBSTRING',    'doc': 'Extracts substring. 1-based indexing.'},
        {'name': 'LEN',          'doc': 'String length. Returns INT.'},
        {'name': 'IN_LIKE',      'doc': 'Pattern matching with wildcards % and _.'},
        {'name': 'MATCH_STRINGS','doc': 'Fuzzy matching by edit distance. Supports TOP_K.'},
        {'name': 'REMAP_VALUES', 'doc': 'Maps STRING values to new values.'},
        {'name': 'LTRIM',        'doc': 'Removes leading whitespace.'},
        {'name': 'RTRIM',        'doc': 'Removes trailing whitespace.'},
        {'name': 'REVERSE',      'doc': 'Reverses a string.'},
    ],
    'Math & Logic': [
        {'name': 'ABS',               'doc': 'Absolute value.'},
        {'name': 'POWER',             'doc': 'Raises to a power. Output: FLOAT.'},
        {'name': 'MODULO',            'doc': 'Remainder of division. Can use % operator.'},
        {'name': 'GREATEST',          'doc': 'Maximum across columns. Good CASE WHEN alternative.'},
        {'name': 'LEAST',             'doc': 'Minimum across columns.'},
        {'name': 'COALESCE',          'doc': 'First non-NULL value.'},
        {'name': 'ISNULL',            'doc': 'Returns 1 if NULL, 0 otherwise.'},
        {'name': 'CASE',              'doc': 'CASE WHEN cond THEN val ELSE default END'},
        {'name': 'ROUND',             'doc': 'Rounds to specified decimal places.'},
        {'name': 'CEIL',              'doc': 'Rounds up to nearest integer.'},
        {'name': 'FLOOR',             'doc': 'Rounds down to nearest integer.'},
        {'name': 'SQRT',              'doc': 'Square root. Returns FLOAT.'},
        {'name': 'LOG',               'doc': 'Natural logarithm. Returns FLOAT. Input must be > 0.'},
        {'name': 'QNORM',             'doc': 'Quantile of normal distribution.'},
        {'name': 'BUCKET_UPPER_BOUND','doc': 'Histogram bucket upper bounds.'},
        {'name': 'ZSCORE',            'doc': 'Z-score normalization.'},
    ],
    'Filter & Lookup': [
        {'name': 'FILTER',        'doc': 'Filters result set. Multiple filters merge by AND.'},
        {'name': 'FILTER_TO_NULL','doc': 'Makes columns filter-aware. Never use inside PU functions.'},
        {'name': 'BIND_FILTERS',  'doc': 'Pulls filter to specified table.'},
        {'name': 'BIND',          'doc': 'Pulls value to target table. Used for 1:N:1 relationships.'},
        {'name': 'IN',            'doc': 'Membership test. Syntax: col IN( "val1", "val2" )'},
        {'name': 'MULTI_IN',      'doc': 'Multi-column tuple membership test.'},
        {'name': 'LOOKUP',        'doc': 'Left outer join ignoring predefined joins.'},
        {'name': 'REMAP_VALUES',  'doc': 'Maps STRING values to new values.'},
        {'name': 'DOMAIN_TABLE',  'doc': 'All distinct combinations of columns.'},
        {'name': 'GENERATE_RANGE','doc': 'Creates a value range. Max 10,000 elements.'},
        {'name': 'COALESCE',      'doc': 'First non-NULL value from a list.'},
    ],
    'Event Log & OCPM': [
        {'name': 'CREATE_EVENTLOG',        'doc': 'Creates activity table from OCPM object perspective.'},
        {'name': 'MERGE_EVENTLOG',         'doc': 'Merges columns from two activity tables.'},
        {'name': 'MERGE_EVENTLOG_DISTINCT','doc': 'Like MERGE_EVENTLOG but removes duplicates.'},
        {'name': 'LINK_PATH',              'doc': 'Traverses object links. Supports CONSTRAINED BY.'},
        {'name': 'LINK_SOURCE',            'doc': 'Source objects of Object Link.'},
        {'name': 'LINK_TARGET',            'doc': 'Target objects of Object Link.'},
        {'name': 'LINK_FILTER',            'doc': 'Filters by ANCESTORS or DESCENDANTS link traversal.'},
        {'name': 'LINK_OBJECTS',           'doc': 'All objects in the Object Link graph.'},
        {'name': 'UNION_ALL',              'doc': 'Vertical concatenation of columns.'},
        {'name': 'UNION_ALL_TABLE',        'doc': 'Vertical concatenation of tables (2-16).'},
        {'name': 'EVENTLOG_SOURCE_TABLE',  'doc': 'Source table name for each row in dynamic event log.'},
        {'name': 'CASE_ID_COLUMN',         'doc': 'References case ID column without exact name.'},
        {'name': 'ACTIVITY_TABLE',         'doc': 'References the activity table in OCPM.'},
    ],
    'Currency & Quantity': [
        {'name': 'CURRENCY_CONVERT',     'doc': 'Converts currency using a rates table.'},
        {'name': 'CURRENCY_CONVERT_SAP', 'doc': 'Converts SAP currency using TCURR/TCURF/TCURX.'},
        {'name': 'CURRENCY_SAP',         'doc': 'Adjusts SAP amounts for decimal places.'},
        {'name': 'QUANTITY_CONVERT',     'doc': 'Converts quantity units using a rates table.'},
    ],
    'ML & Clustering': [
        {'name': 'KMEANS',            'doc': 'K-means++ clustering. Simple or advanced with TRAIN_KM.'},
        {'name': 'TRAIN_KM',          'doc': 'Trains a KMeans model.'},
        {'name': 'CLUSTER',           'doc': 'Assigns rows to trained clusters.'},
        {'name': 'LINEAR_REGRESSION', 'doc': 'Linear regression with TRAIN_LM and PREDICT.'},
        {'name': 'CLUSTER_VARIANTS',  'doc': 'Clusters process variants.'},
        {'name': 'DECISION_TREE',     'doc': 'Decision tree classification.'},
        {'name': 'ZSCORE',            'doc': 'Z-score normalization for outlier detection.'},
        {'name': 'MATCH_STRINGS',     'doc': 'Fuzzy string matching by edit distance.'},
    ],
    'Static / Meta': [
        {'name': 'COLUMN_TYPE',     'doc': 'Returns data type of column at query-build time: INT/FLOAT/STRING/DATE.'},
        {'name': 'ARGUMENT_COUNT',  'doc': 'Counts arguments at query-build time. Useful for dynamic queries.'},
        {'name': 'USER_NAME',       'doc': 'Returns the currently logged-in username.'},
        {'name': 'UNIQUE_ID',       'doc': 'Unique INT for each unique tuple of input columns.'},
        {'name': 'COMMON_TABLE',    'doc': 'References the common table of multiple expressions.'},
    ],
}

CATEGORY_ICONS = {
    'Pull-Up (PU) Aggregation': '⬆',
    'Standard Aggregation': '∑',
    'Window Aggregation': '⧉',
    'Process & Conformance': '⚙',
    'DateTime': '📅',
    'String': 'Aa',
    'Math & Logic': '±',
    'Filter & Lookup': '🔍',
    'Event Log & OCPM': '🔗',
    'Currency & Quantity': '💱',
    'ML & Clustering': '🧠',
    'Static / Meta': '🔬',
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 · SMART FUNCTION RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

FUNCTION_NAMES = list(COMPACT_REFS.keys())
PU_FUNCTIONS = [fn for fn in FUNCTION_NAMES if fn.startswith("PU_")]

INTENT_PATTERNS = [
    (r'per\s+(case|vendor|order|customer|supplier|group|\w+)', PU_FUNCTIONS[:8]),
    (r'(aggregate|group\s+by|count\s+per|sum\s+per|average\s+per)', PU_FUNCTIONS[:8]),
    (r'(throughput|cycle.?time|lead.?time|duration|process.?time|elapsed)', ['CALC_THROUGHPUT', 'REMAP_TIMESTAMPS', 'GLOBAL']),
    (r'(first.*last|start.*end|begin.*end).*(time|date|day)', ['CALC_THROUGHPUT', 'REMAP_TIMESTAMPS', 'PU_FIRST', 'PU_LAST', 'DATEDIFF']),
    (r'(rework|repeat|loop|same.?activit|revisit|multiple.?time)', ['CALC_REWORK', 'INDEX_ACTIVITY_LOOP', 'INDEX_ACTIVITY_TYPE']),
    (r'(conform|path|sequence|order.*activit|activit.*order|follow)', ['MATCH_PROCESS', 'MATCH_ACTIVITIES', 'BPMN_CONFORMS']),
    (r'(days?\s+between|hours?\s+between|date.?diff|workday|calendar)', ['DATEDIFF', 'HOURS_BETWEEN', 'WORKDAYS_BETWEEN', 'REMAP_TIMESTAMPS']),
    (r'(automat|system.?activit|manual.?activit|bot|automation.?rate)', ['PU_COUNT', 'CALC_REWORK', 'GLOBAL']),
    (r'(variant|process.?flow|happy.?path)', ['VARIANT', 'MATCH_PROCESS', 'MATCH_PROCESS_REGEX', 'CLUSTER_VARIANTS']),
    (r'(running|cumulative|rolling|window|moving)', ['RUNNING_TOTAL', 'WINDOW_AVG', 'MOVING_AVG', 'INDEX_ORDER']),
    (r'(filter|where|only.*cases|exclude)', ['FILTER', 'MATCH_ACTIVITIES', 'BIND_FILTERS', 'FILTER_TO_NULL']),
    (r'(ocpm|object.?centric|multi.?object|link)', ['LINK_PATH', 'LINK_FILTER', 'CREATE_EVENTLOG', 'LINK_OBJECTS']),
    (r'(sap|currency|amount|convert)', ['CURRENCY_CONVERT', 'CURRENCY_CONVERT_SAP', 'CURRENCY_SAP']),
    (r'(cluster|segment|kmeans|ml|machine.?learn|predict|regression)', ['KMEANS', 'CLUSTER_VARIANTS', 'LINEAR_REGRESSION', 'DECISION_TREE']),
    (r'(outlier|z.?score|anomaly|abnormal)', ['ZSCORE', 'TRIMMED_MEAN', 'BUCKET_UPPER_BOUND']),
    (r'(median|percentile|quantile|p\d\d)', ['MEDIAN', 'QUANTILE', 'PU_MEDIAN', 'PU_QUANTILE']),
    (r'(lag|lead|previous|next).*(activit|event|step)', ['ACTIVITY_LAG', 'ACTIVITY_LEAD']),
    (r'(first|last).*(occurrence|time|activit)', ['PU_FIRST', 'PU_LAST', 'FIRST', 'LAST', 'CALC_THROUGHPUT']),
    (r'(NULL|missing|empty|blank|coalesce|fill)', ['COALESCE', 'ISNULL', 'INTERPOLATE', 'FILTER_TO_NULL']),
    (r'(on.?time|overdue|delay|late|sla|due)', ['DATEDIFF', 'HOURS_BETWEEN', 'WORKDAYS_BETWEEN', 'CALC_THROUGHPUT']),
    (r'(bottleneck|slow|wait|queue)', ['CALC_THROUGHPUT', 'ACTIVITY_LAG', 'SECONDS_BETWEEN', 'PU_AVG']),
]

def detect_functions(text: str):
    text_lower = text.lower()
    found = set()
    NEEDS_WORD_BOUNDARY = {
        'AVG', 'SUM', 'MAX', 'MIN', 'VAR', 'IN', 'OR', 'AND', 'NOT',
        'ADD', 'SUB', 'DIV', 'MULT', 'LOG', 'LEN', 'ABS',
        'CEIL', 'FLOOR', 'ROUND', 'SQRT', 'SQUARE', 'FIRST', 'LAST', 'MODE',
        'DAY', 'MONTH', 'YEAR', 'HOURS', 'MINUTES', 'SECONDS', 'MILLIS', 'QUARTER',
        'CASE', 'WHEN', 'LIKE', 'RANGE', 'REVERSE', 'BETWEEN',
        'STDEV', 'COUNT', 'FILTER', 'BIND', 'LOOKUP', 'UPPER', 'LOWER',
        'PRODUCT', 'VARIANT', 'CONSTANT', 'FORMAT', 'MEDIAN', 'QUANTILE',
    }
    for fn in FUNCTION_NAMES:
        fn_lower = fn.lower()
        if fn in NEEDS_WORD_BOUNDARY:
            if re.search(r'\b' + re.escape(fn_lower) + r'\b', text_lower):
                found.add(fn)
        else:
            if fn_lower in text_lower:
                found.add(fn)
    for pattern, fns in INTENT_PATTERNS:
        if re.search(pattern, text_lower):
            found.update(fns)
    return list(found)

def build_function_context(user_query: str):
    funcs = detect_functions(user_query)
    if not funcs:
        return ""
    docs = []
    seen = set()
    for fn in funcs[:25]:
        if fn in COMPACT_REFS and fn not in seen:
            seen.add(fn)
            docs.append(f"### {fn}\n{COMPACT_REFS[fn]}")
    return "\n\n".join(docs)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3A · PQL CONTEXT ENGINE  ← NEW in v2
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PQLContext:
    """Execution-level context extracted from a PQL query."""
    used_functions: Set[str] = field(default_factory=set)
    pu_functions: Set[str] = field(default_factory=set)
    table_levels: Set[str] = field(default_factory=set)   # CASE | ACTIVITY | OBJECT | VENDOR | ORDER
    has_filter: bool = False
    has_global: bool = False
    has_pu: bool = False
    has_match: bool = False           # MATCH_ACTIVITIES, MATCH_PROCESS, etc.
    has_conformance: bool = False     # BPMN_CONFORMS, CONFORMANCE
    has_calc_throughput: bool = False
    has_calc_rework: bool = False
    has_running: bool = False
    has_sql_keywords: bool = False
    is_match_safe: bool = False       # True → skip quote enforcement
    aggregation_fns: Set[str] = field(default_factory=set)  # AVG, SUM, COUNT, etc.
    standard_agg_fns: Set[str] = field(default_factory=set)


def extract_context(query: str) -> PQLContext:
    """
    Statically analyse a PQL query and extract full execution context.
    This is the foundation of the context-aware validation pipeline.
    """
    ctx = PQLContext()
    q = query.upper()

    # ── Function detection ────────────────────────────────────────────────────
    for fn in FUNCTION_NAMES:
        if re.search(r'\b' + re.escape(fn) + r'\b', q):
            ctx.used_functions.add(fn)
            if fn.startswith("PU_"):
                ctx.pu_functions.add(fn)

    ctx.has_pu            = bool(ctx.pu_functions)
    ctx.has_filter        = bool(re.search(r'\bFILTER\b', q))
    ctx.has_global        = "GLOBAL(" in q
    ctx.has_match         = bool(re.search(r'\bMATCH_(ACTIVITIES|PROCESS|PROCESS_REGEX)\b', q))
    ctx.has_conformance   = bool(re.search(r'\b(BPMN_CONFORMS|CONFORMANCE)\b', q))
    ctx.has_calc_throughput = "CALC_THROUGHPUT" in q
    ctx.has_calc_rework   = "CALC_REWORK" in q
    ctx.has_running       = bool(re.search(r'\b(RUNNING_TOTAL|RUNNING_SUM)\b', q))
    ctx.is_match_safe     = ctx.has_match or ctx.has_conformance

    # ── Aggregation detection ─────────────────────────────────────────────────
    STD_AGGS = {'AVG', 'SUM', 'COUNT', 'MEDIAN', 'MIN', 'MAX', 'STDEV', 'MODE'}
    for fn in STD_AGGS:
        if re.search(r'\b' + fn + r'\s*\(', q):
            ctx.aggregation_fns.add(fn)
            ctx.standard_agg_fns.add(fn)

    # ── Table level detection ─────────────────────────────────────────────────
    TABLE_PATTERNS = {
        'CASE':     [r'"CASES?"', r'"_CEL_CASES?"', r'CASE_TABLE\('],
        'ACTIVITY': [r'"ACTIVITIES?"', r'"_CEL_ACTIVITIES?"', r'ACTIVITY_TABLE\('],
        'OBJECT':   [r'LINK_PATH\(', r'LINK_FILTER\(', r'LINK_SOURCE\(', r'LINK_TARGET\('],
        'VENDOR':   [r'"VENDORS?"', r'"LFA1"'],
        'ORDER':    [r'"ORDERS?"', r'"EKKO"', r'"VBAK"'],
    }
    for level, patterns in TABLE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, q):
                ctx.table_levels.add(level)
                break

    # ── SQL keyword detection ─────────────────────────────────────────────────
    SQL_KWS = ['SELECT', 'FROM', 'JOIN', r'LEFT\s+JOIN', r'RIGHT\s+JOIN',
               r'INNER\s+JOIN', r'GROUP\s+BY', 'HAVING', r'\bWITH\b', r'OVER\s*\(']
    ctx.has_sql_keywords = any(re.search(r'\b' + kw + r'\b', q) for kw in SQL_KWS)

    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3B · VALIDATION ISSUE DATACLASS  ← NEW in v2
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    severity: str      # CRITICAL | ERROR | WARNING | INFO | PERF
    code: str          # machine-readable rule ID
    message: str       # human-readable short message
    why: str           # root cause explanation
    fix: str           # concrete fix suggestion
    auto_fixable: bool = False
    safe_fix: Optional[str] = None  # optional auto-fix string replacement hint


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3C · AST PARSER  (unchanged from v1 — battle-tested)
# ─────────────────────────────────────────────────────────────────────────────

def parse_pql(query: str) -> dict:
    stack = []
    current = {"type": "ROOT", "name": "", "args": [], "children": []}
    token = ""
    in_string = False
    string_char = None

    for char in query:
        if char in ('"', "'") and not in_string:
            in_string = True; string_char = char; token += char; continue
        elif in_string and char == string_char:
            in_string = False; token += char; continue
        if in_string:
            token += char; continue

        if char == '(':
            fn_name = token.strip()
            node = {"type": "FUNCTION", "name": fn_name.upper() if fn_name else "ANON", "args": [], "children": []}
            stack.append(current); current = node; token = ""
        elif char == ')':
            if token.strip(): current["args"].append(token.strip())
            parent = stack.pop() if stack else {"type": "ROOT", "name": "", "args": [], "children": []}
            parent["children"].append(current); current = parent; token = ""
        elif char == ',':
            if token.strip(): current["args"].append(token.strip()); token = ""
        else:
            token += char

    if token.strip() and current["type"] != "ROOT":
        current["args"].append(token.strip())
    return current


def ast_find_functions(node: dict, name_filter=None) -> list:
    results = []
    if node.get("type") == "FUNCTION":
        fn = node.get("name", "")
        if name_filter is None or fn.startswith(name_filter) or fn == name_filter:
            results.append(node)
    for child in node.get("children", []):
        results.extend(ast_find_functions(child, name_filter))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3D · CONTEXT-AWARE RELATIONSHIP RULE ENGINE  ← NEW in v2
# ─────────────────────────────────────────────────────────────────────────────

def rule_sql_keywords(ctx: PQLContext, query: str) -> List[ValidationIssue]:
    """SQL keywords are illegal in PQL."""
    if not ctx.has_sql_keywords:
        return []
    SQL_KWS = ['SELECT', 'FROM', 'JOIN', 'LEFT JOIN', 'RIGHT JOIN',
                'INNER JOIN', 'GROUP BY', 'HAVING', 'WITH', 'OVER(']
    found = [kw for kw in SQL_KWS if re.search(r'\b' + re.escape(kw.replace(' ', r'\s+')), query, re.I)]
    return [ValidationIssue(
        severity="CRITICAL", code="SQL001",
        message=f"SQL keyword(s) detected: {', '.join(found)}",
        why="PQL is a column-based language — there is no SELECT/FROM/JOIN/GROUP BY. "
            "SQL keywords will cause a parse error in Celonis Studio.",
        fix="Remove all SQL. Use PU_* functions to aggregate across tables, "
            "FILTER for row-level filtering, and GLOBAL() for cross-level aggregation.",
        auto_fixable=False
    )]


def rule_filter_inside_pu_text(ctx: PQLContext, query: str) -> List[ValidationIssue]:
    """FILTER keyword as inline text inside PU function args."""
    if not ctx.has_pu:
        return []
    issues = []
    # Look for PU_FUNC( ... FILTER ... ) pattern in raw text
    pattern = re.compile(r'(PU_\w+)\s*\([^)]*\bFILTER\b[^)]*\)', re.IGNORECASE | re.DOTALL)
    for m in pattern.finditer(query):
        issues.append(ValidationIssue(
            severity="CRITICAL", code="PU001",
            message=f"FILTER keyword inside {m.group(1).upper()}()",
            why="PU functions execute on the raw data graph before global filters apply. "
                "Embedding FILTER inside the parentheses is a syntax error — PQL will reject it.",
            fix=f"Use the 3rd positional argument as the filter condition instead:\n"
                f"  {m.group(1).upper()}(target_table, source_col, your_condition_here)",
            auto_fixable=False
        ))
    return issues


def rule_filter_to_null_inside_pu(ctx: PQLContext, ast: dict) -> List[ValidationIssue]:
    """FILTER_TO_NULL nested inside PU functions — wrong pattern."""
    if not ctx.has_pu:
        return []
    issues = []
    pu_nodes = ast_find_functions(ast, "PU_")
    for pu in pu_nodes:
        for child in pu.get("children", []):
            if child.get("name", "").upper() == "FILTER_TO_NULL":
                issues.append(ValidationIssue(
                    severity="CRITICAL", code="PU002",
                    message=f"FILTER_TO_NULL inside {pu['name']}()",
                    why="FILTER_TO_NULL makes columns filter-aware at query result level. "
                        "Inside PU functions it runs at the wrong scope and produces incorrect results.",
                    fix=f"Replace FILTER_TO_NULL(col) with a direct boolean condition as the 3rd arg:\n"
                        f"  {pu['name']}(target, source_col, condition_expression)",
                    auto_fixable=False
                ))
    return issues


def rule_pu_arg_count(ctx: PQLContext, ast: dict) -> List[ValidationIssue]:
    """PU functions must have at least 2 arguments."""
    if not ctx.has_pu:
        return []
    issues = []
    pu_nodes = ast_find_functions(ast, "PU_")
    for pu in pu_nodes:
        total = len(pu.get("args", [])) + len(pu.get("children", []))
        if total < 2:
            issues.append(ValidationIssue(
                severity="ERROR", code="PU003",
                message=f"{pu['name']}() has fewer than 2 arguments",
                why="PU functions always require at minimum: target_table (parent/1-side) "
                    "and source_table.column (child/N-side). Missing either causes a runtime error.",
                fix=f"Syntax: {pu['name']}(\"TARGET_TABLE\", \"SOURCE_TABLE\".\"COLUMN\" [, filter])",
                auto_fixable=False
            ))
    return issues


def rule_missing_global_mixed_levels(ctx: PQLContext) -> List[ValidationIssue]:
    """
    Missing GLOBAL() when CASE and ACTIVITY table columns are used together.
    This is the most common cause of multiplication errors in Celonis.
    """
    mixing_case_and_activity = (
        'CASE' in ctx.table_levels and 'ACTIVITY' in ctx.table_levels
    )
    if not mixing_case_and_activity:
        return []
    if ctx.has_global:
        return []  # already handled

    has_case_agg = bool(ctx.standard_agg_fns) and not ctx.has_pu
    has_any_agg = bool(ctx.aggregation_fns)

    if has_any_agg or (ctx.has_pu and ctx.standard_agg_fns):
        return [ValidationIssue(
            severity="WARNING", code="GL001",
            message="Missing GLOBAL() — mixing CASE and ACTIVITY level columns",
            why="When columns from both the case table and activity table appear in the same query, "
                "Celonis shifts the common table to the activity level. This multiplies case-level "
                "aggregations by the number of activities per case, inflating results.",
            fix="Wrap case-level aggregations with GLOBAL():\n"
                "  GLOBAL(COUNT(\"CASES\".\"CASE_ID\")) / COUNT(\"ACTIVITIES\".\"ACTIVITY\")\n"
                "  GLOBAL(AVG(CALC_THROUGHPUT(...)))",
            auto_fixable=False
        )]
    return []


def rule_calc_throughput_needs_global(ctx: PQLContext, query: str) -> List[ValidationIssue]:
    """CALC_THROUGHPUT combined with standard aggregations needs GLOBAL()."""
    if not ctx.has_calc_throughput:
        return []
    if ctx.has_global:
        return []
    if not ctx.aggregation_fns:
        return []
    return [ValidationIssue(
        severity="WARNING", code="GL002",
        message="CALC_THROUGHPUT + aggregation likely needs GLOBAL()",
        why="CALC_THROUGHPUT returns a value per case. When wrapped with AVG/SUM/COUNT "
            "alongside activity-level columns, the common table shifts to activity level, "
            "causing the throughput to be re-evaluated multiple times per case.",
        fix="Wrap the aggregation with GLOBAL():\n"
            "  GLOBAL(AVG(CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS(..., DAYS))))",
        auto_fixable=False
    )]


def rule_global_inside_filter(ctx: PQLContext, query: str) -> List[ValidationIssue]:
    """GLOBAL() is not allowed inside FILTER statements."""
    if not ctx.has_global:
        return []
    # Check for FILTER ... GLOBAL pattern
    if re.search(r'\bFILTER\b.*?GLOBAL\s*\(', query, re.IGNORECASE | re.DOTALL):
        return [ValidationIssue(
            severity="CRITICAL", code="GL003",
            message="GLOBAL() used inside a FILTER statement",
            why="GLOBAL() resolves to a single value used for comparison, not a row-level filter. "
                "Celonis does not support GLOBAL() inside FILTER clauses — it will throw a parse error.",
            fix="Move the GLOBAL() expression outside FILTER. Use CASE WHEN instead:\n"
                "  CASE WHEN aggregation > GLOBAL(aggregation) THEN ... END",
            auto_fixable=False
        )]
    return []


def rule_outer_pu_wrapping_inner_pu_datediff(ctx: PQLContext, query: str) -> List[ValidationIssue]:
    """
    Outer PU wrapping DATEDIFF of inner PU with same target = double-aggregation bug.
    This is the most dangerous pattern — silently returns wrong results.
    """
    pattern = re.compile(
        r'(PU_\w+)\s*\(\s*("[\w\s]+")\s*,\s*(?:DATEDIFF|HOURS_BETWEEN|SECONDS_BETWEEN|MINUTES_BETWEEN)',
        re.IGNORECASE
    )
    issues = []
    for m in pattern.finditer(query):
        outer_fn = m.group(1).upper()
        outer_table = m.group(2)
        inner_pattern = re.compile(
            r'PU_(?:FIRST|LAST|MIN|MAX|AVG)\s*\(\s*' + re.escape(outer_table),
            re.IGNORECASE
        )
        if inner_pattern.search(query):
            issues.append(ValidationIssue(
                severity="CRITICAL", code="PU004",
                message=f"Double-PU aggregation: outer {outer_fn} wraps DATEDIFF of inner PU on same table {outer_table}",
                why="PU_FIRST/PU_LAST already return a scalar value at the case level. "
                    "Wrapping them in another PU on the same target table applies aggregation twice, "
                    "which produces mathematically incorrect results without any runtime error.",
                fix="Remove the outer PU function. Use DATEDIFF directly:\n"
                    f"  DATEDIFF('dd', PU_FIRST({outer_table}, ...), PU_LAST({outer_table}, ...))",
                auto_fixable=False
            ))
    return issues


def rule_running_sum_deprecated(ctx: PQLContext, query: str) -> List[ValidationIssue]:
    if 'RUNNING_SUM' not in ctx.used_functions:
        return []
    return [ValidationIssue(
        severity="INFO", code="DEP001",
        message="RUNNING_SUM is deprecated",
        why="RUNNING_SUM was an older alias. The official function is now RUNNING_TOTAL.",
        fix="Replace RUNNING_SUM(...) with RUNNING_TOTAL(...) — syntax is identical.",
        auto_fixable=True,
        safe_fix="Replace RUNNING_SUM with RUNNING_TOTAL"
    )]


def rule_process_order_deprecated(ctx: PQLContext, query: str) -> List[ValidationIssue]:
    if 'PROCESS_ORDER' not in ctx.used_functions:
        return []
    return [ValidationIssue(
        severity="INFO", code="DEP002",
        message="PROCESS_ORDER is deprecated",
        why="PROCESS_ORDER was removed in newer PQL versions.",
        fix="Replace PROCESS_ORDER(...) with INDEX_ACTIVITY_ORDER(...) — same behavior.",
        auto_fixable=True,
        safe_fix="Replace PROCESS_ORDER with INDEX_ACTIVITY_ORDER"
    )]


def rule_all_occurrence_deprecated(query: str) -> List[ValidationIssue]:
    if not re.search(r'ALL_OCCURRENCE\s*\[', query, re.IGNORECASE):
        return []
    return [ValidationIssue(
        severity="WARNING", code="DEP003",
        message="ALL_OCCURRENCE['…'] is deprecated since PQL 4.6",
        why="ALL_OCCURRENCE was removed in PQL 4.6. Using it causes a parse error in newer Celonis versions.",
        fix="Replace ALL_OCCURRENCE['…'] with CASE_START (for the beginning specifier) "
            "or CASE_END (for the ending specifier).",
        auto_fixable=False
    )]


def rule_pu_count_distinct_on_likely_key(ctx: PQLContext, query: str) -> List[ValidationIssue]:
    """
    PU_COUNT_DISTINCT on an ID/KEY column — PU_COUNT is cheaper.
    Heuristic: column name ends with _ID, _KEY, _NO, _NUM, ID, KEY.
    """
    if 'PU_COUNT_DISTINCT' not in ctx.used_functions:
        return []
    pattern = re.compile(
        r'PU_COUNT_DISTINCT\s*\([^,]+,\s*"[^"]+"\."([^"]+)"',
        re.IGNORECASE
    )
    issues = []
    for m in pattern.finditer(query):
        col = m.group(1).upper()
        if re.search(r'(_ID|_KEY|_NO|_NUM|CASE_ID|ORDER_ID|VENDOR_ID)$', col):
            issues.append(ValidationIssue(
                severity="PERF", code="PERF001",
                message=f"PU_COUNT_DISTINCT on key column \"{col}\" — use PU_COUNT instead",
                why=f"Column \"{col}\" looks like a primary key. PU_COUNT_DISTINCT performs "
                    "an expensive sort+dedup operation. Since keys are already unique, "
                    "PU_COUNT gives the same result at a fraction of the cost.",
                fix=f"Replace PU_COUNT_DISTINCT with PU_COUNT on this column.",
                auto_fixable=True,
                safe_fix="Replace PU_COUNT_DISTINCT with PU_COUNT for key columns"
            ))
    return issues


def rule_pu_median_performance(ctx: PQLContext) -> List[ValidationIssue]:
    """PU_MEDIAN is expensive — flag when PU_AVG would likely suffice."""
    if 'PU_MEDIAN' not in ctx.used_functions:
        return []
    return [ValidationIssue(
        severity="PERF", code="PERF002",
        message="PU_MEDIAN detected — consider PU_AVG for better performance",
        why="PU_MEDIAN requires a full sort of the source data per target row, "
            "making it 5–10x more expensive than PU_AVG. Unless the true statistical median "
            "is required (skewed distributions), PU_AVG gives a sufficient approximation.",
        fix="Replace PU_MEDIAN with PU_AVG unless the true median is statistically required.",
        auto_fixable=False
    )]


def rule_pu_first_last_no_order_by(ctx: PQLContext, query: str) -> List[ValidationIssue]:
    """PU_FIRST or PU_LAST without ORDER BY — non-deterministic results."""
    issues = []
    for fn in ['PU_FIRST', 'PU_LAST']:
        if fn not in ctx.used_functions:
            continue
        pattern = re.compile(fn + r'\s*\([^)]*\)', re.IGNORECASE | re.DOTALL)
        for m in pattern.finditer(query):
            if 'ORDER BY' not in m.group(0).upper():
                issues.append(ValidationIssue(
                    severity="WARNING", code="PU005",
                    message=f"{fn}() without ORDER BY — non-deterministic result",
                    why=f"Without ORDER BY, {fn}() depends on physical row order, which is "
                        "not guaranteed to be consistent across query runs or data updates.",
                    fix=f"Add ORDER BY to ensure deterministic output:\n"
                        f"  {fn}(\"CASES\", \"ACTIVITIES\".\"TIMESTAMP\", ORDER BY \"ACTIVITIES\".\"TIMESTAMP\" ASC)",
                    auto_fixable=False
                ))
    return issues


def rule_quotes_enforcement(ctx: PQLContext, query: str) -> List[ValidationIssue]:
    """Detect potentially unquoted table.column identifiers."""
    if ctx.is_match_safe:
        return []  # MATCH_* queries use single-quote syntax — skip
    unquoted = re.findall(r'(?<!")\b([A-Z][A-Z0-9_]{2,})\.([A-Z][A-Z0-9_]{2,})\b(?!")', query)
    if not unquoted:
        return []
    examples = [f'{t}.{c}' for t, c in unquoted[:3]]
    return [ValidationIssue(
        severity="ERROR", code="SYN001",
        message=f"Possibly unquoted identifiers: {examples}",
        why="Celonis requires all table and column names to be double-quoted. "
            "Unquoted identifiers will cause a parse error.",
        fix='Use double-quote syntax: "TABLE_NAME"."COLUMN_NAME"',
        auto_fixable=False
    )]


def rule_match_query_protection(ctx: PQLContext) -> List[ValidationIssue]:
    """Flag MATCH_* queries so validation skips irrelevant checks."""
    if not ctx.is_match_safe:
        return []
    return [ValidationIssue(
        severity="INFO", code="INFO001",
        message="MATCH_ACTIVITIES / MATCH_PROCESS / conformance query detected",
        why="These functions use a different quoting syntax (single-quoted activity names in brackets). "
            "Quote enforcement and some structural checks are skipped for this query type.",
        fix="No action needed — this is informational.",
        auto_fixable=False
    )]


def rule_sum_wrapping_filter(ctx: PQLContext, query: str) -> List[ValidationIssue]:
    """SUM(FILTER ...) is invalid — FILTER is a statement, not a function."""
    if re.search(r'\bSUM\s*\(\s*FILTER\b', query, re.IGNORECASE):
        return [ValidationIssue(
            severity="CRITICAL", code="SYN002",
            message="SUM(FILTER …) is invalid — FILTER is a statement, not a function",
            why="FILTER is a top-level statement that narrows the result set. "
                "It cannot be nested inside aggregation functions like SUM().",
            fix="Use FILTER_TO_NULL to make a column filter-aware inside SUM:\n"
                "  SUM(FILTER_TO_NULL(\"ORDERS\".\"AMOUNT\"))\n"
                "Or put FILTER at the top level: FILTER \"CASES\".\"STATUS\" = 'Open'",
            auto_fixable=False
        )]
    return []


def rule_optimization_count_vs_count_distinct(ctx: PQLContext, query: str) -> List[ValidationIssue]:
    """Standard COUNT(DISTINCT …) when column appears to be a key."""
    if not re.search(r'\bCOUNT\s*\(\s*DISTINCT\b', query, re.IGNORECASE):
        return []
    pattern = re.compile(r'COUNT\s*\(\s*DISTINCT\s+"[^"]+"\."([^"]+)"', re.IGNORECASE)
    issues = []
    for m in pattern.finditer(query):
        col = m.group(1).upper()
        if re.search(r'(_ID|_KEY|_NO|_NUM|CASE_ID)$', col):
            issues.append(ValidationIssue(
                severity="PERF", code="PERF003",
                message=f"COUNT(DISTINCT …) on key column \"{col}\" — COUNT is cheaper",
                why="COUNT(DISTINCT) adds a deduplication step. If the column is already a unique key, "
                    "regular COUNT gives the same answer without the overhead.",
                fix=f"Replace COUNT(DISTINCT \"TABLE\".\"{col}\") with COUNT(\"TABLE\".\"{col}\")",
                auto_fixable=True,
                safe_fix="Replace COUNT(DISTINCT) with COUNT on key columns"
            ))
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3E · RULE ORCHESTRATOR  ← NEW in v2
# ─────────────────────────────────────────────────────────────────────────────

def run_context_rule_engine(query: str) -> List[ValidationIssue]:
    """
    Run all context-aware rules. Returns sorted list of ValidationIssues.
    Priority: CRITICAL → ERROR → WARNING → PERF → INFO
    """
    ctx = extract_context(query)
    ast = parse_pql(query)

    all_issues: List[ValidationIssue] = []

    # Structural / SQL
    all_issues += rule_sql_keywords(ctx, query)
    all_issues += rule_sum_wrapping_filter(ctx, query)
    all_issues += rule_quotes_enforcement(ctx, query)

    # PU function rules
    all_issues += rule_filter_inside_pu_text(ctx, query)
    all_issues += rule_filter_to_null_inside_pu(ctx, ast)
    all_issues += rule_pu_arg_count(ctx, ast)
    all_issues += rule_outer_pu_wrapping_inner_pu_datediff(ctx, query)
    all_issues += rule_pu_first_last_no_order_by(ctx, query)

    # GLOBAL rules
    all_issues += rule_missing_global_mixed_levels(ctx)
    all_issues += rule_calc_throughput_needs_global(ctx, query)
    all_issues += rule_global_inside_filter(ctx, query)

    # Deprecation rules
    all_issues += rule_running_sum_deprecated(ctx, query)
    all_issues += rule_process_order_deprecated(ctx, query)
    all_issues += rule_all_occurrence_deprecated(query)

    # Performance rules
    all_issues += rule_pu_count_distinct_on_likely_key(ctx, query)
    all_issues += rule_pu_median_performance(ctx)
    all_issues += rule_optimization_count_vs_count_distinct(ctx, query)

    # Info / protective
    all_issues += rule_match_query_protection(ctx)

    # Sort: CRITICAL first
    SEVERITY_ORDER = {'CRITICAL': 0, 'ERROR': 1, 'WARNING': 2, 'PERF': 3, 'INFO': 4}
    all_issues.sort(key=lambda i: SEVERITY_ORDER.get(i.severity, 9))

    return all_issues


def format_issues_for_display(issues: List[ValidationIssue]) -> List[str]:
    """Convert ValidationIssues to display strings (backwards-compatible with v1 format)."""
    ICONS = {'CRITICAL': '🚨', 'ERROR': '❌', 'WARNING': '⚠', 'PERF': '⚡', 'INFO': 'ℹ'}
    lines = []
    for issue in issues:
        icon = ICONS.get(issue.severity, '•')
        lines.append(f"{icon} [{issue.code}] {issue.message}")
    return lines


def build_why_explanation(issues: List[ValidationIssue]) -> str:
    """Build a WHY explanation block for the LLM context — explains each issue's root cause."""
    if not issues:
        return ""
    parts = []
    for issue in issues:
        parts.append(
            f"Rule {issue.code} ({issue.severity}): {issue.message}\n"
            f"  WHY: {issue.why}\n"
            f"  FIX: {issue.fix}"
        )
    return "\n\n".join(parts)


def apply_safe_auto_fixes(query: str, issues: List[ValidationIssue]) -> tuple:
    """
    Apply only safe, non-destructive auto-fixes.
    Returns (was_modified: bool, fixed_query: str, applied_fixes: list)
    """
    fixed = query
    applied = []

    for issue in issues:
        if not issue.auto_fixable or not issue.safe_fix:
            continue
        # RUNNING_SUM → RUNNING_TOTAL
        if issue.code == "DEP001":
            new = re.sub(r'\bRUNNING_SUM\b', 'RUNNING_TOTAL', fixed, flags=re.IGNORECASE)
            if new != fixed:
                fixed = new
                applied.append(f"DEP001: Replaced RUNNING_SUM → RUNNING_TOTAL")
        # PROCESS_ORDER → INDEX_ACTIVITY_ORDER
        elif issue.code == "DEP002":
            new = re.sub(r'\bPROCESS_ORDER\b', 'INDEX_ACTIVITY_ORDER', fixed, flags=re.IGNORECASE)
            if new != fixed:
                fixed = new
                applied.append(f"DEP002: Replaced PROCESS_ORDER → INDEX_ACTIVITY_ORDER")
        # PU_COUNT_DISTINCT → PU_COUNT for key columns
        elif issue.code == "PERF001":
            new = re.sub(r'\bPU_COUNT_DISTINCT\b', 'PU_COUNT', fixed, flags=re.IGNORECASE)
            if new != fixed:
                fixed = new
                applied.append(f"PERF001: Replaced PU_COUNT_DISTINCT → PU_COUNT on key column")

    return (fixed != query), fixed, applied


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 · GROQ MODELS
# ─────────────────────────────────────────────────────────────────────────────

GROQ_MODELS = {
    'llama-3.3-70b-versatile':  'LLaMA 3.3 70B — best quality',
    'llama-3.1-8b-instant':     'LLaMA 3.1 8B  — fastest',
    'mixtral-8x7b-32768':       'Mixtral 8x7B  — balanced',
    'gemma2-9b-it':             'Gemma 2 9B    — lightweight',
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 · SYSTEM PROMPT BUILDER  (upgraded with context injection)
# ─────────────────────────────────────────────────────────────────────────────

_FUNCTION_SELECTION_GUIDE = """
## ─── OFFICIAL CELONIS FUNCTION SELECTION GUIDE ───

### THROUGHPUT TIME

| Goal | Correct function | WRONG approach |
|------|-----------------|----------------|
| Throughput per CASE (start→end) | CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS(..., DAYS)) | PU_MAX - PU_MIN per case |
| Throughput per CASE (act→act) | CALC_THROUGHPUT(FIRST_OCCURRENCE['A'] TO LAST_OCCURRENCE['B'], ...) | DATEDIFF on activity table |
| Throughput OVER MULTIPLE CASES | DATEDIFF('dd', PU_MIN("GROUP","ACTIVITIES"."TIMESTAMP"), PU_MAX("GROUP","ACTIVITIES"."TIMESTAMP")) | CALC_THROUGHPUT |
| Cycle time first→last event per case | DATEDIFF('dd', PU_FIRST("CASES","ACTIVITIES"."TIMESTAMP", ORDER BY ...), PU_LAST("CASES","ACTIVITIES"."TIMESTAMP", ORDER BY ...)) | PU_AVG wrapping DATEDIFF |
| Average throughput across all cases | AVG(CALC_THROUGHPUT(...)) | PU_AVG wrapping CALC_THROUGHPUT |
| Working-hours throughput | CALC_THROUGHPUT(..., REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", HOURS, WEEKDAY_CALENDAR(MON,TUE,WED,THU,FRI))) | Plain CALC_THROUGHPUT |

### REWORK / REPEATED ACTIVITIES

| Goal | Correct function |
|------|-----------------|
| Count all activities per case | CALC_REWORK() |
| Count specific activities per case | CALC_REWORK("ACTIVITIES"."ACTIVITY" = 'Review') |
| Detect repeated activities (row-level) | INDEX_ACTIVITY_LOOP("ACTIVITIES"."ACTIVITY") > 0 |
| Loop counter for specific activity type | INDEX_ACTIVITY_TYPE("ACTIVITIES"."ACTIVITY") |

### AGGREGATION SELECTION

| Goal | Correct function | Avoid |
|------|-----------------|-------|
| Count rows (key column) | PU_COUNT | PU_COUNT_DISTINCT (much slower) |
| Average values | PU_AVG | PU_MEDIAN (requires full sort) |
| First/Last value | PU_FIRST / PU_LAST with ORDER BY | Without ORDER BY (non-deterministic) |
| Running total | RUNNING_TOTAL | RUNNING_SUM (deprecated) |
| Position in case | INDEX_ACTIVITY_ORDER | PROCESS_ORDER (deprecated) |

### NULL BEHAVIOUR
| Function | No matching rows |
|----------|-----------------|
| PU_COUNT, PU_COUNT_DISTINCT | 0 |
| PU_SUM, PU_AVG, PU_MIN, PU_MAX, PU_FIRST, PU_LAST | NULL |
| CALC_THROUGHPUT | NULL if single activity or end before start |

### GLOBAL() — WHEN TO USE
Use GLOBAL() whenever you mix columns from different table levels in the same query:
- Case-level column + Activity-level column → GLOBAL wraps the case-level aggregation
- CALC_THROUGHPUT + AVG/SUM → GLOBAL(AVG(CALC_THROUGHPUT(...)))
- Percent of total → SUM("ORDERS"."AMOUNT") / GLOBAL(SUM("ORDERS"."AMOUNT"))
"""

_SQL_PROHIBITION = """
## CRITICAL — PQL IS NOT SQL. NEVER WRITE SQL.

NO: SELECT   FROM    JOIN    LEFT JOIN   GROUP BY   HAVING   WITH   OVER(...)   AS (CTE alias)

### WRONG — SQL:
```sql
SELECT "LFA1"."LIFNR", AVG(DATEDIFF(dd, "EKKO"."BEDAT", "EKPO"."LGDAT")) AS LEAD_TIME
FROM "EKKO" JOIN "EKPO" ON "EKKO"."EBELN" = "EKPO"."EBELN"
GROUP BY "LFA1"."LIFNR"
```

### CORRECT — real PQL:
```pql
PU_AVG("LFA1", DATEDIFF('dd', "EKKO"."BEDAT", "EKPO"."LGDAT"))
```
"""

_ADVANCED_PATTERNS = """
## ─── Advanced PQL Patterns (Official) ───

### P1 · GLOBAL() — prevents join multiplication
```pql
-- Average throughput (safe with GLOBAL):
GLOBAL( AVG( CALC_THROUGHPUT( CASE_START TO CASE_END, REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", DAYS) ) ) )

-- Above vs. below average pattern:
CASE WHEN AVG("ORDERS"."AMOUNT") > GLOBAL(AVG("ORDERS"."AMOUNT")) THEN 'Above Avg' ELSE 'Below Avg' END

-- Percent of total:
SUM("ORDERS"."AMOUNT") / GLOBAL(SUM("ORDERS"."AMOUNT")) * 100
```

### P2 · Automation Rate (% of system activities)
```pql
ROUND(
  PU_COUNT("CASES", "ACTIVITIES"."CASE_ID", "ACTIVITIES"."USER" = 'SYSTEM') * 100.0
  / CALC_REWORK(),
  1
)
```

### P3 · Working-hours throughput
```pql
AVG(CALC_THROUGHPUT(
  CASE_START TO CASE_END,
  REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", HOURS, WEEKDAY_CALENDAR(MON,TUE,WED,THU,FRI))
)) / 8
```

### P4 · Rework detection — cases with repeated activity
```pql
CASE WHEN PU_COUNT("CASES", "ACTIVITIES"."CASE_ID", "ACTIVITIES"."ACTIVITY" = 'Review') > 1
     THEN 'Rework' ELSE 'Clean' END

CASE WHEN INDEX_ACTIVITY_LOOP("ACTIVITIES"."ACTIVITY") > 0 THEN 'Rework' ELSE 'First' END
```

### P5 · Cycle time: first to last event per case
```pql
DATEDIFF('dd',
  PU_FIRST("CASES", "ACTIVITIES"."TIMESTAMP", ORDER BY "ACTIVITIES"."TIMESTAMP" ASC),
  PU_LAST("CASES", "ACTIVITIES"."TIMESTAMP", ORDER BY "ACTIVITIES"."TIMESTAMP" ASC)
)
```

### P6 · Late deliveries (SLA breach)
```pql
PU_COUNT("VENDORS", "ORDERS"."ORDER_ID",
  DATEDIFF('dd', "ORDERS"."PROMISED_DATE", "ORDERS"."ACTUAL_DATE") > 7
)
```

### P7 · Transition time (between consecutive activities)
```pql
SECONDS_BETWEEN(ACTIVITY_LAG("ACTIVITIES"."TIMESTAMP"), "ACTIVITIES"."TIMESTAMP") / 3600
AVG(CASE WHEN "ACTIVITIES"."ACTIVITY" = 'Approve'
         THEN SECONDS_BETWEEN(ACTIVITY_LAG("ACTIVITIES"."TIMESTAMP"), "ACTIVITIES"."TIMESTAMP") / 3600
         ELSE NULL END)
```

### P8 · Conforming throughput (only clean cases)
```pql
AVG(CASE WHEN PU_SUM("CASES", ABS("CONFORMANCE_COL")) = 0
    THEN CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", HOURS)) / 24
    ELSE NULL END)
```

### P9 · Edge KPI in Process Explorer
```pql
PU_MAX("_CEL_CASES",
  SECONDS_BETWEEN(TARGET("_CEL_ACTIVITIES"."EVENTTIME"), SOURCE("_CEL_ACTIVITIES"."EVENTTIME"))
)
```

### P10 · Running total with monthly partitions
```pql
RUNNING_TOTAL(
  "ORDERS"."AMOUNT",
  ORDER BY ("ORDERS"."ORDER_DATE" ASC),
  PARTITION BY (ROUND_MONTH("ORDERS"."ORDER_DATE"))
)
```

### P11 · Z-score outlier detection
```pql
CASE WHEN ZSCORE("ORDERS"."AMOUNT", PARTITION BY ("ORDERS"."VENDOR")) > 3
     THEN 'Outlier' ELSE 'Normal' END
```

### P12 · OCPM — Cross-object throughput
```pql
AVG(CALC_THROUGHPUT(
  CASE_START TO CASE_END,
  REMAP_TIMESTAMPS(TIMESTAMP_COLUMN(ACTIVITY_TABLE(LINK_PATH("ORDERS"."ORDER_ID"))), DAYS)
))
```
"""

_EXPERT_FRAMEWORK = """
## ─── Expert Query Construction Framework ───

**Step 1** — Identify tables & relationships. Which is parent (1-side)? Which is child (N-side)?
**Step 2** — Identify result level. Case-level? Activity-level? Mixing both → GLOBAL() required.
**Step 3** — Choose aggregation strategy. PU for cross-table; Standard for same-table.
**Step 4** — Handle filters. FILTER for simple; PU filter_expression for PU functions; BIND_FILTERS for non-common tables.
**Step 5** — Build KPIs innermost first, wrap with GLOBAL() at table-level boundaries.
**Step 6** — Performance: PU_COUNT vs PU_COUNT_DISTINCT, AVG vs MEDIAN, RUNNING_TOTAL vs RUNNING_SUM.
**Step 7** — NULL safety: COALESCE, ISNULL, check what each function returns on no match.

## Anti-patterns — always avoid
1. Missing GLOBAL() when mixing case + activity columns in same query
2. FILTER or FILTER_TO_NULL inside PU-functions → use filter_expression parameter
3. PU_COUNT_DISTINCT on a key/ID column → use cheaper PU_COUNT
4. PU_MEDIAN or MEDIAN when AVG is sufficient (much more expensive)
5. Missing double-quotes on table/column names: "TABLE"."COLUMN" is mandatory
6. Any SQL syntax (SELECT/FROM/JOIN/GROUP BY/HAVING) — PQL is NOT SQL
7. Outer PU wrapping DATEDIFF of inner PU with same target table → wrong nesting
8. RUNNING_SUM (deprecated) → use RUNNING_TOTAL
9. PROCESS_ORDER (deprecated) → use INDEX_ACTIVITY_ORDER
10. ALL_OCCURRENCE['…'] (deprecated since 4.6) → use CASE_START
11. PU_FIRST / PU_LAST without ORDER BY → non-deterministic results
12. Wrapping CALC_THROUGHPUT in GLOBAL inside FILTER (GLOBAL not allowed in FILTER)
"""

def build_system_prompt(complexity: str, show_reasoning: bool) -> str:
    ALWAYS_INCLUDE = [
        'GLOBAL', 'CALC_THROUGHPUT', 'PU_COUNT', 'PU_SUM', 'PU_AVG',
        'PU_FIRST', 'PU_LAST', 'FILTER', 'DATEDIFF', 'REMAP_TIMESTAMPS',
        'CALC_REWORK', 'MATCH_ACTIVITIES', 'RUNNING_TOTAL', 'INDEX_ACTIVITY_LOOP',
    ]
    core_refs = "\n\n".join(
        f"### {fn}\n{COMPACT_REFS[fn]}"
        for fn in ALWAYS_INCLUDE if fn in COMPACT_REFS
    )

    base = f"""You are a world-class Celonis PQL (Process Query Language) engineer — the equivalent of a 
senior Celonis consultant with 10 years of production experience. You know ALL 250+ official PQL functions,
their edge cases, performance implications, and correct usage patterns.

Write ACCURATE, OPTIMIZED, PRODUCTION-READY PQL queries that will work directly in Celonis Studio.

## PQL Absolute Rules (NEVER violate these)
1. Tables and columns MUST be double-quoted: "TABLE"."COLUMN"
2. String literals MUST use single quotes: 'value'
3. PQL is column-based, not row-based — there is NO SELECT/FROM/JOIN
4. Multiple FILTER statements merge by logical AND
5. NULL: most functions skip NULLs; use COALESCE or ISNULL to handle explicitly
6. PU-functions: aggregate FROM child table (many/N-side) TO parent table (one/1-side)
7. FILTER cannot be inside PU functions — use filter_expression parameter instead
8. GLOBAL() is required when mixing columns from different table levels
9. RUNNING_TOTAL replaces deprecated RUNNING_SUM
10. INDEX_ACTIVITY_ORDER replaces deprecated PROCESS_ORDER

{_SQL_PROHIBITION}

{_FUNCTION_SELECTION_GUIDE}

## Core PQL Functions Reference
{core_refs}
"""

    if complexity in ("Advanced", "Expert"):
        base += _ADVANCED_PATTERNS

    if complexity == "Expert":
        base += _EXPERT_FRAMEWORK

    if show_reasoning and complexity in ("Advanced", "Expert"):
        base += """
## Response Format
1. **Analysis** — identify tables, joins, result level, function selection rationale
2. **Query** — complete PQL in a ```pql code block with inline comments
3. **Explanation** — explain each function and why it was chosen
4. **Performance notes** — highlight any optimization choices or warnings
5. **Edge cases** — NULL handling, filter propagation, GLOBAL() requirement, deprecation warnings
"""
    elif complexity == "Intermediate":
        base += """
## Response Format
1. PQL in a ```pql code block
2. Explain each function used and why it was chosen
3. Mention key gotchas: NULL handling, GLOBAL(), filter awareness, PU direction
"""
    else:
        base += """
## Response Format
1. PQL in a ```pql code block
2. Short plain-English explanation (2-4 sentences)
"""

    instructions = {
        "Basic":        "Simple 1-2 function queries. Focus on correctness.\n",
        "Intermediate": "Queries with 2–5 functions. Use filters, CASE WHEN, simple aggregations.\n",
        "Advanced":     "Nested PU-functions, GLOBAL(), throughput patterns, multi-table KPIs. Explain GLOBAL() need.\n",
        "Expert":       "Production-ready multi-KPI queries. BPMN conformance. OCPM. ML. Full chain-of-thought planning.\n",
    }
    base += f"\n## Complexity: {complexity}\n{instructions[complexity]}\n"
    base += """
When table/column names are unknown, use standard placeholders:
"CASES"."CASE_ID", "ACTIVITIES"."ACTIVITY", "ACTIVITIES"."TIMESTAMP", "ACTIVITIES"."USER",
"ORDERS"."AMOUNT", "VENDORS"."VENDOR_ID", "ORDERS"."CREATE_DATE", "ORDERS"."CLOSE_DATE"
"""
    return base


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 · LLM VERIFICATION SYSTEM PROMPT  (upgraded with WHY context)
# ─────────────────────────────────────────────────────────────────────────────

VERIFICATION_SYSTEM = """You are a strict Celonis PQL validator and targeted auto-corrector.
Your ONLY job is to review PQL code and return a corrected version or confirm it's valid.

## Rules to enforce (priority order):
1. NO SQL: SELECT, FROM, JOIN, LEFT JOIN, GROUP BY, HAVING, WITH, AS (CTE), OVER(...) → REMOVE
2. ALL table/column identifiers must be double-quoted: "TABLE"."COLUMN"
3. String literals must use single quotes: 'value'
4. PU_FUNC( target_table, source_table.column [, filter] ) — always 2+ arguments
5. FILTER_TO_NULL inside PU functions → replace with PU filter_expression argument
6. GLOBAL() required when CALC_THROUGHPUT is combined with AVG/SUM/COUNT
7. PU_COUNT_DISTINCT on a key column → replace with PU_COUNT (more efficient)
8. RUNNING_SUM → replace with RUNNING_TOTAL (deprecated)
9. PROCESS_ORDER → replace with INDEX_ACTIVITY_ORDER (deprecated)
10. ALL_OCCURRENCE['...'] → replace with CASE_START or CASE_END (deprecated since 4.6)
11. CRITICAL: Outer PU wrapping DATEDIFF of inner PU with same target table is WRONG.
    FIX: Remove outer PU. Use DATEDIFF(PU_FIRST(...), PU_LAST(...)) directly.
12. PU direction: target_table must be the PARENT (1-side), source must be CHILD (N-side)
13. PU_FIRST / PU_LAST must always have ORDER BY for deterministic results
14. FILTER keyword must NOT appear inside PU function arguments
15. GLOBAL() must NOT appear inside FILTER statements

## IMPORTANT — safe correction only:
- Only fix what you are CERTAIN is wrong
- Do NOT restructure or rewrite entire queries unless absolutely necessary
- Do NOT add features that were not requested
- If unsure whether something is wrong, mark it as a warning rather than changing it

## Response format:
- If the query is correct: respond with exactly: VALID
- If errors exist: respond ONLY with the corrected ```pql block and a brief bullet list of changes made.
- Do NOT add explanations, preamble, or commentary beyond the fix list.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 · UI CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY_DESC = {
    'Basic':        'Simple 1-2 function queries. Great for beginners.',
    'Intermediate': 'Multi-function queries with filters & conditions.',
    'Advanced':     'Nested PU-functions, GLOBAL(), multi-table joins.',
    'Expert':       'Chain-of-thought · BPMN · OCPM · ML · Full optimization.',
}

EXAMPLE_PROMPTS = {
    'Basic': [
        'Count activities per case',
        "Filter cases where status = 'Open'",
        'Convert vendor name to uppercase',
        'Difference in days between two date columns',
        'Get the most common activity (mode)',
    ],
    'Intermediate': [
        'Average invoice amount per vendor',
        'Find cases where Approve happens before Pay',
        'Throughput time per case in days',
        'Running total of PO values by month',
        'Detect cases with more than 2 Review activities',
    ],
    'Advanced': [
        'Count late deliveries per vendor — delivery > promised by 7 days',
        'Rework rate: Review activity repeating more than 2 times per case',
        'Automation rate: % of system activities per case',
        'Flag non-conforming cases and show their throughput',
        'Z-score outlier detection on invoice amounts per vendor',
    ],
    'Expert': [
        'Full KPI: throughput + rework count + automation rate in one query',
        'Multi-level nesting: avg approval time aggregated vendor → order → line item',
        'BPMN conformance check tolerating undesired but not missing activities',
        'OCPM: throughput across linked objects with workday calendar',
        'Working-hours SLA breach detection with conformance scoring',
        'Variant-level rework analysis: first vs. repeated occurrence per activity type',
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 · PAGE CONFIG + CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='PQL Query Assistant',
    page_icon='⚡',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');
*,*::before,*::after{box-sizing:border-box;}
:root{
  --bg-base:#07090f;--bg-surface:#0d1018;--bg-elevated:#131720;--bg-hover:#1a2030;
  --border:#1f2738;--border-bright:#2a3550;
  --accent:#3b82f6;--accent-dim:#1e3a5f;--accent-glow:rgba(59,130,246,0.15);
  --amber:#f59e0b;--amber-dim:#451a03;--green:#10b981;--green-dim:#052e16;
  --red:#ef4444;--red-dim:#450a0a;--violet:#8b5cf6;--perf:#06b6d4;--perf-dim:#0c2340;
  --text-primary:#e8edf5;--text-secondary:#8899b0;--text-muted:#4a5568;
  --font-mono:'IBM Plex Mono',monospace;--font-ui:'Syne',sans-serif;--font-body:'Inter',sans-serif;
  --radius-sm:6px;--radius-md:10px;--radius-lg:16px;
}
.stApp{background:var(--bg-base)!important;font-family:var(--font-body);}
.main .block-container{background:var(--bg-base)!important;padding-top:2rem!important;max-width:920px!important;}
header[data-testid="stHeader"]{background:var(--bg-base)!important;border-bottom:1px solid var(--border)!important;}
[data-testid="stToolbar"]{background:var(--bg-base)!important;}
h1,h2,h3{font-family:var(--font-ui)!important;color:var(--text-primary)!important;letter-spacing:-0.02em;}
h1{font-size:1.75rem!important;font-weight:800!important;}
h2{font-size:1.25rem!important;font-weight:700!important;}
h3{font-size:1rem!important;font-weight:600!important;}
[data-testid="stHeadingWithActionElements"] h1,
[data-testid="stHeadingWithActionElements"] h2,
[data-testid="stHeadingWithActionElements"] h3{color:var(--text-primary)!important;}
div[data-testid="stMarkdownContainer"] p{color:var(--text-secondary)!important;font-size:14px;line-height:1.6;}
[data-testid="stCaptionContainer"] p,.stCaption,.stCaption p{color:var(--text-muted)!important;font-size:12px!important;font-family:var(--font-mono)!important;}
h1 a,h2 a,h3 a,[data-testid="stHeadingWithActionElements"] a,
[data-testid="stHeadingWithActionElements"] button,[data-testid="stHeadingWithActionElements"] svg{display:none!important;}
[data-testid="stSidebar"]{background:var(--bg-surface)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p,[data-testid="stSidebar"] span{color:var(--text-secondary)!important;font-size:13px;}
[data-testid="stSidebar"] input,[data-testid="stSidebar"] select{background:var(--bg-elevated)!important;border:1px solid var(--border-bright)!important;color:var(--text-primary)!important;border-radius:var(--radius-sm)!important;font-family:var(--font-mono)!important;font-size:12px!important;}
[data-testid="stSidebar"] hr{border-color:var(--border)!important;}
[data-testid="stChatMessage"]{background:var(--bg-surface)!important;border:1px solid var(--border)!important;border-radius:var(--radius-lg)!important;margin-bottom:12px!important;transition:border-color 0.2s ease;}
[data-testid="stChatMessage"]:hover{border-color:var(--border-bright)!important;}
[data-testid="stChatMessageContent"],[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,[data-testid="stChatMessageContent"] span{color:var(--text-primary)!important;font-size:14px!important;line-height:1.7!important;font-family:var(--font-body)!important;}
[data-testid="stChatMessageContent"] strong{color:#f0f4ff!important;font-weight:600;}
[data-testid="stChatMessageContent"] code{background:var(--bg-elevated)!important;color:#93c5fd!important;font-family:var(--font-mono)!important;font-size:12px!important;padding:2px 6px!important;border-radius:4px!important;border:1px solid var(--border-bright)!important;}
pre{background:#040810!important;border:1px solid var(--border-bright)!important;border-radius:var(--radius-md)!important;padding:18px!important;overflow-x:auto!important;position:relative;}
pre::before{content:'PQL';position:absolute;top:10px;right:14px;font-family:var(--font-mono);font-size:10px;font-weight:600;color:var(--text-muted);letter-spacing:0.1em;text-transform:uppercase;}
pre code{background:transparent!important;border:none!important;color:#e2e8f0!important;font-family:var(--font-mono)!important;font-size:13px!important;line-height:1.6!important;padding:0!important;}
[data-testid="stBottom"]{background:linear-gradient(to top,var(--bg-base) 70%,transparent)!important;border-top:none!important;padding:16px 0!important;}
[data-testid="stBottom"]>div{background:transparent!important;}
[data-testid="stChatInput"]{background:var(--bg-elevated)!important;border:1px solid var(--border-bright)!important;border-radius:var(--radius-lg)!important;transition:border-color 0.2s,box-shadow 0.2s;}
[data-testid="stChatInput"]:focus-within{border-color:var(--accent)!important;box-shadow:0 0 0 3px var(--accent-glow),0 4px 24px rgba(0,0,0,0.4)!important;}
[data-testid="stChatInput"] textarea{background:var(--bg-elevated)!important;color:#e8edf5!important;caret-color:var(--accent)!important;border:none!important;font-size:14px!important;font-family:var(--font-body)!important;line-height:1.6!important;-webkit-text-fill-color:#e8edf5!important;}
[data-testid="stChatInput"] textarea::placeholder{color:var(--text-muted)!important;}
[data-testid="stChatInputSubmitButton"] button{background:var(--accent)!important;border:none!important;border-radius:8px!important;transition:background 0.2s,transform 0.1s;}
[data-testid="stChatInputSubmitButton"] button:hover{background:#2563eb!important;transform:scale(1.05);}
.stButton>button{background:var(--bg-elevated)!important;border:1px solid var(--border)!important;color:var(--text-secondary)!important;border-radius:var(--radius-sm)!important;font-size:12px!important;font-family:var(--font-mono)!important;transition:all 0.15s ease;text-align:left!important;}
.stButton>button:hover{background:var(--bg-hover)!important;border-color:var(--accent)!important;color:var(--text-primary)!important;}
[data-testid="stSelectbox"]>div>div{background:var(--bg-elevated)!important;border:1px solid var(--border-bright)!important;border-radius:var(--radius-sm)!important;color:var(--text-primary)!important;font-family:var(--font-mono)!important;font-size:12px!important;}
[data-testid="stToggle"] input:checked+div{background:var(--accent)!important;}
[data-testid="stMetric"]{background:var(--bg-elevated);border:1px solid var(--border);border-radius:var(--radius-md);padding:12px 14px;text-align:center;}
[data-testid="stMetricLabel"]{color:var(--text-muted)!important;font-size:11px!important;font-family:var(--font-mono)!important;}
[data-testid="stMetricValue"]{color:var(--text-primary)!important;font-family:var(--font-mono)!important;font-size:1.5rem!important;font-weight:600!important;}
[data-testid="stExpander"]{background:var(--bg-elevated)!important;border:1px solid var(--border)!important;border-radius:var(--radius-sm)!important;margin-bottom:4px!important;}
[data-testid="stExpander"] summary{font-family:var(--font-mono)!important;font-size:12px!important;color:var(--text-secondary)!important;padding:8px 12px!important;}
details{border:1px solid var(--border)!important;border-radius:var(--radius-sm)!important;}
[data-testid="stAlert"]{background:var(--bg-elevated)!important;border:1px solid var(--border-bright)!important;border-radius:var(--radius-md)!important;color:var(--text-primary)!important;}
/* Validation badge styles */
.verify-pass{display:flex;align-items:center;gap:8px;background:var(--green-dim);border:1px solid var(--green);border-radius:var(--radius-sm);padding:8px 14px;color:#6ee7b7;font-size:12px;font-family:var(--font-mono);margin-top:10px;letter-spacing:0.02em;}
.verify-fix{display:flex;align-items:center;gap:8px;background:var(--amber-dim);border:1px solid var(--amber);border-radius:var(--radius-sm);padding:8px 14px;color:#fcd34d;font-size:12px;font-family:var(--font-mono);margin-top:10px;letter-spacing:0.02em;}
.auto-fix{display:flex;align-items:center;gap:8px;background:#1a2e1a;border:1px solid #22c55e;border-radius:var(--radius-sm);padding:8px 14px;color:#86efac;font-size:12px;font-family:var(--font-mono);margin-top:6px;letter-spacing:0.02em;}
/* Issue severity styles */
.issue-critical{display:flex;align-items:flex-start;gap:10px;background:var(--red-dim);border:1px solid var(--red);border-left:3px solid var(--red);border-radius:var(--radius-sm);padding:10px 14px;color:#fca5a5;font-size:12px;font-family:var(--font-mono);margin-top:6px;line-height:1.5;}
.issue-error{display:flex;align-items:flex-start;gap:10px;background:#2d1515;border:1px solid #dc2626;border-left:3px solid #dc2626;border-radius:var(--radius-sm);padding:10px 14px;color:#fca5a5;font-size:12px;font-family:var(--font-mono);margin-top:6px;line-height:1.5;}
.issue-warning{display:flex;align-items:flex-start;gap:10px;background:var(--amber-dim);border:1px solid var(--amber);border-left:3px solid var(--amber);border-radius:var(--radius-sm);padding:10px 14px;color:#fcd34d;font-size:12px;font-family:var(--font-mono);margin-top:6px;line-height:1.5;}
.issue-perf{display:flex;align-items:flex-start;gap:10px;background:var(--perf-dim);border:1px solid var(--perf);border-left:3px solid var(--perf);border-radius:var(--radius-sm);padding:10px 14px;color:#67e8f9;font-size:12px;font-family:var(--font-mono);margin-top:6px;line-height:1.5;}
.issue-info{display:flex;align-items:flex-start;gap:10px;background:#141e30;border:1px solid #3b82f6;border-left:3px solid #3b82f6;border-radius:var(--radius-sm);padding:10px 14px;color:#93c5fd;font-size:12px;font-family:var(--font-mono);margin-top:6px;line-height:1.5;}
.issue-why{color:#aab4c0;font-size:11px;margin-top:4px;line-height:1.4;font-style:italic;}
.issue-fix{color:#c8d8e0;font-size:11px;margin-top:2px;line-height:1.4;}
/* Brand / layout */
.brand-header{display:flex;align-items:center;gap:12px;margin-bottom:20px;padding-bottom:16px;border-bottom:1px solid var(--border);}
.brand-icon{width:38px;height:38px;border-radius:10px;background:linear-gradient(135deg,#1d4ed8,#7c3aed);display:flex;align-items:center;justify-content:center;font-size:18px;box-shadow:0 0 16px rgba(99,102,241,0.3);flex-shrink:0;}
.brand-title{font-family:var(--font-ui)!important;font-size:15px!important;font-weight:800!important;color:var(--text-primary)!important;line-height:1.2;}
.brand-sub{font-family:var(--font-mono)!important;font-size:10px!important;color:var(--text-muted)!important;letter-spacing:0.08em;text-transform:uppercase;margin-top:2px;}
.stat-pill{display:inline-flex;align-items:center;gap:5px;background:var(--bg-elevated);border:1px solid var(--border);border-radius:20px;padding:3px 10px;font-family:var(--font-mono);font-size:11px;color:var(--text-secondary);margin-right:6px;}
.stat-pill b{color:var(--text-primary);}
.sidebar-section{font-family:var(--font-mono)!important;font-size:10px!important;text-transform:uppercase!important;letter-spacing:0.1em!important;color:var(--text-muted)!important;margin:14px 0 6px!important;}
.page-title{font-family:var(--font-ui);font-size:2rem;font-weight:800;color:var(--text-primary);letter-spacing:-0.03em;margin-bottom:4px;}
.page-title span{color:var(--accent);}
.page-meta{display:flex;gap:0;align-items:center;margin-bottom:24px;}
.welcome-card{background:var(--bg-elevated);border:1px solid var(--border-bright);border-radius:var(--radius-lg);padding:24px 28px;margin-bottom:8px;position:relative;overflow:hidden;}
.welcome-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#3b82f6,#8b5cf6,#ec4899);}
.welcome-title{font-family:var(--font-ui);font-size:1.1rem;font-weight:700;color:var(--text-primary);margin-bottom:6px;}
.welcome-sub{font-size:13px;color:var(--text-secondary);margin-bottom:18px;line-height:1.5;}
.welcome-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:18px;}
.welcome-item{background:var(--bg-surface);border:1px solid var(--border);border-radius:var(--radius-sm);padding:10px 14px;font-size:13px;color:var(--text-secondary);line-height:1.4;}
.welcome-item b{color:var(--text-primary);display:block;margin-bottom:2px;}
.welcome-examples{border-top:1px solid var(--border);padding-top:14px;margin-top:4px;}
.welcome-examples p{font-family:var(--font-mono);font-size:12px;color:var(--text-muted);margin-bottom:8px;text-transform:uppercase;letter-spacing:0.06em;}
.example-chip{display:inline-block;background:var(--bg-surface);border:1px solid var(--border);border-radius:20px;padding:4px 12px;font-size:12px;color:#93c5fd;font-family:var(--font-mono);margin:3px 3px 3px 0;cursor:default;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:var(--bg-base);}
::-webkit-scrollbar-thumb{background:var(--border-bright);border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:#3a4a6a;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 · SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

_defaults = {
    'messages':        [],
    'complexity':      'Advanced',
    'model_id':        'llama-3.3-70b-versatile',
    'show_reasoning':  True,
    'total_queries':   0,
    'verified_count':  0,
    'fixed_count':     0,
    'rule_hits':       0,
    'auto_fixed_count': 0,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 · GROQ CLIENT
# ─────────────────────────────────────────────────────────────────────────────

def get_client():
    key = ""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass
    key = key or os.environ.get("GROQ_API_KEY", "")
    return (Groq(api_key=key), key) if key else (None, "")

client, _api_key = get_client()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 · SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div class="brand-header">'
        '<div class="brand-icon">⚡</div>'
        '<div>'
        '<div class="brand-title">PQL Assistant v2</div>'
        '<div class="brand-sub">250+ functions · 5-layer context-aware validation</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section">Model</div>', unsafe_allow_html=True)
    selected_model = st.selectbox(
        'Model', options=list(GROQ_MODELS.keys()),
        index=list(GROQ_MODELS.keys()).index(st.session_state.model_id),
        format_func=lambda k: GROQ_MODELS[k], label_visibility='collapsed',
    )
    st.session_state.model_id = selected_model

    st.markdown('<div class="sidebar-section">Complexity</div>', unsafe_allow_html=True)
    complexity = st.select_slider(
        'Complexity', options=['Basic', 'Intermediate', 'Advanced', 'Expert'],
        value=st.session_state.complexity, label_visibility='collapsed',
    )
    st.session_state.complexity = complexity
    st.caption(COMPLEXITY_DESC[complexity])

    st.session_state.show_reasoning = st.toggle(
        'Show query reasoning', value=st.session_state.show_reasoning,
        help='AI explains planning steps before writing the query',
    )

    st.markdown('<div class="sidebar-section">Quick examples</div>', unsafe_allow_html=True)
    for ex in EXAMPLE_PROMPTS.get(complexity, EXAMPLE_PROMPTS['Advanced']):
        if st.button(f'→ {ex}', key=f'ex_{ex}', use_container_width=True):
            st.session_state['_pending'] = ex

    st.markdown('<div class="sidebar-section">Function reference</div>', unsafe_allow_html=True)
    search = st.text_input('Search', placeholder='Search 250+ functions…', label_visibility='collapsed')

    for cat, funcs in PANEL_DATA.items():
        hits = [f for f in funcs
                if not search
                or search.lower() in f['name'].lower()
                or search.lower() in f['doc'].lower()]
        if not hits:
            continue
        icon = CATEGORY_ICONS.get(cat, '•')
        with st.expander(f'{icon} {cat}  ({len(hits)})'):
            for fn in hits:
                if st.button(fn['name'], key=f'fn_{fn["name"]}_{cat}', use_container_width=True):
                    st.session_state['_pending'] = (
                        f'Write a PQL query using {fn["name"]} and explain the syntax with a practical example. '
                        f'Show edge cases and performance notes.'
                    )
                st.caption(fn['doc'][:120] + '…' if len(fn['doc']) > 120 else fn['doc'])

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Queries', st.session_state.total_queries)
    c2.metric('✅', st.session_state.verified_count)
    c3.metric('🔧', st.session_state.fixed_count)
    c4.metric('⚡', st.session_state.auto_fixed_count)

    if st.button('Clear chat', use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 · MAIN CHAT AREA
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="page-title">PQL Query <span>Assistant</span></div>',
    unsafe_allow_html=True
)
st.markdown(
    f'<div class="page-meta">'
    f'<span class="stat-pill"><b>{complexity}</b></span>'
    f'<span class="stat-pill"><b>{st.session_state.model_id.split("-")[0]}</b></span>'
    f'<span class="stat-pill"><b>{len(COMPACT_REFS)}</b> functions</span>'
    f'<span class="stat-pill">🛡 5-layer context-aware validation</span>'
    f'</div>',
    unsafe_allow_html=True
)

if not _api_key:
    st.warning(
        '**Groq API key not found.**\n\n'
        '**Local:** `export GROQ_API_KEY=gsk_...` then restart.\n\n'
        '**Streamlit Cloud:** App Settings → Secrets → add `GROQ_API_KEY = "gsk_..."`',
        icon='🔑',
    )
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg['role'], avatar='⚡' if msg['role'] == 'assistant' else '🧑'):
        st.markdown(msg['content'])

if not st.session_state.messages:
    with st.chat_message('assistant', avatar='⚡'):
        st.markdown(
            '<div class="welcome-card">'
            '<div class="welcome-title">PQL Query Assistant v2 — Context-Aware & Celonis-Grade</div>'
            '<div class="welcome-sub">Every query passes a <strong>5-layer validation pipeline</strong>: '
            'Context Engine → Relationship Rules → Safe Auto-Fix → WHY Explanations → LLM Verification.</div>'
            '<div class="welcome-grid">'
            '<div class="welcome-item"><b>🧠 Context Engine</b>Detects table levels, function interactions, execution scope</div>'
            '<div class="welcome-item"><b>🔗 Relationship Rules</b>20+ PU/GLOBAL/FILTER interaction rules</div>'
            '<div class="welcome-item"><b>⚡ Safe Auto-Fix</b>Targeted, non-destructive corrections with explanations</div>'
            '<div class="welcome-item"><b>📖 WHY Engine</b>Every issue explains root cause + concrete fix</div>'
            '</div>'
            '<div class="welcome-examples">'
            '<p>Try asking</p>'
            '<span class="example-chip">Avg working-hours throughput per case</span>'
            '<span class="example-chip">Automation rate with rework count</span>'
            '<span class="example-chip">BPMN conformance with tolerances</span>'
            '<span class="example-chip">Z-score outlier detection per vendor</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 · 5-LAYER VALIDATION ENGINE  ← UPGRADED in v2
# ─────────────────────────────────────────────────────────────────────────────

def extract_pql_blocks(text: str) -> list:
    return re.findall(r"```pql\s*(.*?)```", text, re.S)


SEVERITY_CSS = {
    'CRITICAL': 'issue-critical',
    'ERROR':    'issue-error',
    'WARNING':  'issue-warning',
    'PERF':     'issue-perf',
    'INFO':     'issue-info',
}
SEVERITY_ICON = {
    'CRITICAL': '🚨',
    'ERROR':    '❌',
    'WARNING':  '⚠',
    'PERF':     '⚡',
    'INFO':     'ℹ',
}


def render_issue(issue: ValidationIssue):
    css_class = SEVERITY_CSS.get(issue.severity, 'issue-info')
    icon = SEVERITY_ICON.get(issue.severity, '•')
    fix_escaped = issue.fix.replace('\n', '<br>').replace('"', '&quot;')
    why_escaped = issue.why.replace('"', '&quot;')
    st.markdown(
        f'<div class="{css_class}">'
        f'<div><strong>{icon} [{issue.code}] {issue.message}</strong>'
        f'<div class="issue-why">Why: {why_escaped}</div>'
        f'<div class="issue-fix">Fix: {fix_escaped}</div>'
        f'</div></div>',
        unsafe_allow_html=True
    )


def run_llm_verification(pql_query: str, issues: List[ValidationIssue]) -> tuple:
    """
    Layer 5: LLM verification — final safety net for patterns not caught by rules.
    Only runs for Advanced/Expert OR when critical/error issues were found.
    Returns: (was_modified, final_query, fix_notes)
    """
    has_serious_issues = any(i.severity in ('CRITICAL', 'ERROR') for i in issues)
    always_verify = st.session_state.complexity in ('Advanced', 'Expert')

    if not (has_serious_issues or always_verify):
        return False, pql_query, []

    try:
        why_context = build_why_explanation(issues) if issues else ""
        rule_context = f"\n\nContext engine detected these issues:\n{why_context}" if why_context else ""

        verify_prompt = (
            f"Review this PQL query for correctness:{rule_context}\n\n"
            f"```pql\n{pql_query}\n```\n\n"
            "Check ALL rules from your instructions. "
            "Respond with either VALID or the corrected ```pql block + brief bullet list of changes made."
        )
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": VERIFICATION_SYSTEM},
                {"role": "user", "content": verify_prompt},
            ],
            temperature=0,
            max_tokens=1500,
        )
        result = response.choices[0].message.content.strip()

        if result.upper().startswith("VALID"):
            return False, pql_query, []

        match = re.search(r"```pql\s*(.*?)```", result, re.S)
        if match:
            corrected = match.group(1).strip()
            fixes = re.findall(r'^[-•*]\s+(.+)', result, re.MULTILINE)
            return True, corrected, fixes or ["Query corrected by LLM verification pass"]

        return False, pql_query, []

    except Exception as e:
        return False, pql_query, [f"LLM verification skipped ({e})"]


def validate_pql_block(pql_query: str) -> tuple:
    """
    Full 5-layer validation pipeline for a single PQL block.

    Layer 1: Context extraction (table levels, functions, execution scope)
    Layer 2: Relationship rule engine (20+ context-aware rules)
    Layer 3: Safe auto-fix (non-destructive targeted corrections)
    Layer 4: WHY explanation engine (root cause + fix for every issue)
    Layer 5: LLM verification (final safety net for complex patterns)

    Returns: (was_modified, final_query, issues, auto_fix_notes, llm_fix_notes)
    """
    # Layer 1 + 2: Context + Rules
    issues = run_context_rule_engine(pql_query)
    st.session_state.rule_hits += len([i for i in issues if i.severity in ('CRITICAL', 'ERROR', 'WARNING')])

    # Layer 3: Safe auto-fix
    auto_fixed, query_after_autofix, auto_fix_notes = apply_safe_auto_fixes(pql_query, issues)
    if auto_fixed:
        st.session_state.auto_fixed_count += 1
        # Re-run rules after auto-fix to get clean issue list
        issues = run_context_rule_engine(query_after_autofix)

    # Layer 5: LLM verification
    llm_fixed, final_query, llm_fix_notes = run_llm_verification(query_after_autofix, issues)
    if llm_fixed:
        st.session_state.fixed_count += 1

    was_modified = auto_fixed or llm_fixed
    return was_modified, final_query, issues, auto_fix_notes, llm_fix_notes


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 · GROQ STREAMING + VALIDATION DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def stream_groq(prompt_override=None):
    msgs = st.session_state.messages
    user_query = prompt_override if prompt_override else msgs[-1]["content"]

    func_context = build_function_context(user_query)
    system = build_system_prompt(st.session_state.complexity, st.session_state.show_reasoning)

    if func_context:
        system += "\n\n## Relevant PQL Functions (auto-retrieved for this query)\n" + func_context

    if prompt_override:
        msgs = msgs + [{'role': 'user', 'content': prompt_override}]

    with st.chat_message('assistant', avatar='⚡'):
        placeholder = st.empty()
        full = ""

        try:
            stream = client.chat.completions.create(
                model=st.session_state.model_id,
                messages=[
                    {"role": "system", "content": system},
                    *[{"role": m["role"], "content": m["content"]} for m in msgs],
                ],
                max_tokens=3000,
                temperature=0.10,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full += delta
                placeholder.markdown(full + "▌")

            placeholder.markdown(full)
            st.session_state.total_queries += 1

            pql_blocks = extract_pql_blocks(full)

            for pql_block in pql_blocks:
                was_modified, final_query, issues, auto_fix_notes, llm_fix_notes = validate_pql_block(pql_block)

                # Render issues with WHY explanations (Layer 4)
                blocking_issues = [i for i in issues if i.severity in ('CRITICAL', 'ERROR', 'WARNING', 'PERF')]
                info_issues = [i for i in issues if i.severity == 'INFO']

                for issue in blocking_issues:
                    render_issue(issue)

                for issue in info_issues:
                    render_issue(issue)

                # Show auto-fix results (Layer 3)
                if auto_fix_notes:
                    st.markdown(
                        '<div class="auto-fix">⚡ <strong>Safe auto-fix applied</strong> — deterministic corrections</div>',
                        unsafe_allow_html=True
                    )
                    for note in auto_fix_notes:
                        st.caption(f"  • {note}")

                # Show LLM fix results (Layer 5)
                if llm_fix_notes and was_modified:
                    st.markdown(
                        '<div class="verify-fix">🔧 <strong>LLM verification</strong> — additional corrections applied</div>',
                        unsafe_allow_html=True
                    )
                    for note in llm_fix_notes:
                        st.caption(f"  • {note}")

                # Show corrected query if modified
                if was_modified:
                    st.markdown("**Corrected query:**")
                    st.code(final_query, language="sql")
                    full = full.replace(
                        f"```pql\n{pql_block}\n```",
                        f"```pql\n{final_query}\n```"
                    )
                else:
                    st.session_state.verified_count += 1
                    has_critical = any(i.severity in ('CRITICAL', 'ERROR') for i in issues)
                    if not has_critical and not blocking_issues:
                        st.markdown(
                            '<div class="verify-pass">✅ <strong>Verified</strong> — passed all 5 validation layers</div>',
                            unsafe_allow_html=True
                        )
                    elif not has_critical:
                        st.markdown(
                            '<div class="verify-pass">✅ <strong>Structurally correct</strong> — review warnings above</div>',
                            unsafe_allow_html=True
                        )

            st.session_state.messages.append({"role": "assistant", "content": full})

        except Exception as e:
            placeholder.error(f"Groq API error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 · INPUT HANDLING
# ─────────────────────────────────────────────────────────────────────────────

# Handle sidebar button → pending prompt
if '_pending' in st.session_state:
    pending = st.session_state.pop('_pending')
    st.session_state.messages.append({'role': 'user', 'content': pending})
    with st.chat_message('user', avatar='🧑'):
        st.markdown(pending)
    stream_groq()
    st.rerun()

# Main chat input
if prompt := st.chat_input('Describe your PQL query, paste code to validate/optimize, or ask about any function…'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user', avatar='🧑'):
        st.markdown(prompt)
    stream_groq()
