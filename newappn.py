# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PQL Query Assistant  ·  v3 REASONING ENGINE  ·  Celonis-Grade             ║
# ║  Groq + LLaMA  ·  Streamlit Cloud  ·  250+ PQL Functions                  ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  WHAT'S NEW (v3 — REASONING ENGINE UPGRADE):                               ║
# ║  + Intent Engine: understands WHAT user is trying to achieve                ║
# ║  + Execution Simulator: simulates join shifts, aggregation grain            ║
# ║  + Schema Layer: validates relationship paths (1:N, N:1, N:M)              ║
# ║  + Reasoning Combinator: combines all signals into smart insights           ║
# ║  + LLM upgraded: from "validator" to "reasoning assistant"                  ║
# ║  + Full pipeline: Rule → Intent → Simulate → Schema → Reason → LLM        ║
# ║  + Process domain knowledge: O2C, P2P, Hire-to-Retire built-in             ║
# ║  + Query optimizer: suggests better/cheaper alternatives                    ║
# ║  + Paste-and-fix mode: paste any broken PQL, get it fixed                  ║
# ║  + All v2 features retained                                                 ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  LOCAL RUN                                                                  ║
# ║    pip install streamlit groq                                               ║
# ║    export GROQ_API_KEY=gsk_...                                              ║
# ║    streamlit run app.py                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import os
import re
from dataclasses import dataclass, field
from typing import List, Set, Optional, Dict, Tuple
import streamlit as st
from groq import Groq

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 · KNOWLEDGE BASE  (250+ PQL functions)
# ─────────────────────────────────────────────────────────────────────────────

COMPACT_REFS = {
    'COUNT': '''[OFFICIAL] Counts non-NULL rows in the specified column.
Syntax: COUNT( table.column )
- NULL values are ignored; use COUNT_TABLE to include NULLs
- Returns INT
- Wrap with GLOBAL() when mixing case-level count with activity-level columns
- Example: COUNT("CASES"."CASE_ID") → case count
- With GLOBAL: GLOBAL(COUNT("CASES"."CASE_ID"))''',

    'COUNT_DISTINCT': '''[OFFICIAL] Counts distinct non-NULL values per group.
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

    'SUM': '''[OFFICIAL] Sums values per group.
Syntax: SUM( table.column )
- NULL values are ignored
- Returns same data type as input
- Wrap with GLOBAL() when mixing table levels
- Example: SUM("ORDERS"."AMOUNT")''',

    'AVG': '''[OFFICIAL] Average per group.
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
- NULL values ignored
- Example: MIN("ACTIVITIES"."TIMESTAMP")''',

    'MEDIAN': '''[OFFICIAL] Median per group.
Syntax: MEDIAN( table.column )
- SIGNIFICANTLY more expensive than AVG
- Example: MEDIAN("ORDERS"."PROCESSING_DAYS")''',

    'STDEV': 'Standard deviation (n-1 method) per group. Syntax: STDEV( table.column )',
    'VAR': 'Variance (n-1 method) per group. Syntax: VAR( table.column )',
    'QUANTILE': 'Quantile value per group. Syntax: QUANTILE( table.column, quantile ) quantile: 0.0-1.0',
    'TRIMMED_MEAN': 'Mean excluding outliers. Syntax: TRIMMED_MEAN( table.column [, lower [, upper]] )',
    'MODE': 'Most frequent value per group. Syntax: MODE( table.column )',
    'PRODUCT': 'Product of all values per group. Syntax: PRODUCT( table.column )',

    'FIRST': '''[OFFICIAL] First element per group.
Syntax: FIRST( table.column [, ORDER BY table.column [ASC|DESC]] )
- ALWAYS specify ORDER BY for deterministic results
- Example: FIRST("ACTIVITIES"."ACTIVITY", ORDER BY "ACTIVITIES"."TIMESTAMP" ASC)''',

    'LAST': '''[OFFICIAL] Last element per group.
Syntax: LAST( table.column [, ORDER BY table.column [ASC|DESC]] )
- ALWAYS specify ORDER BY for deterministic results''',

    'STRING_AGG': 'Concatenates string values. Syntax: STRING_AGG( table.column, "delimiter" [, ORDER BY col] )',

    'GLOBAL': '''[OFFICIAL DOCS] Isolates aggregation from common table — prevents join multiplication.
Syntax: GLOBAL( aggregation_expression )
- WHEN TO USE: mixing columns from different table levels (case + activity)
- Celonis shifts common table to lowest level causing multiplication
- GLOBAL() anchors aggregation back to original table
- ALWAYS wrap CALC_THROUGHPUT: GLOBAL(AVG(CALC_THROUGHPUT(...)))
- ALWAYS wrap case-level aggs mixed with activity: GLOBAL(COUNT("CASES"."CASE_ID"))
- GLOBAL cannot be used inside FILTER statements
- GLOBAL result cannot be a grouper column''',

    'RUNNING_TOTAL': '''[OFFICIAL] Cumulative running total. (Replaces RUNNING_SUM)
Syntax: RUNNING_TOTAL( table.column [, ORDER BY (...)] [, PARTITION BY (...)] )
- ORDER BY is required for meaningful results
- Example: RUNNING_TOTAL("ORDERS"."AMOUNT", ORDER BY ("ORDERS"."ORDER_DATE" ASC))''',

    'RUNNING_SUM': 'Alias of RUNNING_TOTAL (deprecated). Prefer RUNNING_TOTAL in new queries.',

    'WINDOW_AVG': 'Average over a sliding window. Syntax: WINDOW_AVG( table.column, lower, upper [, ORDER BY (...)] )',
    'INDEX_ORDER': 'Integer row indices from 1. Syntax: INDEX_ORDER( table.column [, ORDER BY (...)] [, PARTITION BY (...)] )',
    'ZSCORE': 'Z-score normalization. Syntax: ZSCORE( table.column [, PARTITION BY (...)] )',
    'INTERPOLATE': 'Fills NULL values. Syntax: INTERPOLATE( column, CONSTANT | LINEAR [, ORDER BY (...)] )',

    'MOVING_AVG': 'Moving average. Syntax: MOVING_AVG( table.col, lower, upper [, ORDER BY ...] )',
    'MOVING_SUM': 'Moving sum. Syntax: MOVING_SUM( table.col, lower, upper [, ORDER BY ...] )',
    'MOVING_COUNT': 'Moving count. Syntax: MOVING_COUNT( table.col, lower, upper [, ORDER BY ...] )',
    'MOVING_COUNT_DISTINCT': 'Moving distinct count. Syntax: MOVING_COUNT_DISTINCT( table.col, lower, upper )',
    'MOVING_MAX': 'Moving maximum. Syntax: MOVING_MAX( table.col, lower, upper [, ORDER BY ...] )',
    'MOVING_MIN': 'Moving minimum. Syntax: MOVING_MIN( table.col, lower, upper [, ORDER BY ...] )',
    'MOVING_MEDIAN': 'Moving median (expensive). Syntax: MOVING_MEDIAN( table.col, lower, upper )',
    'MOVING_STDEV': 'Moving standard deviation. Syntax: MOVING_STDEV( table.col, lower, upper )',
    'MOVING_TRIMMED_MEAN': 'Moving trimmed mean. Syntax: MOVING_TRIMMED_MEAN( table.col, lower, upper )',
    'MOVING_VAR': 'Moving variance. Syntax: MOVING_VAR( table.col, lower, upper )',

    'PU_COUNT': '''[OFFICIAL DOCS] Counts non-NULL rows in source per target row.
Syntax: PU_COUNT( target_table, source_table.column [, filter_expression] )
- Returns 0 (not NULL) when no matching rows exist
- Requires 1:N relationship: target_table is parent (1-side), source is child (N-side)
- PU_COUNT IGNORES global filters — use filter_expression arg for filter-aware counts
- PREFER over PU_COUNT_DISTINCT when column is already a key (much faster)
- Example: PU_COUNT("CASES", "ACTIVITIES"."CASE_ID", "ACTIVITIES"."ACTIVITY" = 'Approve')''',

    'PU_SUM': '''[OFFICIAL DOCS] Sums source column per target row.
Syntax: PU_SUM( target_table, source_table.column [, filter_expression] )
- Returns NULL (not 0) when no matching rows exist
- PU_SUM IGNORES global filters
- Example: PU_SUM("VENDORS", "ORDERS"."AMOUNT")''',

    'PU_AVG': '''[OFFICIAL DOCS] Average of source column per target row.
Syntax: PU_AVG( target_table, source_table.column [, filter_expression] )
- Returns NULL when no matching rows; always returns FLOAT
- MUCH cheaper than PU_MEDIAN
- Example: PU_AVG("VENDORS", "ORDERS"."LEAD_TIME_DAYS")''',

    'PU_MAX': '''[OFFICIAL DOCS] Maximum of source column per target row.
Syntax: PU_MAX( target_table, source_table.column [, filter_expression] )
- Returns NULL when no matching rows''',

    'PU_MIN': '''[OFFICIAL DOCS] Minimum of source column per target row.
Syntax: PU_MIN( target_table, source_table.column [, filter_expression] )
- Returns NULL when no matching rows''',

    'PU_FIRST': '''[OFFICIAL DOCS] Returns first element of source column for each target row.
Syntax: PU_FIRST( target_table, source_table.column [, filter_expression] [, ORDER BY col [ASC|DESC]] )
- Returns NULL when no matching rows
- ALWAYS use explicit ORDER BY for deterministic results
- Example: PU_FIRST("CASES", "ACTIVITIES"."TIMESTAMP", ORDER BY "ACTIVITIES"."TIMESTAMP" ASC)''',

    'PU_LAST': '''[OFFICIAL DOCS] Returns last element of source column for each target row.
Syntax: PU_LAST( target_table, source_table.column [, filter_expression] [, ORDER BY col [ASC|DESC]] )
- ALWAYS use explicit ORDER BY
- Example: PU_LAST("ORDERS", "STATUS_TABLE"."STATUS", ORDER BY "STATUS_TABLE"."CHANGE_DATE" ASC)''',

    'PU_MEDIAN': '''[OFFICIAL DOCS] Median per target row.
Syntax: PU_MEDIAN( target_table, source_table.column [, filter_expression] )
- SIGNIFICANTLY more expensive than PU_AVG
- Only use when true median is required''',

    'PU_COUNT_DISTINCT': '''[OFFICIAL DOCS] Distinct count per target row.
Syntax: PU_COUNT_DISTINCT( target_table, source_table.column [, filter_expression] )
- Returns 0 (not NULL)
- USE PU_COUNT instead when column is already a key (much cheaper)''',

    'PU_MODE': 'Most frequent value per target row. Syntax: PU_MODE( target, source.column )',
    'PU_PRODUCT': 'Product per target row. Syntax: PU_PRODUCT( target, source.column )',
    'PU_QUANTILE': 'Quantile per target row. Syntax: PU_QUANTILE( target, source.column, quantile )',
    'PU_TRIMMED_MEAN': 'Trimmed mean per target row. Syntax: PU_TRIMMED_MEAN( target, source.column )',
    'PU_STRING_AGG': 'Concatenates strings per target row. Syntax: PU_STRING_AGG( target, source.column, delimiter )',
    'PU_STDEV': 'Standard deviation per target row. Syntax: PU_STDEV( target, source.column )',

    'DOMAIN_TABLE': '''[OFFICIAL] Creates a table with all distinct value combinations.
Syntax: DOMAIN_TABLE( table.col1, table.col2, ... )
- Used as target_table in PU-functions
- Example: PU_SUM(DOMAIN_TABLE("ORDERS"."YEAR"), "ORDERS"."AMOUNT")''',

    'CONSTANT': '''[OFFICIAL] Used as target table in PU-functions to produce a single constant result.
Syntax: CONSTANT()
- Example: PU_SUM(CONSTANT(), "ORDERS"."AMOUNT") → grand total''',

    'COMMON_TABLE': 'References the common table. Syntax: COMMON_TABLE( expr1, expr2 )',

    'CALC_THROUGHPUT': '''[OFFICIAL DOCS] Calculates throughput time per case.
Syntax: CALC_THROUGHPUT( begin TO end, REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", unit) )
begin/end: CASE_START | CASE_END | FIRST_OCCURRENCE['act'] | LAST_OCCURRENCE['act']
unit: DAYS | HOURS | MINUTES | SECONDS | MILLISECONDS
- Returns NULL if start > end or case has only one activity
- NOTE: ALL_OCCURRENCE[''] is DEPRECATED since 4.6
- Wrap with GLOBAL() when combined with activity-level columns
- Example: CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", DAYS))
- Avg: AVG(CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", DAYS)))''',

    'REMAP_TIMESTAMPS': '''[OFFICIAL DOCS] Converts DATE column to integer count of time units since epoch.
Syntax: REMAP_TIMESTAMPS( activity_table.timestamp_col, unit [, calendar_specification] )
Units: DAYS | HOURS | MINUTES | SECONDS | MILLISECONDS
- Primary use: provides timestamps argument to CALC_THROUGHPUT
- Supports WEEKDAY_CALENDAR, FACTORY_CALENDAR, WORKDAY_CALENDAR
- Example: REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", HOURS, WEEKDAY_CALENDAR(MON,TUE,WED,THU,FRI))''',

    'CALC_REWORK': '''[OFFICIAL DOCS] Counts number of activities per case.
Syntax: CALC_REWORK() | CALC_REWORK( filter_expression ) | CALC_REWORK( activity_table.column )
- Returns INT column on CASE table
- Example: FILTER CALC_REWORK("ACTIVITIES"."ACTIVITY" = 'Review') > 1''',

    'CALC_CROP': 'Crops cases to event range. Returns 1 inside range, NULL outside. Syntax: CALC_CROP( begin TO end, activity_table.column )',
    'CALC_CROP_TO_NULL': 'Crops cases to event range, keeps values in range, NULL outside. Syntax: CALC_CROP_TO_NULL( begin TO end, activity_table.column )',

    'MATCH_ACTIVITIES': '''[OFFICIAL DOCS] Flags cases containing activities. Order-INDEPENDENT.
Syntax: MATCH_ACTIVITIES( [STARTING node_list] [NODE node_list] [ENDING node_list] [EXCLUDING node_list] )
- Returns 1 matching / 0 non-matching
- Example: FILTER MATCH_ACTIVITIES(NODE('Approve'), EXCLUDING('Cancel')) = 1''',

    'MATCH_PROCESS': '''[OFFICIAL DOCS] Matches cases against ordered node/edge pattern. Order-SENSITIVE.
Syntax: MATCH_PROCESS( node(, node)* CONNECTED BY edge(, edge)* )
- Node types: NODE, OPTIONAL, LOOP, STARTING, ENDING
- Edge types: DIRECT [A, B], EVENTUALLY [A, B]
- Example: FILTER MATCH_PROCESS( STARTING ["Create"] AS n1, ENDING ["Close"] AS n2 CONNECTED BY EVENTUALLY[n1,n2] ) = 1''',

    'MATCH_PROCESS_REGEX': 'Filters variants using regex. Syntax: MATCH_PROCESS_REGEX( "regex_pattern" )',

    'ACTIVITY_LAG': '''[OFFICIAL DOCS] Returns value from preceding row within same case.
Syntax: ACTIVITY_LAG( activity_table.column [, offset] )  Default offset: 1
- Example: ACTIVITY_LAG("ACTIVITIES"."TIMESTAMP") → previous activity timestamp''',

    'ACTIVITY_LEAD': 'Returns value from following row within same case. Syntax: ACTIVITY_LEAD( activity_table.column [, offset] )',

    'INDEX_ACTIVITY_ORDER': '''[OFFICIAL DOCS] Returns 1-based position of each activity within its case.
Syntax: INDEX_ACTIVITY_ORDER( activity_table.column )
- Replaces deprecated PROCESS_ORDER''',

    'INDEX_ACTIVITY_LOOP': '''[OFFICIAL DOCS] Returns how many times an activity has already occurred.
Syntax: INDEX_ACTIVITY_LOOP( activity_table.column )
- Returns 0 for first occurrence, 1 for second, etc.
- Used for rework: FILTER INDEX_ACTIVITY_LOOP("ACTIVITIES"."ACTIVITY") > 0''',

    'INDEX_ACTIVITY_TYPE': 'Returns type-specific loop counter per case. Syntax: INDEX_ACTIVITY_TYPE( activity_table.column )',
    'PROCESS_ORDER': 'DEPRECATED — use INDEX_ACTIVITY_ORDER instead.',
    'VARIANT': 'Returns the process variant string per case. Syntax: VARIANT( activity_table.string_column )',
    'ACTIVATION_COUNT': 'Returns number of times an edge was activated. Syntax: ACTIVATION_COUNT( SOURCE["A"] TARGET["B"] )',

    'BPMN_CONFORMS': '''[OFFICIAL] Binary BPMN conformance check (1=conforming, 0=not conforming).
Syntax: BPMN_CONFORMS( event_table.col, bpmn_model [, ALLOW(...)] )
- Example: BPMN_CONFORMS("ACTIVITIES"."ACTIVITY", SEQUENCE("Create","Approve","Pay"))''',

    'CONFORMANCE': 'Petri net conformance checking. Use with READABLE() for human-readable descriptions.',
    'READABLE': 'Human-readable violation descriptions from CONFORMANCE.',
    'SEQUENCE': 'BPMN_CONFORMS helper: models sequential flow. Syntax: SEQUENCE("A", "B", "C")',
    'PARALLEL': 'BPMN_CONFORMS helper: models parallel paths.',
    'EXCLUSIVE_CHOICE': 'BPMN_CONFORMS helper: models XOR gateway.',
    'ALLOW': 'Allows specific deviations in BPMN_CONFORMS.',
    'PROCESS_EQUALS': 'Checks if case follows exact sequence. Syntax: PROCESS_EQUALS( "A" > "B" > "C" )',

    'DATEDIFF': '''[OFFICIAL DOCS] Date difference. Returns FLOAT.
Syntax: DATEDIFF( unit, table.date1, table.date2 )
Units: ms | ss | mi | hh | dd | mm | yy
- Example: DATEDIFF('dd', "ORDERS"."CREATE_DATE", "ORDERS"."CLOSE_DATE")''',

    'HOURS_BETWEEN': 'Difference in hours. Syntax: HOURS_BETWEEN( date1, date2 [, calendar] )',
    'MINUTES_BETWEEN': 'Difference in minutes. Syntax: MINUTES_BETWEEN( date1, date2 )',
    'SECONDS_BETWEEN': 'Difference in seconds. Syntax: SECONDS_BETWEEN( date1, date2 )',
    'MILLIS_BETWEEN': 'Difference in milliseconds. Syntax: MILLIS_BETWEEN( date1, date2 )',
    'DAYS_BETWEEN': 'Difference in days (FLOAT). Syntax: DAYS_BETWEEN( date1, date2 [, calendar] )',
    'WORKDAYS_BETWEEN': 'Number of workdays. Syntax: WORKDAYS_BETWEEN( calendar, date1, date2 )',
    'DATE_BETWEEN': 'Difference in days (INT). Syntax: DATE_BETWEEN( date1, date2 )',
    'MONTHS_BETWEEN': 'Difference in months. Syntax: MONTHS_BETWEEN( date1, date2 )',
    'YEARS_BETWEEN': 'Difference in years. Syntax: YEARS_BETWEEN( date1, date2 )',

    'ADD_DAYS': 'Adds days. Syntax: ADD_DAYS( base_col, days_col )',
    'ADD_HOURS': 'Adds hours. Syntax: ADD_HOURS( start_col, hours_col [, calendar] )',
    'ADD_MINUTES': 'Adds minutes. Syntax: ADD_MINUTES( start_col, minutes_col )',
    'ADD_SECONDS': 'Adds seconds. Syntax: ADD_SECONDS( start_col, seconds_col )',
    'ADD_MILLIS': 'Adds milliseconds. Syntax: ADD_MILLIS( start_col, ms_col )',
    'ADD_WORKDAYS': 'Adds workdays. Syntax: ADD_WORKDAYS( calendar, date, days )',
    'ADD_MONTHS': 'Adds months. Syntax: ADD_MONTHS( date_col, months_col )',
    'ADD_YEARS': 'Adds years. Syntax: ADD_YEARS( date_col, years_col )',

    'TODAY': 'Current date. Syntax: TODAY( [timezone_id] )',
    'HOUR_NOW': 'Current hour. Syntax: HOUR_NOW( [timezone_id] )',
    'MINUTE_NOW': 'Current minute. Syntax: MINUTE_NOW( [timezone_id] )',

    'ROUND_DAY': 'Rounds down to day. Syntax: ROUND_DAY( date_col )',
    'ROUND_HOUR': 'Rounds down to nearest hour. Syntax: ROUND_HOUR( timestamp_col )',
    'ROUND_MINUTE': 'Rounds down to nearest minute. Syntax: ROUND_MINUTE( timestamp_col )',
    'ROUND_SECOND': 'Rounds down to nearest second. Syntax: ROUND_SECOND( timestamp_col )',
    'ROUND_WEEK': 'Rounds down to Monday of the week. Syntax: ROUND_WEEK( date_col )',
    'ROUND_MONTH': 'Rounds down to first day of month. Syntax: ROUND_MONTH( date_col )',
    'ROUND_QUARTER': 'Rounds down to start of quarter. Syntax: ROUND_QUARTER( col )',
    'ROUND_YEAR': 'Rounds down to start of year. Syntax: ROUND_YEAR( date_col )',

    'CONVERT_TIMEZONE': 'Converts date between timezones. Syntax: CONVERT_TIMEZONE( date_col [, from_tz], to_tz )',
    'DATE_MATCH': 'Returns 1 if date matches filter lists. Syntax: DATE_MATCH( col, [YEARS], [MONTHS], [DAYS] )',
    'DAYS_IN_MONTH': 'Days in the month of the given date. Syntax: DAYS_IN_MONTH( col )',
    'IN_CALENDAR': 'Checks if date is within a calendar period. Syntax: IN_CALENDAR( ts_col, calendar )',

    'CALENDAR_WEEK': 'Calendar week number (1-53). Syntax: CALENDAR_WEEK( date_col )',
    'DAY': 'Day of month (1-31). Syntax: DAY( date_col )',
    'DAY_OF_WEEK': 'Day of week (1=Mon…7=Sun). Syntax: DAY_OF_WEEK( date_col )',
    'MONTH': 'Month number (1-12). Syntax: MONTH( date_col )',
    'QUARTER': 'Quarter (1-4). Syntax: QUARTER( date_col )',
    'YEAR': '4-digit year. Syntax: YEAR( date_col )',
    'HOURS': 'Hour component (0-23). Syntax: HOURS( timestamp_col )',
    'MINUTES': 'Minute component (0-59). Syntax: MINUTES( timestamp_col )',
    'SECONDS': 'Seconds component (0-59). Syntax: SECONDS( timestamp_col )',
    'MILLIS': 'Milliseconds component. Syntax: MILLIS( timestamp_col )',

    'FACTORY_CALENDAR': 'Factory calendar with specific work intervals. Used with REMAP_TIMESTAMPS.',
    'WORKDAY_CALENDAR': 'Work days from a table. Used with ADD_WORKDAYS and date diff functions.',
    'WEEKDAY_CALENDAR': '''[OFFICIAL] Defines which weekdays count as work days.
Syntax: WEEKDAY_CALENDAR( MON, TUE, WED, THU, FRI )
- Example: REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", HOURS, WEEKDAY_CALENDAR(MON,TUE,WED,THU,FRI))''',

    'UPPER': 'Uppercase. Syntax: UPPER( table.column )',
    'LOWER': 'Lowercase. Syntax: LOWER( table.column )',
    'CONCAT': 'Concatenates strings. Syntax: CONCAT( col1, ..., colN ) NULL in any arg = NULL.',
    'STRING_SPLIT': 'Splits string. Zero-based index. Syntax: STRING_SPLIT( col, pattern, index )',
    'TO_STRING': 'Converts INT or DATE to STRING. Syntax: TO_STRING( col [, FORMAT("%Y-%m-%d")] )',
    'FORMAT': 'Specifies date/string format. Syntax: FORMAT( "%Y-%m-%d" )',
    'IN_LIKE': 'Pattern matching with wildcards. Syntax: col IN_LIKE( "pattern%" )',
    'LIKE': 'Pattern matching. Syntax: col LIKE "pattern%"',
    'MATCH_STRINGS': 'Fuzzy matching by edit distance. Syntax: MATCH_STRINGS( col1, col2 [, TOP_K(k)] )',
    'REMAP_VALUES': 'Maps STRING values. Syntax: REMAP_VALUES( col, [old1, new1], ..., [default] )',
    'REMAP_INTS': 'Maps INT values. Syntax: REMAP_INTS( col, [old1, new1], ..., [default] )',
    'LEN': 'String length. Syntax: LEN( string_col ) Returns INT.',
    'SUBSTRING': 'Extracts substring. Syntax: SUBSTRING( string_col, start_pos [, length] ) 1-based.',
    'LTRIM': 'Removes leading whitespace. Syntax: LTRIM( string_col )',
    'RTRIM': 'Removes trailing whitespace. Syntax: RTRIM( string_col )',
    'REVERSE': 'Reverses a string. Syntax: REVERSE( string_col )',
    'STRINGHASH': 'Hash of string as INT. Syntax: STRINGHASH( string_col )',
    'STR_TO_INT': 'Converts string to integer. Syntax: STR_TO_INT( string_col )',

    'ABS': 'Absolute value. Syntax: ABS( table.column )',
    'POWER': 'Raised to a power. Syntax: POWER( col, exponent ) Output: FLOAT.',
    'MODULO': 'Remainder of division. Syntax: MODULO( dividend, divisor ) or dividend % divisor.',
    'GREATEST': 'Maximum value across columns. Syntax: GREATEST( col1, col2, ..., colN )',
    'LEAST': 'Minimum value across columns. Syntax: LEAST( col1, col2, ..., colN )',
    'COALESCE': 'First non-NULL value. Syntax: COALESCE( col1, col2, ..., colN )',
    'ISNULL': 'Returns 1 if NULL, 0 otherwise. Syntax: ISNULL( table.column )',
    'CEIL': 'Rounds up. Syntax: CEIL( table.column ) Returns INT.',
    'FLOOR': 'Rounds down. Syntax: FLOOR( table.column ) Returns INT.',
    'ROUND': 'Rounds to decimal places. Syntax: ROUND( table.column, decimal_places )',
    'SQRT': 'Square root. Syntax: SQRT( table.column ) Returns FLOAT.',
    'SQUARE': 'Squares a value. Syntax: SQUARE( table.column ) Returns FLOAT.',
    'LOG': 'Natural logarithm. Syntax: LOG( table.column ) Input must be > 0.',
    'QNORM': 'Quantile of normal distribution. Syntax: QNORM( probability )',

    'CASE': 'Conditional expression. Syntax: CASE WHEN cond THEN val [WHEN ...] ELSE default END',
    'IN': 'Membership test. Syntax: col IN( "val1", "val2" )',
    'MULTI_IN': 'Multi-column tuple membership. Syntax: MULTI_IN( (col,...), (val1,...) )',
    'BETWEEN': 'Range check (inclusive). Syntax: col BETWEEN lower AND upper',
    'AND': 'Logical AND.',
    'OR': 'Logical OR.',
    'NOT': 'Logical NOT.',

    'FILTER': '''[OFFICIAL] Filters result set.
Syntax: FILTER table.col = "value"
- Multiple FILTER statements merge by logical AND
- Cannot be used inside GLOBAL()
- FILTER cannot be inside PU_* functions — use filter_expression argument instead''',

    'FILTER_TO_NULL': '''[OFFICIAL] Makes column filter-aware.
Syntax: FILTER_TO_NULL( table.col )
- NEVER use FILTER_TO_NULL inside PU functions (will fail or give wrong results)
- Example: SUM(FILTER_TO_NULL("ORDERS"."AMOUNT"))''',

    'BIND_FILTERS': 'Pulls filter to specified table. Syntax: BIND_FILTERS( target_table, condition )',
    'BIND': 'Pulls a value to a target table. Used for 1:N:1 relationships.',
    'LOOKUP': 'Left outer join ignoring predefined joins. Syntax: LOOKUP( target_table, source_col, (join_cond) )',

    'BUCKET_UPPER_BOUND': 'Histogram bucket upper bounds. Syntax: BUCKET_UPPER_BOUND( col [, SUGGESTED_COUNT(n)] )',
    'GENERATE_RANGE': 'Creates a value range. Syntax: GENERATE_RANGE( step_size, range_start, range_end ) Max 10,000.',
    'RANGE_APPEND': 'Creates a range and appends to a column. Syntax: RANGE_APPEND( col, step_size, range_end )',
    'UNIQUE_ID': 'Unique INT for each unique tuple. Syntax: UNIQUE_ID( col1, ..., colN )',

    'CREATE_EVENTLOG': 'Returns activity table from OCPM object perspective. Syntax: CREATE_EVENTLOG( lead_object, event_type_list )',
    'MERGE_EVENTLOG': 'Merges columns from two activity tables. Syntax: MERGE_EVENTLOG( target_table.col, [FILTER ...] )',
    'MERGE_EVENTLOG_DISTINCT': 'Like MERGE_EVENTLOG but removes duplicates.',
    'EVENTLOG_SOURCE_TABLE': 'Source table name for each row in a dynamic event log.',
    'LINK_PATH': 'Traverses object links in OCPM. Syntax: LINK_PATH( table.col [, CONSTRAINED BY ...] )',
    'LINK_SOURCE': 'Source objects of Object Link. Syntax: LINK_SOURCE( link_name, table.col )',
    'LINK_TARGET': 'Target objects of Object Link. Syntax: LINK_TARGET( link_name, table.col )',
    'LINK_FILTER': 'Filters by link traversal. Syntax: LINK_FILTER( filter_expr, ANCESTORS|DESCENDANTS )',
    'LINK_OBJECTS': 'All objects in the Object Link graph.',
    'UNION_ALL': 'Vertical concatenation of columns.',
    'UNION_ALL_TABLE': 'Vertical concatenation of tables (2-16). Syntax: UNION_ALL_TABLE( table1, ..., tableN )',
    'UNION_ALL_PULLBACK': 'Projects UNION_ALL back to source table. Syntax: UNION_ALL_PULLBACK( union_col, index )',

    'CASE_ID_COLUMN': 'References case ID column. Syntax: CASE_ID_COLUMN( [expr] )',
    'CASE_TABLE': 'References the case table. Syntax: CASE_TABLE( [expr] )',
    'ACTIVITY_TABLE': 'References the activity table in OCPM. Syntax: ACTIVITY_TABLE( LINK_PATH(...) )',
    'ACTIVITY_COLUMN': 'References the activity column. Syntax: ACTIVITY_COLUMN( [expr] )',
    'TIMESTAMP_COLUMN': 'References the timestamp column. Syntax: TIMESTAMP_COLUMN( [expr] )',

    'CURRENCY_CONVERT': 'Converts currency. Syntax: CURRENCY_CONVERT( amount, FROM("USD"), TO("EUR"), date, "RATES_TABLE" )',
    'CURRENCY_CONVERT_SAP': 'Converts SAP currency using TCURR/TCURF/TCURX.',
    'CURRENCY_SAP': 'Adjusts SAP amounts for decimal places.',
    'QUANTITY_CONVERT': 'Converts quantity units. Syntax: QUANTITY_CONVERT( amount, FROM("unit1"), TO("unit2"), id_col, "RATES" )',

    'COLUMN_TYPE': 'Returns data type as STRING at query-build time. Syntax: COLUMN_TYPE( col ) Returns: INT/FLOAT/STRING/DATE',
    'ARGUMENT_COUNT': 'Counts arguments at query-build time. Syntax: ARGUMENT_COUNT( arg1, arg2, ... )',
    'USER_NAME': 'Returns logged-in username. Syntax: USER_NAME()',

    'KMEANS': 'K-means++ clustering. Syntax: KMEANS( k, col1, col2 )',
    'TRAIN_KM': 'Trains a KMeans model. Syntax: TRAIN_KM( k, INPUT( col1, ... ) )',
    'CLUSTER': 'Assigns rows to clusters. Syntax: CLUSTER( TRAIN_KM(...), col, ... )',
    'LINEAR_REGRESSION': 'Linear regression. Syntax: LINEAR_REGRESSION( TRAIN_LM( INPUT(...), OUTPUT(...) ), PREDICT( col ) )',
    'CLUSTER_VARIANTS': 'Clusters process variants. Syntax: CLUSTER_VARIANTS( k )',
    'DECISION_TREE': 'Decision tree classification.',
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1B · PANEL DATA
# ─────────────────────────────────────────────────────────────────────────────

PANEL_DATA = {
    'Pull-Up (PU) Aggregation': [
        {'name': 'PU_COUNT', 'doc': 'Count rows in source per target row. Returns 0. Prefer over PU_COUNT_DISTINCT for keys.'},
        {'name': 'PU_SUM', 'doc': 'Sum source column per target row. Returns NULL when no match.'},
        {'name': 'PU_AVG', 'doc': 'Average per target row. Always FLOAT. Cheaper than PU_MEDIAN.'},
        {'name': 'PU_MAX', 'doc': 'Maximum per target row.'},
        {'name': 'PU_MIN', 'doc': 'Minimum per target row.'},
        {'name': 'PU_FIRST', 'doc': 'First element per target row. Always use ORDER BY.'},
        {'name': 'PU_LAST', 'doc': 'Last element per target row. Always use ORDER BY.'},
        {'name': 'PU_MEDIAN', 'doc': 'Median per target row. Very expensive — use PU_AVG when possible.'},
        {'name': 'PU_COUNT_DISTINCT', 'doc': 'Distinct count per target row. Returns 0. Use PU_COUNT for key columns.'},
        {'name': 'PU_MODE', 'doc': 'Most frequent value per target row.'},
        {'name': 'PU_QUANTILE', 'doc': 'Quantile (0.0-1.0) per target row.'},
        {'name': 'PU_STRING_AGG', 'doc': 'Concatenates strings per target row.'},
        {'name': 'PU_STDEV', 'doc': 'Standard deviation per target row.'},
        {'name': 'CONSTANT', 'doc': 'Used as target_table for a single global result.'},
        {'name': 'DOMAIN_TABLE', 'doc': 'All distinct combinations of columns — use as PU target.'},
    ],
    'Standard Aggregation': [
        {'name': 'COUNT', 'doc': 'Count non-NULL rows. Wrap with GLOBAL() when mixing table levels.'},
        {'name': 'COUNT_TABLE', 'doc': 'Counts rows including NULLs. Stable denominator.'},
        {'name': 'SUM', 'doc': 'Sum per group. Respects global filters.'},
        {'name': 'AVG', 'doc': 'Average per group. Returns FLOAT.'},
        {'name': 'MAX', 'doc': 'Maximum per group.'},
        {'name': 'MIN', 'doc': 'Minimum per group.'},
        {'name': 'MEDIAN', 'doc': 'Median per group. Expensive — use AVG unless median required.'},
        {'name': 'STDEV', 'doc': 'Standard deviation per group.'},
        {'name': 'MODE', 'doc': 'Most frequent value per group.'},
        {'name': 'QUANTILE', 'doc': 'Quantile per group. Syntax: QUANTILE( col, quantile )'},
        {'name': 'FIRST', 'doc': 'First element per group. Always use ORDER BY.'},
        {'name': 'LAST', 'doc': 'Last element per group. Always use ORDER BY.'},
        {'name': 'STRING_AGG', 'doc': 'Concatenates strings with a delimiter.'},
        {'name': 'GLOBAL', 'doc': 'Isolates aggregation to prevent join multiplication.'},
    ],
    'Window Aggregation': [
        {'name': 'RUNNING_TOTAL', 'doc': 'Cumulative running total. Needs ORDER BY.'},
        {'name': 'WINDOW_AVG', 'doc': 'Average over a sliding window of rows.'},
        {'name': 'INDEX_ORDER', 'doc': 'Integer indices from 1.'},
        {'name': 'ZSCORE', 'doc': 'Z-score normalization. Supports PARTITION BY.'},
        {'name': 'INTERPOLATE', 'doc': 'Interpolates NULL values (CONSTANT or LINEAR).'},
        {'name': 'MOVING_AVG', 'doc': 'Moving average.'},
        {'name': 'MOVING_SUM', 'doc': 'Moving sum.'},
        {'name': 'MOVING_COUNT', 'doc': 'Moving count.'},
        {'name': 'MOVING_MAX', 'doc': 'Moving maximum.'},
        {'name': 'MOVING_MIN', 'doc': 'Moving minimum.'},
    ],
    'Process & Conformance': [
        {'name': 'CALC_THROUGHPUT', 'doc': 'Throughput time per case. Wrap with GLOBAL() when mixing with activity KPIs.'},
        {'name': 'REMAP_TIMESTAMPS', 'doc': 'Converts timestamp for CALC_THROUGHPUT. Supports calendars.'},
        {'name': 'CALC_REWORK', 'doc': 'Counts activities per case. Returns INT on case table.'},
        {'name': 'CALC_CROP', 'doc': 'Crops cases to event range. Returns 1 in range, NULL outside.'},
        {'name': 'MATCH_ACTIVITIES', 'doc': 'Flags cases with activities (order-independent).'},
        {'name': 'MATCH_PROCESS', 'doc': 'Matches variants against node/edge pattern (order-sensitive).'},
        {'name': 'MATCH_PROCESS_REGEX', 'doc': 'Filters variants using regex.'},
        {'name': 'ACTIVITY_LAG', 'doc': 'Previous row by offset within a case.'},
        {'name': 'ACTIVITY_LEAD', 'doc': 'Next row by offset within a case.'},
        {'name': 'INDEX_ACTIVITY_ORDER', 'doc': '1-based position of each activity within its case.'},
        {'name': 'INDEX_ACTIVITY_LOOP', 'doc': 'Prior occurrences of this activity in case (0=first).'},
        {'name': 'VARIANT', 'doc': 'Process variant string per case.'},
        {'name': 'BPMN_CONFORMS', 'doc': 'Binary BPMN conformance. Supports ALLOW().'},
        {'name': 'CONFORMANCE', 'doc': 'Petri net conformance. Use with READABLE().'},
    ],
    'DateTime': [
        {'name': 'DATEDIFF', 'doc': 'Date difference. Units: ms|ss|mi|hh|dd|mm|yy.'},
        {'name': 'HOURS_BETWEEN', 'doc': 'Difference in hours. Supports calendar.'},
        {'name': 'SECONDS_BETWEEN', 'doc': 'Difference in seconds.'},
        {'name': 'WORKDAYS_BETWEEN', 'doc': 'Number of workdays between dates.'},
        {'name': 'DAYS_BETWEEN', 'doc': 'Difference in days (FLOAT).'},
        {'name': 'ADD_DAYS', 'doc': 'Adds days to a date.'},
        {'name': 'ADD_HOURS', 'doc': 'Adds hours.'},
        {'name': 'ADD_WORKDAYS', 'doc': 'Adds workdays using a calendar.'},
        {'name': 'ROUND_DAY', 'doc': 'Rounds down to day.'},
        {'name': 'ROUND_WEEK', 'doc': 'Rounds down to Monday.'},
        {'name': 'ROUND_MONTH', 'doc': 'Rounds down to first day of month.'},
        {'name': 'TODAY', 'doc': 'Current date. Syntax: TODAY([timezone])'},
        {'name': 'CONVERT_TIMEZONE', 'doc': 'Converts between timezones.'},
        {'name': 'WEEKDAY_CALENDAR', 'doc': 'Calendar specifying work weekdays.'},
        {'name': 'DAY', 'doc': 'Day of month (1-31).'},
        {'name': 'MONTH', 'doc': 'Month number (1-12).'},
        {'name': 'YEAR', 'doc': '4-digit year.'},
        {'name': 'QUARTER', 'doc': 'Quarter (1-4).'},
    ],
    'String': [
        {'name': 'UPPER', 'doc': 'Uppercase.'},
        {'name': 'LOWER', 'doc': 'Lowercase.'},
        {'name': 'CONCAT', 'doc': 'Concatenates strings. NULL in any arg = NULL.'},
        {'name': 'STRING_SPLIT', 'doc': 'Splits string. Zero-based index.'},
        {'name': 'TO_STRING', 'doc': 'Converts INT or DATE to STRING.'},
        {'name': 'SUBSTRING', 'doc': 'Extracts substring. 1-based indexing.'},
        {'name': 'LEN', 'doc': 'String length.'},
        {'name': 'IN_LIKE', 'doc': 'Pattern matching with wildcards % and _.'},
        {'name': 'MATCH_STRINGS', 'doc': 'Fuzzy matching by edit distance.'},
        {'name': 'REMAP_VALUES', 'doc': 'Maps STRING values to new values.'},
        {'name': 'LTRIM', 'doc': 'Removes leading whitespace.'},
        {'name': 'RTRIM', 'doc': 'Removes trailing whitespace.'},
    ],
    'Math & Logic': [
        {'name': 'ABS', 'doc': 'Absolute value.'},
        {'name': 'POWER', 'doc': 'Raises to a power.'},
        {'name': 'MODULO', 'doc': 'Remainder of division.'},
        {'name': 'GREATEST', 'doc': 'Maximum across columns.'},
        {'name': 'LEAST', 'doc': 'Minimum across columns.'},
        {'name': 'COALESCE', 'doc': 'First non-NULL value.'},
        {'name': 'ISNULL', 'doc': 'Returns 1 if NULL.'},
        {'name': 'CASE', 'doc': 'CASE WHEN cond THEN val ELSE default END'},
        {'name': 'ROUND', 'doc': 'Rounds to decimal places.'},
        {'name': 'CEIL', 'doc': 'Rounds up.'},
        {'name': 'FLOOR', 'doc': 'Rounds down.'},
        {'name': 'SQRT', 'doc': 'Square root.'},
        {'name': 'LOG', 'doc': 'Natural logarithm.'},
        {'name': 'ZSCORE', 'doc': 'Z-score normalization.'},
    ],
    'Filter & Lookup': [
        {'name': 'FILTER', 'doc': 'Filters result set. Multiple filters merge by AND.'},
        {'name': 'FILTER_TO_NULL', 'doc': 'Makes columns filter-aware. Never inside PU functions.'},
        {'name': 'BIND_FILTERS', 'doc': 'Pulls filter to specified table.'},
        {'name': 'BIND', 'doc': 'Pulls value to target table. Used for 1:N:1 relationships.'},
        {'name': 'IN', 'doc': 'Membership test.'},
        {'name': 'LOOKUP', 'doc': 'Left outer join ignoring predefined joins.'},
        {'name': 'COALESCE', 'doc': 'First non-NULL value.'},
        {'name': 'GENERATE_RANGE', 'doc': 'Creates a value range. Max 10,000.'},
    ],
    'Event Log & OCPM': [
        {'name': 'CREATE_EVENTLOG', 'doc': 'Creates activity table from OCPM perspective.'},
        {'name': 'MERGE_EVENTLOG', 'doc': 'Merges columns from two activity tables.'},
        {'name': 'LINK_PATH', 'doc': 'Traverses object links.'},
        {'name': 'LINK_FILTER', 'doc': 'Filters by ANCESTORS or DESCENDANTS.'},
        {'name': 'LINK_OBJECTS', 'doc': 'All objects in Object Link graph.'},
        {'name': 'UNION_ALL_TABLE', 'doc': 'Vertical concatenation of tables (2-16).'},
        {'name': 'CASE_ID_COLUMN', 'doc': 'References case ID column.'},
        {'name': 'ACTIVITY_TABLE', 'doc': 'References the activity table in OCPM.'},
    ],
    'Currency & Quantity': [
        {'name': 'CURRENCY_CONVERT', 'doc': 'Converts currency using a rates table.'},
        {'name': 'CURRENCY_CONVERT_SAP', 'doc': 'Converts SAP currency.'},
        {'name': 'CURRENCY_SAP', 'doc': 'Adjusts SAP amounts for decimal places.'},
        {'name': 'QUANTITY_CONVERT', 'doc': 'Converts quantity units.'},
    ],
    'ML & Clustering': [
        {'name': 'KMEANS', 'doc': 'K-means++ clustering.'},
        {'name': 'CLUSTER_VARIANTS', 'doc': 'Clusters process variants.'},
        {'name': 'LINEAR_REGRESSION', 'doc': 'Linear regression.'},
        {'name': 'DECISION_TREE', 'doc': 'Decision tree classification.'},
        {'name': 'ZSCORE', 'doc': 'Z-score normalization for outlier detection.'},
        {'name': 'MATCH_STRINGS', 'doc': 'Fuzzy string matching.'},
    ],
    'Static / Meta': [
        {'name': 'COLUMN_TYPE', 'doc': 'Returns data type at query-build time.'},
        {'name': 'ARGUMENT_COUNT', 'doc': 'Counts arguments at query-build time.'},
        {'name': 'USER_NAME', 'doc': 'Returns logged-in username.'},
        {'name': 'UNIQUE_ID', 'doc': 'Unique INT for each unique tuple.'},
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
    (r'(rework|repeat|loop|same.?activit|revisit|multiple.?time)', ['CALC_REWORK', 'INDEX_ACTIVITY_LOOP', 'INDEX_ACTIVITY_TYPE']),
    (r'(conform|path|sequence|order.*activit|activit.*order|follow)', ['MATCH_PROCESS', 'MATCH_ACTIVITIES', 'BPMN_CONFORMS']),
    (r'(days?\s+between|hours?\s+between|date.?diff|workday|calendar)', ['DATEDIFF', 'HOURS_BETWEEN', 'WORKDAYS_BETWEEN', 'REMAP_TIMESTAMPS']),
    (r'(automat|system.?activit|manual.?activit|bot|automation.?rate)', ['PU_COUNT', 'CALC_REWORK', 'GLOBAL']),
    (r'(variant|process.?flow|happy.?path)', ['VARIANT', 'MATCH_PROCESS', 'MATCH_PROCESS_REGEX']),
    (r'(running|cumulative|rolling|window|moving)', ['RUNNING_TOTAL', 'WINDOW_AVG', 'MOVING_AVG', 'INDEX_ORDER']),
    (r'(filter|where|only.*cases|exclude)', ['FILTER', 'MATCH_ACTIVITIES', 'BIND_FILTERS', 'FILTER_TO_NULL']),
    (r'(ocpm|object.?centric|multi.?object|link)', ['LINK_PATH', 'LINK_FILTER', 'CREATE_EVENTLOG']),
    (r'(sap|currency|amount|convert)', ['CURRENCY_CONVERT', 'CURRENCY_CONVERT_SAP', 'CURRENCY_SAP']),
    (r'(cluster|segment|kmeans|ml|machine.?learn|predict|regression)', ['KMEANS', 'CLUSTER_VARIANTS', 'LINEAR_REGRESSION']),
    (r'(outlier|z.?score|anomaly|abnormal)', ['ZSCORE', 'TRIMMED_MEAN', 'BUCKET_UPPER_BOUND']),
    (r'(median|percentile|quantile|p\d\d)', ['MEDIAN', 'QUANTILE', 'PU_MEDIAN', 'PU_QUANTILE']),
    (r'(lag|lead|previous|next).*(activit|event|step)', ['ACTIVITY_LAG', 'ACTIVITY_LEAD']),
    (r'(first|last).*(occurrence|time|activit)', ['PU_FIRST', 'PU_LAST', 'FIRST', 'LAST', 'CALC_THROUGHPUT']),
    (r'(NULL|missing|empty|blank|coalesce|fill)', ['COALESCE', 'ISNULL', 'INTERPOLATE', 'FILTER_TO_NULL']),
    (r'(on.?time|overdue|delay|late|sla|due)', ['DATEDIFF', 'HOURS_BETWEEN', 'WORKDAYS_BETWEEN', 'CALC_THROUGHPUT']),
    (r'(bottleneck|slow|wait|queue)', ['CALC_THROUGHPUT', 'ACTIVITY_LAG', 'SECONDS_BETWEEN', 'PU_AVG']),
    (r'(o2c|order.?to.?cash|order management)', ['PU_COUNT', 'CALC_THROUGHPUT', 'MATCH_ACTIVITIES', 'DATEDIFF']),
    (r'(p2p|procure.?to.?pay|purchase)', ['PU_COUNT', 'CALC_THROUGHPUT', 'MATCH_PROCESS', 'DATEDIFF']),
    (r'(touchless|straight.?through)', ['PU_COUNT', 'CALC_REWORK', 'MATCH_ACTIVITIES']),
]

def detect_functions(text: str):
    text_lower = text.lower()
    found = set()
    NEEDS_WORD_BOUNDARY = {
        'AVG', 'SUM', 'MAX', 'MIN', 'VAR', 'IN', 'OR', 'AND', 'NOT',
        'ADD', 'LOG', 'LEN', 'ABS', 'CEIL', 'FLOOR', 'ROUND', 'SQRT',
        'FIRST', 'LAST', 'MODE', 'DAY', 'MONTH', 'YEAR', 'HOURS', 'MINUTES',
        'SECONDS', 'MILLIS', 'QUARTER', 'CASE', 'WHEN', 'LIKE', 'REVERSE',
        'BETWEEN', 'STDEV', 'COUNT', 'FILTER', 'BIND', 'LOOKUP', 'UPPER',
        'LOWER', 'PRODUCT', 'VARIANT', 'CONSTANT', 'FORMAT', 'MEDIAN', 'QUANTILE',
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
# SECTION 3 · SCHEMA LAYER  ← NEW in v3
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SchemaRelation:
    parent: str
    child: str
    relation_type: str  # 1:N | N:1 | N:M | 1:1

# Built-in schema knowledge for common Celonis data models
BUILT_IN_SCHEMA: List[SchemaRelation] = [
    # Generic
    SchemaRelation("CASES", "ACTIVITIES", "1:N"),
    SchemaRelation("CASES", "ACTIVITIES", "1:N"),
    # O2C
    SchemaRelation("VBAK", "VBAP", "1:N"),
    SchemaRelation("VBAK", "VBEP", "1:N"),
    SchemaRelation("VBAK", "LIPS", "1:N"),
    SchemaRelation("KNA1", "VBAK", "1:N"),
    # P2P
    SchemaRelation("EKKO", "EKPO", "1:N"),
    SchemaRelation("EKKO", "EKES", "1:N"),
    SchemaRelation("LFA1", "EKKO", "1:N"),
    SchemaRelation("EKPO", "MSEG", "1:N"),
    # Finance
    SchemaRelation("BKPF", "BSEG", "1:N"),
    SchemaRelation("KUNNR", "BSEG", "1:N"),
    # Generic vendor/order
    SchemaRelation("VENDORS", "ORDERS", "1:N"),
    SchemaRelation("ORDERS", "ACTIVITIES", "1:N"),
    SchemaRelation("ORDERS", "ORDER_ITEMS", "1:N"),
    # HR
    SchemaRelation("EMPLOYEES", "ACTIVITIES", "1:N"),
    SchemaRelation("EMPLOYEES", "PAYROLL", "1:N"),
]

class SchemaValidator:
    def __init__(self, relations: List[SchemaRelation] = None):
        self.relations = relations or BUILT_IN_SCHEMA
        self._relation_map: Dict[Tuple[str, str], str] = {}
        for r in self.relations:
            self._relation_map[(r.parent.upper(), r.child.upper())] = r.relation_type

    def get_relationship(self, table_a: str, table_b: str) -> Optional[str]:
        a, b = table_a.upper(), table_b.upper()
        if (a, b) in self._relation_map:
            return self._relation_map[(a, b)]
        if (b, a) in self._relation_map:
            rel = self._relation_map[(b, a)]
            return "N:1" if rel == "1:N" else rel
        return None

    def validate_pu_direction(self, target: str, source: str) -> Optional[str]:
        """Check PU direction: target must be parent (1-side), source must be child (N-side)."""
        rel = self.get_relationship(target, source)
        if rel is None:
            return None  # Unknown relationship — can't validate
        if rel == "1:N":
            return None  # Correct direction
        if rel == "N:1":
            return f"⚠ PU direction may be reversed: '{target}' is the CHILD (N-side) of '{source}'. PU target must be the parent (1-side). Consider swapping: PU_*(\"{source}\", \"{target}\".\"COL\")"
        if rel == "N:M":
            return f"⚠ N:M relationship between '{target}' and '{source}'. PU functions require 1:N — consider using BIND() or an intermediate table."
        return None

SCHEMA = SchemaValidator()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 · INTENT ENGINE  ← NEW in v3
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PQLIntent:
    goal: Optional[str] = None          # throughput | rework | conformance | count | sum | avg | filter | variant | kpi
    target_level: Optional[str] = None  # CASE | ACTIVITY | VENDOR | ORDER | OBJECT
    metric_type: Optional[str] = None   # count | sum | time | rate | ratio | flag
    domain: Optional[str] = None        # O2C | P2P | HR | GENERIC
    is_aggregation: bool = False
    is_comparison: bool = False
    is_filter_query: bool = False
    is_time_query: bool = False
    wants_percentage: bool = False
    wants_automation: bool = False
    wants_rework: bool = False
    wants_throughput: bool = False
    confidence: float = 0.0             # 0.0-1.0


def detect_intent(query: str) -> PQLIntent:
    """
    Analyse user query and detect intent: what they're trying to achieve.
    This is the core of the reasoning upgrade vs pure rule-based validation.
    """
    q = query.lower()
    intent = PQLIntent()
    signals = 0

    # ── Goal detection ────────────────────────────────────────────────────────
    throughput_patterns = [r'throughput', r'cycle.?time', r'lead.?time', r'duration',
                           r'how long', r'time.?between', r'elapsed', r'from.+to']
    if any(re.search(p, q) for p in throughput_patterns):
        intent.goal = "throughput"
        intent.metric_type = "time"
        intent.wants_throughput = True
        intent.is_time_query = True
        signals += 3

    rework_patterns = [r'rework', r'repeated', r'loop', r'same activit', r'multiple time',
                       r'more than once', r'revisit', r'duplicate activit']
    if any(re.search(p, q) for p in rework_patterns):
        intent.goal = "rework"
        intent.wants_rework = True
        signals += 3

    conformance_patterns = [r'conform', r'compliance', r'bpmn', r'petri', r'deviation',
                            r'follows.*process', r'correct.*sequence', r'path.*correct']
    if any(re.search(p, q) for p in conformance_patterns):
        intent.goal = "conformance"
        signals += 3

    automation_patterns = [r'automat', r'system activit', r'bot', r'straight.?through',
                           r'touchless', r'manual vs', r'human activit']
    if any(re.search(p, q) for p in automation_patterns):
        intent.wants_automation = True
        if not intent.goal:
            intent.goal = "automation_rate"
        signals += 2

    variant_patterns = [r'variant', r'happy path', r'process flow', r'sequence of', r'path']
    if any(re.search(p, q) for p in variant_patterns):
        intent.goal = "variant_analysis"
        signals += 2

    kpi_patterns = [r'\bkpi\b', r'metric', r'performance indicator', r'dashboard', r'scorecard']
    if any(re.search(p, q) for p in kpi_patterns):
        intent.goal = intent.goal or "kpi"
        signals += 1

    # ── Target level ──────────────────────────────────────────────────────────
    if re.search(r'\b(per case|case.?level|each case|by case)\b', q):
        intent.target_level = "CASE"
        signals += 2
    elif re.search(r'\b(per activit|activit.?level|each activit|by activit)\b', q):
        intent.target_level = "ACTIVITY"
        signals += 2
    elif re.search(r'\b(per vendor|by vendor|vendor.?level|supplier.?level)\b', q):
        intent.target_level = "VENDOR"
        signals += 2
    elif re.search(r'\b(per order|by order|order.?level)\b', q):
        intent.target_level = "ORDER"
        signals += 2
    elif re.search(r'"ACTIVITIES"', query):
        intent.target_level = "ACTIVITY"
        signals += 1
    elif re.search(r'"CASES"', query):
        intent.target_level = "CASE"
        signals += 1

    # ── Metric type ───────────────────────────────────────────────────────────
    if not intent.metric_type:
        if re.search(r'\b(count|how many|number of)\b', q):
            intent.metric_type = "count"
            intent.is_aggregation = True
        elif re.search(r'\b(sum|total|sum up)\b', q):
            intent.metric_type = "sum"
            intent.is_aggregation = True
        elif re.search(r'\b(average|avg|mean)\b', q):
            intent.metric_type = "avg"
            intent.is_aggregation = True
        elif re.search(r'\b(rate|ratio|percentage|percent|%)\b', q):
            intent.metric_type = "rate"
            intent.wants_percentage = True
            intent.is_comparison = True

    # ── Domain ────────────────────────────────────────────────────────────────
    if re.search(r'\b(o2c|order.?to.?cash|sales order|delivery|invoice.*sales)\b', q):
        intent.domain = "O2C"
    elif re.search(r'\b(p2p|procure.?to.?pay|purchase order|po |vendor.*invoice|goods receipt)\b', q):
        intent.domain = "P2P"
    elif re.search(r'\b(h2r|hire.?to.?retire|employee|payroll|onboard)\b', q):
        intent.domain = "HR"
    elif re.search(r'\b(a2r|account.?to.?report|finance|gl|ledger|bkpf)\b', q):
        intent.domain = "FINANCE"
    else:
        intent.domain = "GENERIC"

    # ── Flags ─────────────────────────────────────────────────────────────────
    intent.is_filter_query = bool(re.search(r'\b(filter|only|exclude|where|when)\b', q))
    intent.is_comparison = intent.is_comparison or bool(re.search(r'\b(compare|vs|versus|difference between|above average|below)\b', q))

    intent.confidence = min(1.0, signals / 6.0)
    return intent


def format_intent_summary(intent: PQLIntent) -> str:
    """Format intent as a human-readable insight block."""
    parts = []
    if intent.goal:
        parts.append(f"**Goal:** {intent.goal.replace('_', ' ').title()}")
    if intent.target_level:
        parts.append(f"**Result level:** {intent.target_level} level")
    if intent.metric_type:
        parts.append(f"**Metric type:** {intent.metric_type}")
    if intent.domain and intent.domain != "GENERIC":
        parts.append(f"**Domain:** {intent.domain}")
    flags = []
    if intent.wants_throughput:
        flags.append("throughput")
    if intent.wants_rework:
        flags.append("rework")
    if intent.wants_automation:
        flags.append("automation rate")
    if intent.wants_percentage:
        flags.append("percentage/ratio")
    if intent.is_comparison:
        flags.append("comparison")
    if flags:
        parts.append(f"**Patterns detected:** {', '.join(flags)}")
    return "\n".join(parts) if parts else ""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 · EXECUTION SIMULATOR  ← NEW in v3
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecutionPlan:
    common_table: Optional[str] = None
    join_shift: bool = False
    aggregation_risk: bool = False
    grain_conflict: bool = False
    tables_involved: Set[str] = field(default_factory=set)
    mixed_levels: bool = False
    global_required: bool = False
    global_present: bool = False
    execution_warnings: List[str] = field(default_factory=list)


def simulate_execution(query: str) -> ExecutionPlan:
    """
    Simulate how Celonis will execute the query at runtime.
    Detects join shifts, aggregation grain conflicts, GLOBAL() necessity.
    """
    plan = ExecutionPlan()
    q = query.upper()

    # Detect tables
    table_patterns = {
        'ACTIVITIES': [r'"ACTIVITIES"', r'"_CEL_ACTIVITIES"'],
        'CASES': [r'"CASES"', r'"_CEL_CASES"'],
        'VENDORS': [r'"VENDORS?"', r'"LFA1"'],
        'ORDERS': [r'"ORDERS?"', r'"EKKO"', r'"VBAK"'],
        'ORDER_ITEMS': [r'"EKPO"', r'"VBAP"', r'"ORDER_ITEMS"'],
        'PAYMENTS': [r'"BKPF"', r'"BSEG"'],
    }

    for tbl, patterns in table_patterns.items():
        for pat in patterns:
            if re.search(pat, q):
                plan.tables_involved.add(tbl)
                break

    # Determine common table (Celonis shifts to lowest level)
    if 'ACTIVITIES' in plan.tables_involved:
        plan.common_table = 'ACTIVITIES'  # Activities always pulls to lowest level
    elif 'ORDER_ITEMS' in plan.tables_involved:
        plan.common_table = 'ORDER_ITEMS'
    elif 'ORDERS' in plan.tables_involved:
        plan.common_table = 'ORDERS'
    elif 'CASES' in plan.tables_involved:
        plan.common_table = 'CASES'

    # Detect join shift risk
    has_case_columns = bool(re.search(r'"CASES?"\.', q))
    has_activity_columns = bool(re.search(r'"ACTIVITIES?"\.', q))

    if has_case_columns and has_activity_columns:
        plan.join_shift = True
        plan.mixed_levels = True
        plan.aggregation_risk = True
        plan.global_required = True

    # Check if GLOBAL is present
    plan.global_present = 'GLOBAL(' in q

    # Detect grain conflict: aggregation on case-level columns mixed with activity-level
    std_aggs = ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(', 'MEDIAN(']
    has_agg = any(agg in q for agg in std_aggs)

    if plan.mixed_levels and has_agg and not plan.global_present:
        plan.grain_conflict = True
        plan.execution_warnings.append(
            "JOIN SHIFT DETECTED: Query mixes CASES and ACTIVITIES columns. "
            "Celonis will shift the common table to ACTIVITIES level, causing case-level "
            "aggregations to be multiplied by the number of activities per case. "
            "Wrap case-level aggregations with GLOBAL()."
        )

    if 'CALC_THROUGHPUT' in q and has_agg and not plan.global_present:
        plan.execution_warnings.append(
            "THROUGHPUT GRAIN RISK: CALC_THROUGHPUT returns a case-level value. "
            "Combined with standard aggregations and activity-level columns, "
            "GLOBAL() is required: GLOBAL(AVG(CALC_THROUGHPUT(...)))"
        )

    return plan


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 · REASONING COMBINATOR  ← NEW in v3
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReasoningInsight:
    category: str       # JOIN_RISK | INTENT_MISMATCH | OPTIMIZATION | CORRECTNESS | SUGGESTION
    severity: str       # HIGH | MEDIUM | LOW
    title: str
    explanation: str
    recommendation: str


def reason_about_query(query: str, intent: PQLIntent, plan: ExecutionPlan) -> List[ReasoningInsight]:
    """
    Combine intent + execution simulation + rules to generate smart insights.
    This is the core reasoning layer that elevates the app from rule-based to reasoning-based.
    """
    insights: List[ReasoningInsight] = []
    q = query.upper()

    # ── Insight 1: Join multiplication warning ────────────────────────────────
    if plan.grain_conflict:
        insights.append(ReasoningInsight(
            category="JOIN_RISK",
            severity="HIGH",
            title="Aggregation grain conflict detected",
            explanation=(
                f"Your query mixes columns from CASES and ACTIVITIES tables. "
                f"Celonis shifts the execution to ACTIVITIES level ({plan.common_table}), "
                f"which means any COUNT or SUM on CASES columns will be repeated once per activity. "
                f"A case with 5 activities will contribute its values 5× to the aggregation."
            ),
            recommendation=(
                "Wrap all case-level aggregations with GLOBAL():\n"
                "  • GLOBAL(COUNT(\"CASES\".\"CASE_ID\"))\n"
                "  • GLOBAL(AVG(\"CASES\".\"AMOUNT\"))\n"
                "  • GLOBAL(AVG(CALC_THROUGHPUT(...)))"
            )
        ))

    # ── Insight 2: Intent mismatch — counting activities when cases expected ──
    if (intent.metric_type == "count" and intent.target_level == "CASE"
            and re.search(r'COUNT\s*\(\s*"ACTIVITIES"', q)
            and 'GLOBAL' not in q):
        insights.append(ReasoningInsight(
            category="INTENT_MISMATCH",
            severity="HIGH",
            title="You are counting ACTIVITIES, but your intent looks like counting CASES",
            explanation=(
                "COUNT(\"ACTIVITIES\".\"CASE_ID\") at ACTIVITY level counts event rows, "
                "not unique cases. If you want the number of distinct cases, "
                "you need COUNT(\"CASES\".\"CASE_ID\") or COUNT_TABLE(\"CASES\")."
            ),
            recommendation=(
                "For unique case count: COUNT(\"CASES\".\"CASE_ID\")\n"
                "For activity count per case: PU_COUNT(\"CASES\", \"ACTIVITIES\".\"CASE_ID\")\n"
                "For total activity rows: COUNT(\"ACTIVITIES\".\"CASE_ID\")"
            )
        ))

    # ── Insight 3: Throughput without REMAP_TIMESTAMPS ───────────────────────
    if intent.wants_throughput and 'DATEDIFF' in q and 'CALC_THROUGHPUT' not in q:
        insights.append(ReasoningInsight(
            category="SUGGESTION",
            severity="MEDIUM",
            title="Consider CALC_THROUGHPUT instead of DATEDIFF for case throughput",
            explanation=(
                "DATEDIFF is good for date arithmetic on stored columns. But for process "
                "throughput (start to end of a case), CALC_THROUGHPUT is the official "
                "recommended function — it handles NULL cases, supports calendars (working hours), "
                "and integrates with FIRST/LAST_OCCURRENCE specifiers."
            ),
            recommendation=(
                "Use: AVG(CALC_THROUGHPUT(CASE_START TO CASE_END, "
                "REMAP_TIMESTAMPS(\"ACTIVITIES\".\"TIMESTAMP\", DAYS)))\n"
                "With working hours: REMAP_TIMESTAMPS(..., HOURS, WEEKDAY_CALENDAR(MON,TUE,WED,THU,FRI))"
            )
        ))

    # ── Insight 4: Rework without proper function ─────────────────────────────
    if intent.wants_rework and 'CALC_REWORK' not in q and 'INDEX_ACTIVITY_LOOP' not in q:
        insights.append(ReasoningInsight(
            category="SUGGESTION",
            severity="MEDIUM",
            title="Rework intent detected — use CALC_REWORK or INDEX_ACTIVITY_LOOP",
            explanation=(
                "Detecting repeated/rework activities requires specific PQL functions. "
                "COUNT alone won't correctly identify repeated activities within the same case."
            ),
            recommendation=(
                "For case-level rework count: CALC_REWORK(\"ACTIVITIES\".\"ACTIVITY\" = 'Review')\n"
                "For row-level rework flag: INDEX_ACTIVITY_LOOP(\"ACTIVITIES\".\"ACTIVITY\") > 0\n"
                "Filter rework cases: FILTER CALC_REWORK() > PU_COUNT(\"CASES\", \"ACTIVITIES\".\"CASE_ID\")"
            )
        ))

    # ── Insight 5: Automation rate — missing denominator ─────────────────────
    if intent.wants_automation:
        has_numerator = bool(re.search(r"PU_COUNT.*SYSTEM|PU_COUNT.*BOT|PU_COUNT.*USER.*=.*'SYSTEM'", q, re.I))
        has_denominator = bool('CALC_REWORK' in q or re.search(r'PU_COUNT.*CASE_ID', q))
        if has_numerator and not has_denominator:
            insights.append(ReasoningInsight(
                category="CORRECTNESS",
                severity="HIGH",
                title="Automation rate missing denominator",
                explanation=(
                    "You're counting system activities (numerator) but there's no total activity "
                    "count (denominator). An automation rate requires: system_count / total_count × 100."
                ),
                recommendation=(
                    "Full automation rate:\n"
                    "  ROUND(\n"
                    "    PU_COUNT(\"CASES\", \"ACTIVITIES\".\"CASE_ID\", \"ACTIVITIES\".\"USER\" = 'SYSTEM') * 100.0\n"
                    "    / CALC_REWORK(),\n"
                    "    1\n"
                    "  )"
                )
            ))

    # ── Insight 6: Percentage without × 100 ──────────────────────────────────
    if intent.wants_percentage and not re.search(r'\*\s*100', q):
        insights.append(ReasoningInsight(
            category="CORRECTNESS",
            severity="MEDIUM",
            title="Percentage calculation missing × 100",
            explanation=(
                "Your query appears to calculate a ratio/percentage, but there's no multiplication by 100. "
                "PQL division returns a decimal (0.0-1.0), not a percentage."
            ),
            recommendation="Multiply the ratio by 100: (numerator / denominator) * 100"
        ))

    # ── Insight 7: GLOBAL inside FILTER ──────────────────────────────────────
    if re.search(r'\bFILTER\b.*GLOBAL\s*\(', q, re.DOTALL):
        insights.append(ReasoningInsight(
            category="CORRECTNESS",
            severity="HIGH",
            title="GLOBAL() inside FILTER — not allowed in Celonis",
            explanation=(
                "GLOBAL() produces a scalar value used for comparison. "
                "Celonis does not support GLOBAL() inside FILTER clauses."
            ),
            recommendation=(
                "Move GLOBAL() outside FILTER. Use CASE WHEN instead:\n"
                "  CASE WHEN SUM(\"ORDERS\".\"AMOUNT\") > GLOBAL(SUM(\"ORDERS\".\"AMOUNT\"))\n"
                "       THEN 'Above Average' ELSE 'Below Average' END"
            )
        ))

    # ── Insight 8: Domain-specific suggestions ────────────────────────────────
    if intent.domain == "P2P" and intent.wants_throughput:
        insights.append(ReasoningInsight(
            category="SUGGESTION",
            severity="LOW",
            title="P2P domain: consider working-days throughput for business relevance",
            explanation=(
                "In Procure-to-Pay processes, throughput in calendar days often includes weekends "
                "and holidays that are not business-relevant. Working-hours throughput gives more "
                "meaningful SLA analysis."
            ),
            recommendation=(
                "CALC_THROUGHPUT(CASE_START TO CASE_END, "
                "REMAP_TIMESTAMPS(\"ACTIVITIES\".\"TIMESTAMP\", HOURS, WEEKDAY_CALENDAR(MON,TUE,WED,THU,FRI)))"
            )
        ))

    if intent.domain == "O2C" and intent.wants_rework:
        insights.append(ReasoningInsight(
            category="SUGGESTION",
            severity="LOW",
            title="O2C domain: consider credit block as rework signal",
            explanation=(
                "In Order-to-Cash, credit blocks (CREDIT_BLOCK activity) are a common rework pattern. "
                "MATCH_ACTIVITIES with EXCLUDING or NODE can target this specifically."
            ),
            recommendation=(
                "FILTER MATCH_ACTIVITIES(NODE('Credit Block')) = 1\n"
                "Or: CASE WHEN PU_COUNT(\"CASES\", \"ACTIVITIES\".\"CASE_ID\", \"ACTIVITIES\".\"ACTIVITY\" = 'Credit Block') > 0 "
                "THEN 'Blocked' ELSE 'Clean' END"
            )
        ))

    return insights


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 · VALIDATION ISSUE DATACLASS (from v2, enhanced)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    severity: str      # CRITICAL | ERROR | WARNING | INFO | PERF
    code: str
    message: str
    why: str
    fix: str
    auto_fixable: bool = False
    safe_fix: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 · AST PARSER (unchanged — battle-tested)
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
# SECTION 9 · CONTEXT ENGINE (from v2, enhanced)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PQLContext:
    used_functions: Set[str] = field(default_factory=set)
    pu_functions: Set[str] = field(default_factory=set)
    table_levels: Set[str] = field(default_factory=set)
    has_filter: bool = False
    has_global: bool = False
    has_pu: bool = False
    has_match: bool = False
    has_conformance: bool = False
    has_calc_throughput: bool = False
    has_calc_rework: bool = False
    has_running: bool = False
    has_sql_keywords: bool = False
    is_match_safe: bool = False
    aggregation_fns: Set[str] = field(default_factory=set)
    standard_agg_fns: Set[str] = field(default_factory=set)


def extract_context(query: str) -> PQLContext:
    ctx = PQLContext()
    q = query.upper()

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

    STD_AGGS = {'AVG', 'SUM', 'COUNT', 'MEDIAN', 'MIN', 'MAX', 'STDEV', 'MODE'}
    for fn in STD_AGGS:
        if re.search(r'\b' + fn + r'\s*\(', q):
            ctx.aggregation_fns.add(fn)
            ctx.standard_agg_fns.add(fn)

    TABLE_PATTERNS = {
        'CASE':     [r'"CASES?"', r'"_CEL_CASES?"', r'CASE_TABLE\('],
        'ACTIVITY': [r'"ACTIVITIES?"', r'"_CEL_ACTIVITIES?"', r'ACTIVITY_TABLE\('],
        'OBJECT':   [r'LINK_PATH\(', r'LINK_FILTER\('],
        'VENDOR':   [r'"VENDORS?"', r'"LFA1"'],
        'ORDER':    [r'"ORDERS?"', r'"EKKO"', r'"VBAK"'],
    }
    for level, patterns in TABLE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, q):
                ctx.table_levels.add(level)
                break

    SQL_KWS = ['SELECT', 'FROM', 'JOIN', r'LEFT\s+JOIN', r'GROUP\s+BY', 'HAVING', r'\bWITH\b']
    ctx.has_sql_keywords = any(re.search(r'\b' + kw + r'\b', q) for kw in SQL_KWS)

    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 · RULE ENGINE (from v2, retained + schema validation added)
# ─────────────────────────────────────────────────────────────────────────────

def rule_sql_keywords(ctx, query):
    if not ctx.has_sql_keywords:
        return []
    SQL_KWS = ['SELECT', 'FROM', 'JOIN', 'LEFT JOIN', 'GROUP BY', 'HAVING', 'WITH']
    found = [kw for kw in SQL_KWS if re.search(r'\b' + re.escape(kw.replace(' ', r'\s+')), query, re.I)]
    return [ValidationIssue("CRITICAL", "SQL001",
        f"SQL keyword(s) detected: {', '.join(found)}",
        "PQL is column-based — no SELECT/FROM/JOIN/GROUP BY. SQL keywords cause parse errors in Celonis Studio.",
        "Remove all SQL. Use PU_* for cross-table aggregation, FILTER for row filtering, GLOBAL() for cross-level aggregation.",
        False)]

def rule_filter_inside_pu_text(ctx, query):
    if not ctx.has_pu:
        return []
    issues = []
    pattern = re.compile(r'(PU_\w+)\s*\([^)]*\bFILTER\b[^)]*\)', re.IGNORECASE | re.DOTALL)
    for m in pattern.finditer(query):
        issues.append(ValidationIssue("CRITICAL", "PU001",
            f"FILTER keyword inside {m.group(1).upper()}()",
            "PU functions execute on raw data before global filters. Embedding FILTER inside parentheses is a syntax error.",
            f"Use the 3rd positional argument: {m.group(1).upper()}(target_table, source_col, your_condition_here)",
            False))
    return issues

def rule_filter_to_null_inside_pu(ctx, ast):
    if not ctx.has_pu:
        return []
    issues = []
    pu_nodes = ast_find_functions(ast, "PU_")
    for pu in pu_nodes:
        for child in pu.get("children", []):
            if child.get("name", "").upper() == "FILTER_TO_NULL":
                issues.append(ValidationIssue("CRITICAL", "PU002",
                    f"FILTER_TO_NULL inside {pu['name']}()",
                    "FILTER_TO_NULL inside PU functions runs at wrong scope and produces incorrect results.",
                    f"Replace FILTER_TO_NULL(col) with direct boolean condition: {pu['name']}(target, source_col, condition)",
                    False))
    return issues

def rule_pu_arg_count(ctx, ast):
    if not ctx.has_pu:
        return []
    issues = []
    pu_nodes = ast_find_functions(ast, "PU_")
    for pu in pu_nodes:
        total = len(pu.get("args", [])) + len(pu.get("children", []))
        if total < 2:
            issues.append(ValidationIssue("ERROR", "PU003",
                f"{pu['name']}() has fewer than 2 arguments",
                "PU functions require at minimum: target_table (parent/1-side) and source_table.column (child/N-side).",
                f"Syntax: {pu['name']}(\"TARGET_TABLE\", \"SOURCE_TABLE\".\"COLUMN\" [, filter])",
                False))
    return issues

def rule_pu_schema_direction(ctx, query):
    """NEW v3: validate PU direction against schema."""
    if not ctx.has_pu:
        return []
    issues = []
    pattern = re.compile(r'PU_\w+\s*\(\s*"([^"]+)"\s*,\s*"([^"]+)"\."([^"]+)"', re.IGNORECASE)
    for m in pattern.finditer(query):
        target = m.group(1)
        source_table = m.group(2)
        warning = SCHEMA.validate_pu_direction(target, source_table)
        if warning:
            issues.append(ValidationIssue("WARNING", "SCH001",
                f"Possible PU direction issue: {target} → {source_table}",
                warning,
                f"Check relationship: {target} should be the 1-side (parent), {source_table} should be N-side (child).",
                False))
    return issues

def rule_missing_global_mixed_levels(ctx):
    mixing = 'CASE' in ctx.table_levels and 'ACTIVITY' in ctx.table_levels
    if not mixing or ctx.has_global:
        return []
    if ctx.aggregation_fns or (ctx.has_pu and ctx.standard_agg_fns):
        return [ValidationIssue("WARNING", "GL001",
            "Missing GLOBAL() — mixing CASE and ACTIVITY level columns",
            "Celonis shifts common table to activity level, multiplying case-level aggregations by activities per case.",
            "Wrap case-level aggregations with GLOBAL():\n  GLOBAL(COUNT(\"CASES\".\"CASE_ID\"))\n  GLOBAL(AVG(CALC_THROUGHPUT(...)))",
            False)]
    return []

def rule_calc_throughput_needs_global(ctx, query):
    if not ctx.has_calc_throughput or ctx.has_global or not ctx.aggregation_fns:
        return []
    return [ValidationIssue("WARNING", "GL002",
        "CALC_THROUGHPUT + aggregation likely needs GLOBAL()",
        "CALC_THROUGHPUT returns a case-level value. Mixed with activity-level columns, common table shifts causing re-evaluation per activity.",
        "Wrap: GLOBAL(AVG(CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS(..., DAYS))))",
        False)]

def rule_global_inside_filter(ctx, query):
    if not ctx.has_global:
        return []
    if re.search(r'\bFILTER\b.*?GLOBAL\s*\(', query, re.IGNORECASE | re.DOTALL):
        return [ValidationIssue("CRITICAL", "GL003",
            "GLOBAL() used inside a FILTER statement",
            "GLOBAL() is not supported inside FILTER clauses — will throw a parse error.",
            "Move GLOBAL() outside FILTER. Use CASE WHEN:\n  CASE WHEN agg > GLOBAL(agg) THEN ... END",
            False)]
    return []

def rule_outer_pu_wrapping_inner_pu_datediff(ctx, query):
    pattern = re.compile(r'(PU_\w+)\s*\(\s*("[\w\s]+")\s*,\s*(?:DATEDIFF|HOURS_BETWEEN|SECONDS_BETWEEN)', re.IGNORECASE)
    issues = []
    for m in pattern.finditer(query):
        outer_fn = m.group(1).upper()
        outer_table = m.group(2)
        inner_pattern = re.compile(r'PU_(?:FIRST|LAST|MIN|MAX|AVG)\s*\(\s*' + re.escape(outer_table), re.IGNORECASE)
        if inner_pattern.search(query):
            issues.append(ValidationIssue("CRITICAL", "PU004",
                f"Double-PU aggregation: outer {outer_fn} wraps DATEDIFF of inner PU on same table {outer_table}",
                "PU_FIRST/PU_LAST already return scalar values. Wrapping in another PU on same target applies aggregation twice — wrong results without runtime error.",
                f"Remove outer PU. Use: DATEDIFF('dd', PU_FIRST({outer_table}, ...), PU_LAST({outer_table}, ...))",
                False))
    return issues

def rule_running_sum_deprecated(ctx, query):
    if 'RUNNING_SUM' not in ctx.used_functions:
        return []
    return [ValidationIssue("INFO", "DEP001", "RUNNING_SUM is deprecated",
        "RUNNING_SUM was an older alias. The official function is now RUNNING_TOTAL.",
        "Replace RUNNING_SUM(...) with RUNNING_TOTAL(...) — syntax identical.",
        True, "Replace RUNNING_SUM with RUNNING_TOTAL")]

def rule_process_order_deprecated(ctx, query):
    if 'PROCESS_ORDER' not in ctx.used_functions:
        return []
    return [ValidationIssue("INFO", "DEP002", "PROCESS_ORDER is deprecated",
        "PROCESS_ORDER was removed in newer PQL versions.",
        "Replace PROCESS_ORDER(...) with INDEX_ACTIVITY_ORDER(...).",
        True, "Replace PROCESS_ORDER with INDEX_ACTIVITY_ORDER")]

def rule_all_occurrence_deprecated(query):
    if not re.search(r'ALL_OCCURRENCE\s*\[', query, re.IGNORECASE):
        return []
    return [ValidationIssue("WARNING", "DEP003", "ALL_OCCURRENCE['…'] is deprecated since PQL 4.6",
        "ALL_OCCURRENCE was removed in PQL 4.6. Causes parse error in newer Celonis versions.",
        "Replace ALL_OCCURRENCE['…'] with CASE_START or CASE_END.",
        False)]

def rule_pu_count_distinct_on_likely_key(ctx, query):
    if 'PU_COUNT_DISTINCT' not in ctx.used_functions:
        return []
    pattern = re.compile(r'PU_COUNT_DISTINCT\s*\([^,]+,\s*"[^"]+"\."([^"]+)"', re.IGNORECASE)
    issues = []
    for m in pattern.finditer(query):
        col = m.group(1).upper()
        if re.search(r'(_ID|_KEY|_NO|_NUM|CASE_ID|ORDER_ID|VENDOR_ID)$', col):
            issues.append(ValidationIssue("PERF", "PERF001",
                f"PU_COUNT_DISTINCT on key column \"{col}\" — use PU_COUNT instead",
                f"Column \"{col}\" looks like a primary key. PU_COUNT_DISTINCT does expensive sort+dedup. PU_COUNT is equivalent and much cheaper.",
                "Replace PU_COUNT_DISTINCT with PU_COUNT for key columns.",
                True, "Replace PU_COUNT_DISTINCT with PU_COUNT for key columns"))
    return issues

def rule_pu_median_performance(ctx):
    if 'PU_MEDIAN' not in ctx.used_functions:
        return []
    return [ValidationIssue("PERF", "PERF002", "PU_MEDIAN — consider PU_AVG for better performance",
        "PU_MEDIAN requires a full sort per target row — 5-10x more expensive than PU_AVG.",
        "Replace PU_MEDIAN with PU_AVG unless true statistical median is required.",
        False)]

def rule_pu_first_last_no_order_by(ctx, query):
    issues = []
    for fn in ['PU_FIRST', 'PU_LAST']:
        if fn not in ctx.used_functions:
            continue
        pattern = re.compile(fn + r'\s*\([^)]*\)', re.IGNORECASE | re.DOTALL)
        for m in pattern.finditer(query):
            if 'ORDER BY' not in m.group(0).upper():
                issues.append(ValidationIssue("WARNING", "PU005",
                    f"{fn}() without ORDER BY — non-deterministic result",
                    f"Without ORDER BY, {fn}() depends on physical row order — not guaranteed consistent across runs.",
                    f"Add ORDER BY: {fn}(\"CASES\", \"ACTIVITIES\".\"TIMESTAMP\", ORDER BY \"ACTIVITIES\".\"TIMESTAMP\" ASC)",
                    False))
    return issues

def rule_quotes_enforcement(ctx, query):
    if ctx.is_match_safe:
        return []
    unquoted = re.findall(r'(?<!")\b([A-Z][A-Z0-9_]{2,})\.([A-Z][A-Z0-9_]{2,})\b(?!")', query)
    if not unquoted:
        return []
    examples = [f'{t}.{c}' for t, c in unquoted[:3]]
    return [ValidationIssue("ERROR", "SYN001",
        f"Possibly unquoted identifiers: {examples}",
        "Celonis requires all table and column names to be double-quoted. Unquoted identifiers cause parse errors.",
        'Use double-quote syntax: "TABLE_NAME"."COLUMN_NAME"',
        False)]

def rule_sum_wrapping_filter(ctx, query):
    if re.search(r'\bSUM\s*\(\s*FILTER\b', query, re.IGNORECASE):
        return [ValidationIssue("CRITICAL", "SYN002",
            "SUM(FILTER …) is invalid — FILTER is a statement, not a function",
            "FILTER is a top-level statement. Cannot be nested inside SUM().",
            "Use FILTER_TO_NULL: SUM(FILTER_TO_NULL(\"ORDERS\".\"AMOUNT\"))\nOr put FILTER at top level.",
            False)]
    return []

def rule_count_distinct_on_key(ctx, query):
    if not re.search(r'\bCOUNT\s*\(\s*DISTINCT\b', query, re.IGNORECASE):
        return []
    pattern = re.compile(r'COUNT\s*\(\s*DISTINCT\s+"[^"]+"\."([^"]+)"', re.IGNORECASE)
    issues = []
    for m in pattern.finditer(query):
        col = m.group(1).upper()
        if re.search(r'(_ID|_KEY|_NO|_NUM|CASE_ID)$', col):
            issues.append(ValidationIssue("PERF", "PERF003",
                f"COUNT(DISTINCT …) on key column \"{col}\" — COUNT is cheaper",
                "COUNT(DISTINCT) adds deduplication. If column is already unique, COUNT gives same result cheaper.",
                f"Replace COUNT(DISTINCT \"TABLE\".\"{col}\") with COUNT(\"TABLE\".\"{col}\")",
                True, "Replace COUNT(DISTINCT) with COUNT on key columns"))
    return issues


def run_context_rule_engine(query: str) -> List[ValidationIssue]:
    ctx = extract_context(query)
    ast = parse_pql(query)
    all_issues: List[ValidationIssue] = []

    all_issues += rule_sql_keywords(ctx, query)
    all_issues += rule_sum_wrapping_filter(ctx, query)
    all_issues += rule_quotes_enforcement(ctx, query)
    all_issues += rule_filter_inside_pu_text(ctx, query)
    all_issues += rule_filter_to_null_inside_pu(ctx, ast)
    all_issues += rule_pu_arg_count(ctx, ast)
    all_issues += rule_outer_pu_wrapping_inner_pu_datediff(ctx, query)
    all_issues += rule_pu_first_last_no_order_by(ctx, query)
    all_issues += rule_pu_schema_direction(ctx, query)   # NEW v3
    all_issues += rule_missing_global_mixed_levels(ctx)
    all_issues += rule_calc_throughput_needs_global(ctx, query)
    all_issues += rule_global_inside_filter(ctx, query)
    all_issues += rule_running_sum_deprecated(ctx, query)
    all_issues += rule_process_order_deprecated(ctx, query)
    all_issues += rule_all_occurrence_deprecated(query)
    all_issues += rule_pu_count_distinct_on_likely_key(ctx, query)
    all_issues += rule_pu_median_performance(ctx)
    all_issues += rule_count_distinct_on_key(ctx, query)

    SEVERITY_ORDER = {'CRITICAL': 0, 'ERROR': 1, 'WARNING': 2, 'PERF': 3, 'INFO': 4}
    all_issues.sort(key=lambda i: SEVERITY_ORDER.get(i.severity, 9))
    return all_issues


def build_why_explanation(issues: List[ValidationIssue]) -> str:
    if not issues:
        return ""
    parts = []
    for issue in issues:
        parts.append(f"Rule {issue.code} ({issue.severity}): {issue.message}\n  WHY: {issue.why}\n  FIX: {issue.fix}")
    return "\n\n".join(parts)


def apply_safe_auto_fixes(query: str, issues: List[ValidationIssue]) -> tuple:
    fixed = query
    applied = []
    for issue in issues:
        if not issue.auto_fixable or not issue.safe_fix:
            continue
        if issue.code == "DEP001":
            new = re.sub(r'\bRUNNING_SUM\b', 'RUNNING_TOTAL', fixed, flags=re.IGNORECASE)
            if new != fixed:
                fixed = new; applied.append("DEP001: Replaced RUNNING_SUM → RUNNING_TOTAL")
        elif issue.code == "DEP002":
            new = re.sub(r'\bPROCESS_ORDER\b', 'INDEX_ACTIVITY_ORDER', fixed, flags=re.IGNORECASE)
            if new != fixed:
                fixed = new; applied.append("DEP002: Replaced PROCESS_ORDER → INDEX_ACTIVITY_ORDER")
        elif issue.code == "PERF001":
            new = re.sub(r'\bPU_COUNT_DISTINCT\b', 'PU_COUNT', fixed, flags=re.IGNORECASE)
            if new != fixed:
                fixed = new; applied.append("PERF001: Replaced PU_COUNT_DISTINCT → PU_COUNT on key column")
    return (fixed != query), fixed, applied


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 · GROQ MODELS
# ─────────────────────────────────────────────────────────────────────────────

GROQ_MODELS = {
    'llama-3.3-70b-versatile': 'LLaMA 3.3 70B — best quality',
    'llama-3.1-8b-instant':    'LLaMA 3.1 8B  — fastest',
    'mixtral-8x7b-32768':      'Mixtral 8x7B  — balanced',
    'gemma2-9b-it':            'Gemma 2 9B    — lightweight',
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 · SYSTEM PROMPT BUILDER (v3 — reasoning-first)
# ─────────────────────────────────────────────────────────────────────────────

_FUNCTION_SELECTION_GUIDE = """
## ─── OFFICIAL CELONIS FUNCTION SELECTION GUIDE ───

### THROUGHPUT TIME
| Goal | Correct | Wrong |
|------|---------|-------|
| Case throughput (start→end) | CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS(..., DAYS)) | PU_MAX - PU_MIN |
| Case throughput (act→act) | CALC_THROUGHPUT(FIRST_OCCURRENCE['A'] TO LAST_OCCURRENCE['B'], ...) | DATEDIFF on activity table |
| Throughput over multiple cases | DATEDIFF('dd', PU_MIN(...), PU_MAX(...)) | CALC_THROUGHPUT |
| Cycle time first→last per case | DATEDIFF('dd', PU_FIRST(..., ORDER BY ...), PU_LAST(..., ORDER BY ...)) | PU_AVG wrapping DATEDIFF |
| Working-hours throughput | CALC_THROUGHPUT(..., REMAP_TIMESTAMPS(..., HOURS, WEEKDAY_CALENDAR(MON,TUE,WED,THU,FRI))) | Plain CALC_THROUGHPUT |

### REWORK / REPEATED ACTIVITIES
| Goal | Correct |
|------|---------|
| Count all activities per case | CALC_REWORK() |
| Count specific activities per case | CALC_REWORK("ACTIVITIES"."ACTIVITY" = 'Review') |
| Detect repeated activities (row-level) | INDEX_ACTIVITY_LOOP("ACTIVITIES"."ACTIVITY") > 0 |
| Loop counter for specific activity | INDEX_ACTIVITY_TYPE("ACTIVITIES"."ACTIVITY") |

### AGGREGATION SELECTION
| Goal | Use | Avoid |
|------|-----|-------|
| Count rows (key column) | PU_COUNT | PU_COUNT_DISTINCT (much slower) |
| Average values | PU_AVG | PU_MEDIAN (requires full sort) |
| First/Last value | PU_FIRST/PU_LAST with ORDER BY | Without ORDER BY (non-deterministic) |
| Running total | RUNNING_TOTAL | RUNNING_SUM (deprecated) |
| Position in case | INDEX_ACTIVITY_ORDER | PROCESS_ORDER (deprecated) |

### NULL BEHAVIOUR
| Function | No matching rows |
|----------|-----------------|
| PU_COUNT, PU_COUNT_DISTINCT | 0 |
| PU_SUM, PU_AVG, PU_MIN, PU_MAX, PU_FIRST, PU_LAST | NULL |
| CALC_THROUGHPUT | NULL if single activity or end before start |

### GLOBAL() — WHEN TO USE
- Mix case + activity columns → GLOBAL wraps case-level aggregation
- CALC_THROUGHPUT + AVG/SUM → GLOBAL(AVG(CALC_THROUGHPUT(...)))
- Percent of total → SUM("ORDERS"."AMOUNT") / GLOBAL(SUM("ORDERS"."AMOUNT"))
"""

_SQL_PROHIBITION = """
## CRITICAL — PQL IS NOT SQL. NEVER WRITE SQL.
NO: SELECT  FROM  JOIN  LEFT JOIN  GROUP BY  HAVING  WITH  OVER(...)  AS (CTE)

WRONG SQL:
  SELECT "LFA1"."LIFNR", AVG(DATEDIFF(dd, ...)) FROM "EKKO" JOIN "EKPO" ON ... GROUP BY "LFA1"."LIFNR"

CORRECT PQL:
  PU_AVG("LFA1", DATEDIFF('dd', "EKKO"."BEDAT", "EKPO"."LGDAT"))
"""

_ADVANCED_PATTERNS = """
## ─── Advanced PQL Patterns ───

### P1 · GLOBAL() — prevents join multiplication
```pql
GLOBAL( AVG( CALC_THROUGHPUT( CASE_START TO CASE_END, REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", DAYS) ) ) )
CASE WHEN AVG("ORDERS"."AMOUNT") > GLOBAL(AVG("ORDERS"."AMOUNT")) THEN 'Above Avg' ELSE 'Below Avg' END
SUM("ORDERS"."AMOUNT") / GLOBAL(SUM("ORDERS"."AMOUNT")) * 100
```

### P2 · Automation Rate
```pql
ROUND(
  PU_COUNT("CASES", "ACTIVITIES"."CASE_ID", "ACTIVITIES"."USER" = 'SYSTEM') * 100.0
  / CALC_REWORK(), 1
)
```

### P3 · Working-hours throughput
```pql
AVG(CALC_THROUGHPUT(
  CASE_START TO CASE_END,
  REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", HOURS, WEEKDAY_CALENDAR(MON,TUE,WED,THU,FRI))
)) / 8
```

### P4 · Rework detection
```pql
CASE WHEN PU_COUNT("CASES", "ACTIVITIES"."CASE_ID", "ACTIVITIES"."ACTIVITY" = 'Review') > 1
     THEN 'Rework' ELSE 'Clean' END
CASE WHEN INDEX_ACTIVITY_LOOP("ACTIVITIES"."ACTIVITY") > 0 THEN 'Rework' ELSE 'First' END
```

### P5 · Cycle time first→last per case
```pql
DATEDIFF('dd',
  PU_FIRST("CASES", "ACTIVITIES"."TIMESTAMP", ORDER BY "ACTIVITIES"."TIMESTAMP" ASC),
  PU_LAST("CASES", "ACTIVITIES"."TIMESTAMP", ORDER BY "ACTIVITIES"."TIMESTAMP" ASC)
)
```

### P6 · Late deliveries SLA
```pql
PU_COUNT("VENDORS", "ORDERS"."ORDER_ID",
  DATEDIFF('dd', "ORDERS"."PROMISED_DATE", "ORDERS"."ACTUAL_DATE") > 7
)
```

### P7 · Transition time between activities
```pql
SECONDS_BETWEEN(ACTIVITY_LAG("ACTIVITIES"."TIMESTAMP"), "ACTIVITIES"."TIMESTAMP") / 3600
```

### P8 · Conforming cases throughput only
```pql
AVG(CASE WHEN PU_SUM("CASES", ABS("CONFORMANCE_COL")) = 0
    THEN CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", HOURS)) / 24
    ELSE NULL END)
```

### P9 · Z-score outlier detection per partition
```pql
CASE WHEN ZSCORE("ORDERS"."AMOUNT", PARTITION BY ("ORDERS"."VENDOR")) > 3 THEN 'Outlier' ELSE 'Normal' END
```

### P10 · Running total with monthly partitions
```pql
RUNNING_TOTAL("ORDERS"."AMOUNT",
  ORDER BY ("ORDERS"."ORDER_DATE" ASC),
  PARTITION BY (ROUND_MONTH("ORDERS"."ORDER_DATE"))
)
```

### P11 · Full KPI: touchless + rework + throughput
```pql
-- Touchless rate
CASE WHEN CALC_REWORK("ACTIVITIES"."TYPE" = 'Manual') = 0 THEN 'Touchless' ELSE 'Manual Touch' END

-- Rework count
PU_COUNT("CASES", "ACTIVITIES"."CASE_ID", "ACTIVITIES"."ACTIVITY" = 'Reprocess')

-- Throughput in working hours
GLOBAL(AVG(CALC_THROUGHPUT(CASE_START TO CASE_END,
  REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", HOURS, WEEKDAY_CALENDAR(MON,TUE,WED,THU,FRI)))))
```

### P12 · OCPM cross-object throughput
```pql
AVG(CALC_THROUGHPUT(CASE_START TO CASE_END,
  REMAP_TIMESTAMPS(TIMESTAMP_COLUMN(ACTIVITY_TABLE(LINK_PATH("ORDERS"."ORDER_ID"))), DAYS)
))
```
"""

_EXPERT_FRAMEWORK = """
## ─── Expert Query Construction Framework ───

**Step 1** — Identify tables & relationships. Which is parent (1-side)? Which is child (N-side)?
**Step 2** — Identify result level. Case? Activity? Vendor? Mixing? → GLOBAL() required.
**Step 3** — Choose aggregation strategy. PU for cross-table; Standard for same-table.
**Step 4** — Handle filters. FILTER top-level; PU filter_expression inside PU; BIND_FILTERS for non-common.
**Step 5** — Build KPIs innermost first, wrap GLOBAL() at table-level boundaries.
**Step 6** — Performance: PU_COUNT vs PU_COUNT_DISTINCT, AVG vs MEDIAN, RUNNING_TOTAL.
**Step 7** — NULL safety: COALESCE, ISNULL, check what each function returns on no match.
**Step 8** — Validate intent: is the query producing what the user actually wants?

## Anti-patterns — always avoid
1. Missing GLOBAL() when mixing case + activity columns
2. FILTER or FILTER_TO_NULL inside PU functions
3. PU_COUNT_DISTINCT on a key/ID column
4. PU_MEDIAN or MEDIAN when AVG is sufficient
5. Unquoted table/column names
6. Any SQL syntax
7. Outer PU wrapping DATEDIFF of inner PU with same target
8. RUNNING_SUM (deprecated) → RUNNING_TOTAL
9. PROCESS_ORDER (deprecated) → INDEX_ACTIVITY_ORDER
10. ALL_OCCURRENCE['…'] (deprecated) → CASE_START
11. PU_FIRST/PU_LAST without ORDER BY
12. GLOBAL() inside FILTER
"""

_REASONING_FRAMEWORK = """
## ─── v3 REASONING FRAMEWORK ───

Before writing any PQL, ALWAYS reason through these steps internally:

1. **INTENT**: What is the user actually trying to achieve? (throughput? count? rate? flag?)
2. **LEVEL**: What level should the result be at? (case? activity? vendor?)
3. **TABLES**: Which tables are involved? What is their relationship (1:N, N:M)?
4. **GRAIN**: Will mixing these tables cause a join shift? Does GLOBAL() protect against it?
5. **FUNCTION CHOICE**: What is the BEST function for this use case (not just any valid function)?
6. **PERFORMANCE**: Is there a cheaper alternative that gives the same result?
7. **NULL SAFETY**: What happens with NULLs? Is COALESCE needed?
8. **CORRECTNESS CHECK**: Does this query answer EXACTLY what was asked?

Only after completing this reasoning should you write the PQL query.
Always explain the reasoning before the code.
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

    base = f"""You are a world-class Celonis PQL (Process Query Language) reasoning engine — equivalent to a 
senior Celonis architect with 10+ years of production experience building enterprise process mining solutions.

You do NOT just validate syntax. You REASON about:
- What the user is trying to achieve (INTENT)
- How Celonis will execute the query (EXECUTION SIMULATION)
- Whether the result will actually match the intent (CORRECTNESS)
- What the best-performing alternative is (OPTIMIZATION)
- Domain knowledge: O2C, P2P, H2R, Finance processes

Write ACCURATE, OPTIMIZED, PRODUCTION-READY PQL that works directly in Celonis Studio.

## PQL Absolute Rules (NEVER violate)
1. Tables/columns MUST be double-quoted: "TABLE"."COLUMN"
2. String literals MUST use single quotes: 'value'
3. PQL is column-based, NOT row-based — no SELECT/FROM/JOIN
4. Multiple FILTER statements merge by logical AND
5. NULL: most functions skip NULLs; use COALESCE or ISNULL explicitly
6. PU-functions: FROM child (N-side) TO parent (1-side)
7. FILTER cannot be inside PU functions — use filter_expression parameter
8. GLOBAL() required when mixing columns from different table levels
9. RUNNING_TOTAL replaces deprecated RUNNING_SUM
10. INDEX_ACTIVITY_ORDER replaces deprecated PROCESS_ORDER

{_SQL_PROHIBITION}

{_FUNCTION_SELECTION_GUIDE}

{_REASONING_FRAMEWORK}

## Core PQL Functions Reference
{core_refs}
"""

    if complexity in ("Advanced", "Expert"):
        base += _ADVANCED_PATTERNS

    if complexity == "Expert":
        base += _EXPERT_FRAMEWORK

    if show_reasoning and complexity in ("Advanced", "Expert"):
        base += """
## Response Format (REQUIRED for Advanced/Expert)
1. **🎯 Intent Analysis** — what is the user trying to achieve?
2. **📊 Execution Planning** — table levels, join behavior, GLOBAL() needed?
3. **⚙ Function Selection** — why these specific functions?
4. **📝 PQL Query** — complete, production-ready in ```pql block with inline comments
5. **⚡ Performance Notes** — cheaper alternatives, optimization choices
6. **🔒 Edge Cases** — NULL handling, filter propagation, deprecation warnings
"""
    elif complexity == "Intermediate":
        base += """
## Response Format
1. **Intent** — what you understood the user wants
2. **PQL** in a ```pql block
3. **Explanation** — each function and why chosen
4. **Watch out for** — NULLs, GLOBAL(), filter awareness
"""
    else:
        base += """
## Response Format
1. PQL in ```pql block
2. Short plain-English explanation (2-4 sentences)
"""

    instructions = {
        "Basic":        "Simple 1-2 function queries. Focus on correctness and clarity.\n",
        "Intermediate": "2–5 function queries with filters, CASE WHEN, aggregations.\n",
        "Advanced":     "Nested PU-functions, GLOBAL(), throughput patterns, multi-table KPIs.\n",
        "Expert":       "Production multi-KPI queries. BPMN conformance. OCPM. ML. Full chain-of-thought reasoning.\n",
    }
    base += f"\n## Complexity: {complexity}\n{instructions[complexity]}\n"
    base += """
Standard placeholders when schema is unknown:
"CASES"."CASE_ID", "ACTIVITIES"."ACTIVITY", "ACTIVITIES"."TIMESTAMP", "ACTIVITIES"."USER",
"ORDERS"."AMOUNT", "VENDORS"."VENDOR_ID", "ORDERS"."CREATE_DATE", "ORDERS"."CLOSE_DATE"
"""
    return base


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 · LLM REASONING SYSTEM PROMPT (v3 — upgraded role)
# ─────────────────────────────────────────────────────────────────────────────

VERIFICATION_SYSTEM = """You are a strict Celonis PQL reasoning validator and targeted corrector.
Your job is to REASON about PQL correctness — not just syntax, but logic and intent.

## Rules to enforce:
1. NO SQL: SELECT/FROM/JOIN/GROUP BY/HAVING/WITH/AS/OVER → REMOVE
2. All table/column identifiers must be double-quoted
3. String literals must use single quotes
4. PU_FUNC needs 2+ arguments (target_table, source_table.column [, filter])
5. FILTER_TO_NULL inside PU functions → replace with PU filter_expression
6. GLOBAL() required when CALC_THROUGHPUT combined with AVG/SUM/COUNT + activity columns
7. PU_COUNT_DISTINCT on key column → replace with PU_COUNT
8. RUNNING_SUM → RUNNING_TOTAL
9. PROCESS_ORDER → INDEX_ACTIVITY_ORDER
10. ALL_OCCURRENCE['...'] → CASE_START or CASE_END
11. Outer PU wrapping DATEDIFF of inner PU same target → remove outer PU
12. PU direction: target must be PARENT (1-side), source must be CHILD (N-side)
13. PU_FIRST/PU_LAST must have ORDER BY
14. FILTER keyword must NOT appear inside PU function arguments
15. GLOBAL() must NOT appear inside FILTER statements
16. Automation rate needs: system_count / total_count × 100
17. Percentage calculations need × 100
18. COUNT("ACTIVITIES"."CASE_ID") when case count intended → flag and suggest COUNT("CASES"."CASE_ID")

## IMPORTANT — safe correction only:
- Only fix what you are CERTAIN is wrong
- Do NOT restructure entire queries unless necessary
- If unsure, mark as warning rather than changing

## Response format:
- If correct: respond exactly: VALID
- If errors: respond ONLY with corrected ```pql block + brief bullet list of changes.
- No preamble, no commentary beyond the fix list.
"""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 · UI CONSTANTS
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
        'Count late deliveries per vendor (delivery > 7 days past promised)',
        'Rework rate: Review activity repeating more than 2 times per case',
        'Automation rate: % of system activities per case',
        'Flag non-conforming cases and show throughput separately',
        'Z-score outlier detection on invoice amounts per vendor',
    ],
    'Expert': [
        'Full KPI: throughput + rework count + automation rate in one query',
        'Multi-level: avg approval time aggregated vendor → order → line item',
        'BPMN conformance tolerating undesired but not missing activities',
        'OCPM: throughput across linked objects with workday calendar',
        'Working-hours SLA breach detection with conformance scoring',
        'Variant-level rework: first vs repeated occurrence per activity type',
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 · PAGE CONFIG + CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='PQL Query Assistant v3',
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
.main .block-container{background:var(--bg-base)!important;padding-top:2rem!important;max-width:960px!important;}
header[data-testid="stHeader"]{background:var(--bg-base)!important;border-bottom:1px solid var(--border)!important;}
h1,h2,h3{font-family:var(--font-ui)!important;color:var(--text-primary)!important;letter-spacing:-0.02em;}
h1{font-size:1.75rem!important;font-weight:800!important;}
h2{font-size:1.25rem!important;font-weight:700!important;}
h3{font-size:1rem!important;font-weight:600!important;}
div[data-testid="stMarkdownContainer"] p{color:var(--text-secondary)!important;font-size:14px;line-height:1.6;}
[data-testid="stCaptionContainer"] p,.stCaption,.stCaption p{color:var(--text-muted)!important;font-size:12px!important;font-family:var(--font-mono)!important;}
h1 a,h2 a,h3 a,[data-testid="stHeadingWithActionElements"] a,
[data-testid="stHeadingWithActionElements"] button,[data-testid="stHeadingWithActionElements"] svg{display:none!important;}
[data-testid="stSidebar"]{background:var(--bg-surface)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p,[data-testid="stSidebar"] span{color:var(--text-secondary)!important;font-size:13px;}
[data-testid="stSidebar"] input,[data-testid="stSidebar"] select{background:var(--bg-elevated)!important;border:1px solid var(--border-bright)!important;color:var(--text-primary)!important;border-radius:var(--radius-sm)!important;font-family:var(--font-mono)!important;font-size:12px!important;}
[data-testid="stChatMessage"]{background:var(--bg-surface)!important;border:1px solid var(--border)!important;border-radius:var(--radius-lg)!important;margin-bottom:12px!important;}
[data-testid="stChatMessageContent"],[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li{color:var(--text-primary)!important;font-size:14px!important;line-height:1.7!important;}
[data-testid="stChatMessageContent"] strong{color:#f0f4ff!important;font-weight:600;}
[data-testid="stChatMessageContent"] code{background:var(--bg-elevated)!important;color:#93c5fd!important;font-family:var(--font-mono)!important;font-size:12px!important;padding:2px 6px!important;border-radius:4px!important;border:1px solid var(--border-bright)!important;}
pre{background:#040810!important;border:1px solid var(--border-bright)!important;border-radius:var(--radius-md)!important;padding:18px!important;overflow-x:auto!important;position:relative;}
pre::before{content:'PQL';position:absolute;top:10px;right:14px;font-family:var(--font-mono);font-size:10px;font-weight:600;color:var(--text-muted);letter-spacing:0.1em;}
pre code{background:transparent!important;border:none!important;color:#e2e8f0!important;font-family:var(--font-mono)!important;font-size:13px!important;line-height:1.6!important;padding:0!important;}
[data-testid="stBottom"]{background:linear-gradient(to top,var(--bg-base) 70%,transparent)!important;border-top:none!important;padding:16px 0!important;}
[data-testid="stChatInput"]{background:var(--bg-elevated)!important;border:1px solid var(--border-bright)!important;border-radius:var(--radius-lg)!important;}
[data-testid="stChatInput"]:focus-within{border-color:var(--accent)!important;box-shadow:0 0 0 3px var(--accent-glow)!important;}
[data-testid="stChatInput"] textarea{background:var(--bg-elevated)!important;color:#e8edf5!important;border:none!important;font-size:14px!important;-webkit-text-fill-color:#e8edf5!important;}
[data-testid="stChatInputSubmitButton"] button{background:var(--accent)!important;border:none!important;border-radius:8px!important;}
[data-testid="stChatInputSubmitButton"] button:hover{background:#2563eb!important;}
.stButton>button{background:var(--bg-elevated)!important;border:1px solid var(--border)!important;color:var(--text-secondary)!important;border-radius:var(--radius-sm)!important;font-size:12px!important;font-family:var(--font-mono)!important;text-align:left!important;}
.stButton>button:hover{background:var(--bg-hover)!important;border-color:var(--accent)!important;color:var(--text-primary)!important;}
[data-testid="stSelectbox"]>div>div{background:var(--bg-elevated)!important;border:1px solid var(--border-bright)!important;border-radius:var(--radius-sm)!important;color:var(--text-primary)!important;font-family:var(--font-mono)!important;font-size:12px!important;}
[data-testid="stToggle"] input:checked+div{background:var(--accent)!important;}
[data-testid="stMetric"]{background:var(--bg-elevated);border:1px solid var(--border);border-radius:var(--radius-md);padding:12px 14px;text-align:center;}
[data-testid="stMetricLabel"]{color:var(--text-muted)!important;font-size:11px!important;font-family:var(--font-mono)!important;}
[data-testid="stMetricValue"]{color:var(--text-primary)!important;font-family:var(--font-mono)!important;font-size:1.5rem!important;font-weight:600!important;}
[data-testid="stExpander"]{background:var(--bg-elevated)!important;border:1px solid var(--border)!important;border-radius:var(--radius-sm)!important;margin-bottom:4px!important;}
[data-testid="stExpander"] summary{font-family:var(--font-mono)!important;font-size:12px!important;color:var(--text-secondary)!important;padding:8px 12px!important;}

/* Validation badges */
.verify-pass{display:flex;align-items:center;gap:8px;background:var(--green-dim);border:1px solid var(--green);border-radius:var(--radius-sm);padding:8px 14px;color:#6ee7b7;font-size:12px;font-family:var(--font-mono);margin-top:10px;}
.verify-fix{display:flex;align-items:center;gap:8px;background:var(--amber-dim);border:1px solid var(--amber);border-radius:var(--radius-sm);padding:8px 14px;color:#fcd34d;font-size:12px;font-family:var(--font-mono);margin-top:10px;}
.auto-fix{display:flex;align-items:center;gap:8px;background:#1a2e1a;border:1px solid #22c55e;border-radius:var(--radius-sm);padding:8px 14px;color:#86efac;font-size:12px;font-family:var(--font-mono);margin-top:6px;}

/* Issue severities */
.issue-critical{background:var(--red-dim);border:1px solid var(--red);border-left:3px solid var(--red);border-radius:var(--radius-sm);padding:10px 14px;color:#fca5a5;font-size:12px;font-family:var(--font-mono);margin-top:6px;line-height:1.5;}
.issue-error{background:#2d1515;border:1px solid #dc2626;border-left:3px solid #dc2626;border-radius:var(--radius-sm);padding:10px 14px;color:#fca5a5;font-size:12px;font-family:var(--font-mono);margin-top:6px;line-height:1.5;}
.issue-warning{background:var(--amber-dim);border:1px solid var(--amber);border-left:3px solid var(--amber);border-radius:var(--radius-sm);padding:10px 14px;color:#fcd34d;font-size:12px;font-family:var(--font-mono);margin-top:6px;line-height:1.5;}
.issue-perf{background:var(--perf-dim);border:1px solid var(--perf);border-left:3px solid var(--perf);border-radius:var(--radius-sm);padding:10px 14px;color:#67e8f9;font-size:12px;font-family:var(--font-mono);margin-top:6px;line-height:1.5;}
.issue-info{background:#141e30;border:1px solid #3b82f6;border-left:3px solid #3b82f6;border-radius:var(--radius-sm);padding:10px 14px;color:#93c5fd;font-size:12px;font-family:var(--font-mono);margin-top:6px;line-height:1.5;}
.issue-why{color:#aab4c0;font-size:11px;margin-top:4px;font-style:italic;line-height:1.4;}
.issue-fix{color:#c8d8e0;font-size:11px;margin-top:2px;line-height:1.4;}

/* Reasoning insights */
.insight-high{background:#2d1515;border:1px solid #f97316;border-left:4px solid #f97316;border-radius:var(--radius-sm);padding:12px 16px;color:#fed7aa;font-size:13px;font-family:var(--font-body);margin-top:8px;line-height:1.5;}
.insight-medium{background:#1a1a30;border:1px solid #a78bfa;border-left:4px solid #a78bfa;border-radius:var(--radius-sm);padding:12px 16px;color:#ddd6fe;font-size:13px;font-family:var(--font-body);margin-top:8px;line-height:1.5;}
.insight-low{background:#131a20;border:1px solid #38bdf8;border-left:4px solid #38bdf8;border-radius:var(--radius-sm);padding:12px 16px;color:#bae6fd;font-size:13px;font-family:var(--font-body);margin-top:8px;line-height:1.5;}
.insight-title{font-weight:700;font-family:var(--font-ui);margin-bottom:4px;font-size:13px;}
.insight-expl{color:#b0b8c8;font-size:12px;margin-top:4px;line-height:1.4;}
.insight-rec{color:#c8d8e0;font-size:12px;margin-top:6px;line-height:1.4;font-family:var(--font-mono);}

/* Intent panel */
.intent-panel{background:linear-gradient(135deg,#0d1a2e,#0d2218);border:1px solid #1e3a5f;border-radius:var(--radius-md);padding:14px 18px;margin-top:8px;margin-bottom:4px;}
.intent-title{font-family:var(--font-mono);font-size:10px;text-transform:uppercase;letter-spacing:0.1em;color:#4a6080;margin-bottom:8px;}
.intent-row{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:4px;}
.intent-badge{background:#0d2035;border:1px solid #1e4060;border-radius:20px;padding:3px 10px;font-size:11px;color:#60a0e0;font-family:var(--font-mono);}

/* Execution plan */
.exec-panel{background:#0d1a10;border:1px solid #1e3a20;border-radius:var(--radius-md);padding:14px 18px;margin-top:6px;}
.exec-title{font-family:var(--font-mono);font-size:10px;text-transform:uppercase;letter-spacing:0.1em;color:#2a5030;margin-bottom:8px;}
.exec-ok{color:#4ade80;font-size:12px;font-family:var(--font-mono);}
.exec-warn{color:#fbbf24;font-size:12px;font-family:var(--font-mono);}

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
.example-chip{display:inline-block;background:var(--bg-surface);border:1px solid var(--border);border-radius:20px;padding:4px 12px;font-size:12px;color:#93c5fd;font-family:var(--font-mono);margin:3px;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:var(--bg-base);}
::-webkit-scrollbar-thumb{background:var(--border-bright);border-radius:3px;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16 · SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

_defaults = {
    'messages':          [],
    'complexity':        'Advanced',
    'model_id':          'llama-3.3-70b-versatile',
    'show_reasoning':    True,
    'show_intent':       True,
    'total_queries':     0,
    'verified_count':    0,
    'fixed_count':       0,
    'rule_hits':         0,
    'auto_fixed_count':  0,
    'insight_count':     0,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 17 · GROQ CLIENT
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
# SECTION 18 · SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div class="brand-header">'
        '<div class="brand-icon">⚡</div>'
        '<div>'
        '<div class="brand-title">PQL Reasoning Engine v3</div>'
        '<div class="brand-sub">Intent · Simulate · Schema · Reason · Verify</div>'
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
        help='AI explains intent, execution planning, and function selection',
    )
    st.session_state.show_intent = st.toggle(
        'Show intent analysis panel', value=st.session_state.show_intent,
        help='Show detected intent, execution plan, and reasoning insights for PQL blocks',
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
                        f'Show edge cases, performance notes, and common mistakes.'
                    )
                st.caption(fn['doc'][:120] + '…' if len(fn['doc']) > 120 else fn['doc'])

    st.divider()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Queries', st.session_state.total_queries)
    c2.metric('✅', st.session_state.verified_count)
    c3.metric('🔧', st.session_state.fixed_count)
    c4.metric('⚡', st.session_state.auto_fixed_count)
    c5.metric('🧠', st.session_state.insight_count)

    if st.button('Clear chat', use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 19 · MAIN CHAT AREA
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="page-title">PQL <span>Reasoning</span> Engine v3</div>',
    unsafe_allow_html=True
)
st.markdown(
    f'<div class="page-meta">'
    f'<span class="stat-pill"><b>{complexity}</b></span>'
    f'<span class="stat-pill"><b>{st.session_state.model_id.split("-")[0]}</b></span>'
    f'<span class="stat-pill"><b>{len(COMPACT_REFS)}</b> functions</span>'
    f'<span class="stat-pill">🧠 Intent · Simulate · Schema · Reason · Verify</span>'
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
            '<div class="welcome-title">PQL Reasoning Engine v3 — Intent-Aware · Execution-Simulated · Schema-Validated</div>'
            '<div class="welcome-sub">Built on a <strong>6-layer reasoning pipeline</strong> that understands WHAT you want, '
            'simulates HOW Celonis will execute it, and validates WHETHER the result matches your intent.</div>'
            '<div class="welcome-grid">'
            '<div class="welcome-item"><b>🎯 Intent Engine</b>Detects goal, result level, domain (O2C/P2P/HR)</div>'
            '<div class="welcome-item"><b>⚙ Execution Simulator</b>Simulates join shifts, aggregation grain conflicts</div>'
            '<div class="welcome-item"><b>🗂 Schema Layer</b>Validates PU direction against 1:N relationships</div>'
            '<div class="welcome-item"><b>🧠 Reasoning Combinator</b>Intent + Execution → smart cross-layer insights</div>'
            '<div class="welcome-item"><b>🔧 Rule Engine</b>20+ context-aware validation rules with WHY explanations</div>'
            '<div class="welcome-item"><b>🤖 LLM Reasoning</b>Final validation pass — reasons about logic, not just syntax</div>'
            '</div>'
            '<div style="border-top:1px solid #1f2738;padding-top:14px;margin-top:4px;">'
            '<p style="font-family:IBM Plex Mono;font-size:12px;color:#4a5568;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.06em;">Try asking</p>'
            '<span class="example-chip">Avg working-hours throughput per case</span>'
            '<span class="example-chip">Automation rate with rework count</span>'
            '<span class="example-chip">BPMN conformance with tolerances</span>'
            '<span class="example-chip">Z-score outlier detection per vendor</span>'
            '<span class="example-chip">Late P2P deliveries SLA breach</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 20 · RENDERING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

SEVERITY_CSS = {'CRITICAL': 'issue-critical', 'ERROR': 'issue-error',
                'WARNING': 'issue-warning', 'PERF': 'issue-perf', 'INFO': 'issue-info'}
SEVERITY_ICON = {'CRITICAL': '🚨', 'ERROR': '❌', 'WARNING': '⚠', 'PERF': '⚡', 'INFO': 'ℹ'}
INSIGHT_CSS = {'HIGH': 'insight-high', 'MEDIUM': 'insight-medium', 'LOW': 'insight-low'}
INSIGHT_ICON = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🔵'}


def render_issue(issue: ValidationIssue):
    css = SEVERITY_CSS.get(issue.severity, 'issue-info')
    icon = SEVERITY_ICON.get(issue.severity, '•')
    fix_html = issue.fix.replace('\n', '<br>')
    st.markdown(
        f'<div class="{css}"><strong>{icon} [{issue.code}] {issue.message}</strong>'
        f'<div class="issue-why">Why: {issue.why}</div>'
        f'<div class="issue-fix">Fix: {fix_html}</div></div>',
        unsafe_allow_html=True
    )


def render_insight(insight: ReasoningInsight):
    css = INSIGHT_CSS.get(insight.severity, 'insight-low')
    icon = INSIGHT_ICON.get(insight.severity, '•')
    rec_html = insight.recommendation.replace('\n', '<br>')
    st.markdown(
        f'<div class="{css}">'
        f'<div class="insight-title">{icon} [{insight.category}] {insight.title}</div>'
        f'<div class="insight-expl">{insight.explanation}</div>'
        f'<div class="insight-rec"><strong>Recommendation:</strong><br>{rec_html}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


def render_intent_panel(intent: PQLIntent, plan: ExecutionPlan):
    badges = []
    if intent.goal:
        badges.append(f'🎯 {intent.goal.replace("_", " ").title()}')
    if intent.target_level:
        badges.append(f'📊 {intent.target_level} level')
    if intent.metric_type:
        badges.append(f'📐 {intent.metric_type}')
    if intent.domain and intent.domain != "GENERIC":
        badges.append(f'🏭 {intent.domain}')
    if intent.wants_throughput:
        badges.append('⏱ throughput')
    if intent.wants_rework:
        badges.append('🔄 rework')
    if intent.wants_automation:
        badges.append('🤖 automation rate')
    if intent.wants_percentage:
        badges.append('% percentage')

    badges_html = "".join(f'<span class="intent-badge">{b}</span>' for b in badges)
    confidence_bar = "▓" * int(intent.confidence * 10) + "░" * (10 - int(intent.confidence * 10))

    exec_lines = []
    if plan.common_table:
        exec_lines.append(f'<div class="exec-ok">✓ Common table: {plan.common_table}</div>')
    if plan.join_shift:
        exec_lines.append(f'<div class="exec-warn">⚠ Join shift detected → ACTIVITIES level</div>')
    if plan.grain_conflict:
        exec_lines.append(f'<div class="exec-warn">⚠ Aggregation grain conflict → GLOBAL() needed</div>')
    if plan.global_present:
        exec_lines.append(f'<div class="exec-ok">✓ GLOBAL() present — join shift protected</div>')
    if plan.mixed_levels:
        exec_lines.append(f'<div class="exec-warn">⚠ Mixed table levels: {", ".join(plan.tables_involved)}</div>')

    exec_html = "".join(exec_lines) if exec_lines else '<div class="exec-ok">✓ Single table level — no join shift risk</div>'

    st.markdown(
        f'<div class="intent-panel">'
        f'<div class="intent-title">Intent Detection (confidence: {confidence_bar} {int(intent.confidence * 100)}%)</div>'
        f'<div class="intent-row">{badges_html}</div>'
        f'</div>'
        f'<div class="exec-panel">'
        f'<div class="exec-title">Execution Simulation</div>'
        f'{exec_html}'
        f'</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 21 · 6-LAYER VALIDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def extract_pql_blocks(text: str) -> list:
    return re.findall(r"```pql\s*(.*?)```", text, re.S)


def run_llm_verification(pql_query: str, issues: List[ValidationIssue], intent: PQLIntent) -> tuple:
    """Layer 6: LLM reasoning — logic and intent correctness check."""
    has_serious = any(i.severity in ('CRITICAL', 'ERROR') for i in issues)
    always_verify = st.session_state.complexity in ('Advanced', 'Expert')

    if not (has_serious or always_verify):
        return False, pql_query, []

    try:
        why_context = build_why_explanation(issues) if issues else ""
        intent_context = format_intent_summary(intent) if intent else ""
        rule_context = f"\n\nContext engine found:\n{why_context}" if why_context else ""
        intent_hint = f"\n\nDetected user intent:\n{intent_context}" if intent_context else ""

        verify_prompt = (
            f"Review this PQL query for correctness and logical accuracy:{rule_context}{intent_hint}\n\n"
            f"```pql\n{pql_query}\n```\n\n"
            "Check all rules. Also verify the query actually produces what the intent suggests. "
            "Respond with VALID or the corrected ```pql block + brief bullet list of changes."
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
            return True, corrected, fixes or ["Query corrected by reasoning verification"]

        return False, pql_query, []
    except Exception as e:
        return False, pql_query, [f"LLM verification skipped ({e})"]


def validate_pql_block(pql_query: str, user_query: str = "") -> tuple:
    """
    Full 6-layer validation pipeline.

    Layer 1: Context extraction
    Layer 2: Intent detection (NEW v3)
    Layer 3: Execution simulation (NEW v3)
    Layer 4: Rule engine + Schema validation (UPGRADED v3)
    Layer 5: Safe auto-fix
    Layer 6: LLM reasoning verification (UPGRADED v3)

    Returns: (was_modified, final_query, issues, insights, intent, plan, auto_fix_notes, llm_fix_notes)
    """
    # Layer 1: Context
    # Layer 2: Intent
    intent = detect_intent(user_query or pql_query)
    # Layer 3: Execution simulation
    plan = simulate_execution(pql_query)
    # Reasoning combinator
    insights = reason_about_query(pql_query, intent, plan)
    st.session_state.insight_count += len(insights)

    # Layer 4: Rule engine
    issues = run_context_rule_engine(pql_query)
    st.session_state.rule_hits += len([i for i in issues if i.severity in ('CRITICAL', 'ERROR', 'WARNING')])

    # Layer 5: Safe auto-fix
    auto_fixed, query_after_autofix, auto_fix_notes = apply_safe_auto_fixes(pql_query, issues)
    if auto_fixed:
        st.session_state.auto_fixed_count += 1
        issues = run_context_rule_engine(query_after_autofix)

    # Layer 6: LLM reasoning
    llm_fixed, final_query, llm_fix_notes = run_llm_verification(query_after_autofix, issues, intent)
    if llm_fixed:
        st.session_state.fixed_count += 1

    was_modified = auto_fixed or llm_fixed
    return was_modified, final_query, issues, insights, intent, plan, auto_fix_notes, llm_fix_notes


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 22 · GROQ STREAMING + FULL VALIDATION DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def stream_groq(prompt_override=None):
    msgs = st.session_state.messages
    user_query = prompt_override if prompt_override else msgs[-1]["content"]

    func_context = build_function_context(user_query)
    system = build_system_prompt(st.session_state.complexity, st.session_state.show_reasoning)

    if func_context:
        system += "\n\n## Relevant PQL Functions (auto-retrieved)\n" + func_context

    # Inject intent context into system prompt
    intent = detect_intent(user_query)
    if intent.goal or intent.target_level:
        intent_summary = format_intent_summary(intent)
        system += f"\n\n## Detected User Intent\n{intent_summary}\n\nEnsure your query matches this intent exactly."

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
                max_tokens=3500,
                temperature=0.08,
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
                (was_modified, final_query, issues, insights,
                 detected_intent, exec_plan, auto_fix_notes, llm_fix_notes) = validate_pql_block(pql_block, user_query)

                # Intent + Execution panel
                if st.session_state.show_intent and (detected_intent.goal or exec_plan.common_table):
                    render_intent_panel(detected_intent, exec_plan)

                # Reasoning insights
                if insights:
                    st.markdown("**🧠 Reasoning Insights:**")
                    for insight in insights:
                        render_insight(insight)

                # Rule issues
                blocking = [i for i in issues if i.severity in ('CRITICAL', 'ERROR', 'WARNING', 'PERF')]
                info_issues = [i for i in issues if i.severity == 'INFO']

                if blocking or info_issues:
                    st.markdown("**🔍 Validation Analysis:**")
                for issue in blocking:
                    render_issue(issue)
                for issue in info_issues:
                    render_issue(issue)

                # Auto-fix
                if auto_fix_notes:
                    st.markdown(
                        '<div class="auto-fix">⚡ <strong>Safe auto-fix applied</strong> — deterministic corrections</div>',
                        unsafe_allow_html=True
                    )
                    for note in auto_fix_notes:
                        st.caption(f"  • {note}")

                # LLM fix
                if llm_fix_notes and was_modified:
                    st.markdown(
                        '<div class="verify-fix">🔧 <strong>LLM reasoning pass</strong> — additional corrections applied</div>',
                        unsafe_allow_html=True
                    )
                    for note in llm_fix_notes:
                        st.caption(f"  • {note}")

                # Show corrected query
                if was_modified:
                    st.markdown("**✨ Corrected & optimized query:**")
                    st.code(final_query, language="sql")
                    full = full.replace(
                        f"```pql\n{pql_block}\n```",
                        f"```pql\n{final_query}\n```"
                    )
                else:
                    st.session_state.verified_count += 1
                    has_critical = any(i.severity in ('CRITICAL', 'ERROR') for i in issues)
                    if not has_critical and not blocking:
                        st.markdown(
                            '<div class="verify-pass">✅ <strong>Verified</strong> — passed all 6 reasoning layers</div>',
                            unsafe_allow_html=True
                        )
                    elif not has_critical:
                        st.markdown(
                            '<div class="verify-pass">✅ <strong>Structurally correct</strong> — review insights above</div>',
                            unsafe_allow_html=True
                        )

            st.session_state.messages.append({"role": "assistant", "content": full})

        except Exception as e:
            placeholder.error(f"Groq API error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 23 · INPUT HANDLING
# ─────────────────────────────────────────────────────────────────────────────

if '_pending' in st.session_state:
    pending = st.session_state.pop('_pending')
    st.session_state.messages.append({'role': 'user', 'content': pending})
    with st.chat_message('user', avatar='🧑'):
        st.markdown(pending)
    stream_groq()
    st.rerun()

if prompt := st.chat_input(
    'Describe your PQL query, paste code to validate/fix, or ask about any function… '
    '(e.g. "throughput time per vendor in working hours for P2P cases")'
):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user', avatar='🧑'):
        st.markdown(prompt)
    stream_groq()
