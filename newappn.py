# ╔══════════════════════════════════════════════════════════════╗
# ║  PQL Query Assistant  ·  Groq + LLaMA  ·  Streamlit Cloud   ║
# ║  Single file — push to GitHub, deploy in 2 clicks           ║
# ╠══════════════════════════════════════════════════════════════╣
# ║  LOCAL RUN                                                   ║
# ║    pip install streamlit groq                                ║
# ║    export GROQ_API_KEY=gsk_...                               ║
# ║    streamlit run app.py                                      ║
# ╠══════════════════════════════════════════════════════════════╣
# ║  STREAMLIT CLOUD DEPLOY                                      ║
# ║    1. Push this file + requirements.txt to GitHub            ║
# ║    2. go to share.streamlit.io → New app → your repo        ║
# ║    3. App Settings → Secrets → paste:                       ║
# ║          GROQ_API_KEY = "gsk_..."                            ║
# ║    4. Deploy ✓                                               ║
# ╚══════════════════════════════════════════════════════════════╝

import os
import re
import streamlit as st
from groq import Groq

# ──────────────────────────────────────────────────────────────
#  SECTION 1 · KNOWLEDGE BASE  (230 PQL functions + categories)
# ──────────────────────────────────────────────────────────────

COMPACT_REFS = {
    'CREATE_EVENTLOG': 'Returns an activity table based on a given lead object and included event types. Used to generate event logs from an object perspective in OCPM. Syntax: CREATE_EVENTLOG( lead_object, event_type_list )',
    'PU_COUNT': '''[OFFICIAL DOCS] Counts non-NULL rows in source per target row.
Syntax: PU_COUNT( target_table, source_table.column [, filter_expression] )
- Returns 0 (not NULL) when no matching rows exist — unique among PU functions
- Requires 1:N relationship: target_table is parent (1-side), source is child (N-side)
- target_table can also be DOMAIN_TABLE(...) or CONSTANT()
- PU_COUNT IGNORES global filters — use filter_expression arg for filter-aware counts
- PREFER over PU_COUNT_DISTINCT when column is already a key (much faster)
- PREFER over PU_SUM for counting; PU_COUNT is less expensive than PU_SUM
- Example: PU_COUNT("CASES", "ACTIVITIES"."CASE_ID", "ACTIVITIES"."ACTIVITY" = 'Approve')''',

    'PU_SUM': '''[OFFICIAL DOCS] Sums source column per target row.
Syntax: PU_SUM( target_table, source_table.column [, filter_expression] )
- Returns NULL (not 0) when no matching rows exist
- Requires 1:N relationship between target_table and source table
- PU_SUM IGNORES global filters — filter via filter_expression argument
- PU_COUNT is less expensive than PU_SUM for counting — prefer PU_COUNT when possible
- Example: PU_SUM("VENDORS", "ORDERS"."AMOUNT")
- Example (filtered): PU_SUM("CASES", "ACTIVITIES"."AMOUNT", "ACTIVITIES"."TYPE" = 'Invoice')''',

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
- Requires 1:N relationship
- Used for throughput over multiple grouped cases:
  DATEDIFF('dd', PU_MIN("VENDORS","ACTIVITIES"."TIMESTAMP"), PU_MAX("VENDORS","ACTIVITIES"."TIMESTAMP"))
- From official Celonis FAQ: PU_MAX("_CEL_CASES", SECONDS_BETWEEN(TARGET("_CEL_ACTIVITIES"."EVENTTIME"), SOURCE("_CEL_ACTIVITIES"."EVENTTIME")))''',

    'PU_MIN': '''[OFFICIAL DOCS] Minimum of source column per target row.
Syntax: PU_MIN( target_table, source_table.column [, filter_expression] )
- Returns NULL when no matching rows exist
- Requires 1:N relationship
- Combined with PU_MAX for throughput over multiple grouped cases (NOT CALC_THROUGHPUT which is per-case)''',

    'PU_FIRST': '''[OFFICIAL DOCS] Returns first element of source column for each target row.
Syntax: PU_FIRST( target_table, source_table.column [, filter_expression] [, ORDER BY source_table.column [ASC|DESC]] )
- Returns NULL when no matching rows exist (not 0)
- ALWAYS use explicit ORDER BY unless on implicit-sorted activity table
- No guaranteed order without ORDER BY clause
- PU_FIRST(..., ORDER BY col DESC) == PU_LAST(..., ORDER BY col ASC)
- Result is a scalar at target_table level — DO NOT wrap in another PU function with same target
- Example (first activity timestamp per case):
  PU_FIRST("CASES", "ACTIVITIES"."TIMESTAMP", ORDER BY "ACTIVITIES"."TIMESTAMP" ASC)
- Example (first activity of specific type):
  PU_FIRST("CASES", "ACTIVITIES"."ACTIVITY", "ACTIVITIES"."TYPE" = 'System', ORDER BY "ACTIVITIES"."TIMESTAMP" ASC)
- BIND example (1:N:1 relationship):
  PU_FIRST("VBAK", BIND("VBPA", "KNKK"."KKBER"), "VBPA"."PARVW" = 'RE')''',

    'PU_LAST': '''[OFFICIAL DOCS] Returns last element of source column for each target row.
Syntax: PU_LAST( target_table, source_table.column [, filter_expression] [, ORDER BY source_table.column [ASC|DESC]] )
- Returns NULL when no matching rows exist
- ALWAYS use explicit ORDER BY unless on implicit-sorted activity table
- PU_LAST(..., ORDER BY col DESC) == PU_FIRST(..., ORDER BY col ASC)
- Result is a scalar at target_table level — DO NOT wrap in another PU function with same target
- Example (last activity timestamp per case):
  PU_LAST("CASES", "ACTIVITIES"."TIMESTAMP", ORDER BY "ACTIVITIES"."TIMESTAMP" ASC)
- Example (last status per order):
  PU_LAST("ORDERS", "STATUS_TABLE"."STATUS", ORDER BY "STATUS_TABLE"."CHANGE_DATE" ASC)''',

    'PU_MEDIAN': '''[OFFICIAL DOCS] Median of source column per target row.
Syntax: PU_MEDIAN( target_table, source_table.column [, filter_expression] )
- SIGNIFICANTLY more expensive than PU_AVG (requires sorting)
- Only use when true median is required — otherwise use PU_AVG
- Returns NULL when no matching rows exist''',

    'PU_COUNT_DISTINCT': '''[OFFICIAL DOCS] Distinct count of source column values per target row.
Syntax: PU_COUNT_DISTINCT( target_table, source_table.column [, filter_expression] )
- Returns 0 (not NULL) when no matching rows exist
- USE PU_COUNT instead when column is already a key (PU_COUNT is less expensive)
- Use PU_COUNT_DISTINCT only when you genuinely need to count distinct non-key values''',
    'PU_MODE': 'Most frequent value per target row. Syntax: PU_MODE( target_table, source_table.column [, filter_expression] )',
    'PU_PRODUCT': 'Product of source column per target row. Syntax: PU_PRODUCT( target_table, source_table.column [, filter_expression] )',
    'PU_QUANTILE': 'Quantile (0.0-1.0) of source column per target row. Syntax: PU_QUANTILE( target_table, source_table.column, quantile [, filter_expression] )',
    'PU_TRIMMED_MEAN': 'Trimmed mean (excludes outliers) per target row. Syntax: PU_TRIMMED_MEAN( target_table, source_table.column [, lower_cutoff [, upper_cutoff]] [, filter_expression] )',
    'PU_STRING_AGG': 'Concatenates strings from source per target row. Syntax: PU_STRING_AGG( target_table, source_table.column, delimiter [, filter_expression] [, ORDER BY col] )',
    'PU_STDEV': 'Standard deviation (n-1 method) per target row. Syntax: PU_STDEV( target_table, source_table.column [, filter_expression] )',
    'COUNT_TABLE': 'Counts rows in a table including NULLs (unlike COUNT). Returns original count even when common table differs. Syntax: COUNT_TABLE( table )',
    'MEDIAN': 'Median per group. Applies to INT, FLOAT, DATE. Syntax: MEDIAN( table.column ) NULLs ignored.',
    'QUANTILE': 'Quantile per group. Syntax: QUANTILE( table.column, quantile ) quantile: float 0.0-1.0.',
    'GLOBAL': '''[OFFICIAL DOCS] Isolates aggregation from the common table — prevents join multiplication.
Syntax: GLOBAL( aggregation_expression )
- When a query mixes columns from different table levels (e.g. case + activity), Celonis performs
  an implicit join. This shifts the common table to the activity level, causing case-level
  aggregations to be multiplied by the number of activities per case.
- GLOBAL() anchors the aggregation back to the original table, ignoring the join shift.
- Official example from Celonis FAQ:
  CASE WHEN AVG("Companies"."Value") > GLOBAL(AVG("Companies"."Value")) THEN 'larger' ELSE 'smaller' END
- ALWAYS wrap CALC_THROUGHPUT when combined with activity-level columns:
  GLOBAL(AVG(CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", DAYS))))
- ALWAYS wrap case-level COUNT/SUM when mixed with activity columns:
  GLOBAL(COUNT("CASES"."CASE_ID")) / GLOBAL(COUNT("ACTIVITIES"."ACTIVITY"))''',
    'RUNNING_SUM': 'Cumulative sum of previous rows. Syntax: RUNNING_SUM( column [, ORDER BY (...)] [, PARTITION BY (...)] )',
    'WINDOW_AVG': 'Average over a sliding window. Syntax: WINDOW_AVG( table.values, lower_bound, upper_bound [, ORDER BY ...] [, PARTITION BY ...] )',
    'STRING_AGG': 'Concatenates strings with delimiter. Syntax: STRING_AGG( table.column, "delim" [, ORDER BY ...] [, PARTITION BY ...] )',
    'INDEX_ORDER': 'Integer indices starting from 1 for ordering rows. Syntax: INDEX_ORDER( column [, ORDER BY (...)] [, PARTITION BY (...)] ) Result: INT.',
    'PARTITION': 'Used in window functions as PARTITION BY ( column, ... ) to define groups.',
    'ZSCORE': 'Z-score (standard deviations from mean). Syntax: ZSCORE( table.column [, PARTITION BY (...)] )',
    'INTERPOLATE': 'Interpolates NULL values. Syntax: INTERPOLATE( column, CONSTANT|LINEAR [, ORDER BY ...] [, PARTITION BY ...] )',
    'CALC_THROUGHPUT': '''Calculates throughput time per case between two event range specifiers.
Syntax: CALC_THROUGHPUT( begin_specifier TO end_specifier, REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", unit) [, activity_table.string_col] )
begin_specifier: CASE_START | FIRST_OCCURRENCE['activity'] | LAST_OCCURRENCE['activity']
end_specifier:   CASE_END   | FIRST_OCCURRENCE['activity'] | LAST_OCCURRENCE['activity']
unit: DAYS | HOURS | MINUTES | SECONDS | MILLISECONDS
Returns NULL if start > end, or case has only one activity, or activity name not found.
IMPORTANT: Preferred over DATEDIFF(PU_MIN, PU_MAX) for case-level throughput.
Wrap with GLOBAL() when combined with activity-level columns to prevent join multiplication.
Official example from Celonis docs:
  CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", DAYS))
Average conforming throughput (official doc pattern):
  AVG(CASE WHEN PU_SUM("CASES", ABS(conformance)) = 0
      THEN CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", HOURS)) / 24
      ELSE NULL END)
ALL_OCCURRENCE[''] is DEPRECATED since 4.6 — use CASE_START instead.''',
    'CALC_REWORK': '''[OFFICIAL DOCS] Counts number of activities per case. Result temporarily added to case table.
Syntax: CALC_REWORK() | CALC_REWORK( filter_expression ) | CALC_REWORK( activity_table.column )
- Returns INT column on CASE table (not activity table)
- NULL case IDs → result is 0; cases without join partner in case table are ignored
- filter_expression: restricts which activities are counted
- activity_table.column: selects event log when multiple exist
- Rework detection (repeated activity): FILTER CALC_REWORK("ACTIVITIES"."ACTIVITY" = 'Review') > 1
- Total step count: CALC_REWORK() counts ALL activities per case''',

    'CALC_CROP': 'Crops cases to event range, returns 1 in range, NULL outside. Syntax: CALC_CROP( begin TO end, activity_table.col )',
    'CALC_CROP_TO_NULL': 'Crops cases to event range, keeps values in range, NULL outside. Syntax: CALC_CROP_TO_NULL( begin TO end, activity_table.col )',

    'MATCH_ACTIVITIES': '''[OFFICIAL DOCS] Flags cases containing specified activities. Order-INDEPENDENT.
Syntax: MATCH_ACTIVITIES( [STARTING node_list] [NODE node_list] [ENDING node_list] [EXCLUDING node_list] )
- Returns 1 matching / 0 non-matching — use with FILTER or CASE WHEN
- STARTING: activity must be first; ENDING: must be last; NODE: anywhere; EXCLUDING: must not appear
- Use MATCH_PROCESS for order-sensitive matching
- Example: FILTER MATCH_ACTIVITIES(NODE('Approve'), NODE('Pay'), EXCLUDING('Cancel')) = 1
- Example: FILTER MATCH_ACTIVITIES(STARTING('Create'), ENDING('Close')) = 1''',

    'MATCH_PROCESS': '''[OFFICIAL DOCS] Matches cases against ordered node/edge pattern. Order-SENSITIVE.
Syntax: MATCH_PROCESS( [activity_table.string_col,] node(, node)* CONNECTED BY edge(, edge)* )
- Returns INT: 1 matching, 0 non-matching
- Node types: NODE | OPTIONAL | LOOP | OPTIONAL_LOOP | STARTING | ENDING
  NODE [act1, act2]: one of act1/act2 must appear. Multiple activities = OR logic
  STARTING [act]: first activity. ENDING [act]: last activity. LOOP [act]: appears 1+ times
- Edge types: DIRECT [nodeA, nodeB] = B directly follows A (no gap)
              EVENTUALLY [nodeA, nodeB] = B eventually follows A (gaps allowed)
- LIKE supports wildcards: NODE [LIKE 'Approve%']
- Example:
  FILTER MATCH_PROCESS(
    STARTING ["Create Order"] AS n1,
    NODE ["Approve"] AS n2,
    ENDING ["Close"] AS n3
    CONNECTED BY EVENTUALLY[n1, n2], EVENTUALLY[n2, n3]
  ) = 1''',
    'MATCH_PROCESS_REGEX': 'Filters variants using regex over activity names. Syntax: MATCH_PROCESS_REGEX( [table.col,] "regex_pattern" )',
    'ACTIVITY_LAG': '''[OFFICIAL DOCS] Returns value from preceding row by offset within same case.
Syntax: ACTIVITY_LAG( activity_table.column [, offset] )  Default offset: 1
- Returns NULL if no preceding row at that offset
- Use for transition time: SECONDS_BETWEEN(ACTIVITY_LAG("ACTIVITIES"."TIMESTAMP"), "ACTIVITIES"."TIMESTAMP")''',
    'ACTIVITY_LEAD': '''[OFFICIAL DOCS] Returns value from following row by offset within same case.
Syntax: ACTIVITY_LEAD( activity_table.column [, offset] )  Default offset: 1
- Returns NULL if no following row at that offset''',
    'PROCESS_ORDER': 'DEPRECATED — use INDEX_ACTIVITY_ORDER instead. Returns position of each activity within a case.',
    'INDEX_ACTIVITY_ORDER': '''[OFFICIAL DOCS] Returns 1-based position of each activity within its case.
Syntax: INDEX_ACTIVITY_ORDER( activity_table.column )
- Returns INT; only non-NULL activities counted
- Replaces deprecated PROCESS_ORDER
- Use to identify first/last activity: CASE WHEN INDEX_ACTIVITY_ORDER("ACTIVITIES"."ACTIVITY") = 1 THEN ...''',
    'INDEX_ACTIVITY_LOOP': '''[OFFICIAL DOCS] Returns how many times an activity has already occurred at that point in the case.
Syntax: INDEX_ACTIVITY_LOOP( activity_table.column )
- Returns INT: 0 = first occurrence, 1 = second, 2 = third, etc.
- Parallel activities ordered by absolute timestamp
- Used for rework analysis: FILTER INDEX_ACTIVITY_LOOP("ACTIVITIES"."ACTIVITY") > 0 finds all rework rows''',
    'INDEX_ACTIVITY_TYPE': '''[OFFICIAL DOCS] Returns how many times a specific activity TYPE has occurred at that point in the case.
Syntax: INDEX_ACTIVITY_TYPE( activity_table.column )
- Returns INT — type-specific loop counter per case
- Used for Rework per Activity analysis''',
    'BPMN_CONFORMS': 'Binary BPMN conformance check (1=conforming, 0=not). Syntax: BPMN_CONFORMS( event_table.col, bpmn_model [, ALLOW(...)] )',
    'CONFORMANCE': 'Petri net conformance checking. Returns INT flags. Use with READABLE() for violation descriptions.',
    'READABLE': 'Human-readable violation descriptions from CONFORMANCE. Syntax: READABLE( conformance_query )',
    'VARIANT': 'Returns process variant string per case. Syntax: VARIANT( activity_table.string_column )',
    'TRANSIT_COLUMN': 'Computes transition edges between related cases from two processes.',
    'MANUAL_MINER': 'Defines manual transitions for TRANSIT_COLUMN. Syntax: MANUAL_MINER( activity_table.col, ["A", "B"] )',
    'ADD_DAYS': 'Adds days to a date. Syntax: ADD_DAYS( table.base_col, table.days_col ) base: DATE, days: INT. Output: DATE.',
    'DATEDIFF': '''[OFFICIAL DOCS] Computes difference between two dates in specified unit. Returns FLOAT.
Syntax: DATEDIFF( unit, table.date1, table.date2 ) unit: ms|ss|mi|hh|dd|mm|yy
- Supported input: DATE column type
- NULL in any parameter → NULL result
- For sub-day precision with calendar support use SECONDS_BETWEEN / HOURS_BETWEEN
- Example: DATEDIFF('dd', "ORDERS"."CREATE_DATE", "ORDERS"."CLOSE_DATE")
- Example (cycle time using PU_FIRST/PU_LAST — correct pattern):
  DATEDIFF('dd', PU_FIRST("CASES","ACTIVITIES"."TIMESTAMP", ORDER BY "ACTIVITIES"."TIMESTAMP" ASC),
                 PU_LAST("CASES","ACTIVITIES"."TIMESTAMP", ORDER BY "ACTIVITIES"."TIMESTAMP" ASC))''',
    'HOURS_BETWEEN': 'Difference in hours. Supports calendar. Syntax: HOURS_BETWEEN( table.date1, table.date2 [, calendar] )',
    'MINUTES_BETWEEN': 'Difference in minutes. Syntax: MINUTES_BETWEEN( table.date1, table.date2 [, calendar] )',
    'SECONDS_BETWEEN': 'Difference in seconds. Syntax: SECONDS_BETWEEN( table.date1, table.date2 [, calendar] )',
    'MILLIS_BETWEEN': 'Difference in milliseconds. Syntax: MILLIS_BETWEEN( table.date1, table.date2 [, calendar] )',
    'WORKDAYS_BETWEEN': 'Number of workdays between two dates. Syntax: WORKDAYS_BETWEEN( calendar, table.date1, table.date2 )',
    'ADD_HOURS': 'Adds hours to a timestamp. Syntax: ADD_HOURS( table.start_col, table.hours_col [, calendar] )',
    'ADD_MINUTES': 'Adds minutes. Syntax: ADD_MINUTES( table.start_col, table.minutes_col [, calendar] )',
    'ADD_SECONDS': 'Adds seconds. Syntax: ADD_SECONDS( table.start_col, table.seconds_col [, calendar] )',
    'ADD_MILLIS': 'Adds milliseconds. Syntax: ADD_MILLIS( table.start_col, table.ms_col [, calendar] )',
    'ADD_WORKDAYS': 'Adds workdays using a calendar. Syntax: ADD_WORKDAYS( calendar, date, number_of_days )',
    'TODAY': 'Current date in specified timezone. Syntax: TODAY( [timezone_id] ) Default: UTC.',
    'HOUR_NOW': 'Current hour in specified timezone. Syntax: HOUR_NOW( [timezone_id] )',
    'MINUTE_NOW': 'Current minute in specified timezone. Syntax: MINUTE_NOW( [timezone_id] )',
    'ROUND_DAY': 'Rounds date down to day. Syntax: ROUND_DAY( table.date_col )',
    'ROUND_WEEK': 'Rounds date down to Monday of the week. Syntax: ROUND_WEEK( table.date_col )',
    'ROUND_MONTH': 'Rounds date down to first day of month. Syntax: ROUND_MONTH( table.date_col )',
    'ROUND_QUARTER': 'Rounds date down to beginning of quarter. Syntax: ROUND_QUARTER( col )',
    'CONVERT_TIMEZONE': 'Converts date between timezones. Syntax: CONVERT_TIMEZONE( table.date_col [, from_tz], to_tz )',
    'DATE_MATCH': 'Returns 1 if date matches filter lists. Syntax: DATE_MATCH( col, [YEARS], [QUARTERS], [MONTHS], [WEEKS], [DAYS] )',
    'DAYS_IN_MONTH': 'Returns number of days in the month of the given date. Syntax: DAYS_IN_MONTH( table.col )',
    'IN_CALENDAR': 'Checks if date is within a calendar. Returns 1 or NULL. Syntax: IN_CALENDAR( ts_col, calendar )',
    'REMAP_TIMESTAMPS': '''[OFFICIAL DOCS] Converts DATE column to integer count of time units since epoch (1970-01-01).
Syntax: REMAP_TIMESTAMPS( activity_table.timestamp_col, unit [, calendar_specification] )
Units: DAYS | HOURS | MINUTES | SECONDS | MILLISECONDS
- Primary use: provides the timestamps argument to CALC_THROUGHPUT
- Also used in SOURCE/TARGET edge throughput calculations
- Supports 3 calendar types: WEEKDAY_CALENDAR, FACTORY_CALENDAR, WORKDAY_CALENDAR
- Multiple calendars can be combined with INTERSECT
- Returns INT (epoch offset in specified unit); NULL input → NULL output
- Official example: REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", DAYS)
- With workday calendar: REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", HOURS, WEEKDAY_CALENDAR(MON,TUE,WED,THU,FRI))
- Process Explorer uses: REMAP_TIMESTAMPS("_CEL_ACTIVITIES"."EVENTTIME", SECONDS) for edge KPIs''',
    'FACTORY_CALENDAR': 'Defines factory calendar with specific work intervals. Used with REMAP_TIMESTAMPS.',
    'WORKDAY_CALENDAR': 'Defines work days from a table. Used with ADD_WORKDAYS and date diff functions.',
    'WEEKDAY_CALENDAR': 'Defines which weekdays count as work days. Syntax: WEEKDAY_CALENDAR( MON, TUE, ... )',
    'TO_TIMESTAMP': 'Deprecated. Use TO_DATE. Converts STRING to DATE with format.',
    'UPPER': 'Uppercases a string. Syntax: UPPER( table.column )',
    'LOWER': 'Lowercases a string. Syntax: LOWER( table.column )',
    'CONCAT': 'Concatenates strings. Syntax: CONCAT( col1, ..., colN ) or col1 || col2. NULL in any arg = NULL result.',
    'STRING_SPLIT': 'Splits string by pattern. Zero-based index. Syntax: STRING_SPLIT( table.col, pattern, index ) Returns NULL if index out of bounds.',
    'TO_STRING': 'Converts INT or DATE to STRING. Syntax: TO_STRING( table.col [, FORMAT("%Y-%m-%d")] )',
    'FORMAT': 'Specifies date/string format. Used in TO_DATE and TO_STRING. Syntax: FORMAT( "%Y-%m-%d" )',
    'MATCH_STRINGS': 'Finds top-k matching strings by edit distance. Syntax: MATCH_STRINGS( table1.col, table2.col [, TOP_K(k)] [, SEPARATOR(sep)] )',
    'IN_LIKE': 'Pattern matching with wildcards % and _. Syntax: table.col IN_LIKE( "pattern%" ) or IN_LIKE( table2.col )',
    'EDIT_THRESHOLD': 'Edit distance threshold for CLUSTER_STRINGS. Syntax: EDIT_THRESHOLD( distance )',
    'TOP_K': 'Number of matches in MATCH_STRINGS. Syntax: TOP_K( k ) where k <= 100.',
    'SEPARATOR': 'Separator between results in MATCH_STRINGS. Syntax: SEPARATOR( "," )',
    'ABS': 'Absolute value. Syntax: ABS( table.column )',
    'POWER': 'Value raised to a power. Syntax: POWER( table.col, exponent ) Output: FLOAT.',
    'MODULO': 'Remainder of division. Syntax: MODULO( dividend, divisor ) or dividend % divisor.',
    'GREATEST': 'Maximum value across multiple columns. Syntax: GREATEST( col1, col2, ..., colN ) Good CASE WHEN alternative.',
    'LEAST': 'Minimum value across multiple columns. Syntax: LEAST( col1, col2, ..., colN )',
    'COALESCE': 'First non-NULL value. Syntax: COALESCE( col1, col2, ..., colN )',
    'ISNULL': 'Returns 1 if NULL, 0 otherwise. Syntax: ISNULL( table.column )',
    'CASE': 'Conditional expression. Syntax: CASE WHEN cond THEN val [WHEN ...] ELSE default END',
    'WHEN': 'Part of CASE WHEN ... THEN ... ELSE ... END conditional.',
    'AND': 'Logical AND. Used in FILTER, CASE WHEN, and join conditions.',
    'OR': 'Logical OR. Used in FILTER and CASE WHEN conditions.',
    'NOT': 'Logical NOT. Used in NOT IN, NOT IN_LIKE, IS NOT NULL.',
    'IN': 'Checks membership in a list. Syntax: table.col IN( "val1", "val2" )',
    'MULTI_IN': 'Multi-column tuple membership. Syntax: MULTI_IN( (col,...), (val1,...), (val2,...) )',
    'BUCKET_UPPER_BOUND': 'Histogram bucket upper bounds. Syntax: BUCKET_UPPER_BOUND( table.col [, SUGGESTED_COUNT(n)] )',
    'SUGGESTED_COUNT': 'Suggests number of buckets in BUCKET functions. Syntax: SUGGESTED_COUNT( count )',
    'SUGGESTED_WIDTH': 'Suggests bucket width. Syntax: SUGGESTED_WIDTH( width )',
    'MAX_COUNT': 'Limits number of buckets in BUCKET functions. Syntax: MAX_COUNT( max )',
    'FILTER': 'Filters result set. Syntax: FILTER table.col = "value"; Multiple filters merge by AND.',
    'FILTER_TO_NULL': 'Makes functions filter-aware. Syntax: FILTER_TO_NULL( table.col ) Prefer PU-function filter arg when possible.',
    'BIND_FILTERS': 'Pulls filter to specified table. Syntax: BIND_FILTERS( target_table, condition [, condition]* )',
    'BIND': 'Pulls a value to a target table. Used in PU-functions for 1:N:1 relationships. Syntax: BIND( target_table, value )',
    'LOOKUP': 'Left outer join ignoring predefined joins. Syntax: LOOKUP( target_table, source_col, (join_cond) )',
    'REMAP_VALUES': 'Maps STRING column values. Syntax: REMAP_VALUES( table.col, [old1, new1], ..., [default] )',
    'REMAP_INTS': 'Maps INT column values. Syntax: REMAP_INTS( table.col, [old1, new1], ..., [default] )',
    'DOMAIN_TABLE': 'Creates table with all distinct combinations of columns. Syntax: DOMAIN_TABLE( table.col1, table.col2, ... )',
    'GENERATE_RANGE': 'Creates a value range. Syntax: GENERATE_RANGE( step_size, range_start, range_end ) Max 10,000 elements.',
    'RANGE_APPEND': 'Creates a range and appends to a column. Syntax: RANGE_APPEND( table.col, step_size, range_end )',
    'VARIABLE': 'Dynamic variable in PQL. Use <% if(VAR != "") { %> FILTER ... <% } %> to guard empty variables.',
    'KPI': 'References a saved KPI in OCPM LINK_PATH context.',
    'UNIQUE_ID': 'Unique INT for each unique tuple of input columns. Syntax: UNIQUE_ID( table.col1, ..., table.colN )',
    'CONSTANT': 'Used as target table in PU-functions to produce a constant result. Syntax: CONSTANT()',
    'COMMON_TABLE': 'References the common table of multiple expressions. Syntax: COMMON_TABLE( expr1, expr2 )',
    'COLUMN_TYPE': 'Returns data type of a column as STRING (INT/FLOAT/STRING/DATE). Syntax: COLUMN_TYPE( table.col )',
    'ARGUMENT_COUNT': 'Counts number of arguments passed. Syntax: ARGUMENT_COUNT( arg1, arg2, ... )',
    'MERGE_EVENTLOG': 'Merges columns from two activity tables into one. Syntax: MERGE_EVENTLOG( target_table.col, [FILTER ...] )',
    'MERGE_EVENTLOG_DISTINCT': 'Like MERGE_EVENTLOG but removes duplicate activities.',
    'EVENTLOG_SOURCE_TABLE': 'Returns source table name for each row in a dynamic event log. Syntax: EVENTLOG_SOURCE_TABLE( eventlog.col )',
    'LINK_PATH': 'Traverses object links. Syntax: LINK_PATH( table.col [, CONSTRAINED BY (START(...), END(...))] )',
    'LINK_SOURCE': 'Source objects of Object Link. Syntax: LINK_SOURCE( link_name, table.col )',
    'LINK_TARGET': 'Target objects of Object Link. Syntax: LINK_TARGET( link_name, table.col )',
    'LINK_FILTER': 'Filters by link traversal. Syntax: LINK_FILTER( filter_expr, ANCESTORS|DESCENDANTS [, hops] )',
    'LINK_FILTER_ORDERED': 'Order-aware LINK_FILTER (only for Signal Link). Considers timestamp order.',
    'LINK_ATTRIBUTES': 'Returns link attribute values. Syntax: LINK_ATTRIBUTES( link_name, attr_col )',
    'LINK_OBJECTS': 'Creates table of all objects in the Object Link graph.',
    'UNION_ALL': 'Vertical concatenation of columns. Use with UNION_ALL_PULLBACK.',
    'UNION_ALL_TABLE': 'Vertical concatenation of tables. Syntax: UNION_ALL_TABLE( table1, ..., tableN ) 2-16 tables.',
    'UNION_ALL_PULLBACK': 'Projects UNION_ALL section back to source table. Syntax: UNION_ALL_PULLBACK( union_col, index )',
    'CASE_ID_COLUMN': 'References case ID column without exact name. Syntax: CASE_ID_COLUMN( [expr] )',
    'CASE_TABLE': 'References the case table. Syntax: CASE_TABLE( [expr] )',
    'ACTIVITY_TABLE': 'References the activity table in OCPM. Syntax: ACTIVITY_TABLE( LINK_PATH(...) )',
    'ACTIVITY_COLUMN': 'References the activity column. Syntax: ACTIVITY_COLUMN( [expr] )',
    'TIMESTAMP_COLUMN': 'References the timestamp column. Syntax: TIMESTAMP_COLUMN( [expr] )',
    'END_TIMESTAMP_COLUMN': 'References the end timestamp column. Syntax: END_TIMESTAMP_COLUMN( [expr] )',
    'CURRENCY_CONVERT': 'Converts currency. Syntax: CURRENCY_CONVERT( amount, FROM("USD"), TO("EUR"), date, "RATES" )',
    'CURRENCY_CONVERT_SAP': 'Converts SAP currency using TCURR/TCURF/TCURX internal tables.',
    'CURRENCY_SAP': 'Adjusts SAP amounts for decimal places. Syntax: CURRENCY_SAP( table.amount_col, table.currency_col )',
    'QUANTITY_CONVERT': 'Converts quantity units. Syntax: QUANTITY_CONVERT( amount, FROM("unit1"), TO("unit2"), id_col, "RATES" )',
    'KMEANS': 'K-means++ clustering. Syntax: KMEANS( k, col1, col2 ) or KMEANS( TRAIN_KM(...), CLUSTER(...) )',
    'TRAIN_KM': 'Trains a KMeans model. Syntax: TRAIN_KM( k, INPUT( table.col1, ... ) )',
    'CLUSTER': 'Assigns rows to clusters. Syntax: CLUSTER( TRAIN_KM(...), table.col, ... )',
    'LINEAR_REGRESSION': 'Linear regression. Syntax: LINEAR_REGRESSION( TRAIN_LM( INPUT(...), OUTPUT(...) ), PREDICT( col ) )',
    'TRAIN_LM': 'Trains a Linear Regression model. Syntax: TRAIN_LM( INPUT( table.col, ... ), OUTPUT( table.col ) )',
    'PREDICT': 'Specifies prediction columns. Syntax: PREDICT( table.col, ... )',
    'BPMN_MATCH_EXCESSIVE': 'Activity occurs at right place but too often — used in BPMN_CONFORMS ALLOW list.',
    'BPMN_MATCH_MISSING': 'Required activity missing from trace — BPMN_CONFORMS shorthand.',
    'BPMN_MATCH_OUT_OF_SEQUENCE': 'Activity at wrong position — BPMN_CONFORMS shorthand.',
    'BPMN_MATCH_UNDESIRED': 'Activity present that should not be — BPMN_CONFORMS shorthand.',
    'BPMN_MATCH_UNMAPPED': 'Activity with no model mapping — BPMN_CONFORMS shorthand.',
    'SEQUENCE': 'Models sequential flow in BPMN_CONFORMS. Syntax: SEQUENCE("A", "B", "C")',
    'PARALLEL': 'Models parallel paths in BPMN_CONFORMS. Syntax: PARALLEL("A", "B")',
    'EXCLUSIVE_CHOICE': 'Models XOR gateway in BPMN_CONFORMS.',
    'ALLOW': 'Allows deviations in BPMN_CONFORMS. Syntax: ALLOW( BPMN_MATCH_UNDESIRED(ANY) )',
    'COUNT': 'Counts non-NULL rows. Syntax: COUNT(table.column). Often wrapped with GLOBAL() when mixing table levels.',

    # ── Standard aggregation (missing) ──────────────────────────
    'AVG': 'Standard average aggregation per group. Syntax: AVG( table.column ) Returns FLOAT. Respects global filters (unlike PU_AVG).',
    'SUM': 'Standard sum aggregation per group. Syntax: SUM( table.column ) Respects global filters (unlike PU_SUM).',
    'MAX': 'Standard maximum per group. Syntax: MAX( table.column )',
    'MIN': 'Standard minimum per group. Syntax: MIN( table.column )',
    'STDEV': 'Standard deviation (n-1 method) per group. Syntax: STDEV( table.column )',
    'VAR': 'Variance (n-1 method) per group. Syntax: VAR( table.column )',
    'TRIMMED_MEAN': 'Trimmed mean per group excluding outliers. Syntax: TRIMMED_MEAN( table.column [, lower_cutoff [, upper_cutoff]] )',
    'COUNT_DISTINCT': 'Distinct count per group. Syntax: COUNT_DISTINCT( table.column ) Use COUNT when column is already a key.',

    # ── Window / Moving aggregation (all missing) ─────────────
    'MOVING_AVG': 'Moving average over ordered rows within partition. Syntax: MOVING_AVG( table.col, lower_bound, upper_bound [, ORDER BY ...] [, PARTITION BY ...] )',
    'MOVING_SUM': 'Moving sum over a window of rows. Syntax: MOVING_SUM( table.col, lower_bound, upper_bound [, ORDER BY ...] [, PARTITION BY ...] )',
    'MOVING_COUNT': 'Moving count over a window of rows. Syntax: MOVING_COUNT( table.col, lower_bound, upper_bound [, ORDER BY ...] [, PARTITION BY ...] )',
    'MOVING_COUNT_DISTINCT': 'Moving distinct count. Syntax: MOVING_COUNT_DISTINCT( table.col, lower_bound, upper_bound [, ORDER BY ...] [, PARTITION BY ...] )',
    'MOVING_MAX': 'Moving maximum over a window. Syntax: MOVING_MAX( table.col, lower_bound, upper_bound [, ORDER BY ...] [, PARTITION BY ...] )',
    'MOVING_MIN': 'Moving minimum over a window. Syntax: MOVING_MIN( table.col, lower_bound, upper_bound [, ORDER BY ...] [, PARTITION BY ...] )',
    'MOVING_MEDIAN': 'Moving median over a window. Expensive — requires sorting. Syntax: MOVING_MEDIAN( table.col, lower_bound, upper_bound [, ORDER BY ...] [, PARTITION BY ...] )',
    'MOVING_STDEV': 'Moving standard deviation. Syntax: MOVING_STDEV( table.col, lower_bound, upper_bound [, ORDER BY ...] [, PARTITION BY ...] )',
    'MOVING_TRIMMED_MEAN': 'Moving trimmed mean. Syntax: MOVING_TRIMMED_MEAN( table.col, lower_bound, upper_bound [, lower_cutoff, upper_cutoff] [, ORDER BY ...] [, PARTITION BY ...] )',
    'MOVING_VAR': 'Moving variance. Syntax: MOVING_VAR( table.col, lower_bound, upper_bound [, ORDER BY ...] [, PARTITION BY ...] )',
    'RUNNING_TOTAL': 'Cumulative running total (alias of RUNNING_SUM in older PQL versions). Syntax: RUNNING_TOTAL( table.col [, ORDER BY ...] [, PARTITION BY ...] )',

    # ── DateTime difference (missing) ────────────────────────────
    'DATE_BETWEEN': 'Difference between two dates in days (integer). Syntax: DATE_BETWEEN( table.date1, table.date2 ) Returns INT.',
    'DAYS_BETWEEN': 'Difference in days (float). Syntax: DAYS_BETWEEN( table.date1, table.date2 [, calendar] )',
    'MONTHS_BETWEEN': 'Difference in months. Syntax: MONTHS_BETWEEN( table.date1, table.date2 )',
    'YEARS_BETWEEN': 'Difference in years. Syntax: YEARS_BETWEEN( table.date1, table.date2 )',

    # ── DateTime modification (missing) ──────────────────────────
    'ADD_MONTHS': 'Adds months to a date. Syntax: ADD_MONTHS( table.date_col, table.months_col )',
    'ADD_YEARS': 'Adds years to a date. Syntax: ADD_YEARS( table.date_col, table.years_col )',

    # ── DateTime projection (missing) ────────────────────────────
    'CALENDAR_WEEK': 'Returns the calendar week number (1-53) of a date. Syntax: CALENDAR_WEEK( table.date_col )',
    'DAY': 'Returns the day of month (1-31) from a date. Syntax: DAY( table.date_col )',
    'DAY_OF_WEEK': 'Returns day of week (1=Monday … 7=Sunday). Syntax: DAY_OF_WEEK( table.date_col )',
    'MONTH': 'Returns the month number (1-12) from a date. Syntax: MONTH( table.date_col )',
    'QUARTER': 'Returns the quarter (1-4) from a date. Syntax: QUARTER( table.date_col )',
    'YEAR': 'Returns the 4-digit year from a date. Syntax: YEAR( table.date_col )',
    'HOURS': 'Returns the hour component (0-23) of a timestamp. Syntax: HOURS( table.timestamp_col )',
    'MINUTES': 'Returns the minute component (0-59). Syntax: MINUTES( table.timestamp_col )',
    'SECONDS': 'Returns the seconds component (0-59). Syntax: SECONDS( table.timestamp_col )',
    'MILLIS': 'Returns the milliseconds component. Syntax: MILLIS( table.timestamp_col )',

    # ── DateTime rounding (missing) ───────────────────────────────
    'ROUND_HOUR': 'Rounds timestamp down to the nearest hour. Syntax: ROUND_HOUR( table.timestamp_col )',
    'ROUND_MINUTE': 'Rounds timestamp down to the nearest minute. Syntax: ROUND_MINUTE( table.timestamp_col )',
    'ROUND_SECOND': 'Rounds timestamp down to the nearest second. Syntax: ROUND_SECOND( table.timestamp_col )',
    'ROUND_YEAR': 'Rounds date down to start of year. Syntax: ROUND_YEAR( table.date_col )',

    # ── Math (missing) ────────────────────────────────────────────
    'ABC': 'Absolute value (alias of ABS). Syntax: ABC( table.column )',
    'ADD': 'Addition operator. Also available as + operator. Syntax: ADD( col1, col2 ) or col1 + col2',
    'CEIL': 'Rounds up to nearest integer. Syntax: CEIL( table.column ) Returns INT.',
    'DIV': 'Integer division (floor division). Syntax: DIV( dividend, divisor ) or dividend DIV divisor.',
    'FLOOR': 'Rounds down to nearest integer. Syntax: FLOOR( table.column ) Returns INT.',
    'LOG': 'Natural logarithm. Syntax: LOG( table.column ) Returns FLOAT. Column must be > 0.',
    'MULT': 'Multiplication operator. Also available as * operator. Syntax: MULT( col1, col2 ) or col1 * col2',
    'ROUND': 'Rounds to specified decimal places. Syntax: ROUND( table.column, decimal_places ) Returns FLOAT.',
    'SQRT': 'Square root. Syntax: SQRT( table.column ) Returns FLOAT.',
    'SQUARE': 'Squares a value. Syntax: SQUARE( table.column ) Returns FLOAT. Equivalent to POWER(col, 2).',
    'SUB': 'Subtraction operator. Also available as - operator. Syntax: SUB( col1, col2 ) or col1 - col2',

    # ── Predicate (missing) ──────────────────────────────────────
    'BETWEEN': 'Checks if value is within range (inclusive). Syntax: table.col BETWEEN lower AND upper Returns 1/0.',
    'IS_NULL': 'Returns 1 if value is NULL. Syntax: table.col IS NULL (operator form, not function). Same as ISNULL().',
    'LIKE': 'Pattern matching with wildcards. Syntax: table.col LIKE "pattern%" where % = any chars, _ = one char.',
    'LEFT': 'Deprecated string function. Use SUBSTRING instead.',
    'LEN': 'Returns length of string. Syntax: LEN( table.string_col ) Returns INT.',
    'LTRIM': 'Removes leading whitespace. Syntax: LTRIM( table.string_col )',
    'REVERSE': 'Reverses a string. Syntax: REVERSE( table.string_col )',
    'RIGHT': 'Deprecated string function. Use SUBSTRING instead.',
    'RTRIM': 'Removes trailing whitespace. Syntax: RTRIM( table.string_col )',
    'STR_TO_INT': 'Converts string to integer. Syntax: STR_TO_INT( table.string_col ) Returns INT or NULL if not numeric.',
    'STRINGHASH': 'Returns hash of string as INT. Syntax: STRINGHASH( table.string_col )',
    'SUBSTRING': 'Extracts substring. Syntax: SUBSTRING( table.string_col, start_pos [, length] ) 1-based indexing.',

    # ── Process (missing) ─────────────────────────────────────────
    'ACTIVATION_COUNT': 'Returns number of times an edge (transition) was activated. Syntax: ACTIVATION_COUNT( SOURCE["A"] TARGET["B"] )',
    'CLUSTER_VARIANTS': 'Clusters process variants. Syntax: CLUSTER_VARIANTS( k [, ESTIMATE_CLUSTER_PARAMS(...)] )',
    'ESTIMATE_CLUSTER_PARAMS': 'Estimates optimal cluster parameters for CLUSTER_VARIANTS. Syntax: ESTIMATE_CLUSTER_PARAMS( max_k )',
    'PROCESS_EQUALS': 'Checks if case follows exact process sequence. Syntax: PROCESS_EQUALS( "A" > "B" > "C" ) Returns 1/0.',
    'SOURCE_TARGET': 'Computes values for process edges (transitions). SOURCE("ACTIVITIES"."TIMESTAMP") gives start timestamp; TARGET gives end timestamp of the edge.',

    # ── Statistics (missing) ──────────────────────────────────────
    'QNORM': 'Quantile of normal distribution. Syntax: QNORM( probability ) Returns FLOAT. probability: 0.0-1.0.',

    # ── ML (missing) ──────────────────────────────────────────────
    'DECISION_TREE': 'Decision tree classification. Syntax: DECISION_TREE( TRAIN_DT( INPUT(...), OUTPUT(...) ), PREDICT(...) )',

    # ── Custom (missing) ──────────────────────────────────────────
    'USER_NAME': 'Returns the currently logged-in username. Syntax: USER_NAME() Returns STRING.',

    # ── Data generation (missing) ─────────────────────────────────
    'RANGE': 'Creates a range of values (older syntax). Syntax: RANGE( start, end, step ) Use GENERATE_RANGE in modern PQL.',
}

PANEL_DATA = {
    'Pull-Up (PU) Aggregation': [
        {'name': 'PU_COUNT',          'doc': 'Count rows in source per target row. Prefer over PU_COUNT_DISTINCT for key cols.'},
        {'name': 'PU_SUM',            'doc': 'Sum source column per target row.'},
        {'name': 'PU_AVG',            'doc': 'Average of source column per target row. Much cheaper than PU_MEDIAN.'},
        {'name': 'PU_MAX',            'doc': 'Maximum of source column per target row.'},
        {'name': 'PU_MIN',            'doc': 'Minimum of source column per target row.'},
        {'name': 'PU_FIRST',          'doc': 'First element of source column per target row. Supports ORDER BY.'},
        {'name': 'PU_LAST',           'doc': 'Last element of source column per target row. Supports ORDER BY.'},
        {'name': 'PU_MEDIAN',         'doc': 'Median per target row. Expensive — use PU_AVG when possible.'},
        {'name': 'PU_COUNT_DISTINCT', 'doc': 'Distinct count per target row. Use PU_COUNT for key columns.'},
        {'name': 'PU_MODE',           'doc': 'Most frequent value per target row.'},
        {'name': 'PU_PRODUCT',        'doc': 'Product of source column per target row.'},
        {'name': 'PU_QUANTILE',       'doc': 'Quantile (0.0-1.0) per target row.'},
        {'name': 'PU_TRIMMED_MEAN',   'doc': 'Trimmed mean (excludes outliers) per target row.'},
        {'name': 'PU_STRING_AGG',     'doc': 'Concatenates strings from source per target row.'},
        {'name': 'PU_STDEV',          'doc': 'Standard deviation (n-1 method) per target row.'},
    ],
    'Standard Aggregation': [
        {'name': 'COUNT_TABLE', 'doc': 'Counts rows including NULLs. Returns original count regardless of common table.'},
        {'name': 'MEDIAN',      'doc': 'Median per group. INT, FLOAT, DATE.'},
        {'name': 'QUANTILE',    'doc': 'Quantile per group. Syntax: QUANTILE( table.col, quantile )'},
        {'name': 'GLOBAL',      'doc': 'Isolates aggregation to prevent join multiplication. Use when mixing table levels.'},
        {'name': 'RUNNING_SUM', 'doc': 'Cumulative sum. Supports ORDER BY and PARTITION BY.'},
        {'name': 'WINDOW_AVG',  'doc': 'Average over a sliding window of rows.'},
        {'name': 'STRING_AGG',  'doc': 'Aggregates strings with a delimiter.'},
        {'name': 'INDEX_ORDER', 'doc': 'Integer indices from 1. Supports ORDER BY and PARTITION BY.'},
        {'name': 'ZSCORE',      'doc': 'Z-score normalization. Supports PARTITION BY.'},
        {'name': 'INTERPOLATE', 'doc': 'Interpolates NULL values (CONSTANT or LINEAR).'},
    ],
    'Process & Conformance': [
        {'name': 'CALC_THROUGHPUT',    'doc': 'Throughput time. Wrap with GLOBAL() when mixing with activity KPIs.'},
        {'name': 'CALC_REWORK',        'doc': 'Counts activities per case. Returns INT on case table.'},
        {'name': 'CALC_CROP',          'doc': 'Crops cases to event range. Returns 1 in range, NULL outside.'},
        {'name': 'CALC_CROP_TO_NULL',  'doc': 'Crops cases to event range. Keeps values in range, NULL outside.'},
        {'name': 'MATCH_ACTIVITIES',   'doc': 'Flags cases with activities (order-independent). Supports EXCLUDING.'},
        {'name': 'MATCH_PROCESS',      'doc': 'Matches variants against node/edge pattern (order-sensitive).'},
        {'name': 'MATCH_PROCESS_REGEX','doc': 'Filters variants using regex over activity names.'},
        {'name': 'ACTIVITY_LAG',       'doc': 'Previous row by offset within a case. Default offset: 1.'},
        {'name': 'ACTIVITY_LEAD',      'doc': 'Next row by offset within a case. Default offset: 1.'},
        {'name': 'BPMN_CONFORMS',      'doc': 'Binary BPMN conformance (1/0). Supports ALLOW() for tolerances.'},
        {'name': 'CONFORMANCE',        'doc': 'Petri net conformance. Use with READABLE() for descriptions.'},
        {'name': 'VARIANT',            'doc': 'Process variant string per case.'},
        {'name': 'SEQUENCE',           'doc': 'BPMN_CONFORMS helper: sequential flow.'},
        {'name': 'PARALLEL',           'doc': 'BPMN_CONFORMS helper: parallel paths.'},
    ],
    'DateTime': [
        {'name': 'ADD_DAYS',          'doc': 'Adds days to a date.'},
        {'name': 'DATEDIFF',          'doc': 'Date difference. Units: ms|ss|mi|hh|dd|mm|yy'},
        {'name': 'HOURS_BETWEEN',     'doc': 'Difference in hours. Supports calendar.'},
        {'name': 'WORKDAYS_BETWEEN',  'doc': 'Number of workdays between dates.'},
        {'name': 'ADD_HOURS',         'doc': 'Adds hours. Supports calendar.'},
        {'name': 'ADD_WORKDAYS',      'doc': 'Adds workdays using a calendar.'},
        {'name': 'ROUND_DAY',         'doc': 'Rounds down to day.'},
        {'name': 'ROUND_WEEK',        'doc': 'Rounds down to Monday of the week.'},
        {'name': 'ROUND_MONTH',       'doc': 'Rounds down to first day of month.'},
        {'name': 'ROUND_QUARTER',     'doc': 'Rounds down to start of quarter.'},
        {'name': 'TODAY',             'doc': 'Current date. Syntax: TODAY([timezone])'},
        {'name': 'CONVERT_TIMEZONE',  'doc': 'Converts date between timezones.'},
        {'name': 'DATE_MATCH',        'doc': 'Returns 1 if date matches filter lists.'},
        {'name': 'REMAP_TIMESTAMPS',  'doc': 'Remaps timestamps per calendar/unit. Used in CALC_THROUGHPUT.'},
        {'name': 'FACTORY_CALENDAR',  'doc': 'Factory calendar with work intervals.'},
        {'name': 'WEEKDAY_CALENDAR',  'doc': 'Calendar specifying work weekdays.'},
        {'name': 'WORKDAY_CALENDAR',  'doc': 'Calendar from a workday table.'},
    ],
    'String': [
        {'name': 'UPPER',        'doc': 'Uppercase. UPPER( table.col )'},
        {'name': 'LOWER',        'doc': 'Lowercase. LOWER( table.col )'},
        {'name': 'CONCAT',       'doc': 'Concatenates strings. NULL in any arg = NULL result.'},
        {'name': 'STRING_SPLIT', 'doc': 'Splits string by pattern. Zero-based index.'},
        {'name': 'TO_STRING',    'doc': 'Converts INT or DATE to STRING.'},
        {'name': 'IN_LIKE',      'doc': 'Pattern matching with wildcards % and _.'},
        {'name': 'MATCH_STRINGS','doc': 'Fuzzy matching by edit distance. Supports TOP_K.'},
        {'name': 'REMAP_VALUES', 'doc': 'Maps STRING values to new values.'},
        {'name': 'STRING_AGG',   'doc': 'Aggregates strings with delimiter.'},
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
        {'name': 'BUCKET_UPPER_BOUND','doc': 'Histogram bucket upper bounds.'},
        {'name': 'ZSCORE',            'doc': 'Z-score normalization.'},
    ],
    'Filter & Lookup': [
        {'name': 'FILTER',       'doc': 'Filters result set. Multiple filters merge by AND.'},
        {'name': 'FILTER_TO_NULL','doc': 'Makes functions filter-aware. Prefer PU-function filter arg.'},
        {'name': 'BIND_FILTERS', 'doc': 'Pulls filter to specified table.'},
        {'name': 'BIND',         'doc': 'Pulls value to target table. Used for 1:N:1 relationships.'},
        {'name': 'IN',           'doc': 'Membership test. Syntax: col IN( "val1", "val2" )'},
        {'name': 'MULTI_IN',     'doc': 'Multi-column tuple membership test.'},
        {'name': 'LOOKUP',       'doc': 'Left outer join ignoring predefined joins.'},
        {'name': 'REMAP_VALUES', 'doc': 'Maps STRING values to new values.'},
        {'name': 'DOMAIN_TABLE', 'doc': 'All distinct combinations of columns.'},
        {'name': 'GENERATE_RANGE','doc': 'Creates a value range. Max 10,000 elements.'},
    ],
    'Event Log & OCPM': [
        {'name': 'CREATE_EVENTLOG',       'doc': 'Creates activity table from OCPM object perspective.'},
        {'name': 'MERGE_EVENTLOG',        'doc': 'Merges columns from two activity tables.'},
        {'name': 'MERGE_EVENTLOG_DISTINCT','doc': 'Like MERGE_EVENTLOG but removes duplicates.'},
        {'name': 'LINK_PATH',             'doc': 'Traverses object links. Supports CONSTRAINED BY.'},
        {'name': 'LINK_SOURCE',           'doc': 'Source objects of Object Link.'},
        {'name': 'LINK_TARGET',           'doc': 'Target objects of Object Link.'},
        {'name': 'LINK_FILTER',           'doc': 'Filters by ANCESTORS or DESCENDANTS link traversal.'},
        {'name': 'LINK_OBJECTS',          'doc': 'All objects in the Object Link graph.'},
        {'name': 'UNION_ALL',             'doc': 'Vertical concatenation of columns.'},
        {'name': 'UNION_ALL_TABLE',       'doc': 'Vertical concatenation of tables (2-16).'},
        {'name': 'EVENTLOG_SOURCE_TABLE', 'doc': 'Source table name for each row in dynamic event log.'},
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
        {'name': 'TRAIN_LM',          'doc': 'Trains a linear regression model.'},
        {'name': 'PREDICT',           'doc': 'Specifies prediction columns in LINEAR_REGRESSION.'},
        {'name': 'MATCH_STRINGS',     'doc': 'Fuzzy string matching by edit distance.'},
        {'name': 'ZSCORE',            'doc': 'Z-score normalization.'},
    ],
}

CATEGORY_ICONS = {
    'Pull-Up (PU) Aggregation': '⬆',
    'Standard Aggregation': '∑',
    'Process & Conformance': '⚙',
    'DateTime': '📅',
    'String': 'Aa',
    'Math & Logic': '±',
    'Filter & Lookup': '🔍',
    'Event Log & OCPM': '🔗',
    'Currency & Quantity': '💱',
    'ML & Clustering': '🧠',
}

# ──────────────────────────────────────────────────────────────
# SMART FUNCTION RETRIEVAL
# ──────────────────────────────────────────────────────────────

FUNCTION_NAMES = list(COMPACT_REFS.keys())
PU_FUNCTIONS = [fn for fn in FUNCTION_NAMES if fn.startswith("PU_")]

# Intent → function mapping based on official Celonis docs
INTENT_PATTERNS = [
    # PU aggregation hints
    (r'per\s+(case|vendor|order|customer|supplier|group|\w+)', PU_FUNCTIONS[:8]),
    (r'(aggregate|group\s+by|count\s+per|sum\s+per|average\s+per)', PU_FUNCTIONS[:8]),
    # Throughput hints → always suggest CALC_THROUGHPUT
    (r'(throughput|cycle.?time|lead.?time|duration|process.?time|elapsed)', ['CALC_THROUGHPUT', 'REMAP_TIMESTAMPS', 'GLOBAL']),
    (r'(first.*last|start.*end|begin.*end).*(time|date|day)', ['CALC_THROUGHPUT', 'REMAP_TIMESTAMPS', 'PU_FIRST', 'PU_LAST', 'DATEDIFF']),
    # Rework hints
    (r'(rework|repeat|loop|same.?activit|revisit|multiple.?time)', ['CALC_REWORK', 'INDEX_ACTIVITY_LOOP', 'INDEX_ACTIVITY_TYPE']),
    # Process path hints
    (r'(conform|path|sequence|order.*activit|activit.*order|follow)', ['MATCH_PROCESS', 'MATCH_ACTIVITIES', 'CALC_THROUGHPUT']),
    # Date/time hints
    (r'(days?\s+between|hours?\s+between|date.?diff|workday|calendar)', ['DATEDIFF', 'HOURS_BETWEEN', 'WORKDAYS_BETWEEN', 'REMAP_TIMESTAMPS']),
    # Automation rate hint
    (r'(automat|system.?activit|manual.?activit|bot)', ['PU_COUNT', 'CALC_REWORK', 'GLOBAL']),
    # Variant hints
    (r'(variant|process.?flow|happy.?path)', ['VARIANT', 'MATCH_PROCESS', 'MATCH_PROCESS_REGEX']),
    # Running/window hints
    (r'(running|cumulative|rolling|window|moving)', ['RUNNING_SUM', 'WINDOW_AVG', 'INDEX_ORDER']),
    # Filter hints
    (r'(filter|where|only.*cases|exclude)', ['FILTER', 'MATCH_ACTIVITIES', 'BIND_FILTERS']),
]

def detect_functions(text: str):
    text_lower = text.lower()
    found = set()

    # Functions that are short or match common English words need word-boundary matching
    # to avoid false positives (e.g. 'IN' matching 'filter', 'MIN' matching 'minimum', etc.)
    NEEDS_WORD_BOUNDARY = {
        'AVG', 'SUM', 'MAX', 'MIN', 'VAR', 'IN', 'OR', 'AND', 'NOT',
        'ADD', 'SUB', 'DIV', 'MULT', 'LOG', 'LEN', 'ABS', 'ABC',
        'CEIL', 'FLOOR', 'ROUND', 'SQRT', 'SQUARE',
        'DAY', 'MONTH', 'YEAR', 'HOURS', 'MINUTES', 'SECONDS', 'MILLIS', 'QUARTER',
        'CASE', 'WHEN', 'LEFT', 'RIGHT', 'LIKE', 'RANGE',
        'REVERSE', 'BETWEEN', 'SOURCE_TARGET', 'END', 'BY', 'TO',
        'STDEV', 'COUNT', 'FILTER', 'BIND', 'LOOKUP', 'UPPER', 'LOWER',
    }

    # 1. Direct function name detection — word-boundary aware
    for fn in FUNCTION_NAMES:
        fn_lower = fn.lower()
        if fn in NEEDS_WORD_BOUNDARY:
            # Use word boundary so 'IN' doesn't match 'filter', 'MIN' doesn't match 'minimum'
            if re.search(r'\b' + re.escape(fn_lower) + r'\b', text_lower):
                found.add(fn)
        else:
            # Long unambiguous names: safe to use simple substring match
            if fn_lower in text_lower:
                found.add(fn)

    # 2. Intent-based detection (unchanged)
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
    for fn in funcs[:20]:  # expanded from 12
        if fn in COMPACT_REFS and fn not in seen:
            seen.add(fn)
            docs.append(f"### {fn}\n{COMPACT_REFS[fn]}")
    return "\n\n".join(docs)

# ──────────────────────────────────────────────────────────────
#  SECTION 2 · GROQ MODELS
# ──────────────────────────────────────────────────────────────

GROQ_MODELS = {
    'llama-3.3-70b-versatile':  'LLaMA 3.3 70B — best quality',
    'llama-3.1-8b-instant':     'LLaMA 3.1 8B  — fastest',
    'mixtral-8x7b-32768':       'Mixtral 8x7B  — balanced',
    'gemma2-9b-it':             'Gemma 2 9B    — lightweight',
}

# ──────────────────────────────────────────────────────────────
#  SECTION 3 · SYSTEM PROMPT BUILDER  (hardened)
# ──────────────────────────────────────────────────────────────

_FUNCTION_SELECTION_GUIDE = """
## ─── OFFICIAL CELONIS FUNCTION SELECTION GUIDE ───
## Source: docs.celonis.com — use this to choose the RIGHT function every time

### THROUGHPUT TIME — which function to use?

| Goal | Correct function | WRONG approach |
|------|-----------------|----------------|
| Throughput per CASE (start→end) | CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS(..., DAYS)) | PU_MAX - PU_MIN per case |
| Throughput per CASE (activity→activity) | CALC_THROUGHPUT(FIRST_OCCURRENCE['A'] TO LAST_OCCURRENCE['B'], REMAP_TIMESTAMPS(...)) | DATEDIFF on activity table |
| Throughput OVER MULTIPLE CASES (grouped) | DATEDIFF('dd', PU_MIN("GROUP","ACTIVITIES"."TIMESTAMP"), PU_MAX("GROUP","ACTIVITIES"."TIMESTAMP")) | CALC_THROUGHPUT |
| Cycle time first→last event per case | DATEDIFF('dd', PU_FIRST("CASES","ACTIVITIES"."TIMESTAMP", ORDER BY ...), PU_LAST("CASES","ACTIVITIES"."TIMESTAMP", ORDER BY ...)) | PU_AVG wrapping DATEDIFF |
| Average throughput across all cases | AVG(CALC_THROUGHPUT(...)) | PU_AVG wrapping CALC_THROUGHPUT |
| Edge throughput (activity→next activity) | SOURCE/TARGET + SECONDS_BETWEEN or REMAP_TIMESTAMPS | CALC_THROUGHPUT |

### REWORK / REPEATED ACTIVITIES — which function to use?

| Goal | Correct function |
|------|-----------------|
| Count activities per case | CALC_REWORK() |
| Count specific activity per case | CALC_REWORK("ACTIVITIES"."ACTIVITY" = 'Review') |
| Detect repeated activities (row-level) | INDEX_ACTIVITY_LOOP("ACTIVITIES"."ACTIVITY") > 0 |
| Count per activity type at a point | INDEX_ACTIVITY_TYPE("ACTIVITIES"."ACTIVITY") |
| Cases where activity repeats > N times | FILTER PU_COUNT("CASES","ACTIVITIES"."CASE_ID","ACTIVITIES"."ACTIVITY"='Review') > N |

### AGGREGATION — which PU function to use?

| Goal | Correct function | Avoid |
|------|-----------------|-------|
| Count rows (key column) | PU_COUNT | PU_COUNT_DISTINCT (slower) |
| Count distinct values | PU_COUNT_DISTINCT | - |
| Sum values | PU_SUM | - |
| Average values | PU_AVG | PU_MEDIAN (much slower) |
| True median | PU_MEDIAN | - |
| First/Last value | PU_FIRST / PU_LAST with ORDER BY | Without ORDER BY (undefined) |
| Simple count vs sum | PU_COUNT (less expensive than PU_SUM) | - |

### FILTERING — which approach?

| Goal | Correct approach |
|------|-----------------|
| Simple row filter | FILTER "TABLE"."COL" = 'value' |
| Filter within PU function | PU_COUNT("CASES","ACTIVITIES"."COL", "ACTIVITIES"."COL" = 'X') — use filter arg |
| Filter on aggregation result | FILTER PU_COUNT(...) > 5 |
| Filter-aware PU (avoid) | FILTER_TO_NULL inside PU — BREAKS CACHING, avoid unless no alternative |
| Filter to different table | BIND_FILTERS( target_table, condition ) |

### PROCESS MATCHING — which function?

| Goal | Correct function |
|------|-----------------|
| Cases containing specific activities (no order) | MATCH_ACTIVITIES |
| Cases following specific ordered path | MATCH_PROCESS |
| Cases matching activity name pattern (regex) | MATCH_PROCESS_REGEX |
| Cases where activity is at specific position | INDEX_ACTIVITY_ORDER |

### WHEN TO USE GLOBAL()

Use GLOBAL() when:
1. Query has BOTH case-level AND activity-level columns → common table shifts to activity level
2. CALC_THROUGHPUT combined with COUNT/SUM/AVG
3. Comparing per-group value to overall average: CASE WHEN AVG(...) > GLOBAL(AVG(...)) THEN ...
Do NOT use GLOBAL() when: query is purely case-level or purely activity-level

### COMMON TABLE RULES (from Celonis docs)
- PU-functions IGNORE global filters (cached)
- Standard aggregations (AVG, COUNT, SUM) RESPECT global filters (not cached)
- FILTER_TO_NULL makes PU filter-aware BUT breaks caching — avoid
- PU functions result type: PU_COUNT → INT, PU_SUM → same as source, PU_AVG → always FLOAT
- PU_COUNT returns 0 for no match; all other PU functions return NULL for no match

### NULL BEHAVIOUR REFERENCE
| Function | No matching rows | NULL in source |
|----------|-----------------|----------------|
| PU_COUNT | 0 | ignored (row excluded) |
| PU_COUNT_DISTINCT | 0 | ignored |
| PU_SUM, PU_AVG, PU_MIN, PU_MAX | NULL | ignored |
| PU_FIRST, PU_LAST | NULL | treated as non-existent |
| CALC_THROUGHPUT | NULL if single activity or end<start | - |
| DATEDIFF | NULL if any param is NULL | - |
"""

_SQL_PROHIBITION = """
## CRITICAL — PQL IS NOT SQL. NEVER WRITE SQL.

These SQL keywords DO NOT EXIST in PQL. If you write any of them, the query is WRONG:
  NO: SELECT   FROM    JOIN    LEFT JOIN   GROUP BY
  NO: HAVING   WITH    AS (CTE alias)   OVER(...)   ORDER BY (as standalone SQL clause)

### WRONG — SQL (never write this):
```sql
SELECT "LFA1"."LIFNR",
       AVG(DATEDIFF(dd, "EKKO"."BEDAT", "EKPO"."LGDAT")) AS LEAD_TIME
FROM "EKKO"
JOIN "EKPO" ON "EKKO"."EBELN" = "EKPO"."EBELN"
GROUP BY "LFA1"."LIFNR"
```

### CORRECT — real PQL:
```pql
-- Average lead time per vendor (PU aggregates child → parent)
PU_AVG(
  "LFA1",
  DATEDIFF(dd, "EKKO"."BEDAT", "EKPO"."LGDAT")
)
```

PQL works by referencing columns directly and using PU-functions to
aggregate across table relationships. There is NO SELECT, NO FROM, NO JOIN.
Each expression is a single column-level formula evaluated per row of the
result table.
"""

_ADVANCED_PATTERNS = """
## Advanced PQL Patterns

### P1 · GLOBAL() — prevents join multiplication
```pql
-- WRONG: CALC_THROUGHPUT multiplied by activity count
AVG( CALC_THROUGHPUT( CASE_START TO CASE_END, REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", DAYS) ) )
-- CORRECT
GLOBAL( AVG( CALC_THROUGHPUT( CASE_START TO CASE_END, REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", DAYS) ) ) )
```

### P2 · Nested PU aggregation across 3 levels
```pql
PU_SUM( "VENDORS", PU_SUM( "ORDERS", "LINE_ITEMS"."AMOUNT" ) )
```

### P3 · PU with filter argument (caching-friendly, preferred)
```pql
PU_COUNT( "CASES", "ACTIVITIES"."CASE_ID", "ACTIVITIES"."ACTIVITY" = 'Approve' )
```

### P4 · Throughput per case — official Celonis doc patterns
```pql
-- Case start to end (days) — from official Celonis docs
CALC_THROUGHPUT(
  CASE_START TO CASE_END,
  REMAP_TIMESTAMPS( "ACTIVITIES"."TIMESTAMP", DAYS )
)

-- Case start to end (hours, then /24 for days) — from official Celonis FAQ pattern
CALC_THROUGHPUT(
  CASE_START TO CASE_END,
  REMAP_TIMESTAMPS( "ACTIVITIES"."TIMESTAMP", HOURS )
) / 24

-- Between specific activities
CALC_THROUGHPUT(
  FIRST_OCCURRENCE['Create Order'] TO LAST_OCCURRENCE['Ship'],
  REMAP_TIMESTAMPS( "ACTIVITIES"."TIMESTAMP", DAYS )
)

-- Average throughput across all cases (standard aggregation wrapping CALC_THROUGHPUT)
AVG(
  CALC_THROUGHPUT(
    CASE_START TO CASE_END,
    REMAP_TIMESTAMPS( "ACTIVITIES"."TIMESTAMP", DAYS )
  )
)

-- Conforming throughput — official Celonis docs pattern
AVG(
  CASE WHEN PU_SUM( "CASES", ABS( conformance_col ) ) = 0
  THEN CALC_THROUGHPUT(
         CASE_START TO CASE_END,
         REMAP_TIMESTAMPS( "ACTIVITIES"."TIMESTAMP", HOURS )
       ) / 24
  ELSE NULL END
)
```
NOTE: CALC_THROUGHPUT is the preferred approach for case throughput. It is better than
DATEDIFF(PU_MIN("CASES", ...), PU_MAX("CASES", ...)) because it handles edge cases
(single-activity cases → NULL, start > end → NULL) and supports calendars via REMAP_TIMESTAMPS.
Throughput OVER MULTIPLE CASES uses PU_MIN and PU_MAX (not CALC_THROUGHPUT):
```pql
-- Throughput over multiple cases grouped by vendor (from Celonis docs)
DATEDIFF(
  'dd',
  PU_MIN( "VENDORS", "ACTIVITIES"."TIMESTAMP" ),
  PU_MAX( "VENDORS", "ACTIVITIES"."TIMESTAMP" )
)
```

### P5 · Rework detection
```pql
FILTER PU_COUNT(
  "CASES", "ACTIVITIES"."CASE_ID",
  "ACTIVITIES"."ACTIVITY" = 'Review'
) > 2;
```

### P6 · Running total partitioned by group and month
```pql
RUNNING_SUM(
  "ORDERS"."AMOUNT",
  ORDER BY ( ROUND_MONTH( "ORDERS"."ORDER_DATE" ) ASC ),
  PARTITION BY ( "ORDERS"."VENDOR_ID" )
)
```

### P7 · Safe KPI ratio
```pql
GLOBAL( COUNT( "ACTIVITIES"."TIMESTAMP" ) ) /
GLOBAL( COUNT( "CASES"."CASE_ID" ) )
```

### P8 · Workdays between dates
```pql
WORKDAYS_BETWEEN(
  WORKDAY_CALENDAR( WEEKDAY_CALENDAR( MON, TUE, WED, THU, FRI ) ),
  "ORDERS"."CREATE_DATE",
  "ORDERS"."CLOSE_DATE"
)
```

### P9 · Cycle time: first to last event per case (days)
CRITICAL RULE: PU_FIRST / PU_LAST pull scalar DATE values UP to the case table.
Once they are at case-table level, use DATEDIFF directly between them.
NEVER wrap an outer PU_AVG around DATEDIFF(PU_FIRST(...), PU_LAST(...)) —
PU_FIRST and PU_LAST are already case-level scalars; there is nothing left to aggregate over.

```pql
-- WRONG — PU_AVG wrapping DATEDIFF of already-aggregated PU values:
PU_AVG(
  "CASES",
  DATEDIFF( 'dd',
    PU_FIRST( "CASES", "ACTIVITIES"."TIMESTAMP", ORDER BY "ACTIVITIES"."TIMESTAMP" ASC ),
    PU_LAST(  "CASES", "ACTIVITIES"."TIMESTAMP", ORDER BY "ACTIVITIES"."TIMESTAMP" ASC )
  )
)

-- CORRECT — cycle time per case row (shows on case table):
DATEDIFF(
  'dd',
  PU_FIRST(
    "CASES",
    "ACTIVITIES"."TIMESTAMP",
    ORDER BY ( "ACTIVITIES"."TIMESTAMP" ASC )
  ),
  PU_LAST(
    "CASES",
    "ACTIVITIES"."TIMESTAMP",
    ORDER BY ( "ACTIVITIES"."TIMESTAMP" ASC )
  )
)

-- CORRECT — average cycle time across all cases (single KPI number):
AVG(
  DATEDIFF(
    'dd',
    PU_FIRST(
      "CASES",
      "ACTIVITIES"."TIMESTAMP",
      ORDER BY ( "ACTIVITIES"."TIMESTAMP" ASC )
    ),
    PU_LAST(
      "CASES",
      "ACTIVITIES"."TIMESTAMP",
      ORDER BY ( "ACTIVITIES"."TIMESTAMP" ASC )
    )
  )
)
```

### P10 · General rule — never nest PU inside PU value argument
PU functions return a scalar at the TARGET table level.
Once a value is at the target table level it is no longer a child-table column.
You CANNOT feed a PU_* result as the value argument of another PU_* that shares the same target table.
```pql
-- WRONG:
PU_AVG( "CASES", PU_FIRST( "CASES", "ACTIVITIES"."TIMESTAMP" ) )

-- CORRECT: PU_FIRST already IS a case-level value. Use it directly:
PU_FIRST( "CASES", "ACTIVITIES"."TIMESTAMP", ORDER BY ( "ACTIVITIES"."TIMESTAMP" ASC ) )

-- Valid nesting: different target tables (3-level hierarchy)
PU_SUM( "VENDORS", PU_SUM( "ORDERS", "LINE_ITEMS"."AMOUNT" ) )
-- ✓ outer target="VENDORS" (vendor level), inner target="ORDERS" (order level) — different levels, OK.
```
"""

_EXPERT_FRAMEWORK = """
## Expert Query Construction Framework

**Step 1 — Tables & joins**
What is the case/base table? What child tables? Join direction: 1:N or N:1?

**Step 2 — Result level**
Case level or activity level? Mixing levels → GLOBAL() required.

**Step 3 — Filters first**
FILTER for simple conditions. BIND_FILTERS for non-common tables.

**Step 4 — Compose KPIs**
Build innermost aggregation first. Wrap with GLOBAL() at table boundaries.

**Step 5 — Performance check**
PU_COUNT vs PU_COUNT_DISTINCT · AVG vs MEDIAN · PU filter arg vs FILTER_TO_NULL

**Step 6 — Final query**
Write in pql block. Explain each section. Flag NULL handling.

## Anti-patterns — always avoid
1. Missing GLOBAL() when mixing case + activity columns
2. FILTER_TO_NULL inside PU-functions
3. PU_COUNT_DISTINCT on key column
4. MEDIAN when AVG is sufficient
5. Missing double-quotes on table/column names
6. Single-quoting column names
7. ANY SQL syntax (SELECT/FROM/JOIN/GROUP BY)
8. **CRITICAL — Wrapping PU_FIRST/PU_LAST inside another PU function with the SAME target table.**
   PU_FIRST("CASES", ...) and PU_LAST("CASES", ...) already return case-level scalars.
   Using PU_AVG("CASES", DATEDIFF(..., PU_FIRST("CASES",...), PU_LAST("CASES",...))) is WRONG.
   The outer PU has nothing to aggregate — both inner values are already at "CASES" level.
   CORRECT: DATEDIFF('dd', PU_FIRST("CASES",...), PU_LAST("CASES",...)) — no outer PU needed.
   To get an overall average: AVG( DATEDIFF(...) ) — standard aggregation, not PU.
"""


def build_system_prompt(complexity: str, show_reasoning: bool) -> str:
    # Only embed the most critical "always-needed" functions (not the full 230)
    # Full function docs are injected dynamically per-query via build_function_context()
    ALWAYS_INCLUDE = [
        'GLOBAL', 'CALC_THROUGHPUT', 'PU_COUNT', 'PU_SUM', 'PU_AVG',
        'PU_FIRST', 'PU_LAST', 'FILTER', 'DATEDIFF', 'REMAP_TIMESTAMPS',
        'CALC_REWORK', 'MATCH_ACTIVITIES',
    ]
    core_refs = "\n\n".join(
        f"### {fn}\n{COMPACT_REFS[fn]}"
        for fn in ALWAYS_INCLUDE if fn in COMPACT_REFS
    )

    base = f"""You are an expert Celonis PQL (Process Query Language) engineer with deep knowledge of official Celonis documentation.
Write ACCURATE, OPTIMIZED, PRODUCTION-READY PQL queries.

## PQL Core Rules
- Tables and columns MUST be double-quoted: "TABLE"."COLUMN"
- String literals use single quotes: 'value'
- PQL is column-based, not row-based like SQL
- Multiple FILTER statements merge by logical AND
- NULL: most functions skip NULLs; use COALESCE or ISNULL to handle explicitly
- PU-functions aggregate FROM child table (many-side) TO parent table (one-side)
- Standard tables: "CASES"."CASE_ID", "ACTIVITIES"."ACTIVITY", "ACTIVITIES"."TIMESTAMP"

{_SQL_PROHIBITION}

{_FUNCTION_SELECTION_GUIDE}

## Core PQL Functions (always available)
{core_refs}

## Note
Additional relevant function docs are injected dynamically based on the user query.
The full library has 230 functions — only relevant ones are shown per query.
"""

    if complexity in ("Advanced", "Expert"):
        base += _ADVANCED_PATTERNS

    if complexity == "Expert":
        base += _EXPERT_FRAMEWORK

    if show_reasoning and complexity in ("Advanced", "Expert"):
        base += """
## Response Format
1. **Analysis** — identify tables, joins, and which function(s) the selection guide points to
2. **Query** — complete PQL in a ```pql code block
3. **Explanation** — explain each part of the query
4. **Performance notes** — optimization choices (e.g. why PU_COUNT over PU_COUNT_DISTINCT)
5. **Edge cases** — NULL handling, filter propagation, GLOBAL() requirement
"""
    elif complexity == "Intermediate":
        base += """
## Response Format
1. PQL in a ```pql code block
2. Explain each function used and why it was chosen
3. Mention important gotchas (NULL, GLOBAL, filter awareness)
"""
    else:
        base += """
## Response Format
1. PQL in a ```pql code block
2. Short plain-English explanation
"""

    instructions = {
        "Basic": "Simple queries. Use one or two functions maximum. Clear placeholder table names.\n",
        "Intermediate": "Queries may contain 2–5 functions. Use filters, CASE WHEN logic, and simple aggregations.\n",
        "Advanced": "Use nested functions, GLOBAL(), and PU aggregations. Always explain why GLOBAL() is required.\n",
        "Expert": "Write production-ready Celonis PQL with multi-KPI queries, nested PU, throughput, rework, and automation rate.\n",
    }

    base += f"\n## Complexity: {complexity}\n{instructions[complexity]}\n"

    base += """
When table/column names are unknown use:
"CASES"."CASE_ID"
"ACTIVITIES"."ACTIVITY"
"ACTIVITIES"."TIMESTAMP"
"ORDERS"."AMOUNT"
"VENDORS"."VENDOR_ID"
"""
    return base

# ──────────────────────────────────────────────────────────────
#  SECTION 3B · VERIFICATION PASS SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────

VERIFICATION_SYSTEM = """You are a strict Celonis PQL validator and corrector.

Your ONLY job: review PQL code and fix any errors. Return the corrected query.

## Rules to enforce:
1. NO SQL keywords: SELECT, FROM, JOIN, LEFT JOIN, GROUP BY, HAVING, WITH, AS (CTE), OVER(...)
2. ALL table/column names must be double-quoted: "TABLE"."COLUMN"
3. String literals must use single quotes: 'value'
4. PU_FUNC( target_table, source_table.column [, filter] ) — always 2+ args
5. CALC_THROUGHPUT used with standard aggregation → must be wrapped in GLOBAL()
6. COUNT( "TABLE"."COL" ) mixed with activity-level → wrap in GLOBAL()
7. PU_COUNT_DISTINCT on a key column → replace with PU_COUNT
8. FILTER_TO_NULL inside PU functions → replace with PU filter argument
9. PU-function direction: target_table is the PARENT (1-side), source is CHILD (N-side)
10. MEDIAN used when AVG would work → add a comment noting performance cost
11. CRITICAL — Outer PU wrapping DATEDIFF of inner PU with same target table is WRONG.
    Example of the ERROR:
      PU_AVG( "CASES", DATEDIFF( 'dd', PU_FIRST( "CASES", ... ), PU_LAST( "CASES", ... ) ) )
    WHY it's wrong: PU_FIRST("CASES",...) already returns a scalar at the CASES level.
    Passing it as the value-arg of PU_AVG("CASES",...) means the outer PU has no child rows
    to aggregate over — both inner values are already case-level scalars.
    FIX: Remove outer PU. Use DATEDIFF directly:
      DATEDIFF( 'dd', PU_FIRST( "CASES", ... ), PU_LAST( "CASES", ... ) )
    For an overall average: AVG( DATEDIFF(...) )  ← standard aggregation, NOT PU_AVG.
    Valid nesting is ONLY when outer and inner PU have DIFFERENT target tables (3-level hierarchy).

## Response format:
- If the query is correct: respond with exactly: VALID
- If the query has errors: respond with only the corrected ```pql code block and a brief bullet list of what was fixed. Nothing else.
"""

# ──────────────────────────────────────────────────────────────
#  SECTION 4 · UI CONSTANTS
# ──────────────────────────────────────────────────────────────

COMPLEXITY_DESC = {
    'Basic':        'Simple 1-2 function queries. Great for beginners.',
    'Intermediate': 'Multi-function queries with filters & conditions.',
    'Advanced':     'Nested PU-functions, GLOBAL(), multi-table joins.',
    'Expert':       'Chain-of-thought planning · BPMN · OCPM · ML · full optimization.',
}

EXAMPLE_PROMPTS = {
    'Basic': [
        'Count activities per case',
        "Filter cases where status = 'open'",
        'Convert vendor name to uppercase',
        'Difference in days between two date columns',
    ],
    'Intermediate': [
        'Average invoice amount per vendor',
        'Find cases where Approve happens before Pay',
        'Throughput time per case in days',
        'Running total of PO values grouped by month',
    ],
    'Advanced': [
        'Count late deliveries per vendor — delivery > promised by 7 days',
        'Detect rework: Review activity repeating more than 2 times per case',
        'First and last activity timestamp per case using INDEX_ORDER',
        'Flag non-conforming cases using MATCH_ACTIVITIES',
    ],
    'Expert': [
        'Full KPI query: throughput time + rework count + automation rate in one query',
        'Multi-level nesting: avg approval time aggregated vendor → order → line item',
        'BPMN conformance check that tolerates undesired activities',
        'OCPM: throughput across linked objects with workday calendar',
    ],
}

# ──────────────────────────────────────────────────────────────
#  SECTION 5 · PAGE CONFIG + CSS
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='PQL Query Assistant',
    page_icon='⚡',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
.stApp { background:#0a0c10; }
.main .block-container { background:#0a0c10 !important; padding-top:2rem !important; }
header[data-testid="stHeader"] { background:#0a0c10 !important; border-bottom:1px solid #1e2531 !important; }
.stApp > header { background:#0a0c10 !important; }
[data-testid="stToolbar"] { background:#0a0c10 !important; }
h1, h2, h3, h1 *, h2 *, h3 * { color:#f1f5f9 !important; }
[data-testid="stHeadingWithActionElements"] h1,
[data-testid="stHeadingWithActionElements"] h2,
[data-testid="stHeadingWithActionElements"] h3 { color:#f1f5f9 !important; }
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3 { color:#f1f5f9 !important; }
[data-testid="stCaptionContainer"] p, .stCaption, .stCaption p { color:#64748b !important; font-size:13px !important; }
div[data-testid="stMarkdownContainer"] p { color:#cbd5e1 !important; }
h1 a, h2 a, h3 a { display:none !important; }
[data-testid="stHeadingWithActionElements"] a { display:none !important; }
[data-testid="stHeadingWithActionElements"] button,
[data-testid="stHeadingWithActionElements"] svg { display:none !important; }
[data-testid="stSidebar"] { background:#0d1117; border-right:1px solid #1e2531; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color:#94a3b8 !important; }
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select { background:#161b22 !important; border:1px solid #30363d !important; color:#e2e8f0 !important; border-radius:6px !important; }
[data-testid="stChatMessage"] { background:#0d1117 !important; border:1px solid #1e2531; border-radius:12px !important; margin-bottom:10px; box-shadow:0 4px 14px rgba(0,0,0,0.35); }
[data-testid="stChatMessageContent"],
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] span,
[data-testid="stChatMessageContent"] ul { color:#e2e8f0 !important; font-size:14px; }
[data-testid="stChatMessageContent"] strong { color:#f8fafc !important; }
[data-testid="stBottom"] { background:#0a0c10 !important; border-top:1px solid #1e2531 !important; }
[data-testid="stBottom"] > div { background:#0a0c10 !important; }
[data-testid="stChatInput"] { background:#161b22 !important; border:1px solid #30363d !important; border-radius:10px !important; }
[data-testid="stChatInput"]:focus-within { border:1px solid #6366f1 !important; box-shadow:0 0 0 2px rgba(99,102,241,0.15) !important; }
[data-testid="stChatInput"] textarea { background:#161b22 !important; color:#f1f5f9 !important; caret-color:#6366f1 !important; border:none !important; border-radius:10px !important; font-size:14px !important; }
[data-testid="stChatInput"] textarea::placeholder { color:#475569 !important; }
[data-testid="stChatInputSubmitButton"] button { background:linear-gradient(135deg,#6366f1,#8b5cf6) !important; border:none !important; }
pre { background:#020617 !important; border:1px solid #1e293b !important; border-radius:10px !important; padding:14px !important; overflow-x:auto !important; }
pre code, code { background:#020617 !important; border:1px solid #1e293b !important; border-radius:8px !important; color:#f1f5f9 !important; font-family:"JetBrains Mono", monospace !important; font-size:13px !important; line-height:1.5 !important; }
.stButton > button { background:#161b22 !important; border:1px solid #30363d !important; color:#c7d2fe !important; border-radius:8px !important; font-size:12px !important; }
.stButton > button:hover { background:#1e293b !important; border-color:#6366f1 !important; color:#e0e7ff !important; transform:scale(1.02); }
details { border:1px solid #1e2531 !important; border-radius:8px !important; }
[data-testid="stMetric"] { background:#0d1117; border:1px solid #1e2531; border-radius:10px; padding:10px 14px; }

/* Verification badge styles */
.verify-pass { background:#052e16; border:1px solid #16a34a; border-radius:8px; padding:8px 14px; color:#4ade80; font-size:13px; margin-top:8px; }
.verify-fix  { background:#1c1004; border:1px solid #d97706; border-radius:8px; padding:8px 14px; color:#fbbf24; font-size:13px; margin-top:8px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  SECTION 6 · SESSION STATE
# ──────────────────────────────────────────────────────────────

_defaults = {
    'messages':       [],
    'complexity':     'Advanced',
    'model_id':       'llama-3.3-70b-versatile',
    'show_reasoning': True,
    'total_queries':  0,
    'verified_count': 0,
    'fixed_count':    0,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────────────────────
#  SECTION 7 · GROQ CLIENT
# ──────────────────────────────────────────────────────────────

def get_client():
    key = ""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass
    key = key or os.environ.get("GROQ_API_KEY", "")
    return (Groq(api_key=key), key) if key else (None, "")

client, _api_key = get_client()

# ──────────────────────────────────────────────────────────────
#  SECTION 8 · SIDEBAR
# ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">'
        '<div style="width:36px;height:36px;border-radius:9px;'
        'background:linear-gradient(135deg,#6366f1,#8b5cf6);'
        'display:flex;align-items:center;justify-content:center;font-size:18px;">⚡</div>'
        '<div><div style="font-size:15px;font-weight:700;color:#f1f5f9;">PQL Assistant</div>'
        '<div style="font-size:11px;color:#475569;">230 functions · auto-verified</div></div></div>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown('**🤖 Model**')
    selected_model = st.selectbox(
        'Model', options=list(GROQ_MODELS.keys()),
        index=list(GROQ_MODELS.keys()).index(st.session_state.model_id),
        format_func=lambda k: GROQ_MODELS[k], label_visibility='collapsed',
    )
    st.session_state.model_id = selected_model
    st.caption(f'`{selected_model}`')

    st.divider()

    st.markdown('**🎛 Complexity**')
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

    st.divider()

    st.markdown('**💡 Quick examples**')
    for ex in EXAMPLE_PROMPTS.get(complexity, EXAMPLE_PROMPTS['Advanced']):
        if st.button(f'→ {ex}', key=f'ex_{ex}', use_container_width=True):
            st.session_state['_pending'] = ex

    st.divider()

    st.markdown('**📚 Function Reference**')
    search = st.text_input('Search functions', placeholder='e.g. PU_COUNT, DATEDIFF…', label_visibility='collapsed')

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
                        f'Write a PQL query using {fn["name"]} and explain the syntax with a practical example.'
                    )
                st.caption(fn['doc'][:120] + '…' if len(fn['doc']) > 120 else fn['doc'])

    st.divider()

    c1, c2, c3 = st.columns(3)
    c1.metric('Queries', st.session_state.total_queries)
    c2.metric('✅ Valid', st.session_state.verified_count)
    c3.metric('🔧 Fixed', st.session_state.fixed_count)

    if st.button('🗑 Clear chat', use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ──────────────────────────────────────────────────────────────
#  SECTION 9 · MAIN CHAT AREA
# ──────────────────────────────────────────────────────────────

st.markdown('## ⚡ PQL Query Assistant')
st.caption(
    f'Complexity: **{complexity}** · Model: `{st.session_state.model_id}` · '
    f'{len(COMPACT_REFS)} functions · 🔍 auto-verified'
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
        st.markdown("""
**Welcome! I'm your PQL Query Assistant.**

Every query I generate is automatically **verified and corrected** by a second AI pass before you see it — so you always get valid PQL.

**What I can do:**
- 🔨 **Write** PQL from a plain-English description
- 🔍 **Explain** existing PQL line by line
- ⚡ **Optimize** slow or incorrect queries
- 📚 **Teach** any of the 230 PQL functions with examples
- ✅ **Auto-verify** every generated query for correctness

**Try asking:**
- *"Calculate average throughput time per case in days"*
- *"How do I use PU_COUNT with a filter condition?"*
- *"Detect rework loops where the same activity repeats more than twice"*

→ Use the sidebar to change complexity or browse all 230 functions.
""")

# ──────────────────────────────────────────────────────────────
#  SECTION 10 · VERIFICATION ENGINE (two-pass)
# ──────────────────────────────────────────────────────────────

def extract_pql_blocks(text: str) -> list[str]:
    """Extract all PQL code blocks from a response."""
    return re.findall(r"```pql\s*(.*?)```", text, re.S)


def verify_and_fix_pql(pql_query: str) -> tuple[bool, str, list[str]]:
    """
    Two-pass verification:
    Pass 1 — fast rule-based checks (no LLM call needed if clean)
    Pass 2 — LLM review + correction (only if Pass 1 finds issues OR for Advanced/Expert)

    Returns: (was_modified, final_query, list_of_issues)
    """
    issues = []

    # ── Pass 1: Rule-based checks ──────────────────────────────
    SQL_KEYWORDS = [r'\bSELECT\b', r'\bFROM\b', r'\bJOIN\b', r'\bGROUP BY\b',
                    r'\bHAVING\b', r'\bOVER\s*\(', r'\bAS\s+\w+\s*(?:,|\n|$)']
    for kw in SQL_KEYWORDS:
        if re.search(kw, pql_query, re.IGNORECASE):
            issues.append(f"Contains SQL keyword: `{kw.strip()}`")

    # Unquoted table.column references
    unquoted = re.findall(r'(?<!")\b([A-Z][A-Z0-9_]+)\.([A-Z][A-Z0-9_]+)\b(?!")', pql_query)
    if unquoted:
        issues.append(f"Possible unquoted identifiers: {unquoted[:3]}")

    # PU function with wrong arg count
    pu_calls = re.findall(r'(PU_\w+)\s*\(([^)]*)\)', pql_query)
    for fn_name, args in pu_calls:
        arg_count = len([a for a in args.split(',') if a.strip()])
        if arg_count < 2:
            issues.append(f"{fn_name} needs at least 2 arguments (target_table, source_col)")

    # CALC_THROUGHPUT without GLOBAL
    if 'CALC_THROUGHPUT' in pql_query and 'GLOBAL(' not in pql_query:
        if re.search(r'\b(AVG|COUNT|SUM|MEDIAN)\b', pql_query):
            issues.append("CALC_THROUGHPUT mixed with other aggregations — consider GLOBAL()")

    # FILTER_TO_NULL inside PU
    if 'FILTER_TO_NULL' in pql_query and 'PU_' in pql_query:
        issues.append("FILTER_TO_NULL inside PU function — use PU filter argument instead")

    # CRITICAL: Outer PU wrapping DATEDIFF of inner PU with same target table
    # Pattern: PU_xxx( "TABLE", DATEDIFF( ..., PU_yyy( "TABLE", ... ), PU_zzz( "TABLE", ... ) ) )
    # Detect: outer PU whose value arg contains inner PU calls with the same quoted table
    outer_pu_match = re.search(
        r'PU_\w+\s*\(\s*("[\w]+")\s*,\s*(?:DATEDIFF|HOURS_BETWEEN|MINUTES_BETWEEN|SECONDS_BETWEEN|MILLIS_BETWEEN|WORKDAYS_BETWEEN)',
        pql_query, re.IGNORECASE
    )
    if outer_pu_match:
        outer_table = outer_pu_match.group(1)
        # Check if any inner PU uses the same target table
        inner_pu_same = re.search(
            r'PU_(?:FIRST|LAST|MIN|MAX|AVG|SUM|COUNT)\s*\(\s*' + re.escape(outer_table),
            pql_query, re.IGNORECASE
        )
        if inner_pu_same:
            issues.append(
                f"CRITICAL: Outer PU function wraps DATEDIFF of inner PU_FIRST/PU_LAST with same "
                f"target table {outer_table}. PU_FIRST/PU_LAST already return case-level scalars — "
                f"remove the outer PU and use DATEDIFF(..., PU_FIRST(...), PU_LAST(...)) directly. "
                f"For an overall average, wrap with AVG(...), not PU_AVG(...)."
            )

    # ── Pass 2: LLM review (always runs for Advanced/Expert, or when issues found) ──
    always_verify = st.session_state.complexity in ('Advanced', 'Expert')

    if not issues and not always_verify:
        return False, pql_query, []

    try:
        verify_prompt = f"""Review this PQL query for correctness.

```pql
{pql_query}
```

{f"Rule-based checks flagged: {issues}" if issues else "Do a thorough correctness review."}

Respond with either:
- Exactly the word VALID (if the query is correct)
- Or a corrected ```pql block followed by a brief bullet list of what was fixed
"""
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # pinned: fast & cheap, never uses user's model
            messages=[
                {"role": "system", "content": VERIFICATION_SYSTEM},
                {"role": "user", "content": verify_prompt},
            ],
            temperature=0,
            max_tokens=1200,
        )
        result = response.choices[0].message.content.strip()

        if result.upper().startswith("VALID"):
            return False, pql_query, []

        # Extract corrected query
        match = re.search(r"```pql\s*(.*?)```", result, re.S)
        if match:
            corrected = match.group(1).strip()
            # Extract bullet fixes
            fixes = re.findall(r'^[-•*]\s+(.+)', result, re.MULTILINE)
            return True, corrected, fixes if fixes else ["Query corrected by verification pass"]

        return False, pql_query, []

    except Exception as e:
        return False, pql_query, [f"Verification skipped (API error): {e}"]


# ──────────────────────────────────────────────────────────────
#  SECTION 11 · GROQ STREAMING + VERIFICATION
# ──────────────────────────────────────────────────────────────

def stream_groq(prompt_override=None):
    msgs = st.session_state.messages
    user_query = prompt_override if prompt_override else msgs[-1]["content"]

    func_context = build_function_context(user_query)
    system = build_system_prompt(st.session_state.complexity, st.session_state.show_reasoning)

    if func_context:
        system += "\n\n## Relevant PQL Functions (retrieved for this query)\n" + func_context

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
                max_tokens=2048,
                temperature=0.15,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full += delta
                placeholder.markdown(full + "▌")

            placeholder.markdown(full)
            st.session_state.total_queries += 1

            # ── Two-pass verification ──────────────────────────
            pql_blocks = extract_pql_blocks(full)

            if pql_blocks:
                for pql_block in pql_blocks:
                    was_modified, final_query, fix_notes = verify_and_fix_pql(pql_block)

                    if was_modified:
                        st.session_state.fixed_count += 1
                        st.markdown(
                            '<div class="verify-fix">🔧 <strong>Auto-corrected</strong> — verification pass fixed issues</div>',
                            unsafe_allow_html=True
                        )
                        for note in fix_notes:
                            st.caption(f"  • {note}")
                        st.markdown("**Corrected query:**")
                        st.code(final_query, language="sql")
                        # Update stored message with corrected version
                        full = full.replace(
                            f"```pql\n{pql_block}\n```",
                            f"```pql\n{final_query}\n```"
                        )
                    else:
                        st.session_state.verified_count += 1
                        st.markdown(
                            '<div class="verify-pass">✅ <strong>Verified</strong> — query passed correctness check</div>',
                            unsafe_allow_html=True
                        )

            st.session_state.messages.append({"role": "assistant", "content": full})

        except Exception as e:
            placeholder.error(f"Groq API error: {e}")


# Handle sidebar button → pending prompt
if '_pending' in st.session_state:
    pending = st.session_state.pop('_pending')
    st.session_state.messages.append({'role': 'user', 'content': pending})
    with st.chat_message('user', avatar='🧑'):
        st.markdown(pending)
    stream_groq()
    st.rerun()

# Main input
if prompt := st.chat_input('Describe your query, ask about a function, or paste PQL to optimize…'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user', avatar='🧑'):
        st.markdown(prompt)
    stream_groq()
