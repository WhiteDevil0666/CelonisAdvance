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
import streamlit as st
from groq import Groq

# ──────────────────────────────────────────────────────────────
#  SECTION 1 · KNOWLEDGE BASE  (175 PQL functions + categories)
# ──────────────────────────────────────────────────────────────

COMPACT_REFS = {
    'CREATE_EVENTLOG': 'Returns an activity table based on a given lead object and included event types. Used to generate event logs from an object perspective in OCPM. Syntax: CREATE_EVENTLOG( lead_object, event_type_list )',
    'PU_COUNT': 'Counts rows in source table for each row in target table. Prefer over PU_COUNT_DISTINCT for key columns. Syntax: PU_COUNT( target_table, source_table.column [, filter_expression] )',
    'PU_SUM': 'Sums values from source table for each row in target table. Syntax: PU_SUM( target_table, source_table.column [, filter_expression] )',
    'PU_AVG': 'Average of source column per target row. Significantly cheaper than PU_MEDIAN. Syntax: PU_AVG( target_table, source_table.column [, filter_expression] )',
    'PU_MAX': 'Maximum of source column per target row. Syntax: PU_MAX( target_table, source_table.column [, filter_expression] ) Requires 1:N relationship.',
    'PU_MIN': 'Minimum of source column per target row. Syntax: PU_MIN( target_table, source_table.column [, filter_expression] ) Requires 1:N relationship.',
    'PU_FIRST': 'First element of source column per target row. Syntax: PU_FIRST( target_table, source_table.column [, filter_expression] [, ORDER BY col [ASC|DESC]] )',
    'PU_LAST': 'Last element of source column per target row. Syntax: PU_LAST( target_table, source_table.column [, filter_expression] [, ORDER BY col [ASC|DESC]] )',
    'PU_MEDIAN': 'Median of source column per target row. Requires sorting — expensive. Use PU_AVG when possible. Syntax: PU_MEDIAN( target_table, source_table.column [, filter_expression] )',
    'PU_COUNT_DISTINCT': 'Distinct count per target row. Use PU_COUNT when column is a key. Syntax: PU_COUNT_DISTINCT( target_table, source_table.column [, filter_expression] )',
    'PU_MODE': 'Most frequent value per target row. Syntax: PU_MODE( target_table, source_table.column [, filter_expression] )',
    'PU_PRODUCT': 'Product of source column per target row. Syntax: PU_PRODUCT( target_table, source_table.column [, filter_expression] )',
    'PU_QUANTILE': 'Quantile (0.0-1.0) of source column per target row. Syntax: PU_QUANTILE( target_table, source_table.column, quantile [, filter_expression] )',
    'PU_TRIMMED_MEAN': 'Trimmed mean (excludes outliers) per target row. Syntax: PU_TRIMMED_MEAN( target_table, source_table.column [, lower_cutoff [, upper_cutoff]] [, filter_expression] )',
    'PU_STRING_AGG': 'Concatenates strings from source per target row. Syntax: PU_STRING_AGG( target_table, source_table.column, delimiter [, filter_expression] [, ORDER BY col] )',
    'PU_STDEV': 'Standard deviation (n-1 method) per target row. Syntax: PU_STDEV( target_table, source_table.column [, filter_expression] )',
    'COUNT_TABLE': 'Counts rows in a table including NULLs (unlike COUNT). Returns original count even when common table differs. Syntax: COUNT_TABLE( table )',
    'MEDIAN': 'Median per group. Applies to INT, FLOAT, DATE. Syntax: MEDIAN( table.column ) NULLs ignored.',
    'QUANTILE': 'Quantile per group. Syntax: QUANTILE( table.column, quantile ) quantile: float 0.0-1.0.',
    'GLOBAL': 'Isolates aggregation from common table. Prevents join multiplication. Use when mixing case-level and activity-level columns. Syntax: GLOBAL( aggregation )',
    'RUNNING_SUM': 'Cumulative sum of previous rows. Syntax: RUNNING_SUM( column [, ORDER BY (...)] [, PARTITION BY (...)] )',
    'WINDOW_AVG': 'Average over a sliding window. Syntax: WINDOW_AVG( table.values, lower_bound, upper_bound [, ORDER BY ...] [, PARTITION BY ...] )',
    'STRING_AGG': 'Concatenates strings with delimiter. Syntax: STRING_AGG( table.column, "delim" [, ORDER BY ...] [, PARTITION BY ...] )',
    'INDEX_ORDER': 'Integer indices starting from 1 for ordering rows. Syntax: INDEX_ORDER( column [, ORDER BY (...)] [, PARTITION BY (...)] ) Result: INT.',
    'PARTITION': 'Used in window functions as PARTITION BY ( column, ... ) to define groups.',
    'ZSCORE': 'Z-score (standard deviations from mean). Syntax: ZSCORE( table.column [, PARTITION BY (...)] )',
    'INTERPOLATE': 'Interpolates NULL values. Syntax: INTERPOLATE( column, CONSTANT|LINEAR [, ORDER BY ...] [, PARTITION BY ...] )',
    'CALC_THROUGHPUT': 'Calculates throughput time between events. Wrap with GLOBAL() when mixing with activity KPIs. Syntax: CALC_THROUGHPUT( CASE_START TO CASE_END, REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", DAYS) )',
    'CALC_REWORK': 'Counts activities per case (rework = repeated activities). Syntax: CALC_REWORK() | CALC_REWORK(filter) | CALC_REWORK(activity_table.col) Returns INT on case table.',
    'CALC_CROP': 'Crops cases to event range, returns 1 in range, NULL outside. Syntax: CALC_CROP( begin TO end, activity_table.col )',
    'CALC_CROP_TO_NULL': 'Crops cases to event range, keeps values in range, NULL outside. Syntax: CALC_CROP_TO_NULL( begin TO end, activity_table.col )',
    'MATCH_ACTIVITIES': 'Flags cases with certain activities (order-independent). Syntax: MATCH_ACTIVITIES( [STARTING list] [NODE list] [ENDING list] [EXCLUDING list] )',
    'MATCH_PROCESS': 'Matches process variants against node/edge pattern (order-sensitive). Syntax: MATCH_PROCESS( [table.col,] NODE ... CONNECTED BY edge, ... )',
    'MATCH_PROCESS_REGEX': 'Filters variants using regex over activity names. Syntax: MATCH_PROCESS_REGEX( [table.col,] "regex_pattern" )',
    'ACTIVITY_LAG': 'Returns row preceding current row by offset within a case. Syntax: ACTIVITY_LAG( activity_table.column [, offset] ) Default offset: 1.',
    'ACTIVITY_LEAD': 'Returns row following current row by offset within a case. Syntax: ACTIVITY_LEAD( activity_table.column [, offset] ) Default offset: 1.',
    'PROCESS_ORDER': 'Deprecated. Returns position of each activity within a case. Use INDEX_ACTIVITY_ORDER instead.',
    'BPMN_CONFORMS': 'Binary BPMN conformance check (1=conforming, 0=not). Syntax: BPMN_CONFORMS( event_table.col, bpmn_model [, ALLOW(...)] )',
    'CONFORMANCE': 'Petri net conformance checking. Returns INT flags. Use with READABLE() for violation descriptions.',
    'READABLE': 'Human-readable violation descriptions from CONFORMANCE. Syntax: READABLE( conformance_query )',
    'VARIANT': 'Returns process variant string per case. Syntax: VARIANT( activity_table.string_column )',
    'TRANSIT_COLUMN': 'Computes transition edges between related cases from two processes.',
    'MANUAL_MINER': 'Defines manual transitions for TRANSIT_COLUMN. Syntax: MANUAL_MINER( activity_table.col, ["A", "B"] )',
    'ADD_DAYS': 'Adds days to a date. Syntax: ADD_DAYS( table.base_col, table.days_col ) base: DATE, days: INT. Output: DATE.',
    'DATEDIFF': 'Date difference in specified unit. Syntax: DATEDIFF( unit, table.date1, table.date2 ) unit: ms|ss|mi|hh|dd|mm|yy. Output: FLOAT.',
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
    'REMAP_TIMESTAMPS': 'Remaps timestamps per calendar/unit. Used in CALC_THROUGHPUT. Syntax: REMAP_TIMESTAMPS( ts_col, unit [, calendar] )',
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
    'READABLE': 'Human-readable CONFORMANCE violation descriptions. Syntax: READABLE( conformance_query )',
    'EDIT_THRESHOLD': 'Edit distance threshold for CLUSTER_STRINGS. Syntax: EDIT_THRESHOLD( distance )',
    'TOP_K': 'Number of matches in MATCH_STRINGS. Syntax: TOP_K( k ) where k <= 100.',
    'SEPARATOR': 'Separator between results in MATCH_STRINGS. Syntax: SEPARATOR( "," )',
    'RIGHT': 'Used in join type specifications (RIGHT join in MERGE_EVENTLOG scenarios).',
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
    'ISNULL': 'Returns 1 if NULL, 0 otherwise. Syntax: ISNULL( table.col )',
    'MODULO': 'Remainder of division. Syntax: MODULO( dividend, divisor ) or dividend % divisor.',
    'TODAY': 'Current date. Syntax: TODAY( [timezone_id] ) Default: UTC.',
    'UNIQUE_ID': 'Unique INT for each unique tuple of input columns. Syntax: UNIQUE_ID( table.col1, ..., table.colN )',
    'CONSTANT': 'Used as target table in PU-functions to produce a constant result. Syntax: CONSTANT()',
    'COMMON_TABLE': 'References the common table of multiple expressions. Syntax: COMMON_TABLE( expr1, expr2 )',
    'COLUMN_TYPE': 'Returns data type of a column as STRING (INT/FLOAT/STRING/DATE). Syntax: COLUMN_TYPE( table.col )',
    'ARGUMENT_COUNT': 'Counts number of arguments passed. Syntax: ARGUMENT_COUNT( arg1, arg2, ... )',
    'MERGE_EVENTLOG': 'Merges columns from two activity tables into one. Syntax: MERGE_EVENTLOG( target_table.col, [FILTER ...] )',
    'MERGE_EVENTLOG_DISTINCT': 'Like MERGE_EVENTLOG but removes duplicate activities.',
    'EVENTLOG_SOURCE_TABLE': 'Returns source table name for each row in a dynamic event log. Syntax: EVENTLOG_SOURCE_TABLE( eventlog.col )',
    'CREATE_EVENTLOG': 'Creates activity table from OCPM object perspective.',
    'LINK_PATH': 'Traverses object links. Syntax: LINK_PATH( table.col [, CONSTRAINED BY (START(...), END(...))] )',
    'LINK_SOURCE': 'Source objects of Object Link. Syntax: LINK_SOURCE( link_name, table.col )',
    'LINK_TARGET': 'Target objects of Object Link. Syntax: LINK_TARGET( link_name, table.col )',
    'LINK_FILTER': 'Filters by link traversal. Syntax: LINK_FILTER( filter_expr, ANCESTORS|DESCENDANTS [, hops] )',
    'LINK_FILTER_ORDERED': 'Order-aware LINK_FILTER (only for Signal Link). Considers timestamp order.',
    'LINK_ATTRIBUTES': 'Returns link attribute values. Syntax: LINK_ATTRIBUTES( link_name, attr_col )',
    'LINK_OBJECTS': 'Creates table of all objects in the Object Link graph.',
    'LINK_PATH_SOURCE': 'Source objects used in LINK_PATH traversal.',
    'LINK_PATH_TARGET': 'Target objects used in LINK_PATH traversal.',
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
    'BPMN_CONFORMS': 'Binary BPMN conformance. Syntax: BPMN_CONFORMS( event_table.col, bpmn_model [, ALLOW(...)] )',
    'BPMN_MATCH_EXCESSIVE': 'Activity occurs at right place but too often — used in BPMN_CONFORMS ALLOW list.',
    'BPMN_MATCH_MISSING': 'Required activity missing from trace — BPMN_CONFORMS shorthand.',
    'BPMN_MATCH_OUT_OF_SEQUENCE': 'Activity at wrong position — BPMN_CONFORMS shorthand.',
    'BPMN_MATCH_UNDESIRED': 'Activity present that should not be — BPMN_CONFORMS shorthand.',
    'BPMN_MATCH_UNMAPPED': 'Activity with no model mapping — BPMN_CONFORMS shorthand.',
    'SEQUENCE': 'Models sequential flow in BPMN_CONFORMS. Syntax: SEQUENCE("A", "B", "C")',
    'PARALLEL': 'Models parallel paths in BPMN_CONFORMS. Syntax: PARALLEL("A", "B")',
    'EXCLUSIVE_CHOICE': 'Models XOR gateway in BPMN_CONFORMS.',
    'ALLOW': 'Allows deviations in BPMN_CONFORMS. Syntax: ALLOW( BPMN_MATCH_UNDESIRED(ANY) )',
    'CONSTRAINED': 'Limits LINK_PATH traversal. Syntax: CONSTRAINED BY( START(...), END(...) )',
    'START': 'Used in CALC_CROP, LINK_PATH, and BPMN_CONFORMS for start conditions.',
    'END': 'Used in CALC_CROP and BPMN_CONFORMS for end conditions.',
    'EXCLUDED': 'Used in BPMN conformance for excluded activities.',
    'UNMAPPED': 'Used in BPMN_CONFORMS for unmapped activities.',
    'SYNC': 'Used in BPMN conformance checking to synchronize parallel paths.',
    'TASK': 'Used in BPMN_CONFORMS model definition: BPMN_TASK "ActivityName"',
    'VIA': 'Specifies intermediate steps in process path expressions.',
    'SOURCE': 'Specifies source activities in MERGE_EVENTLOG path expressions.',
    'TARGET': 'Specifies target activities in MERGE_EVENTLOG path expressions.',
    'WITH': 'Used in MERGE_EVENTLOG: WITH START(...) / WITH END(...) for artificial nodes.',
    'INPUT': 'Specifies training input columns in LINEAR_REGRESSION/KMEANS.',
    'OUTPUT': 'Specifies target output column in LINEAR_REGRESSION.',
    'BY': 'Part of ORDER BY and PARTITION BY clauses in window functions.',
    'FROM': 'Used in CURRENCY_CONVERT as FROM("USD") for source currency.',
    'TO': 'Used in CURRENCY_CONVERT as TO("EUR") for target currency.',
    'ADD': 'Used in MERGE_EVENTLOG to add columns/tables.',
    'RANGE': 'Used inside GENERATE_RANGE to define range parameters.',
    'DIV': 'Integer division. Also % operator for modulo.',
    'HOURS': 'Time unit specification used in REMAP_TIMESTAMPS.',
    'WEIGHT': 'Token weight in CLUSTER_STRINGS. Syntax: WEIGHT( "tokens", weight )',
    'SHORTENED': 'Shortens self-loops in VARIANT. Syntax: VARIANT( ..., SHORTENED(max) )',
    'BKPF': 'SAP BKPF table reference used in SAP P2P process examples.',
    'MANUAL_MINER': 'Defines manual transitions for TRANSIT_COLUMN.',
    'COUNT': 'Counts non-NULL rows. Syntax: COUNT(table.column). Often wrapped with GLOBAL() when mixing table levels.',
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
# Faster detection for PU functions + normal functions
# ──────────────────────────────────────────────────────────────

import re

FUNCTION_NAMES = list(COMPACT_REFS.keys())

# Pre-categorize PU functions (most commonly used)
PU_FUNCTIONS = [fn for fn in FUNCTION_NAMES if fn.startswith("PU_")]

# Common query patterns that imply PU usage
PU_HINT_PATTERNS = [
    r'per\s+case',
    r'per\s+vendor',
    r'per\s+order',
    r'per\s+customer',
    r'per\s+\w+',
    r'group\s+by',
    r'aggregate',
    r'count\s+per',
    r'sum\s+per',
]


def detect_functions(text: str):

    text_lower = text.lower()
    found = set()

    # 1️⃣ Direct function mentions
    for fn in FUNCTION_NAMES:
        if fn.lower() in text_lower:
            found.add(fn)

    # 2️⃣ Detect PU usage patterns
    if any(re.search(pattern, text_lower) for pattern in PU_HINT_PATTERNS):
        found.update(PU_FUNCTIONS[:8])  # add common PU docs

    return list(found)


def build_function_context(user_query: str):
    """
    Return only relevant function documentation
    instead of sending the entire 175-function KB.
    """

    funcs = detect_functions(user_query)

    if not funcs:
        return ""

    docs = []

    for fn in funcs[:12]:   # limit context size
        if fn in COMPACT_REFS:
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
#  SECTION 3 · SYSTEM PROMPT BUILDER
# ──────────────────────────────────────────────────────────────

_SQL_PROHIBITION = """
## CRITICAL — PQL IS NOT SQL. NEVER WRITE SQL.

These SQL keywords DO NOT EXIST in PQL. If you write any of them, the query is WRONG:
  ✗ SELECT   ✗ FROM    ✗ JOIN    ✗ LEFT JOIN   ✗ GROUP BY
  ✗ HAVING   ✗ WITH    ✗ AS (CTE)  ✗ OVER(...)  ✗ ORDER BY (as SQL clause)

### WRONG — this is SQL, not PQL. NEVER write this:
```sql
SELECT "LFA1"."LIFNR",
       AVG(DATEDIFF(dd, "EKKO"."BEDAT", "EKPO"."LGDAT")) AS LEAD_TIME
FROM "EKKO"
JOIN "EKPO" ON "EKKO"."EBELN" = "EKPO"."EBELN"
GROUP BY "LFA1"."LIFNR"
```

### CORRECT — this is real PQL:
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
## Advanced PQL Patterns — use as building blocks

### P1 · GLOBAL() — prevents join multiplication
When mixing case-level and activity-level columns, the common table shifts
and values get multiplied. Wrap aggregations with GLOBAL() to isolate them.
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
-- GOOD: filter argument preserves caching
PU_COUNT( "CASES", "ACTIVITIES"."CASE_ID", "ACTIVITIES"."ACTIVITY" = 'Approve' )
-- AVOID: FILTER_TO_NULL breaks caching
PU_COUNT( "CASES", FILTER_TO_NULL("ACTIVITIES"."CASE_ID") )
```

### P4 · Throughput between specific milestones
```pql
CALC_THROUGHPUT(
  FIRST_OCCURRENCE ['Create Order'] TO LAST_OCCURRENCE ['Ship'],
  REMAP_TIMESTAMPS( "ACTIVITIES"."TIMESTAMP", DAYS )
)
```

### P5 · Rework detection — activity repeating > N times
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

### P7 · First/last occurrence per case using INDEX_ORDER
```pql
CASE WHEN INDEX_ORDER(
  "ACTIVITIES"."TIMESTAMP",
  ORDER BY ( "ACTIVITIES"."TIMESTAMP" ASC ),
  PARTITION BY ( "ACTIVITIES"."CASE_ID" )
) = 1 THEN "ACTIVITIES"."ACTIVITY" ELSE NULL END
```

### P8 · Safe KPI ratio — avoid common table issues
```pql
GLOBAL( COUNT( "ACTIVITIES"."TIMESTAMP" ) ) /
GLOBAL( COUNT( "CASES"."CASE_ID" ) )
```

### P9 · Process variant as string
```pql
STRING_AGG(
  "ACTIVITIES"."ACTIVITY", ' → ',
  ORDER BY ( "ACTIVITIES"."TIMESTAMP" ASC ),
  PARTITION BY ( "ACTIVITIES"."CASE_ID" )
)
```

### P10 · SLA compliance check
```pql
CASE WHEN DATEDIFF( dd, "ORDERS"."PROMISE_DATE", "ORDERS"."ACTUAL_DATE" ) > 7
  THEN 1 ELSE 0 END
```

### P11 · Activity filtering with MATCH_ACTIVITIES
```pql
FILTER MATCH_ACTIVITIES(
  NODE( 'Approve' ), NODE( 'Pay' ), EXCLUDING( 'Cancel' )
);
```

### P12 · BIND for 1:N:1 relationships
```pql
PU_SUM( "VENDORS", BIND( "VENDORS", "ORDERS"."AMOUNT" ) )
```

### P13 · DOMAIN_TABLE for all combinations
```pql
PU_SUM(
  DOMAIN_TABLE( "ACTIVITIES"."CASE_ID", "ACTIVITIES"."ACTIVITY" ),
  "ACTIVITIES"."VALUE"
)
```

### P14 · CONSTANT target for single-value aggregation
```pql
PU_AVG( CONSTANT(), "CASES"."THROUGHPUT_TIME" )
```

### P15 · Workdays between dates
```pql
WORKDAYS_BETWEEN(
  WORKDAY_CALENDAR( WEEKDAY_CALENDAR( MON, TUE, WED, THU, FRI ) ),
  "ORDERS"."CREATE_DATE",
  "ORDERS"."CLOSE_DATE"
)
```

### P16 · Supplier performance metrics (SAP tables)
```pql
-- Lead time per vendor
PU_AVG(
  "LFA1",
  DATEDIFF(dd, "EKKO"."BEDAT", "EKPO"."LGDAT")
)

-- Delivery reliability rate per vendor
PU_COUNT("LFA1", "EKPO"."EBELN",
  DATEDIFF(dd, "EKPO"."BEDAT", "EKPO"."LGDAT") <= 0
) /
PU_COUNT("LFA1", "EKPO"."EBELN")

-- On-time delivery count per vendor
PU_COUNT("LFA1", "EKPO"."EBELN",
  DATEDIFF(dd, "EKPO"."BEDAT", "EKPO"."LGDAT") <= 0
)
```
"""

_EXPERT_FRAMEWORK = """
## Expert Query Construction Framework

**Step 1 — Tables & joins**
What is the case/base table? What child tables? Join direction: 1:N or N:1?

**Step 2 — Result level**
Case level or activity level? Mixing levels → GLOBAL() required.

**Step 3 — Filters first**
FILTER for simple conditions. BIND_FILTERS for non-common tables. FILTER_TO_NULL only when no better option.

**Step 4 — Compose KPIs**
Build innermost aggregation first. Wrap with GLOBAL() at table boundaries. CASE WHEN for conditionals.

**Step 5 — Performance check**
→ PU_COUNT vs PU_COUNT_DISTINCT: use COUNT for key columns
→ AVG vs MEDIAN: use AVG unless median is required
→ PU filter arg vs FILTER_TO_NULL: always prefer filter arg
→ Is GLOBAL() missing anywhere?

**Step 6 — Final query**
→ Write in ```pql code block
→ Explain each section
→ Flag NULL handling, empty tables, join edge cases

## Anti-patterns — always avoid
1. Missing GLOBAL() when mixing case + activity columns → value multiplication
2. FILTER_TO_NULL inside PU-functions by default → kills caching
3. PU_COUNT_DISTINCT on key column → use PU_COUNT
4. MEDIAN when AVG is sufficient → expensive sort
5. Missing double-quotes on table/column names
6. Single-quoting column names (only for string constants)
7. Writing SQL (SELECT/FROM/JOIN/GROUP BY) instead of PQL → NEVER do this
"""


def build_system_prompt(complexity: str, show_reasoning: bool) -> str:
    func_ref = "\n".join(f"### {fn}\n{doc}" for fn, doc in COMPACT_REFS.items())

    base = f"""You are an expert Celonis PQL (Process Query Language) engineer.
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

## Full PQL Function Reference (175 functions)
{func_ref}
"""

    if complexity in ("Advanced", "Expert"):
        base += _ADVANCED_PATTERNS

    if complexity == "Expert":
        base += _EXPERT_FRAMEWORK

    if show_reasoning and complexity in ("Advanced", "Expert"):
        base += """
## Response Format
1. **Analysis** — state what tables, joins, functions are needed
2. **Query** — complete PQL in a ```pql code block
3. **Explanation** — explain each part of the query
4. **Performance notes** — describe optimization choices
5. **Edge cases** — flag NULL handling or filter propagation issues
"""

    elif complexity == "Intermediate":
        base += """
## Response Format
1. PQL in a ```pql code block
2. Explain each function used
3. Mention important gotchas
"""

    else:
        base += """
## Response Format
1. PQL in a ```pql code block
2. Short plain-English explanation
"""

    instructions = {
        "Basic": """
Simple queries.
Use one or two functions maximum.
Clear placeholder table names.
""",
        "Intermediate": """
Queries may contain 2–5 functions.
Use filters, CASE WHEN logic, and simple aggregations.
Explain join directions when necessary.
""",
        "Advanced": """
Use nested functions, GLOBAL(), and PU aggregations.
Support multi-table logic and performance optimization.
Always explain why GLOBAL() is required.
""",
        "Expert": """
Write production-ready Celonis PQL.

Capabilities expected:
- Multi-KPI queries
- Nested PU aggregations
- Throughput calculations
- Rework detection
- Automation rate calculations
- Prevent join multiplication using GLOBAL()

Stress-test examples the assistant must solve:

1. Generate a full KPI query calculating throughput time,
   rework count, and automation rate per vendor while avoiding join multiplication.

2. Write a single PQL query calculating:
   - average throughput per case
   - number of rework activities
   - first activity timestamp
   - last activity timestamp
   - automation rate (system activities / total)

Ensure PU functions are used correctly and queries remain performant.
"""
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
[data-testid="stSidebar"] { background:#0d1117; border-right:1px solid #1e2531; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color:#94a3b8 !important; }
[data-testid="stChatMessage"] {
    background:#0d1117 !important;
    border:1px solid #1e2531;
    border-radius:12px !important;
    margin-bottom:10px;
}
[data-testid="stChatMessageContent"] p { color:#cbd5e1 !important; font-size:14px; }
pre, code {
    background:#0d1117 !important;
    border:1px solid #30363d !important;
    border-radius:8px !important;
    color:#e6edf3 !important;
    font-size:12.5px !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select {
    background:#161b22 !important;
    border:1px solid #30363d !important;
    color:#e2e8f0 !important;
    border-radius:6px !important;
}
.stButton > button {
    background:#161b22 !important;
    border:1px solid #30363d !important;
    color:#c7d2fe !important;
    border-radius:8px !important;
    font-size:12px !important;
}
.stButton > button:hover { border-color:#6366f1 !important; color:#e0e7ff !important; }
[data-testid="stChatInputSubmitButton"] button {
    background:linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    border:none !important;
}
[data-testid="stChatInput"] textarea {
    background:#161b22 !important;
    border:1px solid #30363d !important;
    color:#e2e8f0 !important;
    border-radius:10px !important;
}
details { border:1px solid #1e2531 !important; border-radius:8px !important; }
h1,h2,h3 { color:#f1f5f9 !important; }
[data-testid="stMetric"] {
    background:#0d1117; border:1px solid #1e2531;
    border-radius:10px; padding:10px 14px;
}

/* Improve visibility of assistant text */
[data-testid="stChatMessageContent"] {
    color:#e2e8f0 !important;
}

[data-testid="stChatMessageContent"] p {
    color:#e2e8f0 !important;
}

[data-testid="stChatMessageContent"] li {
    color:#e2e8f0 !important;
}

[data-testid="stChatMessageContent"] strong {
    color:#f8fafc !important;
}

[data-testid="stChatMessageContent"] span {
    color:#e2e8f0 !important;
}

/* Fix bullet visibility */
[data-testid="stChatMessageContent"] ul {
    color:#e2e8f0 !important;
}

/* Improve code block readability */
pre, code {
    background:#111827 !important;
    border:1px solid #374151 !important;
    color:#f1f5f9 !important;
}

/* ───────── PQL Syntax Highlight ───────── */

pre code {
    color:#e6edf3 !important;
    font-family: "JetBrains Mono", monospace !important;
    font-size:13px !important;
    line-height:1.5 !important;
}

/* highlight functions */
code .pu, code .fn {
    color:#22c55e !important;
    font-weight:600;
}

/* highlight table names */
code .tbl {
    color:#60a5fa !important;
}

/* highlight strings */
code .str {
    color:#facc15 !important;
}

/* highlight numbers */
code .num {
    color:#fb7185 !important;
}

/* ───────── Function Badges ───────── */

.func-badge {
    display:inline-block;
    padding:2px 8px;
    margin:2px;
    border-radius:6px;
    background:#1e293b;
    border:1px solid #334155;
    color:#38bdf8;
    font-size:11px;
    font-weight:600;
    letter-spacing:0.3px;
}

/* ───────── Better Code Blocks ───────── */

pre {
    background:#020617 !important;
    border:1px solid #1e293b !important;
    border-radius:10px !important;
    padding:14px !important;
    overflow-x:auto !important;
}

/* ───────── Smooth Chat Bubbles ───────── */

[data-testid="stChatMessage"] {
    box-shadow:0 4px 14px rgba(0,0,0,0.35);
}

/* ───────── Sidebar Hover Effects ───────── */

.stButton > button:hover {
    background:#1e293b !important;
    border-color:#6366f1 !important;
    transform:scale(1.02);
}
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
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────────────────────
#  SECTION 7 · GROQ CLIENT
#  Key priority: st.secrets → environment variable
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
        '<div style="font-size:11px;color:#475569;">175 functions · Groq powered</div></div></div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Model selector
    st.markdown('**🤖 Model**')
    selected_model = st.selectbox(
        'Model',
        options=list(GROQ_MODELS.keys()),
        index=list(GROQ_MODELS.keys()).index(st.session_state.model_id),
        format_func=lambda k: GROQ_MODELS[k],
        label_visibility='collapsed',
    )
    st.session_state.model_id = selected_model
    st.caption(f'`{selected_model}`')

    st.divider()

    # Complexity slider
    st.markdown('**🎛 Complexity**')
    complexity = st.select_slider(
        'Complexity',
        options=['Basic', 'Intermediate', 'Advanced', 'Expert'],
        value=st.session_state.complexity,
        label_visibility='collapsed',
    )
    st.session_state.complexity = complexity
    st.caption(COMPLEXITY_DESC[complexity])

    # Show reasoning toggle
    st.session_state.show_reasoning = st.toggle(
        'Show query reasoning',
        value=st.session_state.show_reasoning,
        help='AI explains planning steps before writing the query',
    )

    st.divider()

    # Quick examples
    st.markdown('**💡 Quick examples**')
    for ex in EXAMPLE_PROMPTS.get(complexity, EXAMPLE_PROMPTS['Advanced']):
        if st.button(f'→ {ex}', key=f'ex_{ex}', use_container_width=True):
            st.session_state['_pending'] = ex

    st.divider()

    # Function reference panel
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

    # Stats
    c1, c2 = st.columns(2)
    c1.metric('Queries', st.session_state.total_queries)
    c2.metric('Messages', len(st.session_state.messages))

    if st.button('🗑 Clear chat', use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ──────────────────────────────────────────────────────────────
#  SECTION 9 · MAIN CHAT AREA
# ──────────────────────────────────────────────────────────────

st.markdown('## ⚡ PQL Query Assistant')
st.caption(
    f'Complexity: **{complexity}** · Model: `{st.session_state.model_id}` · '
    f'{len(COMPACT_REFS)} functions loaded'
)

# API key warning
if not _api_key:
    st.warning(
        '**Groq API key not found.**\n\n'
        '**Local:** `export GROQ_API_KEY=gsk_...` then restart.\n\n'
        '**Streamlit Cloud:** App Settings → Secrets → add `GROQ_API_KEY = "gsk_..."`',
        icon='🔑',
    )
    st.stop()

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg['role'], avatar='⚡' if msg['role'] == 'assistant' else '🧑'):
        st.markdown(msg['content'])

# Welcome on first load
if not st.session_state.messages:
    with st.chat_message('assistant', avatar='⚡'):
        st.markdown("""
**Welcome! I'm your PQL Query Assistant.**

I write, explain, and optimize Celonis PQL queries at any complexity level.

**What I can do:**
- 🔨 **Write** PQL from a plain-English description
- 🔍 **Explain** existing PQL line by line
- ⚡ **Optimize** slow or incorrect queries
- 📚 **Teach** any of the 175 PQL functions with examples

**Try asking:**
- *"Calculate average throughput time per case in days"*
- *"How do I use PU_COUNT with a filter condition?"*
- *"Optimize this: AVG( CALC_THROUGHPUT(...) )"*
- *"Detect rework loops where the same activity repeats more than twice"*

→ Use the sidebar to change complexity or browse all 175 functions.
""")




# ─────────────────────────────────────────
# PQL VALIDATOR
# ─────────────────────────────────────────

def validate_pql(query):

    issues = []

    # Missing quotes
    if re.search(r'\b[A-Z]+\.[A-Z]+\b', query) and '"' not in query:
        issues.append("Column names might be missing double quotes.")

    # PU syntax check
    if "PU_" in query and "," not in query:
        issues.append("PU functions require syntax: PU_FUNC(target_table, source_column)")

    # GLOBAL check
    if "CALC_THROUGHPUT" in query and "GLOBAL(" not in query:
        issues.append("Consider wrapping CALC_THROUGHPUT with GLOBAL() when mixing tables.")

    # FILTER_TO_NULL warning
    if "FILTER_TO_NULL" in query:
        issues.append("FILTER_TO_NULL may break caching inside PU functions.")

    return issues

# ─────────────────────────────────────────
# SELF CORRECTING AGENT
# ─────────────────────────────────────────

def correct_query_if_needed(query, issues):

    """
    Uses the LLM to repair invalid PQL queries detected by the validator.
    """

    repair_prompt = f"""
You are a Celonis PQL expert.

The following PQL query may contain mistakes.

Issues detected:
{issues}

Fix the query while keeping the original logic.

Return ONLY the corrected PQL query inside a ```pql block.

Original Query:
{query}
"""

    try:

        response = client.chat.completions.create(
            model=st.session_state.model_id,
            messages=[
                {"role": "system", "content": "You are a Celonis PQL expert."},
                {"role": "user", "content": repair_prompt},
            ],
            temperature=0,
            max_tokens=800
        )

        text = response.choices[0].message.content

        match = re.search(r"```pql(.*?)```", text, re.S)

        if match:
            return match.group(1).strip()

        return text

    except Exception as e:
        return f"Correction failed: {e}"


# ── Helper: call Groq with streaming ──────────────────────────
def stream_groq(prompt_override=None):

    msgs = st.session_state.messages

    # Get latest user query
    user_query = prompt_override if prompt_override else msgs[-1]["content"]

    # Function-aware retrieval
    func_context = build_function_context(user_query)

    # Build system prompt
    system = build_system_prompt(
        st.session_state.complexity,
        st.session_state.show_reasoning
    )

    # Inject retrieved function docs
    if func_context:
        system += "\n\n## Relevant PQL Functions\n" + func_context

    # If sidebar example triggered
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

            # Stream tokens
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full += delta
                placeholder.markdown(full + "▌")

            placeholder.markdown(full)

            # Store assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": full
            })

            st.session_state.total_queries += 1

            # ─────────────────────────────────────────
            # PQL VALIDATION + SELF CORRECTION
            # ─────────────────────────────────────────

            match = re.search(r"```pql(.*?)```", full, re.S)

            if match:

                pql_query = match.group(1).strip()

                issues = validate_pql(pql_query)

                if issues:

                    st.warning("⚠ Validator detected possible PQL issues")

                    corrected = correct_query_if_needed(pql_query, issues)

                    st.markdown("### 🔧 Auto-Corrected Query")

                    st.code(corrected, language="sql")

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




