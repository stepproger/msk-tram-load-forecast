"""Агрегация целевой величины из сырых валидаций и сверка с разметкой организаторов.

boardings(route, date, hour) = count(validation_result == 1), route = число из ngpt_route, дата и час — из tran_date_time.
Результат: data/interim/target_hourly.parquet + отчёт о расхождении с dataset/labels/*.
"""
import duckdb

con = duckdb.connect()
con.sql("""
    CREATE TABLE t AS
    SELECT try_cast(regexp_extract(route_raw, '^(\\d+)') AS INT) AS route, ts::DATE AS d, hour(ts) AS h, count(*) AS boardings
    FROM read_parquet(['data/interim/train.parquet', 'data/interim/test.parquet'])
    WHERE vr = 1 AND ts >= '2025-01-01' AND ts < '2025-11-01'
    GROUP BY ALL
""")
con.sql("COPY (SELECT route, d AS date, h AS hour, boardings FROM t ORDER BY ALL) TO 'data/interim/target_hourly.parquet'")
r = con.sql("""
    WITH l AS (SELECT route, "date" AS d, "hour" AS h, boardings FROM read_csv('dataset/labels/labels_day_*.csv', delim=';'))
    SELECT count(*) AS cells,
           count(*) FILTER (WHERE coalesce(t.boardings, 0) <> coalesce(l.boardings, 0)) AS mismatched_cells,
           sum(coalesce(l.boardings, 0)) AS labels_total, sum(coalesce(t.boardings, 0)) AS ours_total,
           sum(abs(coalesce(t.boardings, 0) - coalesce(l.boardings, 0))) AS abs_diff
    FROM t FULL OUTER JOIN l USING (route, d, h)
""").df()
print(r.to_string(index=False))
print(con.sql("""
    WITH l AS (SELECT route, "date" AS d, "hour" AS h, boardings FROM read_csv('dataset/labels/labels_day_*.csv', delim=';'))
    SELECT coalesce(t.route, l.route) route, coalesce(t.d, l.d) d, coalesce(t.h, l.h) h, t.boardings ours, l.boardings labels
    FROM t FULL OUTER JOIN l USING (route, d, h)
    WHERE coalesce(t.boardings, 0) <> coalesce(l.boardings, 0) ORDER BY abs(coalesce(t.boardings,0)-coalesce(l.boardings,0)) DESC LIMIT 8
""").df().to_string(index=False))
