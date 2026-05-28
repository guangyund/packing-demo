# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding="utf-8")
import pymysql
conn = pymysql.connect(host="127.0.0.1", port=3306, user="root", password="Deng123456*",
                       database="packing_demo", charset="utf8mb4",
                       cursorclass=pymysql.cursors.DictCursor)
with conn.cursor() as cur:
    cur.execute("""
        SELECT p.session_id,
               COUNT(DISTINCT p.result_id) AS plan_count,
               GROUP_CONCAT(p.plan_type ORDER BY p.created_at) AS plan_types,
               COUNT(DISTINCT f.result_id) AS fb_count,
               MAX(p.created_at) AS latest
        FROM pack_results p
        LEFT JOIN feedback f ON f.result_id = p.result_id
        WHERE p.session_id IS NOT NULL
        GROUP BY p.session_id
        ORDER BY MAX(p.created_at) DESC
        LIMIT 8
    """)
    for r in cur.fetchall():
        match = "OK" if r["plan_count"] == r["fb_count"] else "!!"
        print(f"[{match}] plans={r['plan_count']} fb={r['fb_count']}  {r['plan_types']}  {r['latest']}")
conn.close()
