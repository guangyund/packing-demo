# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding="utf-8")
import pymysql

conn = pymysql.connect(host="127.0.0.1", port=3306, user="root",
                       password="Deng123456*", database="packing_demo", charset="utf8mb4")
with conn.cursor() as cur:
    for table in ["pack_results", "feedback"]:
        cur.execute(f"SHOW FULL COLUMNS FROM {table}")
        rows = cur.fetchall()
        print(f"\n=== {table} ===")
        for r in rows:
            print(f"  {r[0]:30s}  {r[1]:20s}  null={r[3]}  default={r[5]}")
conn.close()
