"""
异常监控埋点模块 - 写入 anomaly_log 表
用于记录 AI 调用失败、计算结果异常、计算超时等情况
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import datetime
import json
import logging
import pymysql
import pymysql.cursors

logger = logging.getLogger(__name__)

_DB_CONFIG = {
    "host":        "127.0.0.1",
    "port":        3306,
    "user":        "root",
    "password":    "Deng123456*",
    "database":    "packing_demo",
    "charset":     "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}


def log_anomaly(
    anomaly_type: str,
    severity: str = "error",
    session_id: str = None,
    calc_no: str = None,
    error_code: str = None,
    error_msg: str = None,
    duration_ms: int = None,
    extra: dict = None,
):
    """
    记录一条异常到 anomaly_log 表。
    失败时只打 warning，不影响主流程。

    anomaly_type:
      - ai_failure    : AI API 调用失败（重试耗尽或整个 agent 崩溃）
      - calc_anomaly  : 计算结果异常（装不下 / 利用率过低）
      - calc_timeout  : 接口耗时超阈值

    severity:
      - warning  : 值得关注但不影响结果（如利用率偏低）
      - error    : 单次 API 失败（已降级兜底）
      - critical : 整个 agent 崩溃（完全走兜底逻辑）
    """
    try:
        now = datetime.datetime.now(
            datetime.timezone(datetime.timedelta(hours=8))
        ).strftime("%Y-%m-%d %H:%M:%S")
        conn = pymysql.connect(**_DB_CONFIG)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO anomaly_log
                      (created_at, anomaly_type, severity, session_id, calc_no,
                       error_code, error_msg, duration_ms, extra)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        now,
                        anomaly_type,
                        severity,
                        session_id,
                        calc_no,
                        str(error_code)[:50] if error_code is not None else None,
                        str(error_msg)[:1000] if error_msg is not None else None,
                        duration_ms,
                        json.dumps(extra, ensure_ascii=False) if extra else None,
                    ),
                )
            conn.commit()
        finally:
            conn.close()
        logger.info("[monitor] 异常已记录: type=%s severity=%s code=%s",
                    anomaly_type, severity, error_code)
    except Exception as e:
        logger.warning("[monitor] anomaly_log 写入失败（不影响主流程）: %s", e)
