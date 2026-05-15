"""直接查询电力数据库的适配器"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import pymysql
from pymysql.cursors import DictCursor

logger = logging.getLogger(__name__)


class DirectDBAdapter:
    """直接从数据库查询电力数据"""
    
    REGION_DB_MAPPING = {
        "SN": "electricity_trading_analytics_shaanxi",
        "SX": "electricity_trading_analytics_shanxi",
        "SD": "electricity_trading_analytics_shandong",
        "GS": "electricity_trading_analytics_gansu",
    }
    
    METRIC_TABLE_MAPPING = {
        "price": {
            "table": "clearing_price",
            "fields": ["realtime_clearing_price", "dayahead_clearing_price"],
            "time_field": "settlement_date",
        },
        "load": {
            "table": "trading",
            "fields": ["demand"],
            "time_field": "settlement_date",
        },
        "generation": {
            "table": "trading",
            "fields": ["renewable_output"],
            "time_field": "settlement_date",
        },
        "new_energy": {
            "table": "trading",
            "fields": ["pv_output", "wind_output"],
            "time_field": "settlement_date",
        },
    }
    
    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        available_db: str = "electricity_trading_analytics_shaanxi",
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.available_db = available_db
        self._connection = None
    
    def _get_connection(self) -> pymysql.Connection:
        if self._connection is None or not self._connection.open:
            self._connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.available_db,
                charset="utf8mb4",
                cursorclass=DictCursor,
            )
        return self._connection
    
    def _get_date_range(self, time_range: str) -> tuple:
        today = datetime.now().date()
        
        if time_range == "24h":
            start_date = today - timedelta(days=1)
            end_date = today - timedelta(days=1)
        elif time_range == "7d":
            start_date = today - timedelta(days=7)
            end_date = today
        elif time_range == "30d":
            start_date = today - timedelta(days=30)
            end_date = today
        else:
            start_date = today - timedelta(days=1)
            end_date = today - timedelta(days=1)
        
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    
    def fetch(
        self,
        province: str,
        metric: str,
        time_range: str = "24h",
    ) -> Dict[str, Any]:
        """从数据库直接查询数据"""
        
        db_name = self.REGION_DB_MAPPING.get(province, self.available_db)
        
        # 检查是否有权限访问该数据库
        if province != "SN":
            logger.warning(f"[DirectDBAdapter] 只能查询陕西(SN)数据，province={province} 被限制")
            return self._mock_result(province, metric, time_range)
        
        metric_config = self.METRIC_TABLE_MAPPING.get(metric)
        if not metric_config:
            logger.warning(f"[DirectDBAdapter] 未知的 metric: {metric}")
            return self._mock_result(province, metric, time_range)
        
        start_date, end_date = self._get_date_range(time_range)
        table = metric_config["table"]
        fields = metric_config["fields"]
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            fields_str = ", ".join(fields)
            query = f"""
                SELECT settlement_date, settlement_time, {fields_str}
                FROM {table}
                WHERE settlement_date BETWEEN %s AND %s
                ORDER BY settlement_date, settlement_time
            """
            
            cursor.execute(query, (start_date, end_date))
            rows = cursor.fetchall()
            
            data = []
            timestamps = []
            
            for row in rows:
                timestamps.append(f"{row['settlement_date']} {row['settlement_time']}")
                for field in fields:
                    if row.get(field) is not None:
                        data.append(float(row[field]))
            
            logger.info(f"[DirectDBAdapter] 查询到 {len(rows)} 条记录, {len(data)} 个数据点")
            
            return {
                "province": province,
                "metric": metric,
                "time_range": time_range,
                "data": data,
                "timestamps": timestamps,
                "data_points": len(data),
                "metadata": {
                    "source": "direct_db",
                    "database": self.available_db,
                    "table": table,
                    "query_date": f"{start_date} ~ {end_date}",
                },
            }
            
        except Exception as e:
            logger.exception(f"[DirectDBAdapter] 查询失败: {e}")
            return self._mock_result(province, metric, time_range)
    
    def _mock_result(self, province: str, metric: str, time_range: str) -> Dict[str, Any]:
        """返回模拟数据"""
        from app.agent.graph.tools.mock_data import generate_mock_electricity_data
        mock = generate_mock_electricity_data(province, metric, time_range, 24)
        mock["metadata"]["source"] = "mock_fallback"
        return mock
    
    def fetch_sync(
        self,
        province: str,
        metric: str,
        time_range: str = "24h",
    ) -> Dict[str, Any]:
        return self.fetch(province, metric, time_range)
    
    def close(self):
        if self._connection and self._connection.open:
            self._connection.close()