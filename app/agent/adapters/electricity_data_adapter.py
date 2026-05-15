from abc import ABC, abstractmethod
import httpx
import logging
import os
import subprocess
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from app.agent.adapters.data_cache import DataCache

logger = logging.getLogger(__name__)

SKILLS_PATH = Path(__file__).parent.parent.parent.parent / "data" / "skills" / "agentic-data-analysis"


class ElectricityDataAdapter(ABC):
    @abstractmethod
    async def fetch(
        self,
        province: str,
        metric: str,
        time_range: str,
    ) -> list:
        pass
    
    def fetch_sync(
        self,
        province: str,
        metric: str,
        time_range: str,
    ) -> list:
        import asyncio
        return asyncio.run(self.fetch(province, metric, time_range))


class SkillsScriptAdapter(ElectricityDataAdapter):
    def __init__(self, skills_path: str = None, cache_ttl: int = 3600, cache_max_size: int = 1000):
        if skills_path:
            path = Path(skills_path)
            if not path.is_absolute():
                path = Path.cwd() / path
            self.skills_path = path
        else:
            self.skills_path = SKILLS_PATH
        self.scripts_path = self.skills_path / "scripts"
        self._db_configured = self._check_db_config()
        self._cache = DataCache(ttl=cache_ttl, max_size=cache_max_size)
    
    def _check_db_config(self) -> bool:
        config_path = Path.home() / ".electricity_data_skills.json"
        if config_path.exists():
            logger.info(f"Skills database config found: {config_path}")
            return True
        
        env_vars = ["ELECTRICITY_DB_HOST", "ELECTRICITY_DB_USER", "ELECTRICITY_DB_PASSWORD"]
        if all(os.getenv(v) for v in env_vars):
            logger.info("Skills database config found in environment")
            return True
        
        logger.warning("Skills database not configured. Run config.py create-config first.")
        return False
    
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
    
    def _map_province_to_region(self, province: str) -> str:
        mapping = {
            "SN": "shaanxi",
            "SX": "shanxi",
            "GS": "gansu",
            "SD": "shandong",
            "AH": "anhui",
            "GD": "guangdong",
            "ZJ": "zhejiang",
        }
        return mapping.get(province.upper(), province.lower())
    
    def _map_metric_to_fields(self, metric: str) -> str:
        mapping = {
            "load": "demand",
            "generation": "renewable_output",
            "price": "realtime_clearing_price,dayahead_clearing_price",
            "new_energy": "pv_output,wind_output",
        }
        return mapping.get(metric, metric)
    
    async def fetch(
        self,
        province: str,
        metric: str,
        time_range: str,
    ) -> list:
        return self.fetch_sync(province, metric, time_range)
    
    def fetch_sync(
        self,
        province: str,
        metric: str,
        time_range: str,
    ):
        cache_key = f"{province}:{metric}:{time_range}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info(f"[SkillsScriptAdapter] Cache hit for {cache_key}")
            return cached
        
        if not self._db_configured:
            logger.warning("[SkillsScriptAdapter] Database not configured, returning fallback")
            return self._fallback_data(province, metric, time_range)
        
        region = self._map_province_to_region(province)
        start_date, end_date = self._get_date_range(time_range)
        fields = self._map_metric_to_fields(metric)
        
        tables = "trading" if metric in ["load", "generation", "new_energy"] else "clearing_price"
        
        output_dir = Path(__file__).parent.parent.parent.parent / "data" / "skills_output" / region
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "uv", "run", "python",
            str(self.scripts_path / "run_basic_stats.py"),
            "--region", region,
            "--start-date", start_date,
            "--end-date", end_date,
            "--source", "electricity",
            "--tables", tables,
            "--fields", fields,
            "--output", str(output_dir),
        ]
        
        logger.info(f"[SkillsScriptAdapter] Running: {' '.join(cmd)}")
        
        # 清除 VIRTUAL_ENV 避免 uv 环境冲突，设置 UTF-8 编码
        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)
        env["PYTHONIOENCODING"] = "utf-8"
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.skills_path),
                capture_output=True,
                timeout=60,
                env=env,
            )
            
            # 使用 UTF-8 解码，忽略无法解码的字符
            stdout = result.stdout.decode('utf-8', errors='replace') if result.stdout else ""
            stderr = result.stderr.decode('utf-8', errors='replace') if result.stderr else ""
            
            if result.returncode != 0:
                logger.error(f"[SkillsScriptAdapter] Script failed: {stderr[:500]}")
                return []
            
            logger.info(f"[SkillsScriptAdapter] Script output: {stdout[:500]}")
            
            result_file = output_dir / "basic_stats.json"
            if result_file.exists():
                with open(result_file, encoding='utf-8') as f:
                    data = json.load(f)
                    processed = data.get("data", [])
                    result = {
                        "data": processed,
                        "data_points": data.get("data_points", len(processed)),
                        "chart_path": data.get("chart_path", ""),
                        "statistics": data.get("statistics", {}),
                        "source": "skills_script",
                    }
                    self._cache.set(cache_key, result)
                    return result
            
            # 脚本成功运行但没有 JSON 文件（可能只生成了图表）
            # 返回空列表，让其他 adapter 处理
            logger.info(f"[SkillsScriptAdapter] No JSON output, checking for PNG charts")
            png_files = list(output_dir.glob("*.png"))
            if png_files:
                logger.info(f"[SkillsScriptAdapter] Generated {len(png_files)} chart files")
            
            return []
            
        except subprocess.TimeoutExpired:
            logger.error("[SkillsScriptAdapter] Script timeout")
            return []
        except Exception as e:
            logger.exception(f"[SkillsScriptAdapter] Failed: {e}")
            return []
    
    def _fallback_data(self, province: str, metric: str, time_range: str) -> list:
        logger.info(f"[SkillsScriptAdapter] Fallback data for {province}/{metric}/{time_range} - DB not configured")
        return []


class SkillsAPIAdapter(ElectricityDataAdapter):
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def fetch(
        self,
        province: str,
        metric: str,
        time_range: str,
    ) -> list:
        url = f"{self.base_url}/api/v1/data"
        
        params = {
            "province": province,
            "metric": metric,
            "time_range": time_range,
        }
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Skills API fetch success: {len(data.get('data', []))} points")
            
            return data.get("data", [])
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Skills API HTTP error: {e.response.status_code}")
            return []
        except Exception as e:
            logger.error(f"Skills API fetch failed: {e}")
            return []
    
    async def close(self):
        await self.client.aclose()


class LocalDataAdapter(ElectricityDataAdapter):
    def __init__(self, db_session=None, data_dir: str = None):
        self.db = db_session
        self.data_dir = data_dir
    
    async def fetch(
        self,
        province: str,
        metric: str,
        time_range: str,
    ) -> list:
        if self.data_dir:
            return self._fetch_from_file(province, metric)
        
        return []
    
    def _fetch_from_file(self, province: str, metric: str) -> list:
        try:
            import pandas as pd
            file_path = Path(self.data_dir) / f"{province}_{metric}.csv"
            
            if not file_path.exists():
                logger.warning(f"Data file not found: {file_path}")
                return []
            
            df = pd.read_csv(file_path)
            return df["value"].tolist()
        except Exception as e:
            logger.exception(f"Failed to read data file: {e}")
            return []


class CompositeAdapter(ElectricityDataAdapter):
    def __init__(self, adapters: list):
        self.adapters = adapters
    
    async def fetch(
        self,
        province: str,
        metric: str,
        time_range: str,
    ):
        for adapter in self.adapters:
            try:
                # 检查是否是 async 方法
                if hasattr(adapter.fetch, '__func__'):
                    # sync 方法，直接调用
                    data = adapter.fetch(province, metric, time_range)
                else:
                    # async 方法
                    data = await adapter.fetch(province, metric, time_range)
                
                # 处理返回值（dict 或 list）
                if isinstance(data, dict):
                    data_list = data.get("data", [])
                    if data_list:
                        return data  # 返回完整 dict
                elif isinstance(data, list) and data:
                    return data
                    
            except Exception as e:
                logger.warning(f"Adapter {type(adapter).__name__} failed: {e}")
                continue
        
        return []
    
    def fetch_sync(
        self,
        province: str,
        metric: str,
        time_range: str,
    ):
        """同步版本，遍历所有 adapter 直到获取数据"""
        for adapter in self.adapters:
            try:
                data = adapter.fetch_sync(province, metric, time_range)
                
                # 处理返回值（dict 或 list）
                if isinstance(data, dict):
                    data_list = data.get("data", [])
                    if data_list:
                        return data  # 返回完整 dict
                elif isinstance(data, list) and data:
                    return data
                    
            except Exception as e:
                logger.warning(f"Adapter {type(adapter).__name__} failed: {e}")
                continue
        
        return []


def create_data_adapter(settings) -> ElectricityDataAdapter:
    adapters = []
    
    # Skills 脚本优先（完整功能：数据+图表+统计）
    skills_path = getattr(settings, 'electricity_skills_path', None)
    if skills_path:
        adapters.append(SkillsScriptAdapter(skills_path=skills_path))
    elif SKILLS_PATH.exists():
        adapters.append(SkillsScriptAdapter())
    
    # DirectDBAdapter 作为快速备用（当 Skills 失败时）
    db_host = getattr(settings, 'electricity_db_host', None)
    db_port = getattr(settings, 'electricity_db_port', 3306)
    db_user = getattr(settings, 'electricity_db_user', None)
    db_password = getattr(settings, 'electricity_db_password', None)
    
    if db_host and db_user and db_password:
        from app.agent.adapters.direct_db_adapter import DirectDBAdapter
        available_db = getattr(settings, 'base_db_name', 'electricity_trading_analytics_shaanxi')
        adapters.append(DirectDBAdapter(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            available_db=available_db,
        ))
        logger.info(f"[DataAdapter] DirectDBAdapter created for {available_db}")
    
    # API 作为备用
    skills_url = getattr(settings, 'electricity_skills_url', None)
    if skills_url:
        adapters.append(SkillsAPIAdapter(base_url=skills_url))
    
    # 本地文件作为备用
    data_dir = getattr(settings, 'electricity_data_dir', None)
    if data_dir:
        adapters.append(LocalDataAdapter(data_dir=data_dir))
    
    if adapters:
        return CompositeAdapter(adapters)
    
    return LocalDataAdapter()