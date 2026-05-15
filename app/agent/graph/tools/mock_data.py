import random
from datetime import datetime, timedelta
from typing import Dict, List, Any


def generate_mock_electricity_data(
    province: str,
    metric: str,
    time_range: str = "24h",
    num_points: int = 24,
) -> Dict[str, Any]:
    base_values = {
        "load": {"SN": 35000, "SX": 28000, "GS": 18000},
        "generation": {"SN": 12000, "SX": 8000, "GS": 5000},
        "price": {"SN": 350, "SX": 280, "GS": 200},
        "new_energy": {"SN": 8000, "SX": 5000, "GS": 3000},
    }
    
    metric_defaults = base_values.get(metric, {"SN": 1000})
    base = metric_defaults.get(province, 1000)
    
    if metric == "price":
        variance = base * 0.15
    else:
        variance = base * 0.08
    
    now = datetime.now()
    if time_range == "24h":
        start = now - timedelta(hours=24)
        interval_minutes = 60
    elif time_range == "7d":
        start = now - timedelta(days=7)
        interval_minutes = 60 * 24
    elif time_range == "30d":
        start = now - timedelta(days=30)
        interval_minutes = 60 * 24
    else:
        start = now - timedelta(hours=24)
        interval_minutes = 60
    
    data = []
    timestamps = []
    current = start
    
    for i in range(num_points):
        value = base + random.uniform(-variance, variance)
        if metric == "price":
            hour_factor = 1.0 + 0.2 * (i % 24) / 24.0
            value = value * hour_factor
        
        data.append(round(value, 2))
        timestamps.append(current.strftime("%Y-%m-%d %H:%M"))
        current += timedelta(minutes=interval_minutes)
    
    return {
        "province": province,
        "metric": metric,
        "time_range": time_range,
        "data": data,
        "timestamps": timestamps,
        "data_points": len(data),
        "metadata": {
            "source": "mock",
            "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }


def generate_mock_policy_chunks(query: str, provinces: List[str]) -> List[Dict[str, Any]]:
    mock_policies = [
        {
            "content": "陕西省电力市场交易规则规定，发电企业需在每月15日前提交次月发电计划。",
            "source": "陕西省电力交易中心",
            "title_path": "交易规则/发电企业/计划申报",
            "province": "SN",
        },
        {
            "content": "新能源发电企业参与现货市场交易，需满足最低装机容量要求50MW。",
            "source": "西北能监局",
            "title_path": "现货市场/新能源/准入条件",
            "province": "SN",
        },
        {
            "content": "电力用户可直接参与市场化交易，需在电力交易平台完成注册备案。",
            "source": "国家能源局",
            "title_path": "市场准入/电力用户/注册要求",
            "province": "ALL",
        },
        {
            "content": "陕西省峰谷电价时段划分：峰段9:00-12:00,19:00-22:00；谷段0:00-7:00。",
            "source": "陕西省发改委",
            "title_path": "电价政策/峰谷时段/划分标准",
            "province": "SN",
        },
        {
            "content": "风电、光伏发电上网电价执行政府定价或市场竞价，具体标准见各省政策。",
            "source": "国家发改委",
            "title_path": "新能源电价/上网电价/定价方式",
            "province": "ALL",
        },
    ]
    
    filtered = []
    for policy in mock_policies:
        policy_province = policy.get("province", "ALL")
        if policy_province in provinces or policy_province == "ALL":
            filtered.append({
                "content": policy["content"],
                "source": policy["source"],
                "title_path": policy["title_path"],
                "score": round(random.uniform(0.7, 0.95), 2),
            })
    
    return filtered[:3] if filtered else mock_policies[:2]