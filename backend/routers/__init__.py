from .aqi import router as aqi_router
from .forecast import router as forecast_router
from .carbon import router as carbon_router
from .dashboard import router as dashboard_router
from .simulator import router as simulator_router

__all__ = [
    "aqi_router",
    "forecast_router",
    "carbon_router",
    "dashboard_router",
    "simulator_router",
]

