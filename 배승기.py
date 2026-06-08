import sys
import csv
import math
import os

import numpy as np
from scipy.interpolate import interp1d

# ─────────────────────────────────────────────
#  1. 대기 물성 계산
# ─────────────────────────────────────────────

def saturated_vapor_pressure(T_C):
    """Magnus 공식: 포화수증기압 [Pa]"""
    return 610.94 * math.exp(17.625 * T_C / (243.04 + T_C))


def air_density(T_C, RH_pct, P_Pa=101325.0):
    """
    습윤 공기 밀도 [kg/m³] - 이상기체 모델
    T_C   : 기온 [°C]
    RH_pct: 상대습도 [%]
    P_Pa  : 기압 [Pa] (기본값: 1 atm)

    e = (RH/100) · e_sat(T)
    ρ = (P - e) / (R_d · T) + e / (R_v · T)
    """
    T_K = T_C + 273.15
    Rd  = 287.058    # 건조공기 기체상수 [J/(kg·K)]
    Rv  = 461.495    # 수증기 기체상수 [J/(kg·K)]
    e_s = saturated_vapor_pressure(T_C)
    e   = (RH_pct / 100.0) * e_s
    return (P_Pa - e) / (Rd * T_K) + e / (Rv * T_K)


def dynamic_viscosity(T_C):
    """Sutherland 공식: 동적 점성계수 [Pa·s]"""
    T_K = T_C + 273.15
    mu0 = 1.716e-5   # 기준 점성계수 (273 K)
    T0  = 273.15
    S   = 110.4      # Sutherland 상수 [K]
    return mu0 * (T_K / T0) ** 1.5 * (T0 + S) / (T_K + S)