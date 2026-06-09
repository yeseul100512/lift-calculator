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

# ─────────────────────────────────────────────
#  5. Kutta-Joukowski 양력 계산
# ─────────────────────────────────────────────

def compute_lift(gamma, ds, rho, V_inf, chord_m, span_m):
    """
    Kutta-Joukowski 정리를 이용한 양력 계산.

    Katz & Plotkin §11.2 식:
        CL = -2 · gamma · Σds      (무차원 양력계수, 직접)
        L  = CL · (0.5 · ρ · V∞² · c · b)

    순환:
        Γ_dim = CL · V∞ · c / 2  [m²/s]   (참고용)

    Parameters
    ----------
    gamma   : 패널법 와류 강도 (무차원, 단위 chord당)
    ds      : 패널 길이 배열 (정규화 좌표, chord=1 기준)
    rho     : 공기 밀도 [kg/m³]
    V_inf   : 자유류 속도 [m/s]
    chord_m : chord [m]
    span_m  : 스팬 [m]

    Returns
    -------
    CL : 무차원 양력계수
    L  : 실제 양력 [N]
    """
    CL   = -2.0 * gamma * ds.sum()          # Katz §11.2 직접 공식
    S    = chord_m * span_m                  # 날개 면적 [m²]
    L    = CL * 0.5 * rho * V_inf ** 2 * S  # 양력 [N]
    return CL, L


# ─────────────────────────────────────────────
#  6. 유틸리티
# ─────────────────────────────────────────────

def _get_input(prompt, default, cast=float):
    try:
        raw = input(f"  {prompt} [{default}]: ").strip()
        return cast(raw) if raw else cast(default)
    except ValueError:
        print(f"  [경고] 입력 오류 → 기본값 {default} 사용")
        return cast(default)
