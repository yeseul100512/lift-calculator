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