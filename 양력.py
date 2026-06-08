"""
에어포일 양력 계산기 - Hess-Smith 패널법
=============================================
사용법:
    python airfoil_lift.py  (대화형 입력)

CSV 형식:
    - Name, Chord(mm), Pitch(deg), Origin(%) 등 헤더
    - "Airfoil surface" 섹션: X(mm), Y(mm) 좌표
    - "Chord line" 섹션: 앞전·뒷전 두 점 (회전 복원용)

패널법 이론:
    Hess-Smith 방법 (Katz & Plotkin, 2001, §11.2)
    - 패널별 소스 σ_j + 전역 단일 와류 γ
    - 불침투 조건: 각 패널 중점에서 법선속도 = 0
    - 쿠타 조건: 후연 상·하면 패널의 접선속도 합 = 0
    - 양력: Kutta-Joukowski  L = ρ · V∞ · Γ · b
             Γ = -2γ · Σds · c  (전체 순환, 차원 환산 포함)
"""

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


# ─────────────────────────────────────────────
#  2. CSV 파싱 및 좌표 정규화
# ─────────────────────────────────────────────

def load_airfoil_csv(filepath):
    """
    CSV 파일에서 에어포일 좌표 읽기.
    Chord line 두 점 기준으로 Pitch 역회전 후 chord=1로 정규화.

    Returns
    -------
    xn, yn    : 정규화된 에어포일 좌표 배열 (CCW, 닫힌 윤곽)
    chord_mm  : 실제 chord 길이 [mm]
    alpha_csv : CSV Pitch(deg) 값 → 받음각으로 사용
    """
    xs, ys = [], []
    chord_pts = []
    header = {}
    sec = None

    with open(filepath, newline='', encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if not row or all(c.strip() == '' for c in row):
                sec = None
                continue
            key = row[0].strip()

            if key in ('Name', 'Chord(mm)', 'Pitch(deg)', 'Origin(%)',
                       'Radius(mm)', 'Thickness(%)'):
                if len(row) >= 2 and row[1].strip():
                    header[key] = row[1].strip()
                continue

            if key == 'Airfoil surface':
                sec = 'airfoil'; continue
            if key == 'Chord line':
                sec = 'chord';   continue
            if key == 'Camber line':
                sec = None;      continue
            if key == 'X(mm)':
                continue

            try:
                xv, yv = float(row[0]), float(row[1])
            except (ValueError, IndexError):
                continue

            if sec == 'airfoil':
                xs.append(xv); ys.append(yv)
            elif sec == 'chord':
                chord_pts.append((xv, yv))

    if len(xs) < 4:
        raise ValueError(f"에어포일 좌표 부족: {len(xs)}개")
    if len(chord_pts) < 2:
        raise ValueError("Chord line 두 점 없음")

    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    alpha_csv = float(header.get('Pitch(deg)', '0'))

    # Chord line 기반 역회전 (CSV에 Pitch 적용된 좌표를 수평으로 되돌림)
    le = np.array(chord_pts[0])
    dx_c = chord_pts[1][0] - chord_pts[0][0]
    dy_c = chord_pts[1][1] - chord_pts[0][1]
    theta_inv = -math.atan2(dy_c, dx_c)

    xc = x - le[0]; yc = y - le[1]
    ct = math.cos(theta_inv); st = math.sin(theta_inv)
    xr = xc * ct - yc * st
    yr = xc * st + yc * ct

    chord_mm = float(xr.max())
    xn = xr / chord_mm
    yn = yr / chord_mm

    # TE 강제 닫기 (열린 후연 → 상하면 y 중점으로 병합)
    if not (math.isclose(xn[0], xn[-1], abs_tol=1e-4) and
            math.isclose(yn[0], yn[-1], abs_tol=1e-4)):
        te_y = 0.5 * (yn[0] + yn[-1])
        xn[0] = 1.0;  yn[0] = te_y
        xn[-1] = 1.0; yn[-1] = te_y

    return xn, yn, chord_mm, alpha_csv


# ─────────────────────────────────────────────
#  3. cos-분포 재샘플링
# ─────────────────────────────────────────────

def resample_airfoil(xn, yn, N_panel=200):
    """
    에어포일 윤곽을 cos-분포로 재샘플링.
    앞전 근방 패널 밀도 증가 → 앞전 흡입 피크 정밀도 향상.

    Parameters
    ----------
    N_panel : 총 패널 수 (짝수 권장)

    Returns
    -------
    xr, yr : 재샘플링 좌표 (NP+1 점, 닫힌 윤곽)
    """
    idx_le = int(np.argmin(xn))

    x_up = xn[:idx_le + 1]; y_up = yn[:idx_le + 1]  # TE → LE
    x_lo = xn[idx_le:];     y_lo = yn[idx_le:]        # LE → TE

    def _resample_surface(xs, ys, n_half):
        s = np.concatenate([[0.0],
                            np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
        s /= s[-1]
        beta = np.linspace(0, math.pi, n_half + 1)
        s_new = 0.5 * (1.0 - np.cos(beta))          # cos-분포
        fx = interp1d(s, xs, kind='cubic')
        fy = interp1d(s, ys, kind='cubic')
        return fx(s_new), fy(s_new)

    n_half = N_panel // 2
    xu, yu = _resample_surface(x_up, y_up, n_half)
    xl, yl = _resample_surface(x_lo, y_lo, n_half)

    xr = np.concatenate([xu, xl[1:]])
    yr = np.concatenate([yu, yl[1:]])
    return xr, yr


# ─────────────────────────────────────────────
#  4. Hess-Smith 패널법
# ─────────────────────────────────────────────

def _panel_infl(Xl, Yl, ds_j):
    """
    로컬 좌표 (Xl, Yl)에서 수평 패널 (0,0)→(ds_j,0) 영향계수.
    Katz & Plotkin §11.2 (식 11.9).

    Returns
    -------
    us, vs : 단위 소스 σ=1 의 (u_l, v_l)
    uv, vv : 단위 와류 γ=1 의 (u_l, v_l)
    """
    eps = 1.0e-14
    r1  = max(math.hypot(Xl,        Yl), eps)
    r2  = max(math.hypot(Xl - ds_j, Yl), eps)
    log_r = math.log(r1 / r2)
    dth   = math.atan2(Yl, Xl - ds_j) - math.atan2(Yl, Xl)

    us =  log_r / (2.0 * math.pi)
    vs =  dth   / (2.0 * math.pi)
    uv = -dth   / (2.0 * math.pi)
    vv =  log_r / (2.0 * math.pi)
    return us, vs, uv, vv


def hess_smith(xn, yn, alpha_deg):
    """
    Hess-Smith 패널법 메인 함수.

    좌표 규약
    ---------
    - CCW 방향 (윗면 TE→LE, 아랫면 LE→TE)
    - Shoelace 면적 > 0
    - 바깥 법선 = (sin_t, -cos_t)  [검증: 윗면 ny>0, 아랫면 ny<0]

    Parameters
    ----------
    xn, yn    : 에어포일 좌표 (NP+1 점, 닫힘)
    alpha_deg : 받음각 [deg]

    Returns
    -------
    gamma   : 단일 와류 강도 (무차원)
    Vt      : 접선속도 배열 [V∞ 단위]
    Cp      : 압력계수 배열
    xm, ym  : 패널 중점
    ds      : 패널 길이
    nx, ny  : 바깥 법선
    """
    alpha = math.radians(alpha_deg)
    N     = len(xn) - 1

    # 패널 기하
    dxp = np.diff(xn); dyp = np.diff(yn)
    ds  = np.hypot(dxp, dyp)
    ct  = dxp / ds;  st = dyp / ds
    xm  = 0.5 * (xn[:-1] + xn[1:])
    ym  = 0.5 * (yn[:-1] + yn[1:])

    # 바깥 법선 (CCW 좌표에서 오른쪽 직교 = (st, -ct))
    nx = st.copy()
    ny = -ct.copy()

    Vx = math.cos(alpha)
    Vy = math.sin(alpha)

    # ────── 영향계수 행렬 구성 ──────
    # [A]{σ₀,...,σ_{N-1},γ}ᵀ = {rhs}
    A   = np.zeros((N + 1, N + 1))
    rhs = np.zeros(N + 1)

    # 불침투 조건 (행 0 ~ N-1)
    for i in range(N):
        vn_v = 0.0
        for j in range(N):
            Xl = (xm[i] - xn[j]) * ct[j] + (ym[i] - yn[j]) * st[j]
            Yl = -(xm[i] - xn[j]) * st[j] + (ym[i] - yn[j]) * ct[j]
            if i == j:
                A[i, j] = 0.5          # 자기 패널: 소스 법선 기여 = 1/2
            else:
                us, vs, uv, vv = _panel_infl(Xl, Yl, ds[j])
                ug_s = us * ct[j] - vs * st[j]
                vg_s = us * st[j] + vs * ct[j]
                ug_v = uv * ct[j] - vv * st[j]
                vg_v = uv * st[j] + vv * ct[j]
                A[i, j]  += ug_s * nx[i] + vg_s * ny[i]
                vn_v     += ug_v * nx[i] + vg_v * ny[i]
        A[i, N] = vn_v
        rhs[i]  = -(Vx * nx[i] + Vy * ny[i])

    # 쿠타 조건 (행 N): Vt[0] + Vt[N-1] = 0
    svv = 0.0
    for j in range(N):
        vts = 0.0; vtv = 0.0
        for ki in (0, N - 1):
            Xl = (xm[ki] - xn[j]) * ct[j] + (ym[ki] - yn[j]) * st[j]
            Yl = -(xm[ki] - xn[j]) * st[j] + (ym[ki] - yn[j]) * ct[j]
            if j == ki:
                vtv += -0.5            # 자기 패널: 와류 접선 기여 = -1/2
            else:
                us, vs, uv, vv = _panel_infl(Xl, Yl, ds[j])
                ug_s = us * ct[j] - vs * st[j]
                vg_s = us * st[j] + vs * ct[j]
                ug_v = uv * ct[j] - vv * st[j]
                vg_v = uv * st[j] + vv * ct[j]
                vts += ug_s * ct[ki] + vg_s * st[ki]
                vtv += ug_v * ct[ki] + vg_v * st[ki]
        A[N, j] = vts
        svv     += vtv
    A[N, N] = svv
    rhs[N]  = -(Vx * ct[0] + Vy * st[0] + Vx * ct[N-1] + Vy * st[N-1])

    # ────── 선형계 풀기 ──────
    sol   = np.linalg.solve(A, rhs)
    sigma = sol[:N]
    gamma = sol[N]

    # ────── 접선속도 및 Cp ──────
    Vt = np.zeros(N)
    for i in range(N):
        Vt[i] = Vx * ct[i] + Vy * st[i]
        for j in range(N):
            Xl = (xm[i] - xn[j]) * ct[j] + (ym[i] - yn[j]) * st[j]
            Yl = -(xm[i] - xn[j]) * st[j] + (ym[i] - yn[j]) * ct[j]
            if i == j:
                Vt[i] += -0.5 * gamma
            else:
                us, vs, uv, vv = _panel_infl(Xl, Yl, ds[j])
                ug_s = us * ct[j] - vs * st[j]
                vg_s = us * st[j] + vs * ct[j]
                ug_v = uv * ct[j] - vv * st[j]
                vg_v = uv * st[j] + vv * ct[j]
                Vt[i] += (sigma[j] * (ug_s * ct[i] + vg_s * st[i]) +
                          gamma    * (ug_v * ct[i] + vg_v * st[i]))

    Cp = 1.0 - Vt ** 2
    return gamma, Vt, Cp, xm, ym, ds, nx, ny


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


# ─────────────────────────────────────────────
#  7. 메인
# ─────────────────────────────────────────────

def main():
    print("=" * 58)
    print("  에어포일 최소 비행 속력 계산기  ─  Hess-Smith 패널법")
    print("  이론: Katz & Plotkin, Low-Speed Aerodynamics §11.2")
    print("=" * 58)

    # ── 입력 ──
    csv_path = input("  CSV 파일 경로: ").strip()
    if not csv_path:
        csv_path = 'naca0010-il.csv'
        print(f"  → 기본값 사용: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"[오류] 파일 없음: {csv_path}")
        sys.exit(1)

    T_C       = _get_input("기온 [°C]",          25.0)
    RH_pct    = _get_input("상대습도 [%]",        60.0)
    weight_kg = _get_input("전체 무게 [kg]",       1.0)
    span_cm   = _get_input("날개 스팬 [cm]",      30.0)
    N_panel   = _get_input("패널 수 (100~400)",   200,  int)

    span_m   = span_cm / 100.0
    weight_N = weight_kg * 9.80665  # kg → N (중력)

    # ── 대기 물성 ──
    rho = air_density(T_C, RH_pct)
    mu  = dynamic_viscosity(T_C)
    print(f"\n  [대기] ρ = {rho:.4f} kg/m³   μ = {mu:.3e} Pa·s")

    # ── 파일 로드 ──
    print(f"\n  [파일] {csv_path}")
    xn_raw, yn_raw, chord_mm, alpha_csv = load_airfoil_csv(csv_path)
    chord_m = chord_mm / 1000.0
    print(f"  [좌표] chord = {chord_mm:.1f} mm  |  Pitch = {alpha_csv:.1f}°  (받음각으로 사용)")
    print(f"  [좌표] 원본 패널 수 = {len(xn_raw)-1}")

    # ── 재샘플링 ──
    print(f"  [샘플] cos-분포 재샘플링 → {N_panel} 패널")
    xn, yn = resample_airfoil(xn_raw, yn_raw, N_panel)
    print(f"  [샘플] 완료 (NP = {len(xn)-1})")

    # ── 패널법 ──
    print(f"\n  [패널] Hess-Smith 행렬 풀기 ({len(xn)-1}×{len(xn)-1})...")
    gamma, Vt, Cp, xm, ym, ds, nx, ny = hess_smith(xn, yn, alpha_csv)

    kutta_res = abs(Vt[0] + Vt[-1])
    print(f"  [패널] γ = {gamma:.6f}")
    print(f"  [검증] 쿠타 잔차 = {kutta_res:.2e}  (0에 가까울수록 좋음)")

    # ── 양력계수 (속도와 무관) ──
    CL = -2.0 * gamma * ds.sum()
    S  = chord_m * span_m

    if CL <= 0:
        print(f"\n  [오류] 이 받음각({alpha_csv:.1f}°)에서 양력계수 CL = {CL:.4f} ≤ 0")
        print("         받음각이 너무 낮거나 음수입니다. 비행 불가.")
        sys.exit(1)

    # ── 최소 속력 역산 ──
    # L = CL * 0.5 * rho * V^2 * S = W  →  V_min = sqrt(2W / (CL·ρ·S))
    V_min     = math.sqrt(2.0 * weight_N / (CL * rho * S))
    V_comfort = V_min * 1.2  # 20% 여유 마진

    L_min     = CL * 0.5 * rho * V_min**2 * S
    L_comfort = CL * 0.5 * rho * V_comfort**2 * S
    Re_min     = rho * V_min * chord_m / mu
    Re_comfort = rho * V_comfort * chord_m / mu
    thin_CL    = 2.0 * math.pi * math.sin(math.radians(alpha_csv))

    # ── 결과 출력 ──
    print()
    print("=" * 58)
    print("  최종 결과")
    print("=" * 58)
    print(f"  받음각 alpha           = {alpha_csv:.2f} deg")
    print(f"  양력계수 CL            = {CL:.4f}")
    print(f"  박판이론 CL            = {thin_CL:.4f}  (두께=0 가정, 참고용)")
    print(f"  공기 밀도              = {rho:.4f} kg/m³")
    print(f"  동적 점성계수          = {mu:.3e} Pa·s")
    print(f"  날개 면적              = {S*1e4:.1f} cm²  ({S:.6f} m²)")
    print(f"  전체 무게              = {weight_kg:.3f} kg  ({weight_N:.3f} N)")
    print()
    print(f"  ▶ 최소 비행 속력       = {V_min:.3f} m/s  ({V_min*3.6:.2f} km/h)")
    print(f"    발생 양력            = {L_min:.4f} N  (= 무게 {weight_N:.4f} N)")
    print(f"    레이놀즈수 Re        = {Re_min:.3e}")
    print()
    print(f"  ▶ 여유 비행 권장 속력  = {V_comfort:.3f} m/s  ({V_comfort*3.6:.2f} km/h)  (+20% 마진)")
    print(f"    발생 양력            = {L_comfort:.4f} N  (무게 대비 {L_comfort/weight_N:.2f}배)")
    print(f"    레이놀즈수 Re        = {Re_comfort:.3e}")
    print("=" * 58)


if __name__ == '__main__':
    main()