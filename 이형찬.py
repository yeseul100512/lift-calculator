import sys
import csv
import math
import os

import numpy as np
from scipy.interpolate import interp1d

from 배승기 import air_density, dynamic_viscosity
from 황지운 import load_airfoil_csv, resample_airfoil, hess_smith
from 이지훈 import _get_input

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