import sys
import csv
import math
import os

import numpy as np
from scipy.interpolate import interp1d

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