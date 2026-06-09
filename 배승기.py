import sys
import csv
import math
import os

import numpy as np
from scipy.interpolate import interp1d

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
