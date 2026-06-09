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
