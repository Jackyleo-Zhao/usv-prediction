# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, VPacker, TextArea

# =========================
# 路径设置
# =========================
CSV_PATH = "ais-USV_filled.csv"
OUT_DIR = "innovation1_topjournal_pack_refined4"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 字体配置
# 请确保以下字体文件与本脚本位于同一目录
# =========================
FONT_DIR = "."

FP_SIMSUN = FontProperties(fname=os.path.join(FONT_DIR, "simsun.ttc"))
FP_SIMSUN_BOLD = FontProperties(fname=os.path.join(FONT_DIR, "simsunb.ttf"))
FP_TIMES = FontProperties(fname=os.path.join(FONT_DIR, "times.ttf"))
FP_TIMES_BOLD = FontProperties(fname=os.path.join(FONT_DIR, "timesbd.ttf"))
FP_TIMES_ITALIC = FontProperties(fname=os.path.join(FONT_DIR, "timesi.ttf"))

# 把本地字体注册给 matplotlib，避免 mathtext 找不到 Times New Roman
for fn in ["simsun.ttc", "simsunb.ttf", "times.ttf", "timesbd.ttf", "timesi.ttf"]:
    matplotlib.font_manager.fontManager.addfont(os.path.join(FONT_DIR, fn))

TIMES_NAME = FP_TIMES.get_name()

# 全局设置
rcParams["axes.unicode_minus"] = False
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["font.family"] = TIMES_NAME

# 数学文本里的西文/数字/单位也走 Times New Roman
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.rm"] = TIMES_NAME
rcParams["mathtext.it"] = f"{TIMES_NAME}:italic"
rcParams["mathtext.bf"] = f"{TIMES_NAME}:bold"
rcParams["mathtext.default"] = "rm"

plt.rcParams.update({
    "font.size": 7.5,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.5,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "axes.linewidth": 0.8,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "figure.figsize": (3, 2),
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
})

# =========================
# 通用工具函数
# =========================
def beautify_axes(ax, labelpad=5, tickpad=2):
    """统一坐标轴风格"""
    ax.xaxis.labelpad = labelpad
    ax.yaxis.labelpad = labelpad
    ax.tick_params(
        axis="both", which="major", direction="in",
        pad=tickpad, length=4, width=0.8, labelsize=7.0
    )
    ax.tick_params(
        axis="both", which="minor", direction="in",
        pad=tickpad, length=2, width=0.6
    )
    ax.margins(x=0.03, y=0.03)


def set_ticklabel_font(ax, x_font=FP_TIMES, y_font=FP_TIMES):
    """分别设置 x/y 轴刻度字体"""
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(x_font)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(y_font)
    ax.xaxis.get_offset_text().set_fontproperties(x_font)
    ax.yaxis.get_offset_text().set_fontproperties(y_font)


def _ta(text, fp, fs):
    """创建 TextArea"""
    return TextArea(text, textprops=dict(fontproperties=fp, fontsize=fs))


def add_packed_xlabel(ax, parts, y=-0.18):
    """
    parts: [(text, fontprop, fontsize), ...]
    用于中文+英文混排的横坐标标签
    """
    ax.set_xlabel("")
    box = HPacker(
        children=[_ta(txt, fp, fs) for txt, fp, fs in parts],
        align="center",
        pad=0,
        sep=0
    )
    anchored = AnchoredOffsetbox(
        loc="lower center",
        child=box,
        frameon=False,
        bbox_to_anchor=(0.5, y),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
        pad=0.0
    )
    ax.add_artist(anchored)


def add_mixed_ylabel_plain(ax, cn_text, en_text, x=-0.11, fontsize=7.5):
    """
    中文宋体 + 英文/单位 Times New Roman 的纵坐标
    """
    ax.set_ylabel("")
    # 中文部分放下半段
    ax.text(
        x, 0.5, cn_text,
        transform=ax.transAxes,
        ha="center", va="top",
        rotation=90,
        fontproperties=FP_SIMSUN,
        fontsize=fontsize
    )
    # 英文单位放上半段
    ax.text(
        x, 0.5, en_text,
        transform=ax.transAxes,
        ha="center", va="bottom",
        rotation=90,
        fontproperties=FP_TIMES,
        fontsize=fontsize
    )


def add_s1_stats_box(ax, mae, rmse, mismatch_abs, mismatch_rel):
    """
    Fig.S1 左上角统计框：全部手工排版
    """
    rows = [
        _ta(f"MAE={mae:.3f}, RMSE={rmse:.3f}", FP_TIMES, 5.5),
        _ta(f"|Δv|>{TAU_ABS:.1f} m/s: {mismatch_abs * 100:.1f}%", FP_TIMES, 5.5),
        _ta(f"|Δv|>{TAU_REL:.1f}×SOG: {mismatch_rel * 100:.1f}%", FP_TIMES, 5.5),
    ]
    vbox = VPacker(children=rows, align="left", pad=0, sep=1)

    anchored = AnchoredOffsetbox(
        loc="upper left",
        child=vbox,
        frameon=True,
        pad=0.2,
        borderpad=0.35,
        bbox_to_anchor=(0.02, 0.98),
        bbox_transform=ax.transAxes
    )
    anchored.patch.set_boxstyle("round,pad=0.20")
    anchored.patch.set_facecolor("white")
    anchored.patch.set_edgecolor("0.7")
    anchored.patch.set_alpha(0.90)
    ax.add_artist(anchored)


def add_s2_top_box(ax, peak_corr, corr0, delta_corr):
    """
    Fig.S2 顶部统计框：中文宋体，数字 Times New Roman
    """
    parts = [
        _ta("峰值相关=", FP_SIMSUN, 5.0),
        _ta(f"{peak_corr:.3f}", FP_TIMES, 5.0),
        _ta("  零时滞相关=", FP_SIMSUN, 5.0),
        _ta(f"{corr0:.3f}", FP_TIMES, 5.0),
        _ta("  提升=", FP_SIMSUN, 5.0),
        _ta(f"{delta_corr:.3f}", FP_TIMES, 5.0),
    ]
    hbox = HPacker(children=parts, align="center", pad=0, sep=0)

    anchored = AnchoredOffsetbox(
        loc="upper right",
        child=hbox,
        frameon=True,
        pad=0.18,
        borderpad=0.30,
        bbox_to_anchor=(0.98, 0.98),
        bbox_transform=ax.transAxes
    )
    anchored.patch.set_boxstyle("round,pad=0.18")
    anchored.patch.set_facecolor("white")
    anchored.patch.set_edgecolor("0.7")
    anchored.patch.set_alpha(0.88)
    ax.add_artist(anchored)


def add_fig3_xticklabels(ax, y=-0.08):
    """
    Fig.S3 横轴：第一个标签里的 AIS 用 Times，第二个标签保持中文宋体
    """
    ax.set_xticklabels(["", ""])
    trans = ax.get_xaxis_transform()

    # x=0 位置：AIS航迹链路
    ax.text(
        0, y, "AIS",
        transform=trans,
        ha="right", va="top",
        fontproperties=FP_TIMES,
        fontsize=7.5
    )
    ax.text(
        0, y, "字段估计",
        transform=trans,
        ha="left", va="top",
        fontproperties=FP_SIMSUN,
        fontsize=7.5
    )

    # x=1 位置：位置差分链路
    ax.text(
        1, y, "位置差分估计",
        transform=trans,
        ha="center", va="top",
        fontproperties=FP_SIMSUN,
        fontsize=6.5
    )


def save_fig(fig, stem):
    """保存 PNG 和 PDF"""
    png = os.path.join(OUT_DIR, f"{stem}.png")
    pdf = os.path.join(OUT_DIR, f"{stem}.pdf")
    fig.tight_layout(pad=0.6)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.02, dpi=600)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)
    return png, pdf


def wrap180(deg):
    """角度归一化到 [-180, 180]"""
    return (deg + 180.0) % 360.0 - 180.0


# =========================
# 配置参数
# =========================
V_MAX = 1.5
TAU_ABS = 0.2
TAU_REL = 0.30
MAX_XCORR_SPAN_S = 2 * 3600
XCORR_STEP_S = 5.0
XCORR_MAX_LAG_S = 600.0
USE_DIFF_DETREND = True
BOUNDARY_EPS = 1e-9
V_THR = 0.05
W_THR = 0.003
A_LIM_Q = 0.02
V_STAR = 0.113
BAND = 0.03
GENERATE_TABLES = True

# =========================
# 加载数据
# =========================
cols = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
usecols = [c for c in ["base_date_time", "latitude", "longitude", "sog", "cog"] if c in cols]
df = pd.read_csv(CSV_PATH, usecols=usecols)

df["base_date_time"] = pd.to_datetime(df["base_date_time"], errors="coerce", utc=True)
df = df.dropna(subset=["base_date_time", "latitude", "longitude"]).sort_values("base_date_time").reset_index(drop=True)

t = df["base_date_time"].astype("int64").to_numpy() / 1e9
dt = np.diff(t, prepend=np.nan).astype(np.float64)

lat = df["latitude"].to_numpy(np.float64)
lon = df["longitude"].to_numpy(np.float64)

# =========================
# 位置差分链运动学计算
# =========================
R = 6371000.0
latr = np.deg2rad(lat)
lonr = np.deg2rad(lon)

lat1 = np.roll(latr, 1)
lon1 = np.roll(lonr, 1)
dlat = latr - lat1
dlon = lonr - lon1
dlat[0] = np.nan
dlon[0] = np.nan

latm = 0.5 * (latr + lat1)
dx = (R * dlon * np.cos(latm)).astype(np.float64)
dy = (R * dlat).astype(np.float64)
disp = np.sqrt(dx * dx + dy * dy).astype(np.float64)
disp[0] = np.nan

ok = np.isfinite(dt) & (dt > 0) & np.isfinite(disp) & (disp >= 1.0)

v_pos = np.full(len(df), np.nan, np.float64)
v_pos[ok] = disp[ok] / dt[ok]

psi_pos = ((np.rad2deg(np.arctan2(dx, dy)) + 360.0) % 360.0).astype(np.float64)
psi_pos[0] = np.nan
psi_u = np.rad2deg(np.unwrap(np.deg2rad(np.nan_to_num(psi_pos, nan=0.0)))).astype(np.float64)
psi_u[~np.isfinite(psi_pos)] = np.nan

omega_pos = np.full(len(df), np.nan, np.float64)
omega_pos[1:] = np.deg2rad(psi_u[1:] - psi_u[:-1]) / dt[1:]
omega_pos[~np.isfinite(omega_pos)] = np.nan

a_pos = np.full(len(df), np.nan, np.float64)
a_pos[1:] = (v_pos[1:] - v_pos[:-1]) / dt[1:]
a_pos[~np.isfinite(a_pos)] = np.nan

# =========================
# AIS链运动学计算
# =========================
sog_ms = df["sog"].to_numpy(np.float64) * 0.514444 if "sog" in df.columns else None

a_sog = None
if sog_ms is not None:
    a_sog = np.full(len(df), np.nan, np.float64)
    a_sog[1:] = (sog_ms[1:] - sog_ms[:-1]) / dt[1:]
    a_sog[~np.isfinite(a_sog)] = np.nan

omega_cog = None
if "cog" in df.columns:
    cog = df["cog"].to_numpy(np.float64)
    dcog = np.full(len(df), np.nan, np.float64)
    dcog[1:] = wrap180(cog[1:] - cog[:-1])
    omega_cog = np.full(len(df), np.nan, np.float64)
    omega_cog[ok] = np.deg2rad(dcog[ok]) / dt[ok]
    omega_cog[~np.isfinite(omega_cog)] = np.nan

# ============================================================
# Fig.S1 速度一致性分析
# ============================================================
mae = rmse = mismatch_abs = mismatch_rel = np.nan

if sog_ms is not None:
    m0 = ok & np.isfinite(v_pos) & np.isfinite(sog_ms) & (v_pos >= 0) & (sog_ms >= 0)
    m = m0 & (v_pos <= V_MAX) & (sog_ms <= V_MAX)

    dv = v_pos[m] - sog_ms[m]
    mae = float(np.mean(np.abs(dv)))
    rmse = float(np.sqrt(np.mean(dv * dv)))
    mismatch_abs = float(np.mean(np.abs(dv) > TAU_ABS))
    mismatch_rel = float(np.mean(np.abs(dv) > (TAU_REL * np.maximum(sog_ms[m], 1e-3))))

    idx = np.where(m)[0]
    if idx.size > 3500:
        idx = np.random.default_rng(0).choice(idx, size=3500, replace=False)

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.scatter(sog_ms[idx], v_pos[idx], s=5, alpha=0.45, linewidths=0, color="#1f77b4")
    ax.plot([0, V_MAX], [0, V_MAX], linestyle="--", linewidth=1.0, color="#d62728")
    ax.set_xlim(0, V_MAX)
    ax.set_ylim(0, V_MAX)
    ax.set_xticks(np.linspace(0, V_MAX, 4))
    ax.set_yticks(np.linspace(0, V_MAX, 4))

    add_packed_xlabel(
        ax,
        [
            ("AIS", FP_TIMES, 7.5),
            ("对地航速/", FP_SIMSUN, 7.5),
            ("m·s⁻¹", FP_TIMES, 7.5),
        ],
        y=-0.18
    )
    add_mixed_ylabel_plain(ax, "位置差分对地航速/", "m·s⁻¹", x=-0.11, fontsize=7.5)
    ax.grid(True, linestyle=":", alpha=0.5, linewidth=0.5)

    add_s1_stats_box(ax, mae, rmse, mismatch_abs, mismatch_rel)

    set_ticklabel_font(ax, x_font=FP_TIMES, y_font=FP_TIMES)
    beautify_axes(ax, labelpad=5, tickpad=2)
    save_fig(fig, "FigS1_speed_consistency")

# ============================================================
# Fig.S2 互相关时滞分析
# ============================================================
lag_pk = np.nan
peak_corr = np.nan
corr0 = np.nan
delta_corr = np.nan
boundary_hit = False

if sog_ms is not None:
    m = ok & np.isfinite(v_pos) & np.isfinite(sog_ms)
    ts_all = t[m].astype(np.float64)
    s_all = sog_ms[m].astype(np.float64)
    v_all = v_pos[m].astype(np.float64)

    if ts_all.size >= 400:
        t_mid = 0.5 * (ts_all.min() + ts_all.max())
        sel = (ts_all >= t_mid - MAX_XCORR_SPAN_S / 2) & (ts_all <= t_mid + MAX_XCORR_SPAN_S / 2)

        if np.sum(sel) < 250:
            mid_idx = ts_all.size // 2
            lo = max(0, mid_idx - 2500)
            hi = min(ts_all.size, mid_idx + 2500)
            ts = ts_all[lo:hi]
            s = s_all[lo:hi]
            v = v_all[lo:hi]
        else:
            ts = ts_all[sel]
            s = s_all[sel]
            v = v_all[sel]

        grid = np.arange(ts.min(), ts.max(), XCORR_STEP_S)
        if grid.size >= 250:
            s_i = np.interp(grid, ts, s).astype(np.float64)
            v_i = np.interp(grid, ts, v).astype(np.float64)

            if USE_DIFF_DETREND:
                s_i = np.diff(s_i)
                v_i = np.diff(v_i)

            s_i = s_i - np.mean(s_i)
            v_i = v_i - np.mean(v_i)

            max_k = int(XCORR_MAX_LAG_S / XCORR_STEP_S)
            corr_full = np.correlate(s_i, v_i, mode="full")
            mid = corr_full.size // 2
            seg = corr_full[mid - max_k:mid + max_k + 1]
            denom = (np.std(s_i) * np.std(v_i) * len(v_i) + 1e-12)
            corr = seg / denom
            lags = np.arange(-max_k, max_k + 1) * XCORR_STEP_S

            kpk = int(np.argmax(corr))
            lag_pk = float(lags[kpk])
            peak_corr = float(corr[kpk])
            k0 = int(np.where(lags == 0.0)[0][0])
            corr0 = float(corr[k0])
            delta_corr = float(peak_corr - corr0)

            if abs(lag_pk - lags[0]) < BOUNDARY_EPS or abs(lag_pk - lags[-1]) < BOUNDARY_EPS:
                boundary_hit = True

            fig, ax = plt.subplots(figsize=(3, 2))
            ax.plot(lags, corr, linewidth=1.2, color="#1f77b4")
            ax.axvline(0.0, linestyle=":", linewidth=1.0, color="#2ca02c")
            ax.axvline(lag_pk, linestyle="--", linewidth=1.0, color="#d62728")
            ax.set_xticks(np.linspace(lags.min(), lags.max(), 5))

            add_packed_xlabel(
                ax,
                [
                    ("时滞/", FP_SIMSUN, 7.5),
                    ("s", FP_TIMES, 7.5),
                ],
                y=-0.18
            )
            ax.set_ylabel("相关系数", fontproperties=FP_SIMSUN,fontsize=7.5)
            ax.grid(True, linestyle=":", alpha=0.5, linewidth=0.5)

            add_s2_top_box(ax, peak_corr, corr0, delta_corr)

            set_ticklabel_font(ax, x_font=FP_TIMES, y_font=FP_TIMES)
            beautify_axes(ax, labelpad=5, tickpad=2)
            save_fig(fig, "FigS2_lag_xcorr_refined")

# ============================================================
# Fig.S3 回转直径分位数
# ============================================================
# ============================================================
# Fig.S3 回转半径分位数
# ============================================================
def q(arr):
    if arr.size < 50:
        return (np.nan, np.nan, np.nan)
    return tuple(np.quantile(arr, [0.05, 0.50, 0.90]).tolist())


turn_gps = ok & np.isfinite(v_pos) & np.isfinite(omega_pos) & (v_pos > V_THR) & (np.abs(omega_pos) > W_THR)
R_pos = v_pos[turn_gps] / np.abs(omega_pos[turn_gps])
R_pos = R_pos[np.isfinite(R_pos)]
R_pos = R_pos[(R_pos > 0) & (R_pos < 5000)]
Rpos05, Rpos50, Rpos90 = q(R_pos)

R_ais = np.array([], dtype=np.float64)
turn_ais_n = np.nan
if (sog_ms is not None) and (omega_cog is not None):
    turn_ais = ok & np.isfinite(sog_ms) & np.isfinite(omega_cog) & (sog_ms > V_THR) & (np.abs(omega_cog) > W_THR)
    turn_ais_n = int(np.sum(turn_ais))
    R_ais = sog_ms[turn_ais] / np.abs(omega_cog[turn_ais])
    R_ais = R_ais[np.isfinite(R_ais)]
    R_ais = R_ais[(R_ais > 0) & (R_ais < 5000)]
Rais05, Rais50, Rais90 = q(R_ais) if R_ais.size else (np.nan, np.nan, np.nan)

fig, ax = plt.subplots(figsize=(3, 2))

def draw_quantile(ax, x, p05, p50, p90, color):
    ax.plot([x, x], [p05, p90], linewidth=5, solid_capstyle="butt", zorder=2, color=color)
    ax.scatter([x], [p50], s=50, marker="D", zorder=3, color=color)

draw_quantile(ax, 0, Rais05, Rais50, Rais90, "#1f77b4")
draw_quantile(ax, 1, Rpos05, Rpos50, Rpos90, "#ff7f0e")

ax.set_yscale("log")
ax.set_xlim(-0.25, 1.25)
ax.set_xticks([0, 1])
add_fig3_xticklabels(ax, y=-0.08)

ax.set_yticks([10, 100, 1000])
ax.set_yticklabels(["10", "100", "1000"], fontproperties=FP_TIMES, fontsize=6.5)
add_mixed_ylabel_plain(ax, "回转半径/", "m", x=-0.11, fontsize=7.5)

ax.grid(True, linestyle=":", alpha=0.5, linewidth=0.5, which="both", axis="y")

bbox_kw = dict(boxstyle="round,pad=0.10", fc="white", ec="0.7", alpha=0.90)

# 左侧 AIS 链
ax.text(-0.22, Rais90, f"P90={Rais90:.1f} m", ha="left", va="center",
        bbox=bbox_kw, fontsize=5.0, fontproperties=FP_TIMES)
ax.text(-0.22, Rais50, f"P50={Rais50:.1f} m", ha="left", va="center",
        bbox=bbox_kw, fontsize=5.0, fontproperties=FP_TIMES)
ax.text(-0.22, Rais05, f"P05={Rais05:.1f} m", ha="left", va="center",
        bbox=bbox_kw, fontsize=5.0, fontproperties=FP_TIMES)

# 右侧位置差分链
ax.text(0.92, Rpos90, f"P90={Rpos90:.1f} m", ha="left", va="center",
        bbox=bbox_kw, fontsize=5.0, fontproperties=FP_TIMES)
ax.text(0.92, Rpos50, f"P50={Rpos50:.1f} m", ha="left", va="center",
        bbox=bbox_kw, fontsize=5.0, fontproperties=FP_TIMES)
ax.text(0.92, Rpos05, f"P05={Rpos05:.1f} m", ha="left", va="center",
        bbox=bbox_kw, fontsize=5.0, fontproperties=FP_TIMES)

set_ticklabel_font(ax, x_font=FP_TIMES, y_font=FP_TIMES)
beautify_axes(ax, labelpad=5, tickpad=2)
save_fig(fig, "FigS3_turning_radius_quantiles_log")
# ============================================================
# Table T1 & T2
# ============================================================
if GENERATE_TABLES:
    try:
        T1 = pd.DataFrame({
            "指标": [
                f"MAE(v_pos, SOG)/m/s（截断≤{V_MAX} m/s）",
                f"RMSE(v_pos, SOG)/m/s（截断≤{V_MAX} m/s）",
                f"失配 |v_pos - SOG| > {TAU_ABS:.1f}（截断）",
                f"失配 |v_pos - SOG| > {TAU_REL:.1f}×SOG（截断）",
                f"峰值时滞（Fig.S2, ±{int(XCORR_MAX_LAG_S)} s, 去趋势={'差分' if USE_DIFF_DETREND else '无'}）",
                "峰值相关系数",
                "零时滞相关系数",
                "相关提升量",
                "边界命中（峰值是否落在±max_lag边界）",
                "回转样本数（位置差分链）",
                "回转样本数（AIS链）",
                "D50（位置差分链）/m",
                "D90（位置差分链）/m",
                "D50（AIS链）/m",
                "D90（AIS链）/m",
            ],
            "数值": [
                mae, rmse, mismatch_abs, mismatch_rel,
                lag_pk, peak_corr, corr0, delta_corr, int(boundary_hit),
                int(np.sum(turn_gps)), turn_ais_n,
                Dpos50, Dpos90,
                Dais50, Dais90,
            ]
        })
        T1.to_csv(os.path.join(OUT_DIR, "TableT1_consistency_summary.csv"), index=False, encoding="utf-8-sig")

        def robust_decel_limit(a_series):
            a = a_series[ok & np.isfinite(a_series)].astype(np.float64)
            neg = a[a < 0]
            if neg.size < 50:
                return np.nan
            return float(np.quantile(neg, A_LIM_Q))

        def braking_row(v_series, a_series, name):
            a_lim = robust_decel_limit(a_series)
            m = ok & np.isfinite(v_series) & (v_series >= 0)
            v = v_series[m].astype(np.float64)
            sel = (v >= max(0, V_STAR - BAND)) & (v <= V_STAR + BAND)
            if np.sum(sel) < 30:
                k = min(max(200, 30), v.size)
                idx = np.argsort(np.abs(v - V_STAR))[:k]
                v_band = v[idx]
            else:
                v_band = v[sel]

            if not np.isfinite(a_lim) or a_lim >= 0:
                d = np.full_like(v_band, np.nan, dtype=np.float64)
            else:
                d = (v_band ** 2) / (2.0 * abs(a_lim))

            return {
                "方法": name,
                f"a_lim（负加速度Q{A_LIM_Q:.2f}分位数）/m/s²": a_lim,
                "速度带/m/s": f"{V_STAR:.3f}±{BAND:.3f}",
                "样本数 n": int(v_band.size),
                "制动距离P50/m": float(np.nanquantile(d, 0.50)),
                "制动距离P90/m": float(np.nanquantile(d, 0.90)),
            }

        rows = [braking_row(v_pos, a_pos, "位置差分链（v_pos, a_pos）")]
        if (sog_ms is not None) and (a_sog is not None):
            rows.append(braking_row(sog_ms, a_sog, "AIS链（SOG, a_SOG）"))

        T2 = pd.DataFrame(rows)
        T2.to_csv(os.path.join(OUT_DIR, "TableT2_braking_compare.csv"), index=False, encoding="utf-8-sig")

        print("表格已生成。")
    except Exception as e:
        print(f"表格生成失败：{e}")

print("处理完成。输出文件夹:", OUT_DIR)
print("=" * 60)
print("字体文件使用确认：")
print("  simsun.ttc   -> 中文标签（宋体）")
print("  simsunb.ttf  -> 中文粗体（备用）")
print("  times.ttf    -> 西文/数字（Times New Roman）")
print("  timesbd.ttf  -> 西文粗体（备用）")
print("  timesi.ttf   -> 西文斜体（备用）")
print("=" * 60)