import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.font_manager import FontProperties
import seaborn as sns
import numpy as np
from scipy import stats
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, DrawingArea
from matplotlib.lines import Line2D

# =========================
# 加载自定义字体文件
# =========================
font_paths = [
    "simsun.ttc", "simsunb.ttf", "SimsunExtG.ttf",
    "times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"
]
for fp in font_paths:
    font_manager.fontManager.addfont(fp)

# 字体对象
zh_font = FontProperties(fname="simsun.ttc")   # 中文：宋体
en_font = FontProperties(fname="times.ttf")    # 西文：Times New Roman

# =========================
# 全局设置
# =========================
plt.rcParams["font.family"] = en_font.get_name()   # 默认西文/数字 = Times New Roman
plt.rcParams["axes.unicode_minus"] = False

# 数学文本尽量走 Times New Roman
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = en_font.get_name()
plt.rcParams["mathtext.it"] = f"{en_font.get_name()}:italic"
plt.rcParams["mathtext.bf"] = f"{en_font.get_name()}:bold"
plt.rcParams["mathtext.default"] = "rm"

# =========================
# 通用函数
# =========================
def apply_pub_style(ax, xlabel=None, ylabel=None):
    """中文标签用宋体，刻度数字用 Times New Roman"""
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=7.5, fontproperties=zh_font)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=7.5, fontproperties=zh_font)

    # 刻度数字字体
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(en_font)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(en_font)

    # 科学计数法偏移量字体
    ax.xaxis.get_offset_text().set_fontproperties(en_font)
    ax.yaxis.get_offset_text().set_fontproperties(en_font)

    # 图例字体
    leg = ax.get_legend()
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_fontproperties(zh_font)

def add_mean_box(ax, mean_value):
    """右上角自定义‘均值：6.24 kn’框：中文宋体 + 英文/数字 Times"""
    da = DrawingArea(26, 10, 0, 0)
    line = Line2D([0, 26], [5, 5], color='red', linestyle='--', linewidth=1.5)
    da.add_artist(line)

    txt_cn = TextArea("均值：", textprops=dict(fontproperties=zh_font, fontsize=7.5))
    txt_en = TextArea(f"{mean_value:.2f} kn", textprops=dict(fontproperties=en_font, fontsize=7.5))

    box = HPacker(children=[da, txt_cn, txt_en], align="center", pad=0, sep=4)

    anchored_box = AnchoredOffsetbox(
        loc='upper right',
        child=box,
        frameon=True,
        pad=0.2,
        borderpad=0.35
    )

    anchored_box.patch.set_boxstyle("round,pad=0.25")
    anchored_box.patch.set_facecolor("white")
    anchored_box.patch.set_edgecolor("0.7")
    anchored_box.patch.set_alpha(0.9)

    ax.add_artist(anchored_box)

def add_mixed_xlabel(ax, cn_text, en_text, y=-0.18, fontsize=7.5):
    """横坐标：中文宋体 + 英文/单位 Times"""
    ax.set_xlabel("")
    ax.text(
        0.5, y, cn_text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontproperties=zh_font, fontsize=fontsize
    )
    ax.text(
        0.5, y, en_text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontproperties=en_font, fontsize=fontsize
    )

def add_mixed_ylabel(ax, cn_text, en_text, x=-0.11, fontsize=7.5):
    """纵坐标：中文宋体 + 英文/单位 Times"""
    ax.set_ylabel("")
    # 中文部分放下半段
    ax.text(
        x, 0.5, cn_text,
        transform=ax.transAxes,
        ha="center", va="top",
        rotation=90,
        fontproperties=zh_font, fontsize=fontsize
    )
    # 单位部分放上半段
    ax.text(
        x, 0.5, en_text,
        transform=ax.transAxes,
        ha="center", va="bottom",
        rotation=90,
        fontproperties=en_font, fontsize=fontsize
    )

# =========================
# 加载数据
# =========================
file_path = "ais-USV_filled.csv"
data = pd.read_csv(file_path)

# 时间格式
data['base_date_time'] = pd.to_datetime(data['base_date_time'])

# 插值填补缺失值
data['longitude'] = data['longitude'].interpolate(method='linear')
data['latitude'] = data['latitude'].interpolate(method='linear')
data['sog'] = data['sog'].interpolate(method='linear')
data['cog'] = data['cog'].interpolate(method='linear')

# 标准差与置信区间
sog_std = data['sog'].std()
cog_std = data['cog'].std()

sog_ci = stats.norm.interval(0.95, loc=data['sog'].mean(), scale=sog_std / np.sqrt(len(data)))
cog_ci = stats.norm.interval(0.95, loc=data['cog'].mean(), scale=cog_std / np.sqrt(len(data)))

# 平滑
data['sog_smoothed'] = data['sog'].rolling(window=5).mean()
data['cog_smoothed'] = data['cog'].rolling(window=5).mean()

# =========================
# 1. 船舶历史轨迹
# =========================
fig, ax = plt.subplots(figsize=(3, 2), dpi=600)
ax.plot(data['longitude'], data['latitude'], label="船舶轨迹", color='blue', linewidth=0.5)
ax.scatter(data['longitude'].iloc[0], data['latitude'].iloc[0], color='green', label='起点', zorder=5, s=50, alpha=0.5)
ax.scatter(data['longitude'].iloc[-1], data['latitude'].iloc[-1], color='red', label='终点', zorder=5, s=50, alpha=0.5)

ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right', fontsize=7.5)
ax.tick_params(axis='both', direction='in', labelsize=7.5)

apply_pub_style(ax, xlabel="经度", ylabel="纬度")
fig.tight_layout()
plt.show()

# =========================
# 2. 绘制SOG图（带误差条）
# =========================
fig, ax = plt.subplots(figsize=(3, 2), dpi=600)
ax.plot(
    data['base_date_time'],
    data['sog_smoothed'],
    color='lightblue',
    linewidth=0.5,
    alpha=0.6
)
ax.fill_between(
    data['base_date_time'],
    data['sog_smoothed'] - sog_std,
    data['sog_smoothed'] + sog_std,
    color='lightgray',
    alpha=0.3
)

ax.grid(True, linestyle='--', alpha=0.5)
ax.tick_params(axis='both', direction='in', labelsize=7.5)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# 这里只改纵轴单位字体，其余保持不变
apply_pub_style(ax, xlabel="时间", ylabel=None)
add_mixed_ylabel(ax, "对地航速/", "kn", x=-0.11, fontsize=7.5)

plt.tight_layout()
plt.show()

# =========================
# 3. 绘制COG图（带误差条）
# =========================
fig, ax = plt.subplots(figsize=(3, 2), dpi=600)
ax.plot(
    data['base_date_time'],
    data['cog_smoothed'],
    color='lightcoral',
    linewidth=0.5,
    alpha=0.6
)
ax.fill_between(
    data['base_date_time'],
    data['cog_smoothed'] - cog_std,
    data['cog_smoothed'] + cog_std,
    color='lightgray',
    alpha=0.3
)

ax.grid(True, linestyle='--', alpha=0.5)
ax.tick_params(axis='both', direction='in', labelsize=7.5)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# 这里只改纵轴单位字体，其余保持不变
apply_pub_style(ax, xlabel="时间", ylabel=None)
add_mixed_ylabel(ax, "对地航向/", "°", x=-0.11, fontsize=7.5)

plt.tight_layout()
plt.show()

# =========================
# 4. SOG直方图 + KDE + 均值红线
# =========================
fig, ax = plt.subplots(figsize=(3, 2), dpi=600)
sns.histplot(
    data['sog'],
    kde=True,
    color='green',
    bins=30,
    stat='density',
    alpha=0.5,
    kde_kws={"bw_adjust": 1.5},
    ax=ax
)

mean_sog = data['sog'].mean()

# 画均值竖线
ax.axvline(mean_sog, color='red', linestyle='--', linewidth=1.5)

ax.grid(True, linestyle='--', alpha=0.5)
ax.tick_params(axis='both', direction='in', labelsize=7.5)

apply_pub_style(ax, xlabel=None, ylabel="密度")
add_mixed_xlabel(ax, "对地航速/", "kn")
add_mean_box(ax, mean_sog)

fig.subplots_adjust(left=0.16, bottom=0.24, right=0.98, top=0.98)
plt.show()