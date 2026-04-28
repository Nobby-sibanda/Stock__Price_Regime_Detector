import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .config import REGIME_NAMES, REGIME_COLORS


def plot_dashboard(
    df,
    ticker: str,
    method: str,
    outdir: str = "outputs",
) -> None:
    fig = plt.figure(figsize=(22, 16), facecolor="#0d1117")
    fig.suptitle(
        f"  {ticker}  |  Regime Detector  |  Method: {method.upper()}",
        fontsize=17, color="white", fontweight="bold", x=0.02, ha="left", y=0.99,
    )
    gs    = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.32)
    regs  = df["regime"].values
    close = df["Close"]
    dates = df.index
    n_reg = len(REGIME_NAMES)

    def _style(ax, title):
        ax.set_facecolor("#161b22")
        ax.set_title(title, color="white", fontsize=10, pad=5)
        ax.tick_params(colors="#8b949e", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#30363d")

    def _legend(ax):
        leg = ax.legend(loc="upper left", framealpha=0.25, fontsize=8, labelcolor="white")
        leg.get_frame().set_facecolor("#161b22")

    # 1 — price + regime shading
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(dates, close.values, color="#58a6ff", lw=1.1, zorder=3)
    for rid in range(n_reg):
        ax1.fill_between(dates, close.min(), close.max(),
                         where=(regs == rid), alpha=0.17,
                         color=REGIME_COLORS[rid], label=REGIME_NAMES[rid])
    _style(ax1, "Price  +  Regime Shading"); _legend(ax1)

    # 2 — regime timeline
    ax2 = fig.add_subplot(gs[1, :2])
    for rid in range(n_reg):
        ax2.fill_between(dates, rid - 0.45, rid + 0.45,
                         where=(regs == rid), color=REGIME_COLORS[rid], alpha=0.85)
    ax2.set_yticks(range(n_reg))
    ax2.set_yticklabels([REGIME_NAMES[i] for i in range(n_reg)],
                         color="white", fontsize=8)
    _style(ax2, "Regime Labels Over Time")

    # 3 — cumulative returns
    ax3 = fig.add_subplot(gs[2, :2])
    cum_s = (1 + df["strat_ret"].fillna(0)).cumprod()
    cum_b = (1 + df["returns"].fillna(0)).cumprod()
    ax3.plot(dates, cum_s.values, color="#00d4aa", lw=1.5, label="Regime Strategy")
    ax3.plot(dates, cum_b.values, color="#f5a623", lw=1.5, label="Buy & Hold", alpha=0.75)
    _style(ax3, "Cumulative Returns: Strategy vs Buy & Hold"); _legend(ax3)

    # 4 — position signal
    ax4 = fig.add_subplot(gs[3, :2])
    ax4.plot(dates, df["signal"].values, color="#c9a0dc", lw=0.9)
    ax4.axhline(0, color="white", lw=0.5, ls=":", alpha=0.4)
    ax4.fill_between(dates, 0, df["signal"].values,
                     where=(df["signal"].values > 0), alpha=0.15, color="#00d4aa")
    ax4.fill_between(dates, 0, df["signal"].values,
                     where=(df["signal"].values < 0), alpha=0.15, color="#e74c3c")
    _style(ax4, "Sized Position Signal  (incl. vol-targeting)")

    # 5 — days per regime
    ax5 = fig.add_subplot(gs[0, 2])
    counts = [np.sum(regs == i) for i in range(n_reg)]
    bars   = ax5.bar(
        [REGIME_NAMES[i] for i in range(n_reg)],
        counts,
        color=[REGIME_COLORS[i] for i in range(n_reg)],
        width=0.5,
    )
    for bar, cnt in zip(bars, counts):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{cnt}d", ha="center", color="white", fontsize=8)
    _style(ax5, "Days per Regime")

    # 6 — RSI
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(dates, df["rsi"].values, color="#c9a0dc", lw=0.9)
    ax6.axhline(70, color="#e74c3c", lw=0.8, ls="--", alpha=0.75)
    ax6.axhline(30, color="#00d4aa", lw=0.8, ls="--", alpha=0.75)
    _style(ax6, "RSI (14)")

    # 7 — return distributions by regime
    ax7 = fig.add_subplot(gs[2, 2])
    for rid in range(n_reg):
        sub = df[df["regime"] == rid]["strat_ret"].dropna()
        if len(sub) > 10:
            sub.plot.kde(ax=ax7, color=REGIME_COLORS[rid],
                         label=REGIME_NAMES[rid], lw=1.5)
    ax7.axvline(0, color="white", lw=0.7, ls="--", alpha=0.5)
    _style(ax7, "Return Distribution by Regime"); _legend(ax7)

    # 8 — Z-score
    ax8 = fig.add_subplot(gs[3, 2])
    ax8.plot(dates, df["zscore"].values, color="#f5a623", lw=0.9, alpha=0.9)
    ax8.axhline( 1.5, color="#e74c3c", lw=0.8, ls="--", alpha=0.75)
    ax8.axhline(-1.5, color="#00d4aa", lw=0.8, ls="--", alpha=0.75)
    ax8.axhline(0,    color="white",   lw=0.5, ls=":",  alpha=0.35)
    _style(ax8, "Z-Score  (+/-1.5 sigma signal bands)")

    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{ticker}_{method}_dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"    Chart saved: {os.path.basename(out)}")
