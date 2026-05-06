import os
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .config import (
    REGIME_NAMES, REGIME_COLORS,
    PLOT_BG, PLOT_PANEL, PLOT_ALPHA_FILL, PLOT_FIG_W, PLOT_FIG_H,
)

log = logging.getLogger(__name__)

_GREY = "#8b949e"


# ── Panel helpers ─────────────────────────────────────────────────────────────

def _style(ax, title: str) -> None:
    ax.set_facecolor(PLOT_PANEL)
    ax.set_title(title, color="white", fontsize=10, pad=5)
    ax.tick_params(colors=_GREY, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#30363d")


def _legend(ax) -> None:
    leg = ax.legend(loc="upper left", framealpha=0.25, fontsize=8, labelcolor="white")
    leg.get_frame().set_facecolor(PLOT_PANEL)


# ── Individual panels ─────────────────────────────────────────────────────────

def _plot_price_regime(ax, dates, close, regs) -> None:
    n_reg = len(REGIME_NAMES)
    ax.plot(dates, close.values, color="#58a6ff", lw=1.1, zorder=3)
    for rid in range(n_reg):
        ax.fill_between(dates, close.min(), close.max(),
                        where=(regs == rid), alpha=PLOT_ALPHA_FILL,
                        color=REGIME_COLORS[rid], label=REGIME_NAMES[rid])
    _style(ax, "Price  +  Regime Shading")
    _legend(ax)


def _plot_regime_timeline(ax, dates, regs) -> None:
    n_reg = len(REGIME_NAMES)
    for rid in range(n_reg):
        ax.fill_between(dates, rid - 0.45, rid + 0.45,
                        where=(regs == rid), color=REGIME_COLORS[rid], alpha=0.85)
    ax.set_yticks(range(n_reg))
    ax.set_yticklabels([REGIME_NAMES[i] for i in range(n_reg)],
                       color="white", fontsize=8)
    _style(ax, "Regime Labels Over Time")


def _plot_cumulative_returns(ax, dates, df) -> None:
    cum_s = (1 + df["strat_ret"].fillna(0)).cumprod()
    cum_b = (1 + df["returns"].fillna(0)).cumprod()
    ax.plot(dates, cum_s.values, color="#00d4aa", lw=1.5, label="Regime Strategy")
    ax.plot(dates, cum_b.values, color="#f5a623", lw=1.5, label="Buy & Hold", alpha=0.75)
    _style(ax, "Cumulative Returns: Strategy vs Buy & Hold")
    _legend(ax)


def _plot_signal(ax, dates, df) -> None:
    sig = df["signal"].values
    ax.plot(dates, sig, color="#c9a0dc", lw=0.9)
    ax.axhline(0, color="white", lw=0.5, ls=":", alpha=0.4)
    ax.fill_between(dates, 0, sig, where=(sig > 0), alpha=0.15, color="#00d4aa")
    ax.fill_between(dates, 0, sig, where=(sig < 0), alpha=0.15, color="#e74c3c")
    _style(ax, "Sized Position Signal  (incl. vol-targeting)")


def _plot_days_per_regime(ax, regs) -> None:
    n_reg  = len(REGIME_NAMES)
    counts = [np.sum(regs == i) for i in range(n_reg)]
    bars   = ax.bar(
        [REGIME_NAMES[i] for i in range(n_reg)],
        counts,
        color=[REGIME_COLORS[i] for i in range(n_reg)],
        width=0.5,
    )
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{cnt}d", ha="center", color="white", fontsize=8)
    _style(ax, "Days per Regime")


def _plot_rsi(ax, dates, df) -> None:
    ax.plot(dates, df["rsi"].values, color="#c9a0dc", lw=0.9)
    ax.axhline(70, color="#e74c3c", lw=0.8, ls="--", alpha=0.75)
    ax.axhline(30, color="#00d4aa", lw=0.8, ls="--", alpha=0.75)
    _style(ax, "RSI (14)")


def _plot_return_distributions(ax, df) -> None:
    n_reg = len(REGIME_NAMES)
    for rid in range(n_reg):
        sub = df[df["regime"] == rid]["strat_ret"].dropna()
        if len(sub) > 10:
            sub.plot.kde(ax=ax, color=REGIME_COLORS[rid],
                         label=REGIME_NAMES[rid], lw=1.5)
    ax.axvline(0, color="white", lw=0.7, ls="--", alpha=0.5)
    _style(ax, "Return Distribution by Regime")
    _legend(ax)


def _plot_zscore(ax, dates, df) -> None:
    ax.plot(dates, df["zscore"].values, color="#f5a623", lw=0.9, alpha=0.9)
    ax.axhline( 1.5, color="#e74c3c", lw=0.8, ls="--", alpha=0.75)
    ax.axhline(-1.5, color="#00d4aa", lw=0.8, ls="--", alpha=0.75)
    ax.axhline(0,    color="white",   lw=0.5, ls=":",  alpha=0.35)
    _style(ax, "Z-Score  (+/-1.5 sigma signal bands)")


# ── Dashboard assembly ────────────────────────────────────────────────────────

def plot_dashboard(
    df,
    ticker: str,
    method: str,
    outdir: str = "outputs",
) -> None:
    regs  = df["regime"].values
    close = df["Close"]
    dates = df.index

    fig = plt.figure(figsize=(PLOT_FIG_W, PLOT_FIG_H), facecolor=PLOT_BG)
    fig.suptitle(
        f"  {ticker}  |  Regime Detector  |  Method: {method.upper()}",
        fontsize=17, color="white", fontweight="bold", x=0.02, ha="left", y=0.99,
    )
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.32)

    _plot_price_regime(fig.add_subplot(gs[0, :2]), dates, close, regs)
    _plot_regime_timeline(fig.add_subplot(gs[1, :2]), dates, regs)
    _plot_cumulative_returns(fig.add_subplot(gs[2, :2]), dates, df)
    _plot_signal(fig.add_subplot(gs[3, :2]), dates, df)
    _plot_days_per_regime(fig.add_subplot(gs[0, 2]), regs)
    _plot_rsi(fig.add_subplot(gs[1, 2]), dates, df)
    _plot_return_distributions(fig.add_subplot(gs[2, 2]), df)
    _plot_zscore(fig.add_subplot(gs[3, 2]), dates, df)

    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{ticker}_{method}_dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=PLOT_BG)
    plt.close()
    log.info("Chart saved: %s", os.path.basename(out))
