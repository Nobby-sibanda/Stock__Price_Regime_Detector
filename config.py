# ── Regime definitions ────────────────────────────────────────────────────────
REGIME_NAMES  = {0: "TRENDING", 1: "MEAN-REV", 2: "VOLATILE"}
REGIME_COLORS = {0: "#00d4aa",  1: "#f5a623",  2: "#e74c3c"}
REGIME_ICONS  = {0: "↑",        1: "↔",         2: "⚡"}

STRATEGY_DESC = {
    0: ("MOMENTUM - Follow trend direction via slope sign.\n"
        "        Use EMA 20/50 crossovers for confirmation.\n"
        "        Trail stop-loss at 2×ATR below entry.\n"
        "        Take profit at 3-5×ATR extension."),
    1: ("MEAN REVERSION - Fade Z-score extremes (+/-1.5 sigma).\n"
        "        Short overbought (Z>+1.5), long oversold (Z<-1.5).\n"
        "        Target: revert to rolling 20-day mean.\n"
        "        Use Bollinger Band squeeze as entry filter."),
    2: ("DEFENSIVE - Reduce position size to 25% of normal.\n"
        "        Stay flat unless RSI hits extreme (<25 or >75).\n"
        "        Prefer cash or short-vol options hedges.\n"
        "        Wait for ATR compression before re-entering."),
}

# ── Plot constants ────────────────────────────────────────────────────────────
PLOT_BG         = "#0d1117"
PLOT_PANEL      = "#161b22"
PLOT_ALPHA_FILL = 0.17
PLOT_FIG_W      = 22
PLOT_FIG_H      = 16

# ── Strategy thresholds ───────────────────────────────────────────────────────
MR_ZSCORE_THRESHOLD = 1.5
VOL_RSI_LOW         = 25
VOL_RSI_HIGH        = 75
VOL_SIGNAL_SIZE     = 0.25
MAX_POSITION_SIZE   = 3.0
