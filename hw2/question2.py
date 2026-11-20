"""
Question 2
Please install the numpy and matplotlib modules to run this code

"""

from math import erfc, exp, log, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from question1 import binomial

S0 = 30.0
R = 0.05
T = 1 / 12
SIGMA = 0.2
STRIKES = (25, 30, 35)
MAX_N = 1000


def normal_cdf(x):
    return 0.5 * erfc(-x / sqrt(2))


def black_scholes_put(S0, r, T, sigma, strike):
    """Return a put price and initial (cash, shares), with no dividends."""
    d1 = (log(S0 / strike) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    b0 = -normal_cdf(-d1)  # Put delta.
    a0 = strike * exp(-r * T) * normal_cdf(-d2)
    price = a0 + b0 * S0
    return price, (a0, b0)


def compute_results():
    periods = np.arange(1, MAX_N + 1)
    results = {strike: np.empty((MAX_N, 3)) for strike in STRIKES}
    benchmarks = {}

    for strike in STRIKES:
        price, (a0, b0) = black_scholes_put(S0, R, T, SIGMA, strike)
        benchmarks[strike] = np.array([price, a0, b0])

        def put_payoff(S, strike=strike):
            return max(strike - S, 0.0)

        for N in range(1, MAX_N + 1):
            delta = T / N
            U = exp(SIGMA * sqrt(delta))
            price, (a0, b0) = binomial(
                S0=S0, r=R, N=N, period_length=delta,
                U=U, D=1 / U, h=put_payoff, n=N,
            )
            results[strike][N - 1] = price, a0, b0

    return periods, results, benchmarks

"""Let's have a nice plot"""
def set_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "figure.titlesize": 19,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "0.35",
        "axes.linewidth": 0.7,
        "axes.grid": True,
        "grid.color": "0.9",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "lines.linewidth": 1.4,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


def plot_results(periods, results, benchmarks, output_dir):
    set_plot_style()
    colors = plt.get_cmap("magma")(np.linspace(0.2, 0.7, len(STRIKES)))
    labels = (r"Put price $P_0^N(k)$", r"Cash holding $a_0^N(k)$",
              r"Share holding $b_0^N(k)$")

    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True,
                             layout="constrained")
    fig.suptitle("One-month puts: binomial convergence")
    for col, (strike, color) in enumerate(zip(STRIKES, colors)):
        for row in range(3):
            ax = axes[row, col]
            ax.plot(periods, results[strike][:, row], color=color, label="Binomial")
            ax.axhline(benchmarks[strike][row], color="0.25", linestyle="--",
                       linewidth=1.1, label="Black-Scholes")
            ax.set_ylabel(labels[row])
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4),
                                useOffset=False)
            ax.set_xlim(1, periods[-1])
            if row == 0:
                ax.set_title(rf"Strike $k={strike}$")
                ax.legend(loc="best")
            if row == 2:
                ax.set_xlabel(r"Number of periods $N$")
    fig.savefig(output_dir / "question2.pdf")
    
def main():
    output_dir = Path(__file__).resolve().parent 
    output_dir.mkdir(exist_ok=True)
    periods, results, benchmarks = compute_results()
    plot_results(periods, results, benchmarks, output_dir)
    plt.show()


if __name__ == "__main__":
    main()
