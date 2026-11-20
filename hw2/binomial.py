from math import exp


def binomial(S0, r, N, period_length, U, D, h, n):
    """Question 1: return the derivative price and initial portfolio (a0, b0).

    S0: initial stock price.
    r: annual continuously compounded risk-free interest rate.
    N: total number of periods in the model.
    period_length: length of each period in years (Delta).
    U, D: stock price multipliers for an up move and a down move.
    h: payoff function, e.g. max(S - strike, 0) for a call.
    n: number of periods until payment (1 <= n <= N).

    a0 is the amount invested in the risk-free asset at time 0 (B0 = 1).
    A negative a0 means borrowing. b0 is the number of shares held;
    a negative b0 means short selling. The portfolio costs a0 + b0 * S0.
    """
    # Check the basic conditions of the binomial model.
    if type(N) is not int or type(n) is not int or not 1 <= n <= N:
        raise ValueError("N and n must be integers with 1 <= n <= N.")
    if S0 <= 0 or period_length <= 0 or D <= 0:
        raise ValueError("S0, period_length and D must be positive.")

    # One unit of cash grows to exp(r * Delta) after one period.
    growth = exp(r * period_length)
    if not D < growth < U:
        raise ValueError("No arbitrage requires D < exp(r * period_length) < U.")
    q = (growth - D) / (U - D)  # Risk-neutral up probability used for pricing.

    # 1. Calculate the payoff at each possible stock price at maturity.
    # k up moves and n-k down moves give S0 * U**k * D**(n-k).
    # Use n, not N: the derivative may mature before the model ends.
    values = []
    for k in range(n + 1):
        stock_price = S0 * U**k * D**(n - k)
        values.append(h(stock_price))

    # 2. Work backwards: average the up/down values, then discount one period.
    # Stop at period 1 to keep both values needed to find the portfolio.
    # Store only one level of the tree at a time to keep memory use small.
    for step in range(n - 1, 0, -1):
        values = [
            (q * values[k + 1] + (1 - q) * values[k]) / growth
            for k in range(step + 1)
        ]

    value_down, value_up = values
    price = (q * value_up + (1 - q) * value_down) / growth

    # 3. Choose shares and cash to match the derivative in BOTH next states:
    # a0 * growth + b0 * S0 * U = value_up
    # a0 * growth + b0 * S0 * D = value_down
    b0 = (value_up - value_down) / (S0 * (U - D))
    a0 = (value_down - b0 * S0 * D) / growth
    return price, (a0, b0)


if __name__ == "__main__":
    # Run this file to reproduce Question 1's one-year call with strike 1.
    # Change the inputs and payoff below to price another derivative.
    def call_payoff(S):
        return max(S - 1, 0)

    U = exp(0.02)
    price, (a0, b0) = binomial(
        S0=1, r=0.03, N=100, period_length=0.01,
        U=U, D=1 / U, h=call_payoff, n=100,
    )
    # Expected: price = 0.09393596, a0 = -0.50453025, b0 = 0.59846620.
    print(f"Initial derivative price: {price:.8f}")
    print(f"Cash invested (a0):       {a0:.8f}")
    print(f"Shares held (b0):         {b0:.8f}")
