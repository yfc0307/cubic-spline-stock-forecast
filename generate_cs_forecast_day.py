from futu import *
import traceback
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_KLDay_close(code, days, quote_ctx):
    """
    Input: code -> str, days -> int
    Output: close_prices -> list[float]

    Get the close prices of code for the past days.

    """
    ret, data = quote_ctx.get_cur_kline(code, days, KLType.K_DAY, AuType.NONE)
    if ret == RET_OK:
        close_prices = data["close"]
        return close_prices
    else:
        return "error:", data


def natural_cs_forecast_day(array, fc_days, smooth_days=10):
    """
    Forecast future prices with a natural cubic spline.

    Input:
    array       -> array-like of close prices (typically 120 days)
    fc_days     -> int, number of days to forecast
    smooth_days -> int, knot spacing in days for spline smoothing

    Output:
    forecast_df -> pandas.DataFrame with columns:
                   day_index (future day index), forecast_close
    """
    y = np.asarray(array, dtype=float)

    if y.ndim != 1:
        raise ValueError("array must be a 1D sequence of close prices")
    if y.size < 4:
        raise ValueError("array must contain at least 4 close prices")
    if fc_days <= 0:
        raise ValueError("fc_days must be a positive integer")
    if smooth_days <= 0:
        raise ValueError("smooth_days must be a positive integer")

    n = y.size

    # Use one knot every `smooth_days` to control smoothing strength.
    knot_idx = np.arange(0, n, int(smooth_days), dtype=int)
    if knot_idx[-1] != n - 1:
        knot_idx = np.append(knot_idx, n - 1)

    xk = knot_idx.astype(float)
    yk = y[knot_idx]
    k = yk.size

    # Natural boundary condition: second derivative at both ends is 0.
    m = np.zeros(k, dtype=float)
    if k > 2:
        A = np.zeros((k - 2, k - 2), dtype=float)
        rhs = np.zeros(k - 2, dtype=float)

        for i in range(1, k - 1):
            row = i - 1
            h_prev = xk[i] - xk[i - 1]
            h_next = xk[i + 1] - xk[i]

            if row > 0:
                A[row, row - 1] = h_prev
            A[row, row] = 2.0 * (h_prev + h_next)
            if row < k - 3:
                A[row, row + 1] = h_next

            rhs[row] = 6.0 * (
                (yk[i + 1] - yk[i]) / h_next - (yk[i] - yk[i - 1]) / h_prev
            )

        m[1:-1] = np.linalg.solve(A, rhs)

    def eval_spline(xq):
        i = int(np.searchsorted(xk, xq, side="right") - 1)
        i = max(0, min(i, k - 2))

        xi = xk[i]
        xi1 = xk[i + 1]
        h = xi1 - xi

        term1 = m[i] * (xi1 - xq) ** 3 / (6.0 * h)
        term2 = m[i + 1] * (xq - xi) ** 3 / (6.0 * h)
        term3 = (yk[i] - m[i] * h * h / 6.0) * (xi1 - xq) / h
        term4 = (yk[i + 1] - m[i + 1] * h * h / 6.0) * (xq - xi) / h
        return term1 + term2 + term3 + term4

    future_x = np.arange(n, n + fc_days, dtype=float)
    forecast = np.array([eval_spline(xq) for xq in future_x], dtype=float)

    forecast_df = pd.DataFrame(
        {
            "day_index": future_x.astype(int),
            "forecast_close": np.round(forecast, 4),
        }
    )
    return forecast_df


def plot_close_with_forecast(
    close_prices, forecast_df, title, save_path=None, smooth_days=10
):
    """
    Line plot for historical close prices, fitted spline, and forecast prices.

    Input:
    close_prices -> array-like historical close prices
    forecast_df  -> pandas.DataFrame with columns: day_index, forecast_close
    title        -> str
    save_path    -> str | None, if provided save figure to this path
    smooth_days  -> int, knot spacing used for spline fitting

    Output:
    fig, ax
    """
    close_arr = np.asarray(close_prices, dtype=float)
    hist_x = np.arange(close_arr.size, dtype=int)
    n = close_arr.size
    y = close_arr

    # Reconstruct spline from historical data to display fitted curve
    knot_idx = np.arange(0, n, int(smooth_days), dtype=int)
    if knot_idx[-1] != n - 1:
        knot_idx = np.append(knot_idx, n - 1)

    xk = knot_idx.astype(float)
    yk = y[knot_idx]
    k = yk.size

    # Fit natural cubic spline
    m = np.zeros(k, dtype=float)
    if k > 2:
        A = np.zeros((k - 2, k - 2), dtype=float)
        rhs = np.zeros(k - 2, dtype=float)

        for i in range(1, k - 1):
            row = i - 1
            h_prev = xk[i] - xk[i - 1]
            h_next = xk[i + 1] - xk[i]

            if row > 0:
                A[row, row - 1] = h_prev
            A[row, row] = 2.0 * (h_prev + h_next)
            if row < k - 3:
                A[row, row + 1] = h_next

            rhs[row] = 6.0 * (
                (yk[i + 1] - yk[i]) / h_next - (yk[i] - yk[i - 1]) / h_prev
            )

        m[1:-1] = np.linalg.solve(A, rhs)

    def eval_spline(xq):
        i = int(np.searchsorted(xk, xq, side="right") - 1)
        i = max(0, min(i, k - 2))

        xi = xk[i]
        xi1 = xk[i + 1]
        h = xi1 - xi

        term1 = m[i] * (xi1 - xq) ** 3 / (6.0 * h)
        term2 = m[i + 1] * (xq - xi) ** 3 / (6.0 * h)
        term3 = (yk[i] - m[i] * h * h / 6.0) * (xi1 - xq) / h
        term4 = (yk[i + 1] - m[i + 1] * h * h / 6.0) * (xq - xi) / h
        return term1 + term2 + term3 + term4

    hist_spline = np.array([eval_spline(float(xi)) for xi in hist_x], dtype=float)

    fc_x = forecast_df["day_index"].to_numpy(dtype=int)
    fc_y = forecast_df["forecast_close"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hist_x, close_arr, label="Historical Close", color="steelblue", linewidth=2)
    ax.plot(
        hist_x,
        hist_spline,
        label="Fitted Spline (Historical)",
        color="green",
        linewidth=2,
        alpha=0.5,
    )
    ax.plot(fc_x, fc_y, label="Forecast Close", color="darkorange", linewidth=2)
    ax.axvline(x=close_arr.size - 1, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Day Index")
    ax.set_ylabel("Close Price")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    if save_path:
        fig.savefig(f"img/days/{save_path}", dpi=150, bbox_inches="tight")

    return fig, ax


def generate_cs_forecast_day(quote_ctx, codes_list):
    try:
        ret_sub, err_message = quote_ctx.subscribe(
            codes_list, [SubType.QUOTE, SubType.K_DAY]
        )
        if ret_sub == RET_OK:
            days_options = [720, 480, 360, 240]
            forecast_days_options = [30, 20, 15, 10]
            smooth_days_options = [60, 40, 30, 20]
            for i in range(len(days_options)):
                days = days_options[i]
                forecast_days = forecast_days_options[i]
                smooth_days = smooth_days_options[i]

                # Forecast
                def generate_forecast_plot(
                    codes_list,
                    days=days,
                    forecast_days=forecast_days,
                    smooth_days=smooth_days,
                ):
                    for code in codes_list:
                        close = get_KLDay_close(code, days, quote_ctx)
                        forecast = natural_cs_forecast_day(
                            close, forecast_days, smooth_days
                        )
                        plot_close_with_forecast(
                            close,
                            forecast,
                            f"{code} Close Price: Past {days} Days + Forecast {forecast_days} Days (Smooth: {smooth_days} Days)",
                            f"{code} close price (past {days} days + forecast {forecast_days} days, smooth {smooth_days}).png",
                            smooth_days=smooth_days,
                        )

                generate_forecast_plot(codes_list)

    except Exception as e:
        print("error:", e)
        traceback.print_exc()


if __name__ == "__main__":
    quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    codes_list = [
        "HK.03690",
        "HK.09618",
        "HK.09988",
        "HK.09888",
        "HK.02800",
        "HK.00700",
        "HK.00005",
        "HK.09626",
        "HK.01810",
        "HK.02318",
        "HK.800000",
    ]  # 美团 京东 阿里 百度 盈富 腾讯 汇丰 B站 小米 平安 恒指

    generate_cs_forecast_day(quote_ctx, codes_list)
    quote_ctx.unsubscribe_all()
    quote_ctx.close()
