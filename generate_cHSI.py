from futu import *
import traceback
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_KLDay_compareHI(code, days, quote_ctx):
    """
    Input: code -> str
    Output: day_change_compareHI -> list[float], day_vol_compareHI -> list[float]

    Get the day change / volatility of code compared to HI

    """
    ret, data = quote_ctx.get_cur_kline(code, days, KLType.K_DAY, AuType.NONE)
    ret_HI, data_HI = quote_ctx.get_cur_kline(
        "HK.800000", days, KLType.K_DAY, AuType.NONE
    )
    if ret == RET_OK and ret_HI == RET_OK:
        # print(data["open"])
        try:
            open = data["open"]
            close = data["close"]
            high = data["high"]
            low = data["low"]

            open_HI = data_HI["open"]
            close_HI = data_HI["close"]
            high_HI = data_HI["high"]
            low_HI = data_HI["low"]
        except Exception as e:
            print("error:", e)
            traceback.print_exc()

        day_change = (close - open) / open * 100  # 涨跌幅% = (收市价 - 开市价) / 开市价
        day_vol = abs(high - low) / open * 100  # 波幅% = abs(最高 - 最低) / 开市价

        day_change_HI = (close_HI - open_HI) / open_HI * 100
        day_vol_HI = abs(high_HI - low_HI) / open_HI * 100

        day_change_compareHI = day_change - day_change_HI  # 涨跌幅对比恒指
        day_vol_compareHI = day_vol - day_vol_HI  # 波幅对比恒指

        return day_change_compareHI, day_vol_compareHI
    else:
        return "error:", data_HI


def plot_changeHI(cHI_list, titles):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    means = []
    stds = []

    for idx, (cHI, ax, title) in enumerate(zip(cHI_list, axes, titles)):
        colors = ["blue" if x > 0 else "red" for x in cHI]
        ax.bar(range(len(cHI)), cHI, color=colors)
        ax.set_xlabel("Day Index")
        ax.set_ylabel("Change Compared to HI (%)")
        ax.set_title(title)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        mean_cHI = np.mean(cHI)
        means.append(mean_cHI)
        std_cHI = np.std(cHI)
        stds.append(std_cHI)
        ax.text(
            0.02,
            0.98,
            f"Mean: {mean_cHI:.2f}%\nStd Dev: {std_cHI:.2f}%",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    return fig, means, stds


def generate_cHSI_plots(quote_ctx, codes_list):
    try:
        ret_sub, err_message = quote_ctx.subscribe(
            codes_list, [SubType.QUOTE, SubType.K_DAY]
        )
        if ret_sub == RET_OK:
            days_options = [720, 480, 360, 240]
            day_cHI_list = []
            day_vHI_list = []
            for i in range(len(days_options)):
                days = days_options[i]
                for j in codes_list[0:8]:
                    day_cHI, day_vHI = get_KLDay_compareHI(j, days, quote_ctx)
                    day_cHI_list.append(day_cHI)
                    day_vHI_list.append(day_vHI)

                fig_1, means_1, stds_1 = plot_changeHI(
                    [
                        day_cHI_list[i * 8],
                        day_cHI_list[i * 8 + 1],
                        day_cHI_list[i * 8 + 2],
                        day_cHI_list[i * 8 + 3],
                    ],
                    ["Meituan", "JD", "Alibaba", "Baidu"],
                )
                fig_1.savefig(
                    f"img/cHI/Meituan, JD, Alibaba, Baidu day-change compared to HSI ({days} Days).png"
                )
                fig_2, means_2, stds_2 = plot_changeHI(
                    [
                        day_cHI_list[i * 8 + 4],
                        day_cHI_list[i * 8 + 5],
                        day_cHI_list[i * 8 + 6],
                        day_cHI_list[i * 8 + 7],
                    ],
                    ["TFHK", "Tencent", "HSBC", "Bilibili"],
                )
                fig_2.savefig(
                    f"img/cHI/TFHK, Tencent, HSBC, Bilibili day-change compared to HSI ({days} Days).png"
                )

        code_names = [
            "Meituan",
            "JD",
            "Alibaba",
            "Baidu",
            "TFHK",
            "Tencent",
            "HSBC",
            "Bilibili",
        ]

        change_df = (
            pd.DataFrame(
                {
                    "Code": code_names,
                    "Change to HSI (%)": np.round(means_1 + means_2, 2),
                }
            )
            .sort_values(by="Change to HSI (%)")
            .reset_index(drop=True)
        )

        std_df = (
            pd.DataFrame(
                {
                    "Code": code_names,
                    "Match to HSI (Std %)": np.round(stds_1 + stds_2, 2),
                }
            )
            .sort_values(by="Match to HSI (Std %)")
            .reset_index(drop=True)
        )

        print("Change to HSI (sorted):")
        print(change_df)
        print("Match to HSI (sorted):")
        print(std_df)
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

    generate_cHSI_plots()
    quote_ctx.unsubscribe_all()
    quote_ctx.close()
