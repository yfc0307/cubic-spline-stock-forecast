from futu import *
from generate_cHSI import *
from generate_cs_forecast_day import *
from generate_cs_forecast_hour import *
from generate_cs_forecast_5min import *
from delete_imgs import *
import traceback
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    delete_png_files_in_img_folder()

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
    generate_cs_forecast_hour(quote_ctx, codes_list)
    generate_cs_forecast_fivemin(quote_ctx, codes_list)
    generate_cHSI_plots(quote_ctx, codes_list)
    plt.close("all")

    quote_ctx.unsubscribe_all()
    quote_ctx.close()
