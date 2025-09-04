##套件#######
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import chardet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import statsmodels.api as sm
from numpy.linalg import lstsq
from dateutil.relativedelta import relativedelta
import mplfinance as mpf
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import ttest_1samp
from arch import arch_model
import statsmodels.formula.api as smf
from data import get_combine_data
### import data ####
combined_data = combine_data.copy()
combined_data = combined_data.sort_values(['date','time']).reset_index(drop=True)
combined_data['Price'] = pd.to_numeric(combined_data['Price'], errors='coerce')

combined_data["time"] = pd.to_datetime(combined_data["time"], format="%H:%M:%S")
combined_data["hour"] = combined_data["time"].dt.hour
combined_data["minute"] = combined_data["time"].dt.minute
combined_data["second"] = combined_data["time"].dt.second
combined_data = combined_data.drop(columns=["time"])
combined_data = combined_data.dropna(how="all").reset_index(drop=True)

#####################建立OBI###################################
ofi_trade = combined_data.copy()
ofi_trade = ofi_trade.sort_values(["date","hour","minute","second"]).reset_index(drop=True)

ofi_trade["Pb"] = np.where(ofi_trade["Type"]=="BID", ofi_trade["Price"], np.nan)
ofi_trade["Qb"] = np.where(ofi_trade["Type"]=="BID", ofi_trade["Size"] , np.nan)
ofi_trade["Pa"] = np.where(ofi_trade["Type"]=="ASK", ofi_trade["Price"], np.nan)
ofi_trade["Qa"] = np.where(ofi_trade["Type"]=="ASK", ofi_trade["Size"] , np.nan)


ofi_trade["bid_price_prev"] = ofi_trade["Pb"].replace(0, np.nan).ffill().shift(1)
ofi_trade["bid_size_prev"]  = ofi_trade["Qb"].replace(0, np.nan).ffill().shift(1)
ofi_trade["ask_price_prev"] = ofi_trade["Pa"].replace(0, np.nan).ffill().shift(1)
ofi_trade["ask_size_prev"]  = ofi_trade["Qa"].replace(0, np.nan).ffill().shift(1)
print(ofi_trade)

cols = ["Pb","Qb","Pa","Qa",
        "bid_price_prev","bid_size_prev",
        "ask_price_prev","ask_size_prev"]

ofi_trade[cols] = ofi_trade[cols].fillna(0).astype(float)

ofi_trade["e_n"] = 0.0
ofi_trade.loc[(ofi_trade["Pb"] != 0) & (ofi_trade["Pb"] >= ofi_trade["bid_price_prev"]),
              "e_n"] += ofi_trade["Qb"]

ofi_trade.loc[(ofi_trade["Pb"] != 0) & (ofi_trade["Pb"] <= ofi_trade["bid_price_prev"]),
              "e_n"] -= ofi_trade["bid_size_prev"]

ofi_trade.loc[(ofi_trade["Pa"] != 0) & (ofi_trade["Pa"] <= ofi_trade["ask_price_prev"]),
              "e_n"] -= ofi_trade["Qa"]

ofi_trade.loc[(ofi_trade["Pa"] != 0) & (ofi_trade["Pa"] >= ofi_trade["ask_price_prev"]),
              "e_n"] += ofi_trade["ask_size_prev"]
ofi_per_sec = ofi_trade.groupby(["date", "hour", "minute", "second"])["e_n"].sum().reset_index()
ofi_per_sec.rename(columns={"e_n": "OFI"}, inplace=True)

###找出每一秒的最佳報價，並計算MID PRICE###
last_bid = (
    ofi_trade[ofi_trade["Pb"] != 0]
    .groupby(["date","hour","minute","second"])
    .last()[["Pb"]]
    .reset_index()
)

last_ask = (
    ofi_trade[ofi_trade["Pa"] != 0]
    .groupby(["date","hour","minute","second"])
    .last()[["Pa"]]
    .reset_index()
)

last_per_sec = pd.merge(last_bid, last_ask, on=["date","hour","minute","second"], how="outer")
ofi_per_sec = ofi_per_sec.merge(last_per_sec, on=["date","hour","minute","second"], how="left")

ofi_per_sec["mid_price"] = (ofi_per_sec["Pb"] + ofi_per_sec["Pa"]) / 2

#print(ofi_per_sec)

######跑回歸######
ofi_per_sec["mid_price_change"] = ofi_per_sec["mid_price"].diff()
ofi_per_sec["OFI_lag1"] = ofi_per_sec["OFI"].shift(1)


df_reg = ofi_per_sec.dropna(subset=["mid_price_change", "OFI_lag1"]) 
X = sm.add_constant(df_reg["OFI_lag1"])  
y = df_reg["mid_price_change"]

model = sm.OLS(y, X).fit()
print(model.summary())

####滾動窗口跑######
# 確保時間索引
ofi_per_sec["timestamp"] = pd.to_datetime(ofi_per_sec["date"].astype(str) + " " +
                                          ofi_per_sec["hour"].astype(str) + ":" +
                                          ofi_per_sec["minute"].astype(str) + ":" +
                                          ofi_per_sec["second"].astype(str))

ofi_per_sec = ofi_per_sec.sort_values("timestamp").reset_index(drop=True)

# 定義 rolling window 大小 (30 分鐘 = 1800 秒)
window_size = 1800
df = ofi_per_sec.dropna(subset=["mid_price_change","OFI_lag1"]).copy()
results = []

for start in range(0, len(df) - window_size + 1):
    end = start + window_size
    sample = df.iloc[start:end]

    X = sm.add_constant(sample["OFI_lag1"])
    y = sample["mid_price_change"]

    model = sm.OLS(y, X).fit()

    results.append({
        "end_time": sample["timestamp"].iloc[-1],
        "beta": model.params["OFI_lag1"],
        "tstat": model.tvalues["OFI_lag1"],
        "pval": model.pvalues["OFI_lag1"]
    })

rolling_results = pd.DataFrame(results)

print(rolling_results.head())
#output_path = r"D:\台新\intraday_data\OFI\ofi_per_sec.xlsx"
#ofi_per_sec.to_excel(output_path, index=False)