from alpha_vantage.timeseries import TimeSeries
import pandas as pd

api_key = "GG676GUQC1C1F4II"
ts = TimeSeries(key=api_key, output_format='pandas')

companies = {
    "HDFCBANK.BSE": "HDFC Bank",
    "APOLLOHOSP.BSE": "Apollo Hospitals",
    "IOC.BSE": "Indian Oil Corporation",
    "TATASTEEL.BSE": "Tata Steel",
    "OBEROIRLTY.BSE": "Oberoi Realty",
    "RELIANCE.BSE": "Reliance Jio",  
    "INDIGO.BSE": "Indigo Airlines",
    "PVR.BSE": "PVR Cinemas"
}

all_data = pd.DataFrame()

for symbol, company_name in companies.items():
    print (f"Fetching Data for {company_name} ({symbol})...")

    try:
        data, meta_data = ts.get_daily(symbol=symbol, outputsize="full")

        data["Company"] = company_name
        data["Symbol"] = symbol

        data.reset_index(inplace=True)

        all_data = pd.concat([all_data, data], ignore_index=True)

    except Exception as e:
        print(f"Error fetching data for {company_name} ({symbol}): {e}")

all_data.to_csv("bse_stock_data.csv", index=False)
print("Data saved to stock_data.csv")