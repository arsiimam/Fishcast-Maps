import pandas as pd

df = pd.read_csv("/www/wwwroot/fisheries-server.cloud/Fishcast-Maps/Data/HSI_layang_daily.csv", nrows=5)
print(df.columns.tolist())
print(df["date"].head(20))
print(df["date"].dtype)

# Cek range tanggal yang tersedia
df_full = pd.read_csv("/www/wwwroot/fisheries-server.cloud/Fishcast-Maps/Data/HSI_layang_daily.csv")
df_full["date"] = pd.to_datetime(df_full["date"])
df_full["doy"] = df_full["date"].dt.dayofyear
print("DOY tersedia:", sorted(df_full["doy"].unique()))
print("Tanggal min:", df_full["date"].min())
print("Tanggal max:", df_full["date"].max())
print("Total baris:", len(df_full))