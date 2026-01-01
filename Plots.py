import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

output_folder = r"C:\Users\hp\Downloads\financial_analysis_plots"
os.makedirs(output_folder, exist_ok=True)
file_path = r"C:\Users\hp\Downloads\financial_analysis_output.xlsx"


df1 = pd.read_excel(file_path, sheet_name="All_Companies_Ratios")
df2 = pd.read_excel(file_path, sheet_name="All_Companies_DuPont")
df3 = pd.read_excel(file_path, sheet_name="All_Companies_Zscore")



#restrucre the dataframes
dfs_long = []

for df in [df1, df2, df3]:
    df_long = pd.melt(
        df,
        id_vars=[df.columns[0], df.columns[1]],
        var_name="Year",
        value_name="Value"
    )

    df_long = df_long.rename(columns={
        df.columns[0]: "Indicator",
        df.columns[1]: "Symbol"
    })

    df_long["Indicator"] = df_long["Indicator"].astype(str)
    df_long = df_long.dropna(subset=["Indicator", "Value"])

    dfs_long.append(df_long)

df1_long, df2_long, df3_long = dfs_long


output_file = r"C:\Users\hp\Downloads\output_data.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df1_long.to_excel(writer, sheet_name="Sheet1", index=False)
    df2_long.to_excel(writer, sheet_name="Sheet2", index=False)
    df3_long.to_excel(writer, sheet_name="Sheet3", index=False)



df_all = pd.concat([df1_long, df2_long, df3_long], ignore_index=True)


for indicator, indicator_df in df_all.groupby("Indicator"):


    indicator_df["Year"] = pd.to_numeric(indicator_df["Year"], errors='coerce').astype(int)


    indicator_df = indicator_df.sort_values(by="Year")

    plt.figure(figsize=(12,6))  

    for symbol, symbol_df in indicator_df.groupby("Symbol"):
        symbol_df = symbol_df.sort_values(by="Year")
        plt.plot(symbol_df["Year"], symbol_df["Value"], marker='o', label=symbol)

    plt.title(indicator, fontsize=14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    

    plt.xticks(indicator_df["Year"].unique(), rotation=45)
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    
    safe_indicator_name = indicator.replace("/", "_").replace("\\", "_").replace(" ", "_")
    filename = os.path.join(output_folder, f"{safe_indicator_name}.png")
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close()
