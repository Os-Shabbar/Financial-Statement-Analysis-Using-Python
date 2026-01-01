import requests
import pandas as pd
import time
import os

# Your API key
API_KEY = 'EFD059EYA8J22V4Q'

# List of tickers
tickers = ['PG', 'KO', 'PEP', 'CL', 'KMB']

# Function to get financial statements
def get_financial_statement(symbol, statement_type):
    url = f'https://www.alphavantage.co/query?function={statement_type}&symbol={symbol}&apikey={API_KEY}'
    response = requests.get(url)
    data = response.json()
    
    if 'annualReports' in data:
        df = pd.DataFrame(data['annualReports'])
        df['symbol'] = symbol  # add ticker column
        return df
    else:
        print(f"Error fetching {statement_type} for {symbol}: {data}")
        return pd.DataFrame()

# Dictionaries to store DataFrames
income_statements = {}
balance_sheets = {}
cash_flows = {}

# Loop through tickers
for ticker in tickers:
    print(f"Fetching data for {ticker}...")
    
    # Fetch income statement
    income_statements[ticker] = get_financial_statement(ticker, 'INCOME_STATEMENT')
    time.sleep(12)
    
    # Fetch balance sheet
    balance_sheets[ticker] = get_financial_statement(ticker, 'BALANCE_SHEET')
    time.sleep(12)
    
    # Fetch cash flow
    cash_flows[ticker] = get_financial_statement(ticker, 'CASH_FLOW')
    time.sleep(12)

# Combine all companies' data for each statement type
df_income = pd.concat(income_statements.values(), ignore_index=True)
df_balance = pd.concat(balance_sheets.values(), ignore_index=True)
df_cash = pd.concat(cash_flows.values(), ignore_index=True)

# Save to Excel with separate sheets
output_path = r"C:\Users\hp\Downloads\financial_statements.xlsx"
with pd.ExcelWriter(output_path) as writer:
    df_income.to_excel(writer, sheet_name='Income Statements', index=False)
    df_balance.to_excel(writer, sheet_name='Balance Sheets', index=False)
    df_cash.to_excel(writer, sheet_name='Cash Flows', index=False)

print(f"All financial statements saved to {output_path}")
