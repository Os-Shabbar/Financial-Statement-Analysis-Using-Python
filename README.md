##Financial Staement Analysis
---

###Using Python libraries to calculate and conduct financial ration analysis

---

## Scripts Description

### 1️⃣ `Pull Financial Data from API.py`

- Downloads **Balance Sheet, Income Statement, and Cash Flow Statement** for the companies.
- **Tickers:** `['PG', 'KO', 'PEP', 'CL', 'KMB']`
- Saves the output to:  
  `raw_financial_statement.xls`  
- Data spans **2005–2025**.

---

### 2️⃣ `Financial Ratios_Zscore_Dupont_calculation.py`

- Reads the raw financial statements.
- Computes the following:

  - **Financial Ratios** (liquidity, profitability, leverage, etc.)  
  - **DuPont Analysis** (ROE decomposition)  
  - **Altman Z-Score** (bankruptcy prediction metric)

- Saves the results to:  
  `financial_analysis_output.xls`

---

### 3️⃣ `plot_financial_analysis.py`

- Reads `financial_analysis_output.xls`.
- Generates visual comparison plots for the five companies, including:

  - Key financial ratios trends  
  - DuPont components comparison  
  - Z-score trends over the years  

- Helps to visually compare financial health and performance.


