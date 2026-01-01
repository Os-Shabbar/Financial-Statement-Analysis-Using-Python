import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# DATA TRANSFORMATION FUNCTION

def prepare_statements_for_all_companies(df: pd.DataFrame) -> dict:

    df = df.copy()
    
    df['fiscalDateEnding'] = df['fiscalDateEnding'].astype(str)
    
    df['year'] = df['fiscalDateEnding'].str[:4]
    
    if 'symbol' not in df.columns:
        raise ValueError("Column 'symbol' not found in data. Required for multi-company analysis.")
    

    companies = df['symbol'].unique()
    print(f"Found {len(companies)} companies: {list(companies)}")
    
    company_statements = {}
    
    for company in companies:

        company_df = df[df['symbol'] == company].copy()

        cols_to_drop = ['reportedCurrency', 'fiscalDateEnding']
        company_df = company_df.drop(columns=[col for col in cols_to_drop if col in company_df.columns], errors='ignore')

        company_df = company_df.set_index('year')

        company_df = company_df[~company_df.index.duplicated(keep='first')]
        

        for col in company_df.columns:
            if col != 'symbol':
                company_df[col] = pd.to_numeric(company_df[col], errors='coerce')

        company_df = company_df.T
        

        company_df = company_df.sort_index(axis=1, ascending=False)
        
        company_statements[company] = company_df
    
    return company_statements


# CREATE SEPARATE FINANCIAL STATEMENT SHEETS


def create_income_statement_sheet(income_statements):
    """Create Income Statement sheet for all companies"""
    
    print("\n📊 Creating Income Statement sheet...")
    
    all_income = []
    
    for company, income_stmt in income_statements.items():
        if income_stmt is not None:
            # Add symbol column to each row
            income_with_symbol = income_stmt.copy()
            income_with_symbol['Symbol'] = company
            all_income.append(income_with_symbol)
    
    if not all_income:
        return pd.DataFrame()
    

    combined_income = pd.concat(all_income, axis=0)
    

    combined_income = combined_income.reset_index()
    combined_income = combined_income.rename(columns={'index': 'Financial_Item'})
    

    year_cols = [col for col in combined_income.columns if col not in ['Symbol', 'Financial_Item']]
    col_order = ['Symbol', 'Financial_Item'] + year_cols
    combined_income = combined_income[col_order]
    
    return combined_income

def create_balance_sheet_sheet(balance_sheets):
    """Create Balance Sheet sheet for all companies"""
    
    print("\n📊 Creating Balance Sheet sheet...")
    
    all_balance = []
    
    for company, balance_sheet in balance_sheets.items():
        if balance_sheet is not None:
            # Add symbol column to each row
            balance_with_symbol = balance_sheet.copy()
            balance_with_symbol['Symbol'] = company
            all_balance.append(balance_with_symbol)
    
    if not all_balance:
        return pd.DataFrame()
    

    combined_balance = pd.concat(all_balance, axis=0)

    combined_balance = combined_balance.reset_index()
    combined_balance = combined_balance.rename(columns={'index': 'Financial_Item'})
    

    year_cols = [col for col in combined_balance.columns if col not in ['Symbol', 'Financial_Item']]
    col_order = ['Symbol', 'Financial_Item'] + year_cols
    combined_balance = combined_balance[col_order]
    
    return combined_balance

def create_cash_flow_sheet(cash_flows):
    """Create Cash Flow sheet for all companies"""
    
    print("\n📊 Creating Cash Flow sheet...")
    
    all_cash = []
    
    for company, cash_flow in cash_flows.items():
        if cash_flow is not None:
            # Add symbol column to each row
            cash_with_symbol = cash_flow.copy()
            cash_with_symbol['Symbol'] = company
            all_cash.append(cash_with_symbol)
    
    if not all_cash:
        return pd.DataFrame()

    combined_cash = pd.concat(all_cash, axis=0)
    

    combined_cash = combined_cash.reset_index()
    combined_cash = combined_cash.rename(columns={'index': 'Financial_Item'})
    

    year_cols = [col for col in combined_cash.columns if col not in ['Symbol', 'Financial_Item']]
    col_order = ['Symbol', 'Financial_Item'] + year_cols
    combined_cash = combined_cash[col_order]
    
    return combined_cash


# FINANCIAL ANALYZER

class CompanyFinancialAnalyzer:
    def __init__(self, income_stmt, balance_sheet, cash_flow, company_symbol):
        self.income_stmt = income_stmt
        self.balance_sheet = balance_sheet
        self.cash_flow = cash_flow
        self.company_symbol = company_symbol
        
        # Standardize column names to strings
        self.income_stmt.columns = self.income_stmt.columns.astype(str)
        self.balance_sheet.columns = self.balance_sheet.columns.astype(str)
        self.cash_flow.columns = self.cash_flow.columns.astype(str)


    
    def _get_item(self, statement, item_name, alt_names=None):
        """Get item from statement with alternative name options"""
        # First try exact match
        if item_name in statement.index:
            return statement.loc[item_name]
        
        # Try alternative names
        if alt_names:
            for alt in alt_names:
                if alt in statement.index:
                    return statement.loc[alt]
        

        index_lower = [str(idx).lower() for idx in statement.index]
        item_lower = item_name.lower()
        
        for i, idx_lower in enumerate(index_lower):
            if item_lower in idx_lower or idx_lower in item_lower:
                return statement.iloc[i]
        

        for idx in statement.index:
            if isinstance(idx, str):
                if item_name.lower() in idx.lower():
                    return statement.loc[idx]
        

        return pd.Series([np.nan] * len(statement.columns), index=statement.columns)

    def _calculate_average_series(self, series):
        """Calculate average of current and previous year"""
        if series.empty or series.isna().all():
            return pd.Series([np.nan] * len(series), index=series.index)

        try:
            years = [int(col) for col in series.index if str(col).isdigit()]
            years.sort()
        except:
            return series  # Return original if can't parse years
        
        avg_series = pd.Series(index=series.index, dtype=float)
        
        for year_str in series.index:
            try:
                year_int = int(year_str)
                prev_year_str = str(year_int - 1)
                
                if prev_year_str in series.index and not pd.isna(series[prev_year_str]):
                    avg_series[year_str] = (series[year_str] + series[prev_year_str]) / 2
                else:
                    avg_series[year_str] = series[year_str]
            except:
                avg_series[year_str] = series[year_str]
        
        return avg_series

    def _safe_divide(self, numerator, denominator, default=np.nan):
        """Safely divide two series, handling division by zero and NaN values"""
        if numerator.empty or denominator.empty:
            return pd.Series([np.nan] * len(self.income_stmt.columns), index=self.income_stmt.columns)
        
        # Create result series
        result = pd.Series(index=numerator.index, dtype=float)
        
        for year in numerator.index:
            num = numerator[year]
            den = denominator[year]
            
            # Check if both values are valid numbers and denominator is not zero
            if pd.notna(num) and pd.notna(den) and den != 0:
                result[year] = num / den
            else:
                result[year] = default
        
        return result

    # -------------------- LIQUIDITY --------------------

    def current_ratio(self):
        current_assets = self._get_item(self.balance_sheet, 'totalCurrentAssets', 
                                       ['currentAssets', 'CurrentAssets'])
        current_liabilities = self._get_item(self.balance_sheet, 'totalCurrentLiabilities',
                                           ['currentLiabilities', 'CurrentLiabilities'])
        
        return self._safe_divide(current_assets, current_liabilities)

    def quick_ratio(self):
        current_assets = self._get_item(self.balance_sheet, 'totalCurrentAssets',
                                       ['currentAssets', 'CurrentAssets'])
        current_liabilities = self._get_item(self.balance_sheet, 'totalCurrentLiabilities',
                                           ['currentLiabilities', 'CurrentLiabilities'])
        
        inventory = self._get_item(self.balance_sheet, 'inventory', ['Inventory'])
        
        if inventory.empty:
            return self._safe_divide(current_assets, current_liabilities)
        
        quick_assets = current_assets - inventory
        return self._safe_divide(quick_assets, current_liabilities)

    def cash_ratio(self):
        cash = self._get_item(self.balance_sheet, 'cashAndCashEquivalents',
                            ['cash', 'Cash', 'cashAndCashEquivalentsAtCarryingValue',
                             'cashAndShortTermInvestments'])
        current_liabilities = self._get_item(self.balance_sheet, 'totalCurrentLiabilities',
                                           ['currentLiabilities', 'CurrentLiabilities'])
        
        return self._safe_divide(cash, current_liabilities)

    # -------------------- PROFITABILITY --------------------

    def gross_margin(self):
        gross_profit = self._get_item(self.income_stmt, 'grossProfit',
                                    ['grossProfit', 'GrossProfit'])
        revenue = self._get_item(self.income_stmt, 'totalRevenue',
                               ['revenue', 'Revenue', 'totalRevenue'])
        
        result = self._safe_divide(gross_profit, revenue, default=np.nan)
        return result * 100  # Convert to percentage

    def operating_margin(self):
        operating_income = self._get_item(self.income_stmt, 'operatingIncome',
                                        ['operatingIncome', 'incomeFromOperations'])
        revenue = self._get_item(self.income_stmt, 'totalRevenue',
                               ['revenue', 'Revenue', 'totalRevenue'])
        
        result = self._safe_divide(operating_income, revenue, default=np.nan)
        return result * 100  # Convert to percentage

    def net_margin(self):
        net_income = self._get_item(self.income_stmt, 'netIncome',
                                  ['netIncome', 'netIncomeFromContinuingOperations'])
        revenue = self._get_item(self.income_stmt, 'totalRevenue',
                               ['revenue', 'Revenue', 'totalRevenue'])
        
        result = self._safe_divide(net_income, revenue, default=np.nan)
        return result * 100  # Convert to percentage

    def roa(self):
        net_income = self._get_item(self.income_stmt, 'netIncome',
                                  ['netIncome', 'netIncomeFromContinuingOperations'])
        total_assets = self._get_item(self.balance_sheet, 'totalAssets',
                                    ['totalAssets', 'TotalAssets'])
        
        avg_assets = self._calculate_average_series(total_assets)
        
        result = self._safe_divide(net_income, avg_assets, default=np.nan)
        return result * 100  # Convert to percentage

    def roe(self):
        net_income = self._get_item(self.income_stmt, 'netIncome',
                                  ['netIncome', 'netIncomeFromContinuingOperations'])
        
        equity = self._get_item(self.balance_sheet, 'totalShareholderEquity',
                              ['totalStockholdersEquity', 'totalEquity', 
                               'stockholdersEquity', 'shareholdersEquity',
                               'totalShareholderEquity'])
        
        avg_equity = self._calculate_average_series(equity)
        
        result = self._safe_divide(net_income, avg_equity, default=np.nan)
        return result * 100  # Convert to percentage

    # -------------------- LEVERAGE --------------------

    def debt_to_equity(self):
        total_debt = self._get_item(self.balance_sheet, 'totalDebt',
                                  ['totalDebt', 'shortLongTermDebtTotal'])
        
        if total_debt.empty:
            total_debt = self._get_item(self.balance_sheet, 'totalLiabilities',
                                      ['totalLiabilities'])
        
        equity = self._get_item(self.balance_sheet, 'totalShareholderEquity',
                              ['totalStockholdersEquity', 'totalEquity', 
                               'stockholdersEquity', 'shareholdersEquity',
                               'totalShareholderEquity'])
        
        return self._safe_divide(total_debt, equity, default=np.nan)

    def interest_coverage(self):
        operating_income = self._get_item(self.income_stmt, 'operatingIncome',
                                        ['operatingIncome', 'incomeFromOperations'])
        interest_expense = self._get_item(self.income_stmt, 'interestExpense',
                                        ['interestExpense', 'interestAndDebtExpense'])
        
        # For interest coverage, we need to handle negative interest expense (which is common)
        # Also need to handle zero or missing values
        if interest_expense.empty or operating_income.empty:
            return pd.Series([np.nan] * len(self.income_stmt.columns), index=self.income_stmt.columns)
        
        result = pd.Series(index=operating_income.index, dtype=float)
        
        for year in operating_income.index:
            op_inc = operating_income[year]
            int_exp = interest_expense[year]
            
            # Check if values are valid
            if pd.notna(op_inc) and pd.notna(int_exp):
                # Take absolute value of interest expense (it's often negative)
                int_exp_abs = abs(int_exp)
                
                if int_exp_abs != 0:
                    result[year] = op_inc / int_exp_abs
                else:
                    # If interest expense is zero, can't calculate coverage
                    result[year] = np.nan
            else:
                result[year] = np.nan
        
        return result

    # -------------------- EFFICIENCY --------------------

    def asset_turnover(self):
        revenue = self._get_item(self.income_stmt, 'totalRevenue',
                               ['revenue', 'Revenue', 'totalRevenue'])
        total_assets = self._get_item(self.balance_sheet, 'totalAssets',
                                    ['totalAssets', 'TotalAssets'])
        
        avg_assets = self._calculate_average_series(total_assets)
        
        return self._safe_divide(revenue, avg_assets, default=np.nan)

    # -------------------- ALTMAN Z-SCORE --------------------
    
    def altman_z_score(self):
        """
        Calculate Altman Z-Score for bankruptcy prediction
        
        Z-Score = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
        
        Where:
        X1 = Working Capital / Total Assets
        X2 = Retained Earnings / Total Assets
        X3 = EBIT / Total Assets
        X4 = Market Value of Equity / Total Liabilities (or Book Value if market value unavailable)
        X5 = Sales / Total Assets
        
        Interpretation:
        Z > 2.99: Safe Zone (low bankruptcy risk)
        1.81 < Z < 2.99: Grey Zone (moderate risk)
        Z < 1.81: Distress Zone (high bankruptcy risk)
        """
        
        print(f"  Calculating Z-Score for {self.company_symbol}...")
        

        current_assets = self._get_item(self.balance_sheet, 'totalCurrentAssets',
                                       ['currentAssets', 'CurrentAssets'])
        current_liabilities = self._get_item(self.balance_sheet, 'totalCurrentLiabilities',
                                           ['currentLiabilities', 'CurrentLiabilities'])
        total_assets = self._get_item(self.balance_sheet, 'totalAssets',
                                    ['totalAssets', 'TotalAssets'])
        retained_earnings = self._get_item(self.balance_sheet, 'retainedEarnings',
                                         ['retainedEarnings', 'RetainedEarnings', 
                                          'accumulatedRetainedEarningsDeficit'])
        

        ebit = self._get_item(self.income_stmt, 'operatingIncome',
                             ['operatingIncome', 'incomeFromOperations', 'ebit', 'EBIT'])
        

        equity = self._get_item(self.balance_sheet, 'totalShareholderEquity',
                              ['totalStockholdersEquity', 'totalEquity', 
                               'stockholdersEquity', 'shareholdersEquity',
                               'totalShareholderEquity'])
        
        total_liabilities = self._get_item(self.balance_sheet, 'totalLiabilities',
                                         ['totalLiabilities', 'TotalLiabilities'])
        
        revenue = self._get_item(self.income_stmt, 'totalRevenue',
                               ['revenue', 'Revenue', 'totalRevenue'])
        
        # Calculate each component
        # X1: Working Capital / Total Assets
        working_capital = current_assets - current_liabilities
        x1 = self._safe_divide(working_capital, total_assets, default=0)
        
        # X2: Retained Earnings / Total Assets
        x2 = self._safe_divide(retained_earnings, total_assets, default=0)
        
        # X3: EBIT / Total Assets
        x3 = self._safe_divide(ebit, total_assets, default=0)
        
        # X4: Book Value of Equity / Total Liabilities
        x4 = self._safe_divide(equity, total_liabilities, default=0)
        
        # X5: Sales / Total Assets
        x5 = self._safe_divide(revenue, total_assets, default=0)
        

        z_score = pd.Series(index=total_assets.index, dtype=float)
        
        for year in total_assets.index:
            # Check if we have all components
            if all(pd.notna([x1[year], x2[year], x3[year], x4[year], x5[year]])):
                z_score[year] = (1.2 * x1[year] + 
                                1.4 * x2[year] + 
                                3.3 * x3[year] + 
                                0.6 * x4[year] + 
                                1.0 * x5[year])
            else:
                z_score[year] = np.nan


        interpretation = pd.Series(index=z_score.index, dtype=str)
        for year in z_score.index:
            z = z_score[year]
            if pd.notna(z):
                if z > 2.99:
                    interpretation[year] = "Safe Zone"
                elif z > 1.81:
                    interpretation[year] = "Grey Zone"
                else:
                    interpretation[year] = "Distress Zone"
            else:
                interpretation[year] = "N/A"
        

        return pd.DataFrame({
            'X1 (WC/TA)': x1,
            'X2 (RE/TA)': x2,
            'X3 (EBIT/TA)': x3,
            'X4 (Equity/TL)': x4,
            'X5 (Sales/TA)': x5,
            'Z-Score': z_score,
            'Risk Category': interpretation
        }).T

    # -------------------- DUPONT --------------------

    def dupont_analysis(self):
        try:
            profit_margin = self.net_margin() / 100
            asset_turnover = self.asset_turnover()
            
            total_assets = self._get_item(self.balance_sheet, 'totalAssets',
                                        ['totalAssets', 'TotalAssets'])
            
            equity = self._get_item(self.balance_sheet, 'totalShareholderEquity',
                                  ['totalStockholdersEquity', 'totalEquity', 
                                   'stockholdersEquity', 'shareholdersEquity',
                                   'totalShareholderEquity'])
            

            equity_multiplier = self._safe_divide(total_assets, equity, default=np.nan)
            
            # Calculate DuPont ROE
            roe_dupont = pd.Series(index=profit_margin.index, dtype=float)
            for year in profit_margin.index:
                pm = profit_margin[year]
                at = asset_turnover[year]
                em = equity_multiplier[year]
                
                if pd.notna(pm) and pd.notna(at) and pd.notna(em):
                    roe_dupont[year] = pm * at * em * 100
                else:
                    roe_dupont[year] = np.nan

            return pd.DataFrame({
                'Net Profit Margin (%)': self.net_margin(),
                'Asset Turnover': asset_turnover,
                'Equity Multiplier': equity_multiplier,
                'ROE (DuPont) (%)': roe_dupont,
                'ROE (Direct) (%)': self.roe()
            }).T
        except Exception as e:
            print(f"  Warning in DuPont analysis for {self.company_symbol}: {str(e)}")
            # Return empty DataFrame with correct structure
            return pd.DataFrame(index=['Net Profit Margin (%)', 'Asset Turnover', 
                                      'Equity Multiplier', 'ROE (DuPont) (%)', 
                                      'ROE (Direct) (%)'])

    def calculate_financial_ratios(self):
        """Calculate all financial ratios for this company"""
        
        print(f"  Calculating ratios for {self.company_symbol}...")
        
        ratios = {}
        
        # Liquidity Ratios
        ratios['Current Ratio'] = self.current_ratio()
        ratios['Quick Ratio'] = self.quick_ratio()
        ratios['Cash Ratio'] = self.cash_ratio()
        
        # Profitability Ratios
        ratios['Gross Margin (%)'] = self.gross_margin()
        ratios['Operating Margin (%)'] = self.operating_margin()
        ratios['Net Margin (%)'] = self.net_margin()
        ratios['ROA (%)'] = self.roa()
        ratios['ROE (%)'] = self.roe()
        
        # Leverage Ratios
        ratios['Debt to Equity'] = self.debt_to_equity()
        ratios['Interest Coverage'] = self.interest_coverage()
        
        # Efficiency Ratios
        ratios['Asset Turnover'] = self.asset_turnover()
        

        ratio_df = pd.DataFrame(ratios).T
        

        ratio_df.insert(0, 'Company', self.company_symbol)
        
        for col in ratio_df.columns:
            if col != 'Company':
                ratio_df[col] = ratio_df[col].round(2)
        
        return ratio_df


def main():

    input_path = r"C:\Users\hp\Downloads\financial_statements.xlsx"
    output_path = r"C:\Users\hp\Downloads\financial_analysis_output.xlsx"
    
    print("=" * 60)
    print("MULTI-COMPANY FINANCIAL ANALYSIS TOOL WITH Z-SCORE")
    print("=" * 60)
    
    try:

        print("\n📂 Loading Excel file...")

        xl = pd.ExcelFile(input_path)
        sheet_names = xl.sheet_names
        print(f"Found sheets: {sheet_names}")
        

        if len(sheet_names) >= 3:
            income_raw = pd.read_excel(input_path, sheet_name=sheet_names[0])
            balance_raw = pd.read_excel(input_path, sheet_name=sheet_names[1])
            cash_raw = pd.read_excel(input_path, sheet_name=sheet_names[2])
        else:
            raise ValueError(f"Expected 3 sheets, found {len(sheet_names)}")
        
        print(f"\nIncome Statement shape: {income_raw.shape}")
        print(f"Balance Sheet shape: {balance_raw.shape}")
        print(f"Cash Flow shape: {cash_raw.shape}")
        

        

        print("\n🔄 Transforming data for all companies...")
        income_statements = prepare_statements_for_all_companies(income_raw)
        balance_sheets = prepare_statements_for_all_companies(balance_raw)
        cash_flows = prepare_statements_for_all_companies(cash_raw)
        

        all_companies = list(income_statements.keys())
        print(f"\n✅ Found {len(all_companies)} companies: {all_companies}")
        

        income_sheet = create_income_statement_sheet(income_statements)
        balance_sheet = create_balance_sheet_sheet(balance_sheets)
        cash_flow_sheet = create_cash_flow_sheet(cash_flows)

        all_ratios = []
        all_dupont = []
        all_zscores = []
        
        for company in all_companies:
            print(f"\n{'='*40}")
            print(f"Analyzing: {company}")
            print(f"{'='*40}")

            income_stmt = income_statements.get(company)
            balance_stmt = balance_sheets.get(company)
            cash_flow = cash_flows.get(company)
            
            if income_stmt is None or balance_stmt is None:
                print(f"  ⚠ Missing data for {company}, skipping...")
                continue
            

            analyzer = CompanyFinancialAnalyzer(income_stmt, balance_stmt, cash_flow, company)
            
            try:
                # Calculate ratios
                company_ratios = analyzer.calculate_financial_ratios()
                all_ratios.append(company_ratios)
                
                # Calculate DuPont analysis
                company_dupont = analyzer.dupont_analysis()
                company_dupont.insert(0, 'Company', company)
                all_dupont.append(company_dupont)
                
                # Calculate Z-Score
                company_zscore = analyzer.altman_z_score()
                company_zscore.insert(0, 'Company', company)
                all_zscores.append(company_zscore)
                
                print(f"  ✅ Successfully calculated all metrics for {company}")
                
            except Exception as e:
                print(f"  ❌ Error analyzing {company}: {str(e)}")
                import traceback
                traceback.print_exc()
                # Continue with other companies
        

        if not all_ratios:
            print("\n❌ No companies could be analyzed. Check your data.")
            return
        
        combined_ratios = pd.concat(all_ratios, axis=0)
        combined_dupont = pd.concat(all_dupont, axis=0)
        combined_zscores = pd.concat(all_zscores, axis=0)
        
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            
            income_sheet.to_excel(writer, sheet_name='Income_Statement', index=False)
            balance_sheet.to_excel(writer, sheet_name='Balance_Sheet', index=False)
            cash_flow_sheet.to_excel(writer, sheet_name='Cash_Flow', index=False)
            
            
            combined_ratios.to_excel(writer, sheet_name='All_Companies_Ratios', index=True)
            combined_dupont.to_excel(writer, sheet_name='All_Companies_DuPont', index=True)
            combined_zscores.to_excel(writer, sheet_name='All_Companies_Zscore', index=True)
        
        print("\n✅ Analysis completed successfully!")
        print(f"📄 Output saved to: {output_path}")

        
        print(f"📈 Ratios shape: {combined_ratios.shape}")
        print(f"📉 DuPont shape: {combined_dupont.shape}")
        
    except FileNotFoundError:
        print(f"\n❌ Error: File not found at {input_path}")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
