import warnings
import pandas as pd
import os
from src.data_loader import LocalDataLoader
from src.alpha_model import MultiFactorModel
from src.optimizer import PortfolioOptimizer
from src.analytics import print_full_diagnostics, print_comparison, plot_dashboard, run_attribution

# --- Global Settings ---
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format)
warnings.filterwarnings('ignore')

def main():
    print("=== Systematic Multi-Factor Credit Strategy (Project Mode) ===")
    
    # 1. Data Loading & Proxy Construction
    print("\n[Step 1] Initializing Data Loader...")
    loader = LocalDataLoader()
    
    # Define paths: Try relative path first, fallback to your specific local path
    relative_path = os.path.join("data", "LQD_holdings.csv")
    absolute_path = r"D:/quant/CorpBondMultiFactorStrategy/data/LQD_holdings.csv"
    
    try:
        if os.path.exists(relative_path):
            df = loader.load_data(relative_path)
        else:
            print(f"Note: Relative path not found, attempting absolute path: {absolute_path}")
            df = loader.load_data(absolute_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Alpha Model Execution (Ridge Regression)
    print("\n[Step 2] Running Multi-Factor Alpha Model...")
    model = MultiFactorModel()
    df_scored = model.run_model(df)

    # 3. Portfolio Optimization (Scipy SLSQP)
    # Note: Selecting top 500 candidates to optimize computation speed
    print("\n[Step 3] Constructing Portfolio (Transaction-Cost Aware)...")
    optimizer = PortfolioOptimizer(n_select=500, risk_scalar=100)
    df_final = optimizer.construct(df_scored)
    
    # 4. Performance Attribution
    print("\n[Step 4] Running Performance Attribution...")
    df_final = run_attribution(df_final)

    # 5. Diagnostics & Visualization
    print("\n[Step 5] Generating Reports and Dashboard...")
    print_full_diagnostics(df_final)
    print_comparison(df_final)
    plot_dashboard(df_final)

if __name__ == "__main__":
    main()