# Systematic Multi-Factor Credit Strategy

## Overview
This project implements an institutional-grade **"Alpha-Plus" systematic credit strategy** benchmarked against the **Bloomberg US Corporate Bond Index (LQD)**. 

Unlike traditional screening methods, this engine treats portfolio construction as a **Constrained Convex Optimization** problem. It isolates idiosyncratic alpha using **Ridge Regression** and maximizes risk-adjusted returns while strictly adhering to **Duration-Times-Spread (DTS) Neutrality** and **Key Rate Duration (KRD)** constraints.

## Key Features

### 1. Robust Alpha Generation
* **Ridge Regression (L2 Regularization):** Extracts "Fair OAS" by regressing spreads against Duration, Rating (Log-WARF), and Sector, solving the multi-collinearity problem inherent in credit data.
* **Multi-Factor Composite:** * **Value:** Residual OAS (Mispricing).
    * **Quality:** Leverage implied from Ratings.
    * **Momentum:** Price-to-Par deviation / Spread compression.
    * **Carry:** Raw OAS (Winsorized).

### 2. Microstructure Proxies (Data Engineering)
Due to the lack of real-time transaction data, this project engineers structural proxies:
* **Liquidity Cost:** Modeled using **Log-Market-Cap** via a power law to estimate Bid-Ask spreads (5bps for Jumbo issuers vs. 40bps for tail issuers).
* **Momentum:** Derived from **Price-to-Par** divergence to capture yield compression trends.

### 3. Transaction-Cost Aware Optimization
* **Objective Function:** Maximizes **Net Score** (Alpha Score minus Estimated Liquidity Penalty).
* **Risk Constraints:**
    * **DTS Neutrality:** Matches benchmark risk contribution per sector.
    * **Curve Immunization:** Uses maturity buckets (Short/Med/Long) as proxies for KRD to prevent curve twist risks.
    * **Constraint Handling:** Uses `scipy.optimize` (SLSQP) with a Heuristic Fallback mechanism to ensure production stability.

## Project Structure

```text
CorpBond-MultiFactor-Strategy/
├── data/
│   └── LQD_holdings.csv       # Input holdings data
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data cleaning & Proxy construction
│   ├── alpha_model.py         # Ridge Regression & Scoring logic
│   ├── optimizer.py           # Scipy SLSQP & Heuristic Fallback
│   └── analytics.py           # Performance Attribution & Visualization
├── main.py                    # Execution Entry Point
└── README.md                  # Project Documentation