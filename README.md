import yfinance as yf
import pandas as pd
import numpy as np

# Step 1: Stock Risk Analysis
tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'JPM', 'V', 'PG']  # Replace with your desired stock tickers
portfolio_weights = 1 / len(tickers)

data = yf.download(tickers, period='3mo')
returns = data['Adj Close'].pct_change()

volatility = returns.std() * np.sqrt(252)  # Assuming 252 trading days in a year
beta_spy = returns.cov(returns['SPY']) / returns['SPY'].var()
beta_iwm = returns.cov(returns['IWM']) / returns['IWM'].var()
beta_dia = returns.cov(returns['DIA']) / returns['DIA'].var()

52_week_high = data['Adj Close'].rolling(window=252).max()
52_week_low = data['Adj Close'].rolling(window=252).min()
average_drawdown = (52_week_high - 52_week_low) / 52_week_high
maximum_drawdown = (52_week_high - 52_week_low) / 52_week_high.max()

total_return = (data['Adj Close'].iloc[-1] / data['Adj Close'].iloc[0]) - 1
annualized_total_return = ((1 + total_return) ** (1/10)) - 1

stock_risk_analysis = pd.DataFrame({
    'Ticker': tickers,
    'Portfolio Weight': portfolio_weights,
    'Annualized Volatility': volatility,
    'Beta (SPY)': beta_spy,
    'Beta (IWM)': beta_iwm,
    'Beta (DIA)': beta_dia,
    'Average Drawdown': average_drawdown,
    'Maximum Drawdown': maximum_drawdown,
    'Total Return': total_return,
    'Annualized Total Return': annualized_total_return
})

print("Stock Risk Analysis:")
print(stock_risk_analysis)

# Step 2: Portfolio Risk Analysis against ETFs
etf_tickers = ['SPY', 'IWM', 'DIA']  # Replace with desired ETF tickers

etf_data = yf.download(etf_tickers, period='10y')
etf_returns = etf_data['Adj Close'].pct_change()

correlation = returns.corrwith(etf_returns)
covariance = returns.cov(etf_returns)
tracking_errors = np.sqrt(np.diag(covariance))
risk_free_rate = 0.02  # Replace with the current risk-free rate

sharpe_ratio = (stock_risk_analysis['Annualized Total Return'] - risk_free_rate) / stock_risk_analysis['Annualized Volatility']
portfolio_volatility = volatility.mean()
etf_volatility = etf_returns.std() * np.sqrt(252)
volatility_spread = portfolio_volatility - etf_volatility

portfolio_risk_analysis = pd.DataFrame({
    'ETF Ticker': etf_tickers,
    'Correlation': correlation,
    'Covariance': covariance,
    'Tracking Errors': tracking_errors,
    'Sharpe Ratio': sharpe_ratio,
    'Volatility Spread': volatility_spread
})

print("\nPortfolio Risk Analysis:")
print(portfolio_risk_analysis)

# Step 3: Correlation Matrix
correlation_matrix = returns.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)
