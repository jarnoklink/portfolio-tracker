import pandas as pd
import numpy as np
import yfinance as yf
import os
from hmmlearn.hmm import GaussianHMM

class Portfolio:
    def __init__(self, cache_dir="data_cache", use_cache_only=False):
        self.assets = pd.DataFrame({"Ticker": pd.Series(dtype="str"), "Sector": pd.Series(dtype="str"), "Asset Class": pd.Series(dtype="str"), "Quantity": pd.Series(dtype="float"), "Purchase Price": pd.Series(dtype="float")})
        self.transactions = pd.DataFrame({"Ticker": pd.Series(dtype="str"), "Type": pd.Series(dtype="str"), "Quantity": pd.Series(dtype="float"), "Price": pd.Series(dtype="float"), "Value": pd.Series(dtype="float")})
        self.cache_dir = cache_dir
        self.use_cache_only = use_cache_only

    def filter_by_period(self, df, period):
        if df.empty:
            return df
        end_date = df.index.max()
        if period == "1d":
            start_date = end_date - pd.Timedelta(days=1)
        elif period == "5d":
            start_date = end_date - pd.Timedelta(days=5)
        elif period == "1mo":
            start_date = end_date - pd.DateOffset(months=1)
        elif period == "3mo":
            start_date = end_date - pd.DateOffset(months=3)
        elif period == "6mo":
            start_date = end_date - pd.DateOffset(months=6)
        elif period == "1y":
            start_date = end_date - pd.DateOffset(years=1)
        elif period == "2y":
            start_date = end_date - pd.DateOffset(years=2)
        elif period == "5y":
            start_date = end_date - pd.DateOffset(years=5)
        elif period == "10y":
            start_date = end_date - pd.DateOffset(years=10)
        elif period == "ytd":
            start_date = pd.Timestamp(end_date.year, 1, 1)
        elif period == "max":
            return df
        else:
            return df  # fallback if period is invalid
        return df[df.index >= start_date]
    
    def validate_ticker(self, ticker):
        if self.use_cache_only:
            path = os.path.join(self.cache_dir, f"{ticker}_10yhistory.csv")
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    if not df.empty:
                        return True, f"Ticker {ticker} found in cache"
                    else:
                        return False, f"Ticker {ticker} cache file is empty"
                except Exception as e:
                    return False, f"Error reading cached data for {ticker}: {str(e)}"
            else:
                return False, f"Ticker {ticker} not found in cache. Available tickers: check data_cache folder"
        else:
            try:
                test_ticker = yf.Ticker(ticker)
                info = test_ticker.info
                if 'symbol' in info or 'shortName' in info or 'longName' in info:
                    return True, f"Ticker {ticker} validated successfully"
                else:
                    return False, f"Ticker {ticker} not found on Yahoo Finance"
            except Exception as e:
                return False, f"Error validating {ticker}: Unable to fetch data from Yahoo Finance"

    def add_assets(self, ticker, sector, asset_class, quantity, purchase_price):
        valid, message = self.validate_ticker(ticker)
        if not valid:
            return False, message
        value = quantity * purchase_price
        if ticker in self.assets["Ticker"].values:
            row = self.assets[self.assets["Ticker"] == ticker].index[0]
            existing_qty = self.assets.at[row, "Quantity"]
            existing_price = self.assets.at[row, "Purchase Price"]
            total_qty = existing_qty + quantity
            weighted_price = ((existing_qty * existing_price) + (quantity * purchase_price)) / total_qty
            self.assets.at[row, "Quantity"] = total_qty
            self.assets.at[row, "Purchase Price"] = weighted_price
        else:
            new_row = {"Ticker": ticker, "Sector": sector, "Asset Class": asset_class, "Quantity": quantity, "Purchase Price": purchase_price}
            self.assets = pd.concat([self.assets, pd.DataFrame([new_row])], ignore_index=True)
        new_transaction = {"Ticker": ticker, "Type": "Buy", "Quantity": quantity, "Price": purchase_price, "Value": value}
        self.transactions = pd.concat([self.transactions, pd.DataFrame([new_transaction])], ignore_index=True)
        return True, "Asset added succesfully"

    def _get_cached_price(self, ticker):
        path = os.path.join(self.cache_dir, f"{ticker}_10yhistory.csv")
        try:
            df = pd.read_csv(path)
            return df["Close"].dropna().iloc[-1]
        except Exception as e:
            return None

    def _get_cached_prices(self, tickers, period="10y"):
        all_prices = pd.DataFrame()
        for ticker in tickers:
            path = os.path.join(self.cache_dir, f"{ticker}_10yhistory.csv")
            try:
                df = pd.read_csv(path, parse_dates=["Date"])
                df.set_index("Date", inplace=True)
                df = self.filter_by_period(df, period)
                all_prices[ticker] = df["Close"]
            except Exception as e:
                pass
        return all_prices    

    def get_prices(self, tickers, period="10y"):
        if isinstance(tickers, str):
            tickers = [tickers]
        if self.use_cache_only:
            return self._get_cached_prices(tickers, period)
        else:
            try:
                data = yf.download(tickers, period=period)
                if isinstance(data.columns, pd.MultiIndex):
                    return data["Close"]
                else:
                    return data["Close"]
            except Exception as e:
                return self._get_cached_prices(tickers, period)
        
    def get_current_prices(self, tickers=None):
        if tickers is None:
            tickers = self.assets["Ticker"].unique().tolist()
        results = {}
        for ticker in tickers:
            if self.use_cache_only:
                price = self._get_cached_price(ticker)
                price = round(price, 2)
            else:
                try:
                    fast = yf.Ticker(ticker).fast_info
                    price = fast.get("last_price", None)
                except Exception as e:
                    price=self._get_cached_price(ticker)
            results[ticker] = price
        return results
    
    def get_historical_prices(self, tickers, period="10y", normalize=False):
        prices = self.get_prices(tickers, period=period)
        if prices.empty:
            return pd.DataFrame()
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0] if isinstance(tickers, list) else tickers)
        if normalize:
            prices = prices/prices.iloc[0]
        return prices

    def get_portfolio_data(self):
        if self.assets.empty:
            return None
        df = self.assets.copy()
        tickers = df["Ticker"].tolist()
        price_df = self.get_prices(tickers, period="5d")
        latest_price = price_df.iloc[-1] if not price_df.empty else pd.Series(dtype="float64")
        df["Current Price"] = df["Ticker"].map(latest_price)
        df["Transaction Value"] = df["Quantity"] * df["Purchase Price"]
        df["Current Value"] = df["Quantity"] * df["Current Price"]
        df["Unrealized PnL"] = df["Current Value"] - df["Transaction Value"]
        df["weight"] = df["Current Value"] / df["Current Value"].sum()
        df["Current Price"] = df["Current Price"].round(2)
        df["Transaction Value"] = df["Transaction Value"].round(2)
        df["Current Value"] = df["Current Value"].round(2)
        df["Unrealized PnL"] = df["Unrealized PnL"].round(2)
        df["Unrealized Return"] = (((df["Current Value"] / df["Transaction Value"]) - 1)*100).round(2).astype(str) + "%"
        df["weight"] = (df["weight"] * 100).round(2).astype(str) + "%"
        total_unrealized_PnL = df["Unrealized PnL"].sum().round(2)
        total_return = ((total_unrealized_PnL / df["Transaction Value"].sum())*100).round(2).astype(str) + "%"
        return {"portfolio_df": df, "total_unrealized_pnl": total_unrealized_PnL, "total_return": total_return}
        
    def get_realized_pnl(self):
        realized_PnL = 0.0
        sell_transaction = self.transactions[self.transactions["Type"].str.upper() == "SELL"]
        for _, row in sell_transaction.iterrows():
            ticker = row["Ticker"]
            qty_sold = row["Quantity"]
            sell_price = row["Price"]
            buys = self.transactions[(self.transactions["Ticker"] == ticker) & (self.transactions["Type"].str.upper() == "BUY")]
            total_bought = buys["Quantity"].sum()
            if total_bought == 0:
                continue
            avg_buy_price = (buys["Quantity"] * buys["Price"]).sum() / total_bought
            realized_PnL += (sell_price - avg_buy_price) * qty_sold
        return round(realized_PnL, 2)

    def sell_asset(self, ticker, quantity, sell_price):
        if ticker not in self.assets["Ticker"].values:
            return False, f"Ticker {ticker} not found in portfolio."
        asset_row = self.assets[self.assets["Ticker"] == ticker].iloc[0]
        current_qty = asset_row["Quantity"]
        if quantity > current_qty:
            return False, f"Cannot sell {quantity} of {ticker}. You only own {current_qty} of {ticker}."
        new_qty = current_qty - quantity
        self.assets.loc[self.assets["Ticker"] == ticker, "Quantity"] = new_qty
        if new_qty == 0:
            self.assets=self.assets[self.assets["Ticker"] != ticker]
        value = quantity * sell_price
        new_transaction = {"Ticker": ticker, "Type": "Sell", "Quantity": quantity, "Price": sell_price, "Value": value}
        self.transactions = pd.concat([self.transactions, pd.DataFrame([new_transaction])], ignore_index=True)
        return True, f"Sold {quantity} of {ticker} at {sell_price} each. Total value of transaction: {value:.2f}"

    def get_transactions(self):
        return self.transactions

    def get_portfolio_summary(self, group_by="Sector"):
        if self.assets.empty:
            return None
        df = self.assets.copy()
        tickers = df["Ticker"].tolist()
        price_df = self.get_prices(tickers, period="5d")
        latest_price = price_df.iloc[-1] if not price_df.empty else pd.Series(dtype="float64")
        df["Current Price"] = df["Ticker"].map(latest_price)
        df["Current Value"] = df["Quantity"] * df["Current Price"]
        df["Transaction Value"] = df["Quantity"] * df["Purchase Price"]
        summary = df.groupby(group_by)[["Current Value", "Transaction Value"]].sum().reset_index()
        total = summary["Current Value"].sum()
        summary["Unrealized Return"] = (((summary["Current Value"] / summary["Transaction Value"])-1)*100).round(2).astype(str) + "%"
        summary["Weight"] = (summary["Current Value"] / total * 100).round(2).astype(str) + "%"
        summary["Current Value"] = summary["Current Value"].round(2)
        summary["Transaction Value"] = summary["Transaction Value"].round(2)
        return summary

    def simulation_gbm(self, n_years=15, n_paths=100000, seed=None):
        if self.assets.empty:
            return None
        np.random.seed(seed)
        tickers = self.assets["Ticker"].tolist()
        quantities = self.assets["Quantity"].tolist()
        prices = self.get_prices(tickers, period="5y")
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
        prices = prices.dropna()
        log_returns = np.log(prices/prices.shift(1)).dropna()
        corr_matrix = log_returns.corr().values
        n_assets = len(tickers)
        mean_daily = log_returns.mean().values
        std_daily = log_returns.std().values
        mu_annual = mean_daily * 252
        sigma_annual = std_daily * np.sqrt(252)
        S0 = prices.iloc[-1].values
        L = np.linalg.cholesky(corr_matrix)
        Z_indep = np.random.standard_normal((n_paths, n_assets))
        Z_corr = Z_indep @ L.T
        T = n_years
        S_T = np.zeros_like(Z_corr)
        for i in range(n_assets):
            S_T[:, i] = S0[i] * np.exp((mu_annual[i] - 0.5 * sigma_annual[i]**2) * T + sigma_annual[i] * np.sqrt(T) * Z_corr[:, i])
        total_sim = np.dot(S_T, quantities)

        median_val = np.median(total_sim)
        p5 = np.percentile(total_sim, 5)
        p95 = np.percentile(total_sim, 95)

        current_prices = self.get_prices(tickers, period="5d").iloc[-1]
        current_value = np.dot(current_prices.values, quantities)
        median_return = (median_val/current_value) - 1
        median_annualized_return = ((median_val / current_value) ** (1/n_years)) - 1
        VaR_5 = current_value - p5
        CVaR_5 = current_value - np.mean(total_sim[total_sim <= p5])
        return {
            "current_value": current_value,
            "median_val": median_val,
            "p5": p5,
            "p95": p95,
            "median_return": median_return,
            "median_annualized": median_annualized_return,
            "VaR_5": VaR_5,
            "CVaR_5": CVaR_5
        }

    def simulate_regime_switching(self, n_years=15, n_paths=100000, n_states=2, seed=None):
        if self.assets.empty:
            return None
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        tickers = self.assets["Ticker"].tolist()
        quantities = np.array(self.assets["Quantity"].tolist(), dtype=float)
        prices = self.get_prices(tickers, period="10y")
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
        prices = prices.dropna()
        log_returns = np.log(prices/prices.shift(1)).dropna()
        if log_returns.empty:
            return None
        X = log_returns.values
        model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=500, random_state=seed, tol=1e-3, verbose=False)
        model.fit(X)
        trans_mat = model.transmat_
        start_prob = model.startprob_
        state_means = model.means_
        state_covs = model.covars_
        mu_states = state_means
        sigma_states = []
        chol_states = []
        for s in range(n_states):
            cov_daily = state_covs[s]
            std = np.sqrt(np.diag(cov_daily))
            std[std==0] = 1e-8
            corr = cov_daily / np.outer(std, std)
            L = np.linalg.cholesky(corr)
            sigma_states.append(std)
            chol_states.append(L)
        sigma_states = np.array(sigma_states)

        n_days = 252 * n_years
        n_assets = len(tickers)
        S0 = prices.iloc[-1].values.astype(float)
        total_sim = np.zeros(n_paths)
        Z = rng.standard_normal((n_days, n_paths, n_assets))
        regime_paths = np.zeros((n_days, n_paths), dtype=int)
        current_states = rng.choice(n_states, size=n_paths, p=start_prob)
        regime_paths[0] = current_states

        for day in range(1, n_days):
            for state in range(n_states):
                mask = current_states == state
                if mask.any():
                    current_states[mask] = rng.choice(n_states, size=mask.sum(), p=trans_mat[state])
            regime_paths[day] = current_states

        S = np.tile(S0, (n_paths, 1))

        for day in range(n_days):
            for state in range(n_states):
                mask = regime_paths[day] == state
                if not mask.any():
                    continue
                mu = mu_states[state]
                sigma = sigma_states[state]
                L = chol_states[state]
                Z_corr = Z[day, mask] @ L.T
                daily_increments = mu - 0.5 * sigma**2 + sigma * Z_corr
                S[mask] *= np.exp(daily_increments) 

        total_sim = S @ quantities

        median_val = np.median(total_sim)
        p5 = np.percentile(total_sim, 5)
        p95 = np.percentile(total_sim, 95)

        current_prices = self.get_prices(tickers, period="5d").iloc[-1]
        current_value = np.dot(current_prices.values, quantities)
        median_return = (median_val / current_value) - 1
        median_annualized_return = ((median_val / current_value) ** (1 / n_years)) - 1
        VaR_5 = current_value - p5
        CVaR_5 = current_value - np.mean(total_sim[total_sim <= p5])

        return {
            "current_value": current_value,
            "median_val": median_val,
            "p5": p5,
            "p95": p95,
            "median_return": median_return,
            "median_annualized": median_annualized_return,
            "VaR_5": VaR_5,
            "CVaR_5": CVaR_5
        }
                
    def reset_portfolio(self):
        self.assets = pd.DataFrame({"Ticker": pd.Series(dtype="str"), "Sector": pd.Series(dtype="str"), "Asset Class": pd.Series(dtype="str"), "Quantity": pd.Series(dtype="float"), "Purchase Price": pd.Series(dtype="float")})
        self.transactions = pd.DataFrame({"Ticker": pd.Series(dtype="str"), "Type": pd.Series(dtype="str"), "Quantity": pd.Series(dtype="float"), "Price": pd.Series(dtype="float"), "Value": pd.Series(dtype="float")})
        return True
    
    def get_tickers(self):
        return self.assets["Ticker"].unique().tolist()
    
    def is_empty(self):
        return self.assets.empty
    


    

    


        




    

