from Model.portfolio import Portfolio
from View.cli_view import CLIView


class Controller:    
    def __init__(self, use_cache_only=True):
        self.model = Portfolio(use_cache_only=use_cache_only)
        self.view = CLIView()

    def run(self):
        if self.model.use_cache_only:
            self.view.show_message("Running in cache-only mode (using local data)")
        else:
            self.view.show_message("Running in online mode (fetching from Yahoo Finance)")
        while True:
            self.view.show_menu()
            choice = self.view.get_input("Choose option")
            if choice == "1":
                self.add_asset()
            elif choice == "2":
                self.view_prices()
            elif choice == "3":
                self.view_current_portfolio()
            elif choice == "4":
                self.sell_asset()
            elif choice == "5":
                self.view_transactions()
            elif choice == "6":
                self.view_portfolio_summary()
            elif choice == "7":
                self.run_simulation()
            elif choice == "8":
                self.toggle_data_source()
            elif choice == "9":
                self.reset_portfolio()
            elif choice == "0":
                self.view.show_message("Goodbye!")
                break
            else:
                self.view.show_error("Invalid option. Please try again.")

    def add_asset(self):
        ticker = self.view.get_input("Ticker").upper()
        sector = self.view.get_input("Sector")
        asset_class = self.view.get_input("Asset Class")
        quantity = self.view.get_float_input("Quantity")
        purchase_price = self.view.get_float_input("Purchase price")
        success, message = self.model.add_assets(ticker, sector, asset_class, quantity, purchase_price)
        if success:
            self.view.show_success(message)
            portfolio_data = self.model.get_portfolio_data()
            if portfolio_data:
                self.view.show_portfolio(portfolio_data)
        else:
            self.view.show_error(message)

    def view_prices(self):
        tickers = self.model.get_tickers()
        if not tickers:
            self.view.show_message("Your portfolio is empty. Add assets first.")
            return
        self.view.show_tickers(tickers)
        selected_input = self.view.get_input("Enter tickers to view (separate with spaces, or press Enter for all)")
        if selected_input:
            selected = [t.upper() for t in selected_input.split()]
        else:
            selected = tickers
        price_type = self.view.get_choice(
            "Do you want to see current price or historical prices?", ["current", "historical"])
        if price_type == "current":
            self._show_current_prices(selected)
        elif price_type == "historical":
            self._show_historical_prices(selected)

    def _show_current_prices(self, tickers):
        prices = self.model.get_current_prices(tickers)
        self.view.show_current_prices(prices)

    def _show_historical_prices(self, tickers):
        valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        display_type = self.view.get_choice("Do you want historical prices as table or as graph?", ["table", "graph"])
        period = self.view.get_choice(f"On what period?", valid_periods)
        if display_type == "table":
            prices = self.model.get_prices(tickers, period=period)
            self.view.show_historical_prices_table(prices)
        elif display_type == "graph":
            normalize_choice = self.view.get_choice("Do you want normalized prices?", ["True", "False"])
            normalize = (normalize_choice == "True")
            prices = self.model.get_historical_prices(tickers, period=period, normalize=normalize)
            self.view.plot_prices(prices, tickers, period=period, normalize=normalize)

    def view_current_portfolio(self):
        portfolio_data = self.model.get_portfolio_data()
        self.view.show_portfolio(portfolio_data)
        if not self.model.transactions.empty:
            realized_pnl = self.model.get_realized_pnl()
            self.view.show_realized_pnl(realized_pnl)

    def sell_asset(self):
        if self.model.is_empty():
            self.view.show_message("Your portfolio is empty. Add assets first.")
            return
        ticker = self.view.get_input("Ticker").upper()
        quantity = self.view.get_float_input("Quantity")
        sell_price = self.view.get_float_input("Price per unit")
        success, message = self.model.sell_asset(ticker, quantity, sell_price)
        if success:
            self.view.show_success(message)
        else:
            self.view.show_error(message)

    def view_transactions(self):
        transactions = self.model.get_transactions()
        self.view.show_transactions(transactions)

    def view_portfolio_summary(self):
        if self.model.is_empty():
            self.view.show_message("Your portfolio is empty. Add assets first.")
            return
        group_by = self.view.get_choice("Group by 'Sector' or 'Asset Class'?", ["Sector", "Asset Class"]).title()
        summary = self.model.get_portfolio_summary(group_by=group_by)
        self.view.show_portfolio_summary(summary, group_by)

    def run_simulation(self):
        if self.model.is_empty():
            self.view.show_message("Your portfolio is empty. Add assets first.")
            return
        self.view.show_simulation_menu()
        simulation_method = self.view.get_choice("Enter 1 or 2", ["1", "2"])
        n_years = self.view.get_int_input("Enter number of years to simulate (default = 15)", default=15)
        n_paths = self.view.get_int_input("Enter number of simulation paths (default = 100000)", default=100000)
        self.view.show_message(f"Running simulation with {n_paths} paths over {n_years} years...")
        if simulation_method == "1":
            results = self.model.simulation_gbm(n_years=n_years, n_paths=n_paths)
            if results:
                self.view.show_simulation_results(results, title="Correlated GBM Portfolio Simulation Results")
            else:
                self.view.show_error("Simulation failed. Please check your data.")
        
        elif simulation_method == "2":
            results = self.model.simulate_regime_switching(n_years=n_years, n_paths=n_paths)
            if results:
                self.view.show_simulation_results(results, title="Regime-Switching GBM Portfolio Simulation Results")
            else:
                self.view.show_error("Simulation failed. Please check your data.")

    def reset_portfolio(self):
        if self.view.get_yes_no("Are you sure you want to reset the portfolio?"):
            self.model.reset_portfolio()
            self.view.show_success("Portfolio has been reset.")
        else:
            self.view.show_message("Reset cancelled.")

    def toggle_data_source(self):
        current_mode = "Cache-only" if self.model.use_cache_only else "Online"
        new_mode = "Online" if self.model.use_cache_only else "Cache-Only"
        self.view.show_message(f"Current mode: {current_mode}")
        if self.view.get_yes_no(f"Switch to {new_mode} mode?"):
            self.model.use_cache_only = not self.model.use_cache_only
            self.view.show_success(f"Switched to {new_mode} mode")
            if new_mode == "CACHE-ONLY":
                self.view.show_message("Note: Only cached tickers will be available")
            else:
                self.view.show_message("Note: Will fetch live data from Yahoo Finance (may be slow)")