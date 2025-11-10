import matplotlib.pyplot as plt

class CLIView:
    @staticmethod
    def show_menu():
        print("\n" + "="*50)
        print("Portfolio Tracker")
        print("1. Add asset")
        print("2. Show historical prices and plot")
        print("3. Show current portfolio")
        print("4. Sell asset")
        print("5. View transaction history")
        print("6. View portfolio breakdown by sector or asset class")
        print("7. Simulate portfolio")
        print("8. Toggle data source (online/cache)")
        print("9. Reset portfolio")
        print("0. Exit")
        print("="*50)

    @staticmethod
    def show_message(message):
        print(f"\n{message}")

    @staticmethod
    def show_error(error_message):
        print(f"\nError: {error_message}")

    @staticmethod
    def show_success(success_message):
        print(f"\n{success_message}")

    @staticmethod
    def show_current_prices(price_dict):
        print("\n Current Prices")
        for ticker, price in price_dict.items():
            if price is not None:
                print(f"{ticker}: {price:.2f}")
            else:
                print(f"{ticker}: Price unavailable")
    
    @staticmethod
    def show_historical_prices_table(prices_df):
        if prices_df.empty:
            print("\nNo price data available")
        else:
            print("\nHistorical Prices")
            print(prices_df.to_string())

    @staticmethod
    def show_portfolio(portfolio_data):
        if portfolio_data is None:
            print("\nNo current holdings.")
            return
        df = portfolio_data["portfolio_df"]
        total_pnl = portfolio_data["total_unrealized_pnl"]
        total_return = portfolio_data["total_return"]
        print("\nCurrent Portfolio")
        print(df.to_string(index=False))
        print(f"\nTotal unrealized profit/loss: {total_pnl} ({total_return})")
        
    @staticmethod
    def show_realized_pnl(realized_pnl):
        print(f"Total realized profit/loss: {realized_pnl}")

    @staticmethod
    def show_transactions(transactions_df):
        if transactions_df.empty:
            print("\nNo transactions yet.")
        else:
            print("\nTransaction History")
            print(transactions_df.to_string(index=False))

    @staticmethod
    def show_portfolio_summary(summary_df, group_by):
        if summary_df is None:
            print(f"\nNo current holdings to summarize by {group_by}.")
        else:
            print(f"\nPortfolio breakdown by {group_by}")
            print(summary_df.to_string(index=False))
            
    @staticmethod
    def show_simulation_results(results, title="Portfolio Simulation Results"):
        if not results:
            print("No results to display.")
            return
        print(f"\n{title}")
        display_order = [("current_value", "Current Portfolio Value", False),
            ("median_val", "Median Future Value", False),
            ("p5", "5th Percentile (Worst Case)", False),
            ("p95", "95th Percentile (Best Case)", False),
            ("median_return", "Median Total Return", True),
            ("median_annualized", "Median Annualized Return", True),
            ("VaR_5", "Value at Risk (5%)", False),
            ("CVaR_5", "Conditional VaR (5%)", False),]
        for key, label, is_percentage in display_order:
            if key in results:
                value = results[key]
                if is_percentage:
                    print(f"{label}: {value:.2%}")
                else:
                    print(f"{label}: {value:,.2f}")

    @staticmethod
    def plot_prices(prices, tickers, period="10y", normalize=False):
        if prices.empty:
            print(f"\nNo data available for {tickers}.")
            return
        prices.plot(figsize=(12,6))
        title = f"{'Normalized ' if normalize else ''}Historical Closing Prices ({period})"
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Normalized Price" if normalize else "Price", fontsize=12)
        plt.legend(title="Ticker")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def get_input(prompt):
        return input(f"{prompt}: ").strip()
    
    @staticmethod
    def get_float_input(prompt):
        while True:
            try:
                value = float(input(f"{prompt}: ").strip())
                return value
            except ValueError:
                print("Invalid input. Please enter a number.")
            
    @staticmethod
    def get_int_input(prompt, default=None):
        while True:
            try:
                user_input = input(f"{prompt}: ").strip()
                if not user_input and default is not None:
                    return default
                return int(user_input)
            except ValueError:
                print("Invalid input. Please enter a whole number.")
    
    @staticmethod
    def get_choice(prompt, valid_choices):
        while True:
            choice = input(f"{prompt} {valid_choices}: ").strip()
            if choice in valid_choices:
                return choice
            print(f"Invalid choice. Please select from: {valid_choices}")
        
    @staticmethod
    def get_yes_no(prompt):
        while True:
            response = input(f"{prompt} (yes/no): ").strip().lower()
            if response in ['yes', 'y']:
                return True
            elif response in ['no', 'n']:
                return False
            print("Please enter 'yes' or 'no'.") 

    @staticmethod
    def show_tickers(tickers):
        print(f"\nYour portfolio tickers: {', '.join(tickers)}")

    @staticmethod
    def show_simulation_menu():
        print("\nChoose Simulation Method")
        print("1. Correlated GBM Portfolio Simulation")
        print("2. Regime-Switching GBM Portfolio Simulation")
                    


    
