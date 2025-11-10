from Controller.controller import Controller

def main():
    print("\nWelcome to Portfolio Tracker!")
    controller = Controller(use_cache_only=True)
    controller.run()


if __name__ == "__main__":
    main()
