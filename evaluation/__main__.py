
if __name__ == "__main__":
    import sys
    from evaluation.cli import main

    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)
