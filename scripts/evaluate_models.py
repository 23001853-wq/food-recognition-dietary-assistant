"""Evaluate frozen candidates on the same locked test manifest. Not implemented yet."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate frozen candidates on the same locked test manifest')
    parser.parse_args()
    parser.exit(2, "Script not implemented yet; see docs/tasks.md.\n")


if __name__ == "__main__":
    main()
