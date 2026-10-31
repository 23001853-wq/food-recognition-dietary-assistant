"""Create seeded stratified manifests without modifying original test. Not implemented yet."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description='Create seeded stratified manifests without modifying original test')
    parser.parse_args()
    parser.exit(2, "Script not implemented yet; see docs/tasks.md.\n")


if __name__ == "__main__":
    main()
