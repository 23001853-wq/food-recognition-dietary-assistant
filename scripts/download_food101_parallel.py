"""Compatibility CLI wrapper for the shared verified parallel archive downloader."""

import json

try:
    from scripts.download_food101 import run_download_cli
except ModuleNotFoundError:
    from download_food101 import run_download_cli


def main() -> None:
    """Run parallel Extract with the same approval, config and archive locks as sequential."""
    result = run_download_cli("parallel")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
