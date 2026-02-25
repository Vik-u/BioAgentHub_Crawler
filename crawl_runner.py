"""Unified entrypoint for simple or agentic crawl."""
from __future__ import annotations

import argparse
import sys

import agentic_crawl
import simple_crawl


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--agentic", action="store_true", help="Use agentic crawler.")
    parser.add_argument("--mode", choices=["simple", "agentic"], help="Select crawler mode.")
    args, rest = parser.parse_known_args()

    mode = "agentic"
    if args.mode == "simple":
        mode = "simple"
    elif args.mode == "agentic" or args.agentic:
        mode = "agentic"
    sys.argv = [sys.argv[0]] + rest
    if mode == "agentic":
        agentic_crawl.main()
    else:
        simple_crawl.main()


if __name__ == "__main__":
    main()
