"""Entry point for the SIFT demo."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SIFT scratch implementation demo")
    parser.add_argument("image1", help="Path to first image")
    parser.add_argument("image2", help="Path to second image")
    args = parser.parse_args(argv)

    # TODO: wire up pipeline after implementation
    print(f"Image 1: {args.image1}")
    print(f"Image 2: {args.image2}")
    print("Pipeline not yet implemented.")


if __name__ == "__main__":
    main()
