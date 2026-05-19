#!/usr/bin/env python3
"""Collate the count results from micronuclAI."""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Collate the count results from micronuclAI."
    )
    _ = parser.add_argument(
        "dir",
        nargs="+",
        help="Directory containing micronuclAI count results",
    )
    _ = parser.add_argument(
        "--report",
        type=int,
        choices=[1, 2],
        default=1,
        metavar="n",
        help="""Report (default: %(default)s)

1: micronuclei count
2: micronuclei fraction
""",
    )
    parser.add_argument(
        "--print",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Print the report (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--out",
        help="Output filename",
    )
    args = parser.parse_args()

    import glob
    import os

    import pandas as pd

    data = []
    for fn in args.dir:
        if os.path.isfile(fn):
            if fn.endswith("_counts.csv"):
                df = pd.read_csv(fn)
                df["source"] = fn
                data.append(df)
        elif os.path.isdir(fn):
            for file in glob.glob(os.path.join(fn, "*_counts.csv")):
                # file = os.path.join(fn, file)
                df = pd.read_csv(file)
                df["source"] = fn
                data.append(df)
        else:
            raise RuntimeError("Not a file or directory: " + fn)

    df = pd.concat(data) if data else pd.DataFrame()

    if args.report == 1:
        df = df.groupby(["source", "micronuclei"], as_index=False).sum()
        if args.print:
            print(df.to_markdown(index=False, tablefmt="psql"))
        if args.out:
            df.to_csv(args.out, index=False)
    elif args.report == 2:
        df["mni"] = df["micronuclei"] > 0
        df = df.groupby(["source", "mni"], as_index=False).sum()
        df = df.pivot(index="source", columns="mni", values="count").fillna(0)
        df.rename(columns={True: "mni", False: "none"}, inplace=True)

        # Add fraction
        if "mni" not in df.columns:
            df["mni"] = 0
        if "none" not in df.columns:
            df["none"] = 0

        df["total"] = df["mni"] + df["none"]
        df["fraction"] = df["mni"] / df["total"]

        if args.print:
            print(df.to_markdown(index=True, tablefmt="psql"))
        if args.out:
            df.to_csv(args.out, index=True)

if __name__ == "__main__":
    main()
