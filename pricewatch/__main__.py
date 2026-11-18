from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .runner import run, write_site_data
from .store import History


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pricewatch", description="Compare prices across stores.")
    p.add_argument("--watchlist", default="watchlist.json")
    p.add_argument("--history", default="data/history.jsonl")
    p.add_argument("--out", default="site/site.json")
    p.add_argument("--source", action="append", help="override sources (repeatable), e.g. --source sample")
    args = p.parse_args(argv)

    config = json.loads(Path(args.watchlist).read_text(encoding="utf-8"))
    history = History(Path(args.history))
    data = run(config, history, source_overrides=args.source)
    history.save()
    write_site_data(data, Path(args.out))

    for prod in data["products"]:
        s = prod["summary"]
        if s:
            print(f"{prod['name']}: lowest {s['lowest']['price']:,.0f} at {s['lowest']['store']}, "
                  f"median {s['median']:,.0f}, {s['store_count']} stores")
        else:
            print(f"{prod['name']}: no offers ({len(prod['errors'])} source errors)")
        for err in prod["errors"]:
            print(f"  ! {err['source']}: {err['error']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
