#!/bin/sh
# Writes the MASSIVE train utterances as JSON lines of {"lang", "text"}, from the parquet files
# of mteb/amazon_massive_intent in $1 (one <lang>-train.parquet per language).
set -eu
duckdb -json -c "select regexp_replace(filename, '.*/|-train.parquet', '', 'g') as lang, text from read_parquet('$1/*-train.parquet', filename=true) order by lang, id" |
  python3 -c 'import json,sys; [print(json.dumps(r, ensure_ascii=False)) for r in json.load(sys.stdin)]'
