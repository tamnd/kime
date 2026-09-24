#!/bin/sh
# Writes the language id data as JSON lines of {"text", "lang", "source", "split"}.
#   $1: MASSIVE parquet directory with <lang>-train.parquet (tools/route/massive.sh)
#   $2: MASSIVE test and validation directory with <lang>.parquet and <lang>-val.parquet
#   $3: directory with papluca_language-identification-{train,validation,test}.parquet and
#       fancyzhx_ag_news-{train,test}.parquet from the Hugging Face parquet API
#   $4: leetcode.sqlite with a questions table, whose statements are English technical text with
#       code in it (test only)
# AG News train rows 0 to 59,999 are for training, 60,000 to 79,999 for validation and
# 100,000 to 119,999 plus the test split for the English check.
set -eu
duckdb -json -c "
select text, regexp_replace(filename, '.*/|-train.parquet', '', 'g') as lang, 'massive' as source, 'train' as split
  from read_parquet('$1/*-train.parquet', filename=true)
union all
select text, regexp_replace(filename, '.*/|-val.parquet', '', 'g'), 'massive', 'val'
  from read_parquet('$2/*-val.parquet', filename=true)
union all
select text, regexp_replace(filename, '.*/|.parquet', '', 'g'), 'massive', 'test'
  from read_parquet('$2/*.parquet', filename=true) where filename not like '%-val.parquet'
union all
select text, labels, 'papluca', 'train' from '$3/papluca_language-identification-train.parquet'
union all
select text, labels, 'papluca', 'val' from '$3/papluca_language-identification-validation.parquet'
union all
select text, labels, 'papluca', 'test' from '$3/papluca_language-identification-test.parquet'
union all
select text, 'en', 'agnews', case when r < 60000 then 'train' when r < 80000 then 'val' else 'test' end
  from (select text, row_number() over () - 1 as r from '$3/fancyzhx_ag_news-train.parquet')
  where r < 80000 or r >= 100000
union all
select text, 'en', 'agnews', 'test' from '$3/fancyzhx_ag_news-test.parquet'
" | python3 -c '
import html, json, re, sqlite3, sys
for r in json.load(sys.stdin):
    print(json.dumps(r, ensure_ascii=False))
db = sqlite3.connect(sys.argv[1])
for (h,) in db.execute("select content_html from questions where length(content_html) > 0"):
    text = html.unescape(re.sub(r"<[^>]+>", " ", h))
    text = re.sub(r"[ \t]+", " ", text).strip()
    print(json.dumps({"text": text, "lang": "en", "source": "leetcode", "split": "test"}, ensure_ascii=False))
' "$4"
