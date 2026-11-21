"""Build the curated Twin Falls precipitation supplement without changing SQLite."""
import argparse
import json
import sqlite3
from pathlib import Path

from idwx.precipitation import build_archive

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('source', type=Path)
parser.add_argument('--database', type=Path, default=Path('data/idwx.sqlite3'))
parser.add_argument('--output', type=Path, default=Path('data/precipitation/twin_falls.json'))
args = parser.parse_args()
with sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True) as connection:
    digest, = connection.execute("SELECT sha256 FROM source_files WHERE station_id='twin_falls'").fetchone()
archive = build_archive(args.source, expected_sha256=digest)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(archive, separators=(',', ':'), allow_nan=False) + '\n')
print(json.dumps({'daily_rows': len(archive['days']), 'content_sha256': archive['content_sha256'], 'output': str(args.output)}))
