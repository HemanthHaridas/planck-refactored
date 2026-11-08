"""Create the same fixed-seed, normal-verbosity input for both FCIQMC profilers."""
from pathlib import Path
import re
import sys

source = Path(sys.argv[1]).read_text()
seed = re.search(r'(?im)^([ \t]*)fciqmc_seed\b', source)
if not seed:
    raise SystemExit('fixture has no explicit fciqmc_seed')
profile = re.findall(r'(?im)^[ \t]*fciqmc_profile[ \t]+([^\s#]+)', source)
if not profile:
    source = source[:seed.start()] + seed[1] + 'fciqmc_profile         true\n' + source[seed.start():]
elif len(profile) != 1 or profile[0].lower() not in ('true', '.true.'):
    raise SystemExit('fixture explicitly sets a conflicting fciqmc_profile value')
source = re.sub(r'(?im)^([ \t]*verbosity[ \t]+)verbose\b', r'\g<1>normal', source)
Path(sys.argv[2]).write_text(source)
