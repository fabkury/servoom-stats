# servoom-stats

The data pipeline behind the statistics pages of [servoom.pages.dev](https://servoom.pages.dev/stats/):
an independent, read-only measurement of the public Divoom gallery. Unofficial and not
affiliated with Divoom.

This repository holds the collector, the aggregation, the workflows that run them, and
their output in [`data/`](data/). The pages that draw the data live in the
[servoom](https://github.com/fabkury/servoom) repository (`docs/public/stats/`), whose
build downloads `data/` from here. The design is documented in
[servoom/docs/community-stats](https://github.com/fabkury/servoom/tree/main/docs/community-stats).

## What runs

| Job | When | What it does |
|-----|------|--------------|
| `pulse` | hourly, 44 minutes of reading | Reads every upload of the past 30 days, the Popular ordering, and who gave each new like. About 1,300 requests, all its budget allows. |
| `refresh` | right after each pulse, on its own runner, until 2.5 minutes before the next pulse | Reads new artwork ids (the walk, which counts private and held uploads too), new comments and a sample of new accounts; every 4 hours it rebuilds `data/pulse/` and the pulse health lanes whether or not those reads fit. About 300 requests. |
| `probe` | daily | Reads every list with a long look-ahead, the category totals and the outcome of held uploads. About 3,000 requests. |
| `snapshot` | every two days | Reads the whole public catalog (about 1.5 million artworks), new files, vanished artworks and featured-artist profiles, then rebuilds `data/daily/`. About 66,000 requests, roughly 3.5 hours. The crawl is checkpointed to the raw repository every 20 minutes, and a run started within 6 hours of a failed one resumes from the checkpoint. |

Each job commits `data/` when it changed and asks Cloudflare to rebuild the site.

Divoom serves one source 30 requests a minute (since 2026-10-23). Every job reads within
a time budget and logs what the budget cut; the pulse and the refresh run one after the
other so they never share the minute, and the client slows itself down after a refusal
(`stats/api.py`) in case two runners share an address.

Every run also appends one row to `runs.parquet` on its state branch, and `stats/health.py`
turns the run logs and the rollups into `data/health/*.json`: for each collector (hourly
pulse, 4-hour refresh, snapshot, catalog reach), contiguous spans marked `ok`, `degraded`
or `missing` with the evidence (`gap`, `spread`, `partial`, `no_token`, `reach`,
`skipped`). The site shades those spans on every chart that draws from the collector and
shows the whole history on the overview page. Nothing in it is written by hand:
`config/health_seed.json` only lists the refresh times from before the run log existed.

## What starts the jobs

GitHub's own schedule for Actions proved unreliable (one scheduled run in the first eight
hours), so a small Cloudflare Worker in [`worker/`](worker/) starts the workflows through
the `workflow_dispatch` API: the pulse at :17 every hour, the probe at 02:27 UTC and the
snapshot at 05:43 UTC. The workflows keep cron lines at other minutes as a fallback. A
pulse exits at once when the previous one is under 50 minutes old, and a snapshot when the previous one is under
36 hours old (the catalog is read every two days, to halve the load on each polling
account), so extra triggers cost a few seconds.

The Worker holds one secret, `GITHUB_TOKEN`: a fine-grained personal access token limited
to this repository with the permission "Actions: read and write". Deploy with
`npx wrangler deploy` from `worker/`; set the secret with `npx wrangler secret put GITHUB_TOKEN`.

The collector only sends read commands (`stats/api.py` keeps the list) and never likes,
views or uploads anything. A failed request is retried with backoff for up to 15 minutes
(`API_OUTAGE_SECONDS`) before a job gives up, so a short server outage does not end a run.

## Layout

```
stats/api.py        throttled access to the Divoom API
stats/accounts.py   polling account pool: token reuse, health checks, rotation, capped registration
stats/rawrepo.py    git plumbing for the private raw-data repository
stats/pulse.py      hourly job
stats/refresh.py    the job after each pulse: id walk, comments, sign-ups, data/pulse/*.json
stats/probe.py      daily job: list survey, category totals, held uploads
stats/snapshot.py   whole-catalog job, every two days
stats/daily.py      data/daily/*.json
stats/files.py      artwork file features and hashes, avatars
stats/health.py     collection health lanes, data/health/*.json
config/             automated account ranges, artists who asked not to be listed
data/               published aggregates (JSON) and featured artists' avatars (WebP)
```

## What is public and what is not

`data/` holds aggregates only. Individual accounts are named only when Divoom itself
features them (Recommend picks, the expert list, an ambassador badge), and any figure
counted in accounts needs at least five of them. No artworks are stored here; the only
images are featured artists' avatars.

Raw observations (per-artwork counters, like events, account ids) and the polling
accounts live in a separate private repository. This repository reaches it with a
deploy key held as the Actions secret `RAW_REPO_DEPLOY_KEY`.

## Asking to be removed

If you are listed among the featured artists and would rather not be, open an issue.
Your account id is added to `config/excluded_artists.json` and your row, page and
avatar disappear at the next daily run.

## Running locally

```
pip install -r requirements.txt -r requirements-snapshot.txt
set RAW_REPO_URL=https://github.com/<you>/<your-raw-repo>.git
python -m stats.pulse        # PULSE_DAYS=1 for a quick run
python -m stats.refresh      # REFRESH_BUDGET_SECONDS=120 for a quick run
python -m stats.snapshot     # LIMIT_LISTS=16_1,5_4 to read only a few lists
```

`FORCE_REFRESH=1` makes the refresh rebuild `data/pulse/` at once; `STRICT=1` turns skipped
parts into errors.
