# Phase 6 seeded lifecycle artifact

This directory publishes the aggregate-only output of an offline deterministic
seeded demonstration. It contains a self-contained HTML report and one
full-page screenshot of that local HTML file. It contains no request or
response text, request-level records, database, event export, or provider/API
payload capture.

## Reproduce

From the repository root, regenerate the report without credentials, a server,
or any network call:

```bash
uv run python -m costpilot.phase6 docs/artifacts/phase6/seeded-lifecycle-report.html
```

The runner reads the repository's fixed synthetic fixture in file order six
times, assigns `phase6-0001` through `phase6-0060`, applies fixed UTC
timestamps, uses `FakeProvider`, and stores its temporary local audit data only
for the duration of the process. Running the command again produces identical
HTML bytes.

The screenshot is a 1440x1753 full-page local capture. It was captured with
Chrome DevTools `Page.captureScreenshot` using `captureBeyondViewport: true`,
so it visibly includes every report section and the closing disclaimer:

```bash
google-chrome --headless --disable-gpu --hide-scrollbars --window-size=1440,1200 \
  --remote-debugging-port=9222 \
  "file://$PWD/docs/artifacts/phase6/seeded-lifecycle-report.html"
```

Then use `Page.captureScreenshot` with `format: "png"` and
`captureBeyondViewport: true` in the local Chrome DevTools session. No server
or network connection is involved.

## Interpretation boundary

**Offline deterministic seeded demonstration.** This artifact replays 60
request lifecycles from the repository's fixed synthetic prompt fixture using
`FakeProvider`, fixed constants, and fixed timestamps. It makes no network or
live-provider calls and incurs no provider spend. Any tokens, latency,
verification results, USD figures, or deltas shown are deterministic simulated
values, not measurements of real spend, answer quality, routing efficacy,
throughput, reliability, or realized savings. The routing dataset remains
`ai_drafted_pending_human_review`; its labels and derived metrics are not
real-world ground truth.

A real evaluation would need separately authorized and privacy-reviewed data,
human-reviewed labels, defined evaluation criteria, and an approved measurement
protocol. None of that evidence is produced by this artifact.
