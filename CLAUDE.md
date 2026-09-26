# Notes for Claude

## Keep the project diary

`DIARY.md` is organised **by day**, so the user can see what state the project
was in on any given day. After every piece of work on this project (a fix, a
feature, a measurement run, a cleanup):

1. **Add it as one row of today's step table** (`| # | 🧩 Problem | 🔧 Fix |
   📈 Result |`) in today's `📅 <date>` section at the bottom of `DIARY.md`. If
   today has no section yet, create one from the template at the end of the
   file.
2. **Update that day's "State at the end of the day" table.**
3. **Update the two sections at the top of the file:** *🧭 Where the project
   stands* (status table plus key numbers, with the "last updated" date) and
   *🗓️ Timeline at a glance* (one row per day).

Keep it easy to scan and **short**: one short phrase per cell, a failed attempt
folded into the Fix cell (`❌ tried X → why it failed`), ✅ ⚠️ ❌ markers. The
user found 600 lines too long, then 300; the whole diary was cut to ~160 lines
on 23 Sep 2026. A day should fit on one screen. Detail belongs in
`UPGRADE_NOTES.md` and commit messages, not here.

Take the numbers from `tools/run_report.py` / `reports/run-log.json` or from
`tools/evaluate.py`. Don't estimate them. If something was not run or not
verified, say so. If new work shows an earlier entry was wrong, add a ✏️
*Correction* note under that entry instead of rewriting it.

Also add the change to `CHANGELOG.md`.
