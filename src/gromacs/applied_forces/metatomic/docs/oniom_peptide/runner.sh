#!/usr/bin/env bash
# Work through jobs.txt one job at a time. Started by cron every 15 minutes
# under flock, so at most one runner is active; a job that fails is retried on
# the next start, up to MAX_TRIES, then skipped. When no job is left, the
# runner removes its own cron entry.
set -uo pipefail
CAMPAIGN=$(cd "$(dirname "$0")" && pwd)
MAX_TRIES=${MAX_TRIES:-3}
PY=${PY:-/home/boittier/metawork/.venv/bin/python}
cd "$CAMPAIGN"
log() { echo "$(date -Is) $*"; }

pending=0
while read -r name box ml seed; do
    [[ -z $name || $name == \#* ]] && continue
    job=jobs/$name
    [[ -f $job/.analysis.done ]] && continue
    tries=$(cat "$job/.tries" 2>/dev/null || echo 0)
    if (( tries >= MAX_TRIES )); then continue; fi
    mkdir -p "$job"; echo $((tries + 1)) > "$job/.tries"
    log "start $name (box $box nm, ML $ml, seed $seed, try $((tries + 1)))"
    if ./run_job.sh "$name" "$box" "$ml" "$seed" >> "$job/job.log" 2>&1; then
        log "finished $name"
    else
        log "FAILED $name (see $job/job.log)"; pending=1
    fi
    "$PY" summarize.py > /dev/null 2>&1 || log "summary failed"
done < jobs.txt

if (( pending == 0 )); then
    log "no jobs left; removing the cron entry"
    ./finish.sh || log "finish.sh failed"
    crontab -l 2>/dev/null | grep -v "oniom-campaign" | crontab -
fi
