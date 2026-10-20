# ONIOM campaign: repeats, box sizes, ML water shell

18 jobs (`jobs.txt`): cubic boxes of 4, 5 and 6 nm, two ML regions, three
seeds. Each job builds segetalin A in TIP3P, minimises and heats with MM
(50 ps NVT, velocities from the seed), then runs ONIOM with PET-MAD xs:
20 ps NpT (C-rescale, V-rescale 300 K) and 20 ps NVE, dt 0.5 fs,
rcoulomb 1.0 nm, rlist 1.55 nm (`templates/`).

| ML region | atoms | notes |
| --- | --- | --- |
| `peptide` | 87 | the peptide only |
| `shell` | ~435 | peptide + whole waters with any atom within 0.5 nm after heating (116 in the 5 nm box); their SETTLEs are removed, so they are flexible PET-MAD waters. The set is fixed for the run, so shell waters can diffuse away over 40 ps. The region is 2-3 nm wide, well beyond rcoulomb, so it relies on the embedded Coulomb correction. |

Binary: `gromacs-oniom-torch` (PR #11 + metatomic + the embedded Coulomb
correction, Torch build), PET-MAD on the GPU, 16 OpenMP threads.

## Running unattended

    ./install_cron.sh      # adds the cron entry and starts the runner now

Cron starts `runner.sh` every 15 minutes under `flock`, so only one runner
is ever active and the campaign resumes after a reboot or crash. The runner
works through `jobs.txt` in order; each stage is skipped once done and
mdrun continues from its checkpoint. A failed job is retried on the next
start, up to 3 tries, then skipped. After every job it rewrites
`summary.md` and `summary.csv`. When nothing is left, it removes its own
cron entry (tagged `# oniom-campaign`).

Expected time: ~30 min per peptide job, ~45 min per shell job, ~12 h total.

* progress: `runner.log`, `summary.md`, `jobs/<job>/job.log`
* stop: `crontab -l | grep -v oniom-campaign | crontab -` and kill `runner.sh`
* add jobs: append lines to `jobs.txt` (the entry must be reinstalled if the
  runner already finished)
* rerun a job: delete `jobs/<job>`

## Per-job metrics (`analyze.py` → `jobs/<job>/metrics.json`)

NVE drift (linear fit of the total energy) and residual std, NVE
temperature, NpT density (second half), number of ML atoms, the ring bond
(ALA6 C - GLY1 N) at the end, PET-MAD uncertainty warnings, and ms/step.
