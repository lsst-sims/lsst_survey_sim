# batch/ — Prenight Simulation Batch Scripts

This directory contains SLURM batch scripts that run nightly prenight
simulations at the USDF (SLAC S3DF) and manage the disk space they consume.

## Scripts

### run_prenight_sims.sh

Runs Simonyi (main telescope) prenight simulations.  Produces multiple
simulations per night with varying conditions (nominal, delayed start,
anomalous overhead, good/poor seeing), archives results to S3 and the
visit-sequence metadata database, and updates the prenight index.

### run_auxtel_prenight_sims.sh

Runs AuxTel (auxiliary telescope) prenight simulations.  Produces a single
nominal simulation, archives results, and updates the prenight index.

### cleanup_prenight.sh

Generates a human-reviewable cleanup script that archives and removes
completed work directories and their associated conda virtual environments.
Does NOT directly delete anything; the operator reviews and executes the
generated script.

## Quick Start

These scripts are designed to be submitted via SLURM, typically from a cron
job on `sdfcron001`.  The production crontab submits them daily:

    15 6 * * * /opt/slurm/slurm-curr/bin/sbatch /sdf/data/rubin/shared/scheduler/packages/lsst_survey_sim/batch/run_auxtel_prenight_sims.sh 2>&1 >> /sdf/data/rubin/shared/scheduler/prenight/daily/daily_auxtel_cron.out
    55 6 * * * /opt/slurm/slurm-curr/bin/sbatch /sdf/data/rubin/shared/scheduler/packages/lsst_survey_sim/batch/run_prenight_sims.sh 2>&1 >> /sdf/data/rubin/shared/scheduler/prenight/daily/daily_simonyi_cron.out

`cleanup_prenight.sh` is not run from cron; submit it manually when disk
space management is needed:

    bash cleanup_prenight.sh

To simulate a specific night (YYYYMMDD format):

    DAYOBS=20260715 sbatch run_prenight_sims.sh
    DAYOBS=20260715 sbatch run_auxtel_prenight_sims.sh

If DAYOBS is not set, the scripts default to the current observing day
(UTC date minus 12 hours).

## Prerequisites

- SLAC S3DF account with `rubin:developers` allocation
- Membership in `rubin_users` group
- Valid access token at `~/.lsst/usdf_access_token`
- AWS profile `prenight` configured for S3 archive access
- Gate file present (see "Cron Gate Mechanism" below)
- Conda available via `/sdf/group/rubin/sw/w_latest/loadLSST.sh`

## Cron Gate Mechanism

Each script checks for a "gate file" before running:

    /sdf/data/rubin/shared/scheduler/cron_gates/<script_name>/<username>

If the file is missing, the script exits immediately.  This allows any
scheduler group member to stop another member's cron job by deleting their
gate file, without needing access to their crontab.  See:

    /sdf/data/rubin/shared/scheduler/cron_gates/README.txt

## Key Paths

| Purpose               | Path                                                               |
|-----------------------|--------------------------------------------------------------------|
| Installed scripts     | /sdf/data/rubin/shared/scheduler/packages/lsst_survey_sim/batch/   |
| Simonyi work dirs     | /sdf/data/rubin/shared/scheduler/prenight/work/run_prenight_sims   |
| AuxTel work dirs      | /sdf/data/rubin/shared/scheduler/prenight/work/run_auxtel_prenight_sims |
| Conda venvs (scratch) | /sdf/scratch/users/<initial>/<user>/prenight_venvs                 |
| SLURM logs            | /sdf/data/rubin/shared/scheduler/prenight/sbatch/                  |
| Cron logs             | /sdf/data/rubin/shared/scheduler/prenight/daily/                   |
| Cleanup scripts       | /sdf/data/rubin/shared/scheduler/prenight/cleanup_scripts/         |
| S3 archive            | s3://rubin:rubin-scheduler-prenight/opsim/vseq/                    |

## Shared Access (ACLs)

All scripts grant POSIX ACL read/write access to the scheduler group
members on every file and directory they create.  This ensures that
any group member can debug, re-run, or clean up artifacts regardless
of which user's cron job created them.

## See Also

- batch/design.md — Detailed design documentation
- notebooks/lsst_eval.ipynb — Interactive simulation example
- /sdf/data/rubin/shared/scheduler/cron_gates/README.txt — Gate file docs
