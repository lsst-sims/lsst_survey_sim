  # Prenight Simulation Batch Scripts — Design

## Overview

The prenight simulation system produces nightly scheduler simulations for the
Rubin Observatory's Simonyi Survey Telescope and Auxiliary Telescope (AuxTel).
These simulations help observers and the scheduling team anticipate what the
scheduler will do on upcoming nights under various conditions.

The system is composed of three batch scripts:

1. **`run_prenight_sims.sh`** — Simonyi telescope simulations
2. **`run_auxtel_prenight_sims.sh`** — AuxTel simulations
3. **`cleanup_prenight.sh`** — Disk space management

All scripts are designed to run as SLURM batch jobs on SLAC S3DF and
are typically triggered by cron.

---

## Architecture

### Data Flow

1. Fetch completed visits from the consolidated database (consdb)
2. Build a scheduler instance from the current configuration
3. Run forward simulations from the current survey state
4. Archive simulation outputs (opsim.db, rewards.h5) to S3
5. Record metadata and statistics in the PostgreSQL metadata database
6. Update the prenight index so downstream consumers (schedview, Times Square)
   can discover the new simulations

---

## Script Design Details

### run_prenight_sims.sh

#### Purpose

Simulates multiple nights of Simonyi telescope observations under several
scenarios that vary conditions such as seeing, start-time delay, overhead
multiplier, and whether detailed reward data is recorded.

#### Execution Phases

1. **Gate check** — Verify the cron gate file exists
2. **Group switch** — Re-execute under `rubin_users` group via `sg`
3. **Date computation** — Determine DAYOBS (today or from environment)
4. **Preflight** — Verify commands, paths, disk space, and ACLs
5. **Working directory** — Create a timestamped directory under the work root;
   wait and retry if a collision occurs
6. **Environment setup** — Create a fresh conda environment on scratch,
   install `lsst_survey_sim` from the configured git reference
7. **Configuration** — Clone `ts_config_scheduler` at the `develop` branch
8. **Fetch visits** — Query consdb for all completed visits through last night
9. **Build inputs** — Create scheduler pickle, model observatory, and band
   scheduler
10. **Simulations** — Run each scenario via `run_and_archive_sim`, archiving
    outputs and updating the index incrementally
11. **Completion marker** — Touch `.done` to signal that this work directory
    is eligible for cleanup

#### Error Handling

- `set -euo pipefail` ensures any failure aborts the script
- An `EXIT` trap logs the final status
- The prenight index is updated after each successful simulation so partial
  results are visible even if later simulations fail

---

### run_auxtel_prenight_sims.sh

#### Purpose

Simulates multiple nights of AuxTel observations under nominal conditions.
AuxTel simulations start from an empty visit history (no prior completed
visits are relevant).

#### Key Differences from Simonyi

- Uses `ideal_model_observatory` instead of `make_model_observatory`
- Does not use `make_band_scheduler`
- Passes `--telescope auxtel` when recording metadata
- Runs a single simulation scenario (nominal, with rewards)
- Separate work root to avoid directory collisions with Simonyi runs
- Separate SLURM output file prefix

---

### cleanup_prenight.sh

#### Purpose

Manages disk space consumed by completed simulation runs.  Rather than
directly deleting files, it **generates a shell script** containing explicit
`tar`, `rm`, and `mv` commands.  A human operator reviews and executes the
generated script.

#### Design Rationale

Automated deletion of shared data carries risk.  The two-phase approach
(generate then review) provides:

- **Auditability** — Every deletion is recorded in the generated script
- **Safety** — A human verifies the commands before execution
- **Recoverability** — Work directories are archived to `.tgz` before removal

#### Generated Script Contents

1. **Work directory archival** — For each completed work directory (containing
   a `.done` marker), emit commands to:
   - `tar -czf` the work directory
   - `tar -czf` and `rm -r` the associated conda venv
   - `rm -r` the work directory (after removing large git pack files first)

2. **Conditional archive offload** — If free space on the work filesystem is
   below 10 GiB, emit `mv` commands to relocate `.tgz` files older than 30
   days to a secondary storage location.

3. **Conditional venv archive cleanup** — If free space on the scratch
   filesystem is below 10 GiB, emit `rm` commands for venv `.tgz` archives
   older than 30 days.

#### Safety Guards

- Only processes directories whose names match the expected timestamp format
  (`YYYY-MM-DDTHHMMSS`)
- Only processes venvs whose names match the expected pattern
  (`prenight-YYYY-MM-DDTHHMMSS-XXXXXX`)
- Verifies that venv symlink targets are under the expected root
- Uses `--one-file-system` on all `rm -r` commands to prevent crossing
  filesystem boundaries
- Skips the cron gate check when run interactively (stdin is a terminal)

---

## Shared Design Patterns

### Cron Gate Mechanism

Each script checks for a sentinel file at:

```
/sdf/data/rubin/shared/scheduler/cron_gates/<script_name>/<username>
```

If the file is absent, the script exits with a message.  This allows any
scheduler group member to halt another user's cron-triggered job by deleting
the gate file — useful when the cron owner is unavailable but a job needs to
be stopped.

### Group Switching

Scripts re-execute themselves under the `rubin_users` group using:

```bash
exec sg rubin_users -c "$(printf '%q ' "$0" "$@")"
```

This ensures all created files have the correct group ownership regardless of
the invoking user's primary group.

### ACL Management

All created files and directories receive POSIX ACL entries granting `rwX`
access to each scheduler group member.  The `run_prenight_sims.sh` preflight
additionally verifies that the work root has the correct default ACLs, failing
early if they are misconfigured.

### Working Directory Isolation

Each run creates a new timestamped directory.  If a name collision occurs (two
jobs starting in the same second), the script waits and retries.  This ensures
concurrent runs never interfere with each other.

### Environment Reproducibility

Each run creates a dedicated conda environment on scratch and installs
`lsst_survey_sim` from a pinned git reference.  The environment specification
is hashed and recorded in the metadata database (`conda_env_sha256`), enabling
exact reproduction of any past simulation.

### Completion Markers

Simulation scripts touch a `.done` file in the work directory upon successful
completion.  The cleanup script only processes directories containing this
marker, ensuring in-progress or failed runs are never cleaned up automatically.

---

## Configuration

### Key Constants

| Variable | Script | Purpose |
|----------|--------|---------|
| `LSST_SURVEY_SIM_REFERENCE` | sim scripts | Git ref for lsst_survey_sim (default: `main`; empty = latest semver tag) |
| `TS_CONFIG_SCHEDULER_REFERENCE` | sim scripts | Git branch/ref for scheduler config (default: `develop`) |
| `SIM_NIGHTS` | sim scripts | Number of nights to simulate (default: 3) |
| `SCHEDULER_GROUP_USERS` | all | Users granted ACL access |
| `MIN_WORK_FREE_KB` | cleanup | Free-space threshold for archive offload (10 GiB) |
| `MIN_VENV_FREE_KB` | cleanup | Free-space threshold for venv archive cleanup (10 GiB) |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `DAYOBS` | Override the observing day (YYYYMMDD); if unset, computed from UTC clock |
| `AWS_PROFILE` | AWS credentials profile for S3 archive access (set to `prenight`) |
| `VSARCHIVE_PG*` | PostgreSQL connection parameters for the metadata database |

---

## Operational Procedures

### Starting Nightly Automation

1. Ensure gate files exist:
   ```bash
   touch /sdf/data/rubin/shared/scheduler/cron_gates/run_prenight_sims/$USER
   touch /sdf/data/rubin/shared/scheduler/cron_gates/run_auxtel_prenight_sims/$USER
   touch /sdf/data/rubin/shared/scheduler/cron_gates/cleanup_prenight/$USER
   ```

2. Add cron entries on `sdfcron001` (the current production schedule):
   ```
   15 6 * * * /opt/slurm/slurm-curr/bin/sbatch /sdf/data/rubin/shared/scheduler/packages/lsst_survey_sim/batch/run_auxtel_prenight_sims.sh 2>&1 >> /sdf/data/rubin/shared/scheduler/prenight/daily/daily_auxtel_cron.out
   55 6 * * * /opt/slurm/slurm-curr/bin/sbatch /sdf/data/rubin/shared/scheduler/packages/lsst_survey_sim/batch/run_prenight_sims.sh 2>&1 >> /sdf/data/rubin/shared/scheduler/prenight/daily/daily_simonyi_cron.out
   ```

   Note: `cleanup_prenight.sh` is not run from cron — it is submitted
   manually when disk space management is needed.

### Stopping Another User's Job

Delete their gate file:
```bash
rm /sdf/data/rubin/shared/scheduler/cron_gates/run_prenight_sims/<username>
```

The next cron-triggered invocation will exit immediately.  Recreate the file
to re-enable.

### Running Cleanup

1. Submit the cleanup generator:
   ```bash
   sbatch cleanup_prenight.sh
   ```

2. After the job completes, review the generated script:
   ```bash
   cat /sdf/data/rubin/shared/scheduler/prenight/cleanup_scripts/cleanup_prenight_<timestamp>.sh
   ```

3. Execute it if the commands look correct:
   ```bash
   bash /sdf/data/rubin/shared/scheduler/prenight/cleanup_scripts/cleanup_prenight_<timestamp>.sh
   ```

### Diagnosing Failures

- Check SLURM output logs in `/sdf/data/rubin/shared/scheduler/prenight/sbatch/`
- Look for `ERROR:` lines in the log output
- Verify preflight conditions (disk space, ACLs, token validity)
- Check that the work directory does not contain a `.done` file (indicates
  the job did not complete successfully)

---

## Dependencies

### System Commands

`date`, `id`, `sg`, `git`, `curl`, `jq`, `df`, `awk`, `tar`, `find`,
`mktemp`, `mkdir`, `ln`, `rm`, `chmod`, `setfacl`, `getfacl`

### Python Packages (installed at runtime)

- `lsst_survey_sim` (provides: `fetch_lsst_visits`, `make_lsst_scheduler`,
  `make_model_observatory`, `ideal_model_observatory`, `make_band_scheduler`,
  `run_lsst_sim`, `vseqarchive`)
- `rubin-scheduler`
- `ts_fbs_utils`

### External Services

- consdb (consolidated database) — visit history
- PostgreSQL metadata database — simulation metadata and indices
- S3 (`rubin-scheduler-prenight` bucket) — simulation artifact archive
- GitHub — `lsst_survey_sim` and `ts_config_scheduler` source
