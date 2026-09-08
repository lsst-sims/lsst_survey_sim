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
7. Report completion status and basic statistics to Sasquatch for monitoring

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
- On completion (success or failure), the script reports its status to
  Sasquatch; reporting failures are logged but never alter the script's exit
  status

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

### Sasquatch Status Reporting

Both simulation scripts report their completion status to
[Sasquatch](https://sasquatch.lsst.io), the project's timeseries database, so
that downstream monitoring can detect when simulations fail or do not run.

**Endpoint and namespace.**  Records are POSTed to the Sasquatch REST Proxy at
`https://<host>/sasquatch-rest-proxy/topics/lsst.survey` using content type
`application/vnd.kafka.json.v2+json`.  The measurement name is
`lsst.survey.pre_night`.  The current deployment targets the `usdf-rsp-dev`
instance; production cutover requires changing the URL and enabling
authentication.

**Payload.**  Each record contains:

| Field | Type | Present | Description |
|-------|------|---------|-------------|
| `measurement` | string | Always | `"lsst.survey.pre_night"` |
| `telescope` | string | Always | `"simonyi"` or `"auxtel"` |
| `dayobs` | string | Always | `YYYYMMDD` or `"unknown"` on early failure |
| `timestamp` | integer | Always | Event time as Unix milliseconds |
| `success` | boolean | Always | `true` or `false` |
| `uuid` | string | On success | UUID of the nominal simulation |
| `total_visit_count` | integer | On success | Total visits across all simulated nights |
| `download_url` | string | On success | Public URL for the visits file |

**Authentication.**  When `SASQUATCH_REQUIRE_AUTH=true`, the script reads a
bearer token from `~/.lsst/sasquatch_access_token` (requires `write:sasquatch`
scope).  The token is never stored in a shell variable or passed in curl's
argv; it is fed through a process-substitution curl config file with xtrace
disabled.  Preflight validates the token file (ownership, mode 400/600, no
symlinks, no named POSIX ACL entries, single-line bounded-length content)
whenever the file exists, regardless of the auth-required setting.
Unauthenticated reporting is permitted only when `SASQUATCH_URL` equals the
explicitly allow-listed development endpoint.

**Failure isolation.**  All calls to `report_to_sasquatch` are guarded with
`|| true`.  If Sasquatch is unreachable, the script logs a warning and
continues; the simulation results remain the primary deliverable.

**Reporting boundary.**  Success is reported just before the `.done` marker.
Failure is reported in the `on_exit` trap when `$? != 0`.  Failures before
trap installation (gate rejection, group-switch failure, invalid DAYOBS) are
not reported to Sasquatch.

**Viewing results.**  Records can be queried in Chronograf at
`https://usdf-rsp-dev.slac.stanford.edu/chronograf` under the
`lsst.survey.pre_night` measurement.

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
| `SASQUATCH_URL` | sim scripts | Sasquatch REST Proxy endpoint for status reporting |
| `SASQUATCH_DEV_URL` | sim scripts | Allow-listed dev endpoint (unauthenticated reporting permitted only when URL matches this) |
| `SASQUATCH_REQUIRE_AUTH` | sim scripts | `true` to require bearer-token authentication; `false` only for the dev endpoint |
| `TELESCOPE` | sim scripts | Telescope identifier sent in the Sasquatch record (`simonyi` or `auxtel`) |
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
- Query Sasquatch/Chronograf for the `lsst.survey.pre_night` measurement to
  see whether the script reported success, failure, or did not report at all
  (indicating a pre-trap failure or Sasquatch unreachable)
- Look for `WARNING: Sasquatch reporting failed` in the logs if reporting
  appears to be missing

---

## Dependencies

### System Commands

`date`, `id`, `sg`, `git`, `curl`, `jq`, `df`, `awk`, `tar`, `find`,
`mktemp`, `mkdir`, `ln`, `rm`, `chmod`, `setfacl`, `getfacl`, `stat`, `tr`

### Python Packages (installed at runtime)

- `lsst_survey_sim` (provides: `fetch_lsst_visits`, `make_lsst_scheduler`,
  `make_model_observatory`, `ideal_model_observatory`, `make_band_scheduler`,
  `run_lsst_sim`, `vseqarchive`)
- `rubin-scheduler`
- `ts_fbs_utils`

### Credential Files

| File | Purpose | Required |
|------|---------|----------|
| `~/.lsst/usdf_access_token` | consdb / metadata database access | Always |
| `~/.lsst/sasquatch_access_token` | Sasquatch bearer token (`write:sasquatch` scope) | Only when `SASQUATCH_REQUIRE_AUTH=true` (production) |

Both files must be non-symlink regular files owned by the effective user with
mode `400` or `600` and no named POSIX ACL entries.

### External Services

- consdb (consolidated database) — visit history
- PostgreSQL metadata database — simulation metadata and indices
- S3 (`rubin-scheduler-prenight` bucket) — simulation artifact archive
- Sasquatch / InfluxDB (`lsst.survey` namespace) — simulation status reporting
- GitHub — `lsst_survey_sim` and `ts_config_scheduler` source
