# LSST-3 — Report prenight simulation status to Sasquatch

| Field | Value |
|-------|--------|
| **Issue** | LSST-3 |
| **Branch** | `feature/LSST-3-sasquatch-reporting` |
| **Author** | Eric Neilsen |
| **Status** | Drafting |
| **Scope Tier** | T2 |
| **QA Level** | Low |
| **Estimate** | 2 days |
| **Created / Updated** | 2026-08-27 / 2026-08-27 |

---

## 1. Abstract

The prenight simulation batch scripts currently run on a daily cron schedule but have no mechanism to report their completion status to a monitoring system. If a simulation silently fails or never starts, no alarm fires. This issue adds reporting from the prenight bash scripts to Sasquatch (https://sasquatch.lsst.io), the project's timeseries database backed by InfluxDB, so that downstream monitoring can detect when simulations do not complete on their expected schedule. Basic statistics from the nominal simulation (e.g., number of visits produced) may also be sent. Actually configuring or triggering alarms in Sasquatch is out of scope; this issue only ensures the data needed to support such alarms is present in the timeseries database.

### 1.1 Scope

**In scope.**
- Add logic to `run_prenight_sims.sh` to send a record to Sasquatch upon successful completion.
- Add logic to `run_auxtel_prenight_sims.sh` to send a record to Sasquatch upon successful completion.
- Include basic statistics from the nominal simulation (e.g., visit count, simulated nights) in the Sasquatch record.
- Report failure status to Sasquatch (if the script reaches the reporting point in the exit trap after a failure).

**Out of scope.**
- Configuring Sasquatch alarm rules or alert routing.
- Changes to the Python `lsst_survey_sim` package itself.
- Changes to the cleanup script.
- Modifying the simulation logic or its outputs.
- Creating a Sasquatch dashboard or visualization.

**Done when.**
Both prenight simulation scripts send a timestamped record to Sasquatch on each run indicating success/failure status and basic nominal-simulation statistics.

---

## 2. Concept of Operations

The prenight simulation scripts (`run_prenight_sims.sh` and `run_auxtel_prenight_sims.sh`) are triggered daily by cron on SLAC S3DF. They run under SLURM and produce simulation outputs archived to S3 and a PostgreSQL metadata database. Currently, the only way to discover a failed or missing run is to manually inspect SLURM logs or notice missing entries in the prenight index.

After this change, each script will, at the end of its execution, send a record to a Sasquatch topic via HTTP (the Sasquatch REST Proxy or kafka-rest-proxy endpoint). The record will include at minimum: a timestamp, the telescope identifier (simonyi or auxtel), the DAYOBS being simulated, and a success/failure flag. On success, basic statistics from the nominal simulation (such as the number of visits simulated) will also be included.

The exit trap (`on_exit`) already captures the exit status. The Sasquatch reporting will be integrated near this point so that both success and failure outcomes are reported. If the reporting call itself fails (e.g., network issue), the script will log a warning but not change its own exit status — the simulation results remain the primary deliverable.

A downstream consumer (configured separately, outside this issue) can then query Sasquatch for the expected daily heartbeat and raise an alarm if it is absent or indicates failure.

---

## 3. External Requirements and Design Evaluation Criteria

### 3.1 External Requirements

1. **R-1: Success reporting.** On successful completion of `run_prenight_sims.sh`, a record is sent to Sasquatch containing at minimum: timestamp, telescope identifier ("simonyi"), DAYOBS, and a success indicator.

2. **R-2: AuxTel success reporting.** On successful completion of `run_auxtel_prenight_sims.sh`, a record is sent to Sasquatch containing at minimum: timestamp, telescope identifier ("auxtel"), DAYOBS, and a success indicator.

3. **R-3: Failure reporting.** If either script exits with a non-zero status and execution reaches the exit trap, a record is sent to Sasquatch containing at minimum: timestamp, telescope identifier, DAYOBS (if computed), and a failure indicator.

4. **R-4: Nominal statistics.** On success, the Sasquatch record includes at least the number of visits produced by the nominal simulation, the UUID of the nominal simulation, and a download URL for its visits file.

5. **R-5: Reporting failure isolation.** If the Sasquatch reporting call itself fails, the script logs a warning but does not alter its own exit status or prevent other post-simulation steps from completing.

6. **R-6: No new binary dependencies.** The reporting mechanism uses only tools already available in the script environment (e.g., `curl`, `jq`) or standard Python from the created conda env.

### 3.2 Design Evaluation Criteria

Omitted — only one plausible design approach (HTTP POST to Sasquatch REST Proxy).

---

## 4. Context

- The `lsst.survey` namespace has been set up in Sasquatch by Angelo Fausti and is working at `usdf-rsp-dev` for experimentation.
- Data is sent as JSON via HTTP POST to the Sasquatch REST Proxy. URL pattern: `https://<host>/sasquatch-rest-proxy/topics/{namespace}`
  - Dev: `https://usdf-rsp-dev.slac.stanford.edu/sasquatch-rest-proxy/topics/lsst.survey`
- Content-Type header: `application/vnd.kafka.json.v2+json`
- Authentication: Bearer token with `write:sasquatch` scope (required on prod; dev currently does not enforce it).
- The measurement name for prenight sim reporting is `lsst.survey.pre_night`.
- A separate measurement `lsst.survey.night_summary` exists for nightly summary statistics (not in scope for this issue but shares the namespace).
- Payload structure (JSON, one or more records):
  ```json
  {
    "records": [
      {
        "value": {
          "measurement": "lsst.survey.pre_night",
          "telescope": "simonyi",
          "simulation_type": "nominal",
          "success": true,
          "uuid": "<sim-uuid>",
          "download_url": "<url>"
        }
      }
    ]
  }
  ```
- Although Sasquatch can supply ingestion time when a timestamp is omitted, prenight records explicitly include the event time as a Unix timestamp in milliseconds. This satisfies R-1--R-3 and distinguishes batch completion time from delayed ingestion time.
- Tags (e.g., `telescope`, `simulation_type`) must be listed in the Sasquatch configuration and require a PR to add new ones.
- The existing scripts already have `curl`, `jq`, `awk`, `stat`, and `getfacl` available (or will verify them in preflight). A consdb access token lives at `~/.lsst/usdf_access_token`; for Sasquatch reporting, a separate token at `~/.lsst/sasquatch_access_token` is used (requires `write:sasquatch` scope on prod). The token file may be absent only when the configured URL is the explicitly allow-listed development endpoint, where authentication is not enforced. All other endpoints require a token.
- Lynne Jones is adding a `put` method to `rubin_nights` `InfluxQueryClient` on a branch, and it may eventually be the correct approach to use a corresponding command from `rubin_nights` when it is ready, but for now the bash scripts a direct `curl` POST is the appropriate approach.
- Results can be viewed at `usdf-rsp-dev.slac.stanford.edu/chronograf`.
- Hyphens are preferred over underscores in names that appear in URLs (per Angelo Fausti).
- **Script structure constraints (from codebase exploration, 2026-08-27):**
  - Both scripts run under `set -euo pipefail`; any unguarded failing command (including a failed `curl`) would abort the entire script. Reporting calls must be explicitly isolated (e.g., `|| true` or a subshell).
  - `DAYOBS` is computed early (before environment setup), so it is available even if the script fails after preflight. However, variables set later (e.g., `SIM_UUID`, `CONDA_ENV_HASH`) may not be defined if the script fails during environment setup or simulation.
  - In `run_prenight_sims.sh`, `run_and_archive_sim()` cleans up `opsim.db` at the end of each simulation invocation (line 316). The function also runs `vseqarchive add-nightly-stats` (when `ADD_STATS=true`) which computes per-night statistics and stores them in PostgreSQL. These stats can later be retrieved with `vseqarchive query-nightly-stats <UUID>`, which outputs a TSV table with columns: `day_obs`, `value_name`, `count`, `mean`, `std`, `min`, `p05`, `q1`, `median`, `q3`, `p95`, `max`, `accumulated`. The `count` column gives the number of visits per night per value_name.
  - The `compute_nightly_stats()` function in `rubin_sim.sim_archive.vseqarchive` groups visits by `day_obs` and calls `pandas.describe()` on the specified columns. Since the simulation spans 3 nights, summing the `count` values for a single `value_name` across all nights gives the total visit count.
  - `vseqarchive get-visitseq-url <UUID>` prints the download URL for the visits file of a visit sequence to stdout. This is the S3/HTTP URL stored in the archive metadata at archival time.
  - The first nominal simulation (`prenight_nominal_noreward`) has `ADD_STATS=false` and `KEEP_REWARDS=false`; the second nominal simulation (`prenight_nominal`) has `ADD_STATS=true` and `KEEP_REWARDS=true`. The second is the one whose stats are already computed and stored in the database.
  - `SIM_UUID` is a local variable inside `run_and_archive_sim()` and is not visible to the outer script scope. The Sasquatch report would need either a variable exported from the function or a separate mechanism to track the nominal sim's UUID.
  - The `on_exit` trap (`on_exit()`) captures `$?` but has no access to simulation-specific variables unless they are stored in script-global scope before the trap fires.
  - In `run_auxtel_prenight_sims.sh`, there is only one simulation, and `SIM_UUID` is set at outer scope (line 347), making it directly available for reporting. That script also calls `vseqarchive add-nightly-stats` directly at outer scope.
  - Sasquatch tags have low cardinality (< 10,000 distinct values). `telescope` (2 values: simonyi, auxtel) and `simulation_type` are appropriate tags. Numeric fields like `visit_count` are InfluxDB *fields*, not tags.
  - The Sasquatch REST Proxy with a JSON connector does not require a pre-registered Avro schema; the JSON `value` object is passed through directly. New fields can be added freely to the value without a schema evolution PR — only new *tags* require a Sasquatch config PR.

---

## 5. Critical Design Decisions (when applicable)

None — only one plausible approach exists (HTTP POST via `curl`). The design details below are the single obvious realization of that approach.

---

## 6. Architecture and Design

- [ ] Design reviewed and approved by ______EHN______ on __2026-08-27______ .

### 6.1 Overview

A new shell function `report_to_sasquatch()` is added to each script. It builds a JSON payload and POSTs it to the Sasquatch REST Proxy. The function is called:
- **On success:** at the end of the script (just before the `.done` marker), with status=success and nominal-simulation statistics.
- **On failure:** inside the `on_exit` trap when `$? != 0`, with status=failure and whatever information is available.

All calls to `report_to_sasquatch` are guarded with `|| true` so that a reporting failure never changes the script's exit status.

### 6.2 Shared helper function: `report_to_sasquatch`

Added to both scripts (duplicated; these are standalone shell scripts, not a shared library).

```bash
# Report prenight simulation status to Sasquatch.
# Arguments:
#   $1 - success: "true" or "false"
#   $2 - total_visit_count: integer or "" if unavailable
#   $3 - sim_uuid: UUID string or "" if unavailable
#   $4 - download_url: URL string or "" if unavailable
# Uses globals: DAYOBS, SASQUATCH_URL, TELESCOPE, SASQUATCH_TOKEN_FILE_VALIDATED
report_to_sasquatch() {
    local SUCCESS="$1"
    local TOTAL_VISIT_COUNT="${2:-}"
    local NOMINAL_SIM_UUID="${3:-}"
    local DOWNLOAD_URL="${4:-}"

    # Record event time rather than relying on possibly delayed ingestion time.
    local EVENT_TIMESTAMP_MS
    EVENT_TIMESTAMP_MS=$(date -u +%s%3N)

    # Build the value object with jq for safe JSON construction.
    local PAYLOAD
    PAYLOAD=$(jq -n \
        --arg measurement "lsst.survey.pre_night" \
        --arg telescope "${TELESCOPE}" \
        --arg dayobs "${DAYOBS:-unknown}" \
        --argjson timestamp "${EVENT_TIMESTAMP_MS}" \
        --argjson success "${SUCCESS}" \
        --arg uuid "${NOMINAL_SIM_UUID}" \
        --arg visit_count "${TOTAL_VISIT_COUNT}" \
        --arg download_url "${DOWNLOAD_URL}" \
        '{
            records: [{
                value: (
                    {
                        measurement: $measurement,
                        telescope: $telescope,
                        dayobs: $dayobs,
                        timestamp: $timestamp,
                        success: $success
                    }
                    + (if $uuid != "" then {uuid: $uuid} else {} end)
                    + (if $visit_count != "" then {total_visit_count: ($visit_count | tonumber)} else {} end)
                    + (if $download_url != "" then {download_url: $download_url} else {} end)
                )
            }]
        }')

    log "Reporting to Sasquatch: success=${SUCCESS} visits=${TOTAL_VISIT_COUNT:-n/a} uuid=${NOMINAL_SIM_UUID:-n/a}"

    # Construct a private curl config via process substitution so that
    # neither the token nor the payload (which may contain a presigned
    # download URL) appears in curl's argv or xtrace output.
    local CURL_OK=false
    local USE_TOKEN=false
    if [ -n "${SASQUATCH_TOKEN_FILE_VALIDATED:-}" ]; then
        USE_TOKEN=true
    fi

    local XTRACE_WAS_SET=false
    [[ $- == *x* ]] && XTRACE_WAS_SET=true
    { set +x; } 2>/dev/null
    if printf '%s' "${PAYLOAD}" | curl -K <(
        printf 'silent\nfail\noutput = /dev/null\nmax-time = 30\n'
        printf 'request = POST\n'
        printf 'url = "%s"\n' "${SASQUATCH_URL}"
        printf 'header = "Content-Type: application/vnd.kafka.json.v2+json"\n'
        if [ "${USE_TOKEN}" = "true" ]; then
            printf 'header = "Authorization: Bearer '
            tr -d '\r\n' < "${SASQUATCH_TOKEN_FILE_VALIDATED}"
            printf '"\n'
        fi
        printf 'data = @-\n'
    ) 2>/dev/null; then
        CURL_OK=true
    fi
    [ "${XTRACE_WAS_SET}" = "true" ] && set -x

    if [ "${CURL_OK}" = "true" ]; then
        log "Sasquatch report sent successfully."
    else
        echo "WARNING: Sasquatch reporting failed. Continuing." >&2
    fi
}
```

### 6.3 Constants added to each script

```bash
readonly SASQUATCH_URL="https://usdf-rsp-dev.slac.stanford.edu/sasquatch-rest-proxy/topics/lsst.survey"
readonly SASQUATCH_DEV_URL="https://usdf-rsp-dev.slac.stanford.edu/sasquatch-rest-proxy/topics/lsst.survey"
readonly SASQUATCH_REQUIRE_AUTH=false
readonly TELESCOPE="simonyi"   # or "auxtel" in the auxtel script
SASQUATCH_TOKEN_FILE_VALIDATED=""
```

`SASQUATCH_REQUIRE_AUTH=false` is permitted only when `SASQUATCH_URL` exactly equals `SASQUATCH_DEV_URL`. A production cutover must set a production URL and `SASQUATCH_REQUIRE_AUTH=true`; changing only the URL is rejected by preflight. This prevents a production deployment from silently attempting unauthenticated reporting.

### 6.4 Preflight check for Sasquatch token

Both scripts' `preflight_check()` functions gain a check for `~/.lsst/sasquatch_access_token`. The file may be absent only for the allow-listed development endpoint when `SASQUATCH_REQUIRE_AUTH=false`. If present, it must be a non-symlink regular file owned by the effective user, have mode `400` or `600`, have no named POSIX ACL entries, and contain exactly one non-empty token of bounded length using the allowed character set. Every metadata, ACL, or content validation failure is a hard error: an insecure or malformed token indicates a security misconfiguration and must abort the job rather than being ignored.

R-5 applies to failures of the isolated HTTP reporting operation. It does not require the simulation to proceed in the presence of an insecure credential. The approved policy is therefore to abort during preflight for any insecure or malformed token.

```bash
local SASQUATCH_TOKEN_FILE="${HOME}/.lsst/sasquatch_access_token"
if [ "${SASQUATCH_REQUIRE_AUTH}" != "true" ] \
   && [ "${SASQUATCH_URL}" != "${SASQUATCH_DEV_URL}" ]; then
    echo "ERROR: Unauthenticated Sasquatch reporting is allowed only for the development endpoint." >&2
    exit 1
fi

if [ -e "${SASQUATCH_TOKEN_FILE}" ] || [ -L "${SASQUATCH_TOKEN_FILE}" ]; then
    if [ -L "${SASQUATCH_TOKEN_FILE}" ] || [ ! -f "${SASQUATCH_TOKEN_FILE}" ]; then
        echo "ERROR: ${SASQUATCH_TOKEN_FILE} must be a non-symlink regular file." >&2
        exit 1
    fi

    local TOKEN_OWNER TOKEN_PERMS TOKEN_ACL
    TOKEN_OWNER=$(stat -c '%u' "${SASQUATCH_TOKEN_FILE}")
    TOKEN_PERMS=$(stat -c '%a' "${SASQUATCH_TOKEN_FILE}")
    TOKEN_ACL=$(getfacl -cp "${SASQUATCH_TOKEN_FILE}")
    if [ "${TOKEN_OWNER}" != "$(id -u)" ]; then
        echo "ERROR: ${SASQUATCH_TOKEN_FILE} is not owned by the effective user." >&2
        exit 1
    fi
    if [ "${TOKEN_PERMS}" != "600" ] && [ "${TOKEN_PERMS}" != "400" ]; then
        echo "ERROR: ${SASQUATCH_TOKEN_FILE} has permissions ${TOKEN_PERMS}; expected 600 or 400." >&2
        exit 1
    fi
    if grep -qE '^(user|group):[^:]+' <<< "${TOKEN_ACL}"; then
        echo "ERROR: ${SASQUATCH_TOKEN_FILE} has named ACL entries; access tokens must be private." >&2
        exit 1
    fi
    # Validate token content with xtrace suppressed so the value is never logged.
    local XTRACE_WAS_SET=false
    [[ $- == *x* ]] && XTRACE_WAS_SET=true
    { set +x; } 2>/dev/null
    if ! awk '
        BEGIN { valid = 1 }
        NR != 1 { valid = 0 }
        NR == 1 && (length($0) == 0 || length($0) > 8192 ||
                    $0 !~ /^[A-Za-z0-9._~+\/=\-]+$/) { valid = 0 }
        END { exit !(valid && NR == 1) }
    ' "${SASQUATCH_TOKEN_FILE}"; then
        echo "ERROR: ${SASQUATCH_TOKEN_FILE} must contain exactly one valid token of at most 8192 characters." >&2
        exit 1
    fi
    [ "${XTRACE_WAS_SET}" = "true" ] && set -x

    # Record that the token file passed validation (store path, not content).
    SASQUATCH_TOKEN_FILE_VALIDATED="${SASQUATCH_TOKEN_FILE}"
elif [ "${SASQUATCH_REQUIRE_AUTH}" = "true" ]; then
    echo "ERROR: Missing required Sasquatch token file ${SASQUATCH_TOKEN_FILE}." >&2
    exit 1
fi
```

The token value is never assigned to a shell variable, passed as an argument, or logged. All metadata (`stat`, `getfacl`) and content (`awk`) checks run directly against the validated filesystem path — safe because symlinks have already been rejected. The content check runs under `set +x` so the token value never appears in xtrace output. `report_to_sasquatch` re-reads the file with `tr -d '\r\n' < "${SASQUATCH_TOKEN_FILE_VALIDATED}"` inside a process-substitution curl config while xtrace is still disabled, so the token is never in a shell variable, never in curl's argv, and never in logs. Both scripts add `getfacl` and `stat` to their preflight command checks.

### 6.5 Changes to `run_prenight_sims.sh`

**6.5.1 Global state for nominal sim UUID.**

A script-global variable is initialized before the simulations section:

```bash
NOMINAL_SIM_UUID=""
```

**6.5.2 Capturing the nominal sim's UUID.**

The `run_and_archive_sim` function currently declares `SIM_UUID` as local. After the second nominal simulation call (the one with `ADD_STATS=true`), the UUID is captured into the global:

```bash
run_and_archive_sim observatory.p \
    "prenight_nominal" \
    "Nominal start and overhead, ideal conditions" \
    true true \
    "prenight ideal nominal rewards" \
    --delay 0 --anom_overhead_scale 0

NOMINAL_SIM_UUID="${LAST_SIM_UUID}"
```

To enable this, `run_and_archive_sim` is modified to set a script-global variable at the end of the function:

```bash
    # Export UUID to caller (not local)
    LAST_SIM_UUID="${SIM_UUID}"
```

This is a minimal change: `LAST_SIM_UUID` is not declared `local` inside the function, so assigning it writes to the enclosing scope.

**6.5.3 Obtaining the total visit count and download URL.**

After capturing `NOMINAL_SIM_UUID`, the total visit count and download URL are obtained by querying the metadata database:

```bash
# Sum the per-night visit counts from the nightly stats (one value_name is sufficient;
# each has the same count per night).
NOMINAL_VISIT_COUNT=""
NOMINAL_DOWNLOAD_URL=""
if [ -n "${NOMINAL_SIM_UUID}" ]; then
    NOMINAL_VISIT_COUNT=$(vseqarchive query-nightly-stats "${NOMINAL_SIM_UUID}" \
        | awk -F'\t' 'NR>1 && !seen[$1]++ {sum += $3} END {print sum+0}') || NOMINAL_VISIT_COUNT=""
    NOMINAL_DOWNLOAD_URL=$(vseqarchive get-visitseq-url "${NOMINAL_SIM_UUID}") || NOMINAL_DOWNLOAD_URL=""
fi
```

Explanation: The TSV output has one row per (day_obs, value_name). Since `add-nightly-stats` was called with columns `azimuth altitude`, there are two rows per night. We take only the first unseen `day_obs` row (via `!seen[$1]++`) to avoid double-counting, sum the `count` column (`$3`), and print the total. The download URL is the S3/HTTP URL for the visits file, as stored in the archive metadata.

**6.5.4 Success reporting (end of script, before `.done`).**

Inserted just before `touch "${WORK_DIR}/.done"`:

```bash
report_to_sasquatch "true" "${NOMINAL_VISIT_COUNT}" "${NOMINAL_SIM_UUID}" "${NOMINAL_DOWNLOAD_URL}" || true
```

**6.5.5 Failure reporting (in `on_exit` trap).**

The `on_exit` function is modified:

```bash
on_exit() {
    local STATUS=$?
    if [ "${STATUS}" -ne 0 ]; then
        echo "run_prenight_sims.sh FAILED with exit status ${STATUS}" >&2
        report_to_sasquatch "false" "" "" "" || true
    fi
    echo "Design docs: https://github.com/lsst-sims/lsst_survey_sim/blob/main/batch/design.md"
    echo "******** END of run_prenight_sims.sh (status ${STATUS}) **********"
    date --iso=s
}
```

The trap is installed only after `report_to_sasquatch` has been defined, so every failure that reaches this trap can call the function. Failures before trap installation—including gate rejection, failure to re-execute under `rubin_users`, invalid `DAYOBS`, or failure while deriving date variables—are not reported to Sasquatch. This is the explicit boundary of R-3. For failures after trap installation, `${DAYOBS:-unknown}` remains defensive even though `DAYOBS` is normally already set.

### 6.6 Changes to `run_auxtel_prenight_sims.sh`

The structure is simpler because there is only one simulation and `SIM_UUID` is at outer scope.

**6.6.1 Constants.**

```bash
readonly SASQUATCH_URL="https://usdf-rsp-dev.slac.stanford.edu/sasquatch-rest-proxy/topics/lsst.survey"
readonly TELESCOPE="auxtel"
```

**6.6.2 Obtaining the total visit count and download URL.**

After the existing `vseqarchive add-nightly-stats` call:

```bash
NOMINAL_VISIT_COUNT=$(vseqarchive query-nightly-stats "${SIM_UUID}" \
    | awk -F'\t' 'NR>1 && !seen[$1]++ {sum += $3} END {print sum+0}') || NOMINAL_VISIT_COUNT=""
NOMINAL_DOWNLOAD_URL=$(vseqarchive get-visitseq-url "${SIM_UUID}") || NOMINAL_DOWNLOAD_URL=""
```

**6.6.3 Success reporting.**

Inserted just before `touch "${WORK_DIR}/.done"`:

```bash
report_to_sasquatch "true" "${NOMINAL_VISIT_COUNT}" "${SIM_UUID}" "${NOMINAL_DOWNLOAD_URL}" || true
```

**6.6.4 Failure reporting (in `on_exit` trap).**

Same pattern as Simonyi:

```bash
on_exit() {
    local STATUS=$?
    if [ "${STATUS}" -ne 0 ]; then
        echo "run_auxtel_prenight_sims.sh FAILED with exit status ${STATUS}" >&2
        report_to_sasquatch "false" "" "" "" || true
    fi
    echo "Design docs: https://github.com/lsst-sims/lsst_survey_sim/blob/main/batch/design.md"
    echo "******** END of run_auxtel_prenight_sims.sh (status ${STATUS}) **********"
    date --iso=s
}
```

### 6.7 Sasquatch record schema

The JSON value object sent to Sasquatch:

| Field | Type | Tag/Field | Present | Description |
|-------|------|-----------|---------|-------------|
| `measurement` | string | (routing) | Always | `"lsst.survey.pre_night"` |
| `telescope` | string | Tag | Always | `"simonyi"` or `"auxtel"` |
| `dayobs` | string | Tag | Always | YYYYMMDD or `"unknown"` on early failure |
| `timestamp` | integer | Field/time | Always | Event time as Unix milliseconds, generated immediately before reporting |
| `success` | boolean | Field | Always | `true` or `false` |
| `uuid` | string | Field | On success | UUID of the nominal simulation |
| `download_url` | string | Field | On success | URL to download the visits file for the nominal simulation |
| `total_visit_count` | integer | Field | On success | Total visits across all simulated nights |

Tags requiring a Sasquatch configuration PR: `telescope`, `dayobs`. (The `simulation_type` tag from Angelo's example is not needed here since we report only the overall run status, not per-simulation-variant records.)

### 6.8 Requirement-to-Verification Mapping

| Requirement | Verification |
|---|---|
| R-1: Success reporting | Manual test: run `run_prenight_sims.sh` to completion; verify the record appears in Chronograf at `usdf-rsp-dev` with `telescope=simonyi`, correct DAYOBS, `success=true`, and an integer event timestamp in milliseconds. |
| R-2: AuxTel success reporting | Manual test: run `run_auxtel_prenight_sims.sh` to completion; verify the record appears in Chronograf with `telescope=auxtel`, correct DAYOBS, `success=true`, and an integer event timestamp in milliseconds. |
| R-3: Failure reporting | Manual test: inject a forced failure after trap installation (e.g., invalid `SCHED_CONFIG_FNAME`); verify that the trap sends a record with `success=false` and an integer event timestamp. Verify separately that documented pre-trap failures do not report. |
| R-4: Nominal statistics | Manual test: on the success record from R-1 or R-2, verify `total_visit_count` is a positive integer consistent with a 3-night simulation (~900–1500 visits), `uuid` is a valid UUID, and `download_url` is a non-empty URL. |
| R-5: Reporting failure isolation | Manual test: temporarily set `SASQUATCH_URL` to an unreachable host; verify the script still completes normally (exits 0, `.done` is created) and logs a WARNING. |
| R-6: No new binary dependencies | Code inspection: the function uses only `curl`, `jq`, `cat`, `awk`, and `vseqarchive` — all already present in the script environment. |

### 6.9 Implementation outline

1. Add `SASQUATCH_URL`, `SASQUATCH_DEV_URL`, `SASQUATCH_REQUIRE_AUTH`, and `TELESCOPE` constants to each script's constants block.
2. Add `report_to_sasquatch()` to each script's helper functions section.
3. Add endpoint/auth-policy enforcement and secure validation of `~/.lsst/sasquatch_access_token` to `preflight_check()` in both scripts; reject symlinks, wrong ownership/mode, named ACLs, malformed content, and missing credentials when auth is required.
4. Initialize `NOMINAL_SIM_UUID=""` (Simonyi only) before the simulations block.
5. Add `LAST_SIM_UUID="${SIM_UUID}"` at the end of `run_and_archive_sim()` (Simonyi only).
6. After the second nominal sim call, capture `NOMINAL_SIM_UUID="${LAST_SIM_UUID}"` (Simonyi only).
7. After the final index update (both scripts), compute `NOMINAL_VISIT_COUNT` via `vseqarchive query-nightly-stats` and `NOMINAL_DOWNLOAD_URL` via `vseqarchive get-visitseq-url`.
8. Call `report_to_sasquatch "true" ...` before `touch .done` (both scripts).
9. Modify `on_exit()` to call `report_to_sasquatch "false" ...` on non-zero status (both scripts).

---

## 7. Acceptance Criteria and Evidence

### 7.1 Design Conformance (Code Inspection)

All 9 steps of the implementation outline (§6.9) are verified present in the branch `tickets/LSST-3` (commit `30b2697`):

| §6.9 Step | `run_prenight_sims.sh` | `run_auxtel_prenight_sims.sh` | Status |
|---|---|---|---|
| 1. Sasquatch constants | Lines 87–91 | Lines 84–88 | ✅ |
| 2. `report_to_sasquatch()` function | Lines 313–393 | Lines 254–334 | ✅ |
| 3. Preflight token validation | Lines 225–284 | Lines 176–235 | ✅ |
| 4. Init `NOMINAL_SIM_UUID=""` | Line 671 | N/A (outer `SIM_UUID`) | ✅ |
| 5. `LAST_SIM_UUID="${SIM_UUID}"` in function | Line 472 | N/A | ✅ |
| 6. Capture UUID after 2nd nominal sim | Line 692 | N/A | ✅ |
| 7. Compute visit count & download URL | Lines 761–767 | Lines 548–553 | ✅ |
| 8. Success report before `.done` | Line 769 (`.done` at 775) | Line 556 (`.done` at 562) | ✅ |
| 9. Failure report in `on_exit` trap | Line 479 | Line 340 | ✅ |

The `report_to_sasquatch` function body is identical in both scripts (as specified in §6.2).

### 7.2 Requirement Verification

| Req | Criterion | Verification Method | Evidence / Instructions |
|---|---|---|---|
| R-1 | Simonyi success record sent to Sasquatch | Manual test | Run `run_prenight_sims.sh` to completion. Verify in Chronograf at `usdf-rsp-dev.slac.stanford.edu/chronograf` that a record appears in measurement `lsst.survey.pre_night` with `telescope=simonyi`, `dayobs` matching the simulated DAYOBS, `success=true`, and `timestamp` as an integer (Unix ms). |
| R-2 | AuxTel success record sent to Sasquatch | Manual test | Run `run_auxtel_prenight_sims.sh` to completion. Verify in Chronograf that a record appears with `telescope=auxtel`, correct DAYOBS, `success=true`, and an integer `timestamp`. |
| R-3 | Failure record sent on non-zero exit | Manual test | Inject a failure after trap installation (e.g., set `SCHED_CONFIG_FNAME` to a non-existent path, or `exit 1` after `preflight_check`). Verify in Chronograf that a record appears with `success=false` and an integer `timestamp`. Separately, verify that a failure *before* trap installation (e.g., missing gate file) does NOT produce a Sasquatch record. |
| R-4 | Nominal statistics included on success | Manual test | On the success record from R-1 or R-2, verify: `total_visit_count` is a positive integer consistent with a 3-night simulation (~900–1500 visits for simonyi, fewer for auxtel), `uuid` is a valid UUID (8-4-4-4-12 hex), and `download_url` is a non-empty S3/HTTP URL. |
| R-5 | Reporting failure does not alter script exit | Manual test | Temporarily set `SASQUATCH_URL` to an unreachable host (e.g., `https://unreachable.example.com/topics/lsst.survey`), keeping it equal to `SASQUATCH_DEV_URL` for the auth check. Run the script. Verify: (a) the script completes with exit 0, (b) `.done` marker is created, (c) a WARNING about Sasquatch reporting failure appears in the log. |
| R-6 | No new binary dependencies | Code inspection | The function uses only `curl`, `jq`, `awk`, `date`, `printf`, `tr`, `stat`, `getfacl`, and `vseqarchive` — all already available in the script environment. `stat` and `getfacl` were already used by the simonyi script's `check_dir_ready` and are now also checked in the auxtel script's `require_commands`. ✅ |

### 7.3 Manual Verification Procedure

**Prerequisites:**
- Access to S3DF with SLURM job submission.
- The `usdf-rsp-dev` Chronograf instance is accessible.
- The `lsst.survey` namespace is configured in Sasquatch (already done by Angelo Fausti).

**Steps for R-1 / R-2 / R-4 (success path):**

1. Submit the script under test:
   ```bash
   export DAYOBS=20260828  # or omit to use today
   sbatch batch/run_prenight_sims.sh     # for R-1
   sbatch batch/run_auxtel_prenight_sims.sh  # for R-2
   ```
2. Wait for the SLURM job to complete (check with `squeue` or inspect output file).
3. Confirm the job exited 0 and `.done` exists in the work directory.
4. Open Chronograf at `https://usdf-rsp-dev.slac.stanford.edu/chronograf`.
5. Query `lsst.survey.pre_night` filtered by `telescope` and `dayobs` matching the run.
6. Verify the record contains:
   - `success: true`
   - `timestamp`: integer, in reasonable range (within minutes of job completion time)
   - `total_visit_count`: positive integer
   - `uuid`: valid UUID format
   - `download_url`: non-empty URL beginning with `https://` or `s3://`

**Steps for R-3 (failure path):**

1. Edit the script (or set an environment variable) to force a failure after `trap on_exit EXIT` but before completion. For example, add `exit 1` after the `preflight_check` call.
2. Submit the script and wait for it to finish (non-zero exit).
3. In Chronograf, verify a record with `success: false` and a valid `timestamp` appeared.
4. Verify that `uuid`, `total_visit_count`, and `download_url` are absent from the failure record.

**Steps for R-5 (reporting failure isolation):**

1. Temporarily change both `SASQUATCH_URL` and `SASQUATCH_DEV_URL` to `https://unreachable.example.com/topics/lsst.survey`.
2. Run the script to completion.
3. Verify: exit status is 0, `.done` is created, and the log contains `WARNING: Sasquatch reporting failed. Continuing.`

### 7.4 Scope Compliance

- No changes to the Python `lsst_survey_sim` package. ✅
- No changes to `cleanup_prenight.sh`. ✅
- No changes to simulation logic or outputs. ✅
- No Sasquatch alarm rules or dashboards configured. ✅
- Only the two prenight scripts were modified. ✅

### 7.5 Deviations from Design

None identified. The implementation matches §6 exactly.

---

## 8. Open Questions

**Q-1.** What is the correct Sasquatch REST endpoint URL and topic name for ingesting records from batch jobs at USDF/S3DF?
- *Impact:* Cannot implement the HTTP POST without knowing the target URL and topic.
- *Answer:* Dev: `https://usdf-rsp-dev.slac.stanford.edu/sasquatch-rest-proxy/topics/lsst.survey`. Prod URL TBD (same pattern, different host). Topic/namespace is `lsst.survey`; measurement name is `lsst.survey.pre_night`. (Resolved per Angelo Fausti, 2026-08-22 Slack.)

**Q-2.** What authentication mechanism does the Sasquatch REST Proxy require at USDF (bearer token, mTLS, unauthenticated from internal network)?
- *Impact:* Determines whether a credential file or token must be sourced in the script.
- *Answer:* Bearer token with `write:sasquatch` scope. Dev does not currently enforce auth; prod does. A dedicated token file `~/.lsst/sasquatch_access_token` is used for Sasquatch reporting (separate from the consdb token at `~/.lsst/usdf_access_token`). (Resolved per Angelo Fausti, 2026-08-22 Slack.)

**Q-3.** Is there an existing Sasquatch schema/topic for batch-job heartbeats, or must a new topic be created?
- *Impact:* Determines whether we define a new schema or conform to an existing one.
- *Answer:* The `lsst.survey` namespace and a JSON connector have been configured by Angelo Fausti. The schema is flexible (JSON value object); new tags require a PR to the Sasquatch configuration repo. No separate topic creation needed — records are POSTed to the namespace topic. (Resolved per Angelo Fausti, 2026-08-22 Slack.)

**Q-4.** What is the production Sasquatch REST Proxy hostname?
- *Impact:* Needed for production deployment; not a blocker since initial work targets the dev instance.
- *Answer:* Deferred — initial implementation targets `usdf-rsp-dev.slac.stanford.edu`. Production cutover is a follow-up configuration change.

**Q-5.** Does `~/.lsst/sasquatch_access_token` need to be provisioned with `write:sasquatch` scope?
- *Impact:* Needed for production deployment; dev does not enforce auth so not a blocker for initial work.
- *Answer:* Deferred — dev instance does not require auth. The file is optional; when absent, the request is sent without an Authorization header. Will be resolved before production cutover.

---

## 9. Notes, Risks, and Future Considerations (Optional)

- The alarm configuration in Sasquatch (e.g., "alert if no success record for simonyi in 26 hours") is explicitly deferred to a follow-up issue.
- If Sasquatch's REST Proxy is not reachable from S3DF compute nodes, a network or firewall change may be required (infrastructure dependency outside this issue's control).
- Future enhancement: report per-simulation timing data to Sasquatch for performance trending.

---

## 10. Change Log (Optional)

| Date | Author | Summary |
|------|--------|---------|
| 2026-08-27 | Eric Neilsen | Initial IWD creation (Inception) |
| 2026-08-27 | Eric Neilsen | Updated Context and resolved Q-1–Q-3 from Slack conversation with Angelo Fausti and Lynne Jones (2026-08-19–22) |
| 2026-08-27 | Eric Neilsen | Frame phase: added script-structure constraints to Context from codebase, Sasquatch repo, and rubin_sim sim_archive exploration |
| 2026-08-27 | Eric Neilsen | Design phase: Architecture and Design (§6) drafted |
| 2026-08-28 | Eric Neilsen | Security review amendments: endpoint-specific auth policy, strict token ownership/mode/ACL/content validation without storing token content in shell variables, explicit event timestamps, and corrected exit-trap boundary. |

---

## Implementation Notes

*(To be provided by the implementation agent at closeout.)*

- Design steps completed:
- Files and symbols changed:
- Tests added or updated and results:
- Outcome evidence:
- Approved design amendments:
- Unresolved deviations:

## Definition of Done

- [ ] Scope (§1.1) respected — nothing implemented from the out-of-scope list.
- [ ] Architecture and Design (§6) approved; for T2 before implementation, for T1 before merge.
- [ ] Every External Requirement (§3.1) has a passing test or a recorded manual check (§6, §7).
- [ ] Material deviations from Architecture and Design (§6) are approved and recorded.
- [ ] A targeted diff review was completed; detailed review was performed for applicable risk triggers.
- [ ] Durable content has been promoted to project documentation where useful and logged in the Change Log (§10).
- [ ] CI is green; the change has been reviewed per the review discipline (process §4.1).
