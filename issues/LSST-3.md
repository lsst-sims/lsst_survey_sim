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

4. **R-4: Nominal statistics.** On success, the Sasquatch record includes at least the number of visits produced by the nominal simulation.

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
- Timestamp field is optional in the value; if omitted, system time at ingestion is used. Format when provided: unix timestamp in milliseconds.
- Tags (e.g., `telescope`, `simulation_type`) must be listed in the Sasquatch configuration and require a PR to add new ones.
- The existing scripts already have `curl` and `jq` available (verified in preflight checks), and an access token at `~/.lsst/usdf_access_token`.
- Lynne Jones is adding a `put` method to `rubin_nights` `InfluxQueryClient` on a branch, but for the bash scripts a direct `curl` POST is the appropriate approach.
- Results can be viewed at `usdf-rsp-dev.slac.stanford.edu/chronograf`.
- Hyphens are preferred over underscores in names that appear in URLs (per Angelo Fausti).

---

## 5. Critical Design Decisions (when applicable)

*(To be populated during Design phase if genuine alternatives arise.)*

---

## 6. Architecture and Design

*(To be populated during Design phase.)*

- [ ] Design reviewed and approved by _______________ on _______________ .

| Requirement | Verification |
|---|---|
| R-1: Success reporting | *(to be specified)* |
| R-2: AuxTel success reporting | *(to be specified)* |
| R-3: Failure reporting | *(to be specified)* |
| R-4: Nominal statistics | *(to be specified)* |
| R-5: Reporting failure isolation | *(to be specified)* |
| R-6: No new binary dependencies | *(to be specified)* |

---

## 7. Acceptance Criteria and Evidence

*(To be populated after implementation.)*

---

## 8. Open Questions

**Q-1.** What is the correct Sasquatch REST endpoint URL and topic name for ingesting records from batch jobs at USDF/S3DF?
- *Impact:* Cannot implement the HTTP POST without knowing the target URL and topic.
- *Answer:* Dev: `https://usdf-rsp-dev.slac.stanford.edu/sasquatch-rest-proxy/topics/lsst.survey`. Prod URL TBD (same pattern, different host). Topic/namespace is `lsst.survey`; measurement name is `lsst.survey.pre_night`. (Resolved per Angelo Fausti, 2026-08-22 Slack.)

**Q-2.** What authentication mechanism does the Sasquatch REST Proxy require at USDF (bearer token, mTLS, unauthenticated from internal network)?
- *Impact:* Determines whether a credential file or token must be sourced in the script.
- *Answer:* Bearer token with `write:sasquatch` scope. Dev does not currently enforce auth; prod does. The scripts already source a token from `~/.lsst/usdf_access_token`. (Resolved per Angelo Fausti, 2026-08-22 Slack.)

**Q-3.** Is there an existing Sasquatch schema/topic for batch-job heartbeats, or must a new topic be created?
- *Impact:* Determines whether we define a new schema or conform to an existing one.
- *Answer:* The `lsst.survey` namespace and a JSON connector have been configured by Angelo Fausti. The schema is flexible (JSON value object); new tags require a PR to the Sasquatch configuration repo. No separate topic creation needed — records are POSTed to the namespace topic. (Resolved per Angelo Fausti, 2026-08-22 Slack.)

**Q-4.** What is the production Sasquatch REST Proxy hostname?
- *Impact:* Needed for production deployment; not a blocker since initial work targets the dev instance.
- *Answer:* Deferred — initial implementation targets `usdf-rsp-dev.slac.stanford.edu`. Production cutover is a follow-up configuration change.

**Q-5.** Does the existing `~/.lsst/usdf_access_token` already have `write:sasquatch` scope, or must a new token be provisioned?
- *Impact:* Needed for production deployment; dev does not enforce auth so not a blocker for initial work.
- *Answer:* Deferred — dev instance does not require auth. Will be resolved before production cutover.

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
