#!/usr/bin/env bash
#SBATCH --account=rubin:developers      # Account name
#SBATCH --job-name=prenight_simonyi_daily   # Job name
#SBATCH --output=/sdf/data/rubin/shared/scheduler/prenight/sbatch/run_prenight_sims_%A_%a.out # Output file (stdout)
#SBATCH --error=/sdf/data/rubin/shared/scheduler/prenight/sbatch/run_prenight_sims_%A_%a.out  # Error file (stderr)
#SBATCH --partition=milano              # Partition (queue) names
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks run in parallel
#SBATCH --cpus-per-task=1               # Number of CPUs per task
#SBATCH --mem=8G                        # Requested memory
#SBATCH --time=2:30:00                  # Wall time (hh:mm:ss)

# Design documentation: batch/design.md in the lsst_survey_sim repository
# https://github.com/lsst-sims/lsst_survey_sim/blob/main/batch/design.md

echo "******** START of run_prenight_sims.sh **********"
echo "Design docs: https://github.com/lsst-sims/lsst_survey_sim/blob/main/batch/design.md"

# Source global definitions
if [ -f /etc/bashrc ]; then
    . /etc/bashrc
fi

# SLAC S3DF - source all files under ~/.profile.d
if [[ -d ~/.profile.d ]]; then
    shopt -s nullglob
    for S3DF_CONF in ~/.profile.d/*-rubin/*.conf; do
        source "$S3DF_CONF"
    done
    shopt -u nullglob
fi

date --iso=s

# The gate files provide a mechanism that scheduler group members
# can use to stop this script from running, so if a cron job is
# running it it can still be stopped when the owner is not
# available.
# This is accomplished by deleting the gate file
# with the name of the owner of the cron job.
CRONGATE="/sdf/data/rubin/shared/scheduler/cron_gates/run_prenight_sims/${USER}"
if [ ! -e "${CRONGATE}" ]; then
    echo "Aborting because ${CRONGATE} does not exist."
    echo "See /sdf/data/rubin/shared/scheduler/cron_gates/README.txt"
    exit 1
fi

# Re-execute under the rubin_users group if we are not already in it.
# (Using "newgrp" directly in a non-interactive script spawns a new
# shell and does not affect the remainder of the script.)
if [ "$(id -gn)" != "rubin_users" ]; then
    exec sg rubin_users -c "$(printf '%q ' "$0" "$@")"
fi

set -x
set -euo pipefail

##############################################################################
# Dates to run
##############################################################################

if [ "${DAYOBS+x}" = "x" ]; then
    if [[ ! "${DAYOBS}" =~ ^[0-9]{8}$ ]]; then
        echo "ERROR: DAYOBS is set but not in YYYYMMDD format: '${DAYOBS}'" >&2
        exit 1
    fi
    if ! date -u --date="${DAYOBS}" +'%Y%m%d' >/dev/null 2>&1; then
        echo "ERROR: DAYOBS is set but is not a valid calendar date: '${DAYOBS}'" >&2
        exit 1
    fi
    export DAYOBS
else
    export DAYOBS="$(date -u --date='-12 hours' +'%Y%m%d')"
fi

export NEXT_DAYOBS="$(date -u --date="${DAYOBS} +1 day" +'%Y%m%d')"
export LAST_DAYOBS="$(date -u --date="${DAYOBS} +2 days" +'%Y%m%d')"
export DAYOBS_SIMULATED="${DAYOBS} ${NEXT_DAYOBS} ${LAST_DAYOBS}"
export LASTNIGHTISO="$(date -u --date="${DAYOBS} -1 day" +'%F')"

##############################################################################
# Constants
##############################################################################

LSST_SURVEY_SIM_REFERENCE="main"


readonly SIM_NIGHTS=3
readonly TS_CONFIG_SCHEDULER_REFERENCE="develop"
readonly SCHEDULER_GROUP_USERS="lynnej neilsen yoachim"
readonly SCHED_CONFIG_REPO_BASE="https://github.com/lsst-ts/ts_config_scheduler"
readonly RELATIVE_SCHED_CONFIG_FNAME="Scheduler/feature_scheduler/maintel/fbs_config_lsst_survey.py"
readonly SCHED_CONFIG_FNAME="ts_config_scheduler/${RELATIVE_SCHED_CONFIG_FNAME}"
readonly ARCHIVE="s3://rubin:rubin-scheduler-prenight/opsim/vseq/"

readonly PRENIGHT_WORK_ROOT="/sdf/data/rubin/shared/scheduler/prenight/work/run_prenight_sims"
readonly PRENIGHT_VENV_ROOT="/sdf/scratch/users/${USER:0:1}/${USER}/prenight_venvs"

export AWS_PROFILE=prenight
export VSARCHIVE_PGDATABASE="opsim_log"
export VSARCHIVE_PGHOST="usdf-maf-visit-seq-archive-tx.sdf.slac.stanford.edu"
export VSARCHIVE_PGUSER="writer"
export VSARCHIVE_PGSCHEMA="vsmd"

##############################################################################
# Helper functions
##############################################################################

log() {
    echo "$*"
    date --iso=s
}

require_commands() {
    local MISSING=()
    local CMD
    for CMD in "$@"; do
        if ! command -v "${CMD}" >/dev/null 2>&1; then
            MISSING+=("${CMD}")
        fi
    done

    if [ "${#MISSING[@]}" -ne 0 ]; then
        echo "ERROR: Missing required command(s): ${MISSING[*]}" >&2
        return 1
    fi
}

# Check that a directory exists, is writable, has the expected ACLs,
# and resides on a filesystem with at least MIN_KB of free space.
# ACL behavior can be strict (missing ACLs fail) or warn-only.
check_dir_ready() {
    local DIR="$1"
    local MIN_KB="$2"
    local LABEL="$3"
    local ACL_MODE="${4:-strict}"  # strict|warn
    local SCHEDULER_USER FACL AVAIL_KB ACL_ISSUES=0

    if [ ! -d "${DIR}" ]; then
        echo "ERROR: ${LABEL} directory does not exist: '${DIR}'." \
             "Create it and set ACLs." >&2
        return 1
    fi
    if [ ! -w "${DIR}" ]; then
        echo "ERROR: ${LABEL} directory is not writable: '${DIR}'" >&2
        return 1
    fi

    FACL=$(getfacl -p --omit-header "${DIR}")
    for SCHEDULER_USER in ${SCHEDULER_GROUP_USERS}; do
        if ! grep -q "^default:user:${SCHEDULER_USER}:rwx$" <<< "${FACL}"; then
            if [ "${ACL_MODE}" = "warn" ]; then
                echo "WARNING: ${LABEL} '${DIR}' is missing default ACL for user '${SCHEDULER_USER}'." \
                     "Recommended: setfacl -m u:${SCHEDULER_USER}:rwX -d -m u:${SCHEDULER_USER}:rwX '${DIR}'" >&2
                ACL_ISSUES=1
            else
                echo "ERROR: ${LABEL} '${DIR}' is missing default ACL for" \
                     "user '${SCHEDULER_USER}'. Run: setfacl -m u:${SCHEDULER_USER}:rwX -d -m u:${SCHEDULER_USER}:rwX '${DIR}'" >&2
                return 1
            fi
        fi
    done

    if grep -q '#effective' <<< "${FACL}"; then
        if [ "${ACL_MODE}" = "warn" ]; then
            echo "WARNING: ACL mask on '${DIR}' is clamping entries; check getfacl output." >&2
            ACL_ISSUES=1
        else
            echo "ERROR: ACL mask on '${DIR}' is clamping entries; check getfacl output." >&2
            return 1
        fi
    fi

    AVAIL_KB=$(df -Pk "${DIR}" | awk 'NR==2 {print $4}')

    if [ -z "${AVAIL_KB}" ]; then
        echo "ERROR: Could not determine free disk space for ${LABEL} path '${DIR}'" >&2
        return 1
    fi

    if [ "${AVAIL_KB}" -lt "${MIN_KB}" ]; then
        echo "ERROR: Insufficient disk space for ${LABEL} at '${DIR}': ${AVAIL_KB} KB available, ${MIN_KB} KB required." >&2
        return 1
    fi

    if [ "${ACL_ISSUES}" -eq 1 ] && [ "${ACL_MODE}" = "warn" ]; then
        log "Directory check passed for ${LABEL} at '${DIR}' with ACL warnings: ${AVAIL_KB} KB available (required ${MIN_KB} KB)."
    else
        log "Directory check passed for ${LABEL} at '${DIR}': ${AVAIL_KB} KB available (required ${MIN_KB} KB)."
    fi
}

preflight_check() {
    log "Running preflight dependency checks"

    # Available from base system / loaded profile.
    require_commands \
        date id sg ls find cat mktemp mkdir ln rm chmod setfacl \
        git curl jq df awk tar || {
        echo "ERROR: Base dependency preflight failed." >&2
        exit 1
    }

    # Check static paths/files needed later.
    if [ ! -f /sdf/group/rubin/sw/w_latest/loadLSST.sh ]; then
        echo "ERROR: Missing /sdf/group/rubin/sw/w_latest/loadLSST.sh" >&2
        exit 1
    fi

    if [ ! -f ~/.lsst/usdf_access_token ]; then
        echo "ERROR: Missing token file ~/.lsst/usdf_access_token" >&2
        exit 1
    fi

    # Checks for planned working directories.
    check_dir_ready "${PRENIGHT_WORK_ROOT}" 5242880 "WORK_ROOT" strict || exit 1
    check_dir_ready "${PRENIGHT_VENV_ROOT}" 2097152 "VENV_ROOT" warn || true
}

# Validate that a string is a canonical UUID (8-4-4-4-12 hex chars).
is_valid_uuid() {
    local UUID_CANDIDATE="$1"
    [[ "${UUID_CANDIDATE}" =~ ^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ ]]
}

# Grant read/write access on the given paths to all scheduler group users.
grant_group_access() {
    local PATH_ITEM SCHEDULER_GROUP_USER
    for PATH_ITEM in "$@"; do
        for SCHEDULER_GROUP_USER in ${SCHEDULER_GROUP_USERS}; do
            setfacl -m "u:${SCHEDULER_GROUP_USER}:rwX" "${PATH_ITEM}"
        done
    done
}

# Rebuild the prenight index for every night we simulate.
# Called repeatedly so partial progress remains visible even if a
# later simulation fails.
update_prenight_index() {
    local DAYOBS_TO_INDEX
    for DAYOBS_TO_INDEX in ${DAYOBS_SIMULATED}; do
        vseqarchive make-prenight-index "${DAYOBS_TO_INDEX}" simonyi
    done
}

# Run one simulation, record its metadata, archive its outputs,
# and clean up its intermediate files.
#
# Usage:
#   run_and_archive_sim OBSERVATORY_FILE RUN_PREFIX LABEL_DESC \
#       KEEP_REWARDS(true|false) ADD_STATS(true|false) "TAGS..." \
#       [extra run_lsst_sim args...]
run_and_archive_sim() {
    local OBSERVATORY_FILE="$1"
    local RUN_PREFIX="$2"
    local LABEL_DESC="$3"
    local KEEP_REWARDS="$4"
    local ADD_STATS="$5"
    local TAGS="$6"
    shift 6

    local TIMESTAMP OPSIMRUN LABEL SIM_UUID
    TIMESTAMP=$(date --iso=s)
    OPSIMRUN="${RUN_PREFIX}_${TIMESTAMP}"
    LABEL="${LABEL_DESC}, run at ${TIMESTAMP}"

    local REWARD_ARGS=()
    if [ "${KEEP_REWARDS}" = "true" ]; then
        REWARD_ARGS+=(--keep_rewards)
    fi

    log "Running simulation ${OPSIMRUN}"
    run_lsst_sim scheduler.p "${OBSERVATORY_FILE}" "" "${DAYOBS}" "${SIM_NIGHTS}" "${OPSIMRUN}" \
        ${REWARD_ARGS[@]+"${REWARD_ARGS[@]}"} \
        --label "${LABEL}" \
        "$@" \
        --results "${OPSIM_RESULT_DIR}"

    log "Creating entry in metadata database"
    SIM_UUID=$(vseqarchive record-visitseq-metadata \
        simulations \
        "${OPSIM_RESULT_DIR}/opsim.db" \
        "${LABEL}" \
        --first_day_obs "${DAYOBS}" \
        --last_day_obs "${LAST_DAYOBS}")

    if ! is_valid_uuid "${SIM_UUID}"; then
        echo "ERROR: Invalid SIM_UUID returned by vseqarchive: '${SIM_UUID}'" >&2
        exit 1
    fi

    log "SIM_SUMMARY LABEL=\"${LABEL}\" SIM_UUID=\"${SIM_UUID}\" TAGS=\"${TAGS}\""

    vseqarchive update-visitseq-metadata "${SIM_UUID}" parent_visitseq_uuid "${COMPLETED}"
    vseqarchive update-visitseq-metadata "${SIM_UUID}" parent_last_day_obs "${LASTNIGHTISO}"
    vseqarchive update-visitseq-metadata "${SIM_UUID}" scheduler_version "${RUBIN_SCHEDULER_VERSION}"
    vseqarchive update-visitseq-metadata "${SIM_UUID}" conda_env_sha256 "${CONDA_ENV_HASH}"
    vseqarchive update-visitseq-metadata "${SIM_UUID}" config_url "${SCHED_CONFIG_URL}"

    vseqarchive archive-file "${SIM_UUID}" "${OPSIM_RESULT_DIR}/opsim.db" visits --archive-base "${ARCHIVE}"
    if [ "${KEEP_REWARDS}" = "true" ]; then
        vseqarchive archive-file "${SIM_UUID}" "${OPSIM_RESULT_DIR}/rewards.h5" rewards --archive-base "${ARCHIVE}"
    fi

    # Word-splitting of ${TAGS} is intentional.
    vseqarchive tag "${SIM_UUID}" ${TAGS}

    if [ "${ADD_STATS}" = "true" ]; then
        vseqarchive get-file "${SIM_UUID}" visits visits.h5
        vseqarchive add-nightly-stats "${SIM_UUID}" visits.h5 azimuth altitude
        rm -f visits.h5
    fi

    # Clean up per-simulation outputs so the next run starts fresh.
    rm -f \
        "${OPSIM_RESULT_DIR}/opsim.db" \
        "${OPSIM_RESULT_DIR}/rewards.h5" \
        "${OPSIM_RESULT_DIR}/obs_stats.txt" \
        "${OPSIM_RESULT_DIR}/sim_metadata.yaml" \
        "${OPSIM_RESULT_DIR}"/*.p
}

on_exit() {
    local STATUS=$?
    if [ "${STATUS}" -ne 0 ]; then
        echo "run_prenight_sims.sh FAILED with exit status ${STATUS}" >&2
    fi
    echo "Design docs: https://github.com/lsst-sims/lsst_survey_sim/blob/main/batch/design.md"
    echo "******** END of run_prenight_sims.sh (status ${STATUS}) **********"
    date --iso=s
}
trap on_exit EXIT

preflight_check

##############################################################################
# Working directory
##############################################################################

WORK_DIR_WAIT_START=$(date +%s)
WORK_DIR_WAIT_TIMEOUT=3600

while true; do
    WORK_DATE=$(date '+%Y-%m-%dT%H%M%S' --utc)
    WORK_DIR="${PRENIGHT_WORK_ROOT}/${WORK_DATE}"
    if [ -e "${WORK_DIR}" ]; then
        WORK_DIR_WAIT_ELAPSED=$(( $(date +%s) - WORK_DIR_WAIT_START ))
        if [ "${WORK_DIR_WAIT_ELAPSED}" -ge "${WORK_DIR_WAIT_TIMEOUT}" ]; then
            echo "ERROR: Timed out after ${WORK_DIR_WAIT_TIMEOUT} seconds waiting for a unique working directory." >&2
            exit 1
        fi
        echo "Working directory already exists: ${WORK_DIR}; waiting before retrying to avoid collision."
        sleep 3
        continue
    fi
    break
done

echo "Working in ${WORK_DIR}"
mkdir -p "${WORK_DIR}"
grant_group_access "${WORK_DIR}"
cd "${WORK_DIR}"

##############################################################################
# Environment setup
##############################################################################

# Install required python packages in a new conda env.
# The true environment is from the one created below,
# but we source loadLSST.sh first to get conda into
# our path.
# Using a plain (non-conda) env would mostly work,
# but using a conda env better supports getting
# more version information into the simulation
# metadata database, even though the packages
# involved will be installed with pip.

# loadLSST.sh fails when set -u is on
set +u
source /sdf/group/rubin/sw/w_latest/loadLSST.sh
set -u

PRENIGHT_VENV=$(mktemp -d "${PRENIGHT_VENV_ROOT}/prenight-${WORK_DATE}-XXXXXX")
conda create --prefix "${PRENIGHT_VENV}" --yes python=3.13 --quiet
ln -s "${PRENIGHT_VENV}" "${WORK_DIR}/venv"

echo "activating environment ${PRENIGHT_VENV}"
set +u
conda activate "${PRENIGHT_VENV}"
set -u

# proj not available from pip
conda install proj --yes --quiet

# If not set in the constants block above, set
# LSST_SURVEY_SIM_REFERENCE to the semantic
# version in github.
if [ -z "${LSST_SURVEY_SIM_REFERENCE:-}" ]; then
    # The github tags API is paginated, so a single request may not
    # return all tags. Walk through the pages (using the maximum
    # allowed page size) and accumulate all tag names before
    # selecting the highest semantic version.
    LSST_SURVEY_SIM_TAGS_URL="https://api.github.com/repos/lsst-sims/lsst_survey_sim/tags"
    ALL_TAG_NAMES=""
    TAGS_PAGE=1
    MAX_TAGS_PAGES=100
    while [ "${TAGS_PAGE}" -le "${MAX_TAGS_PAGES}" ]; do
        PAGE_TAG_NAMES=$(curl -s "${LSST_SURVEY_SIM_TAGS_URL}?per_page=100&page=${TAGS_PAGE}" \
            | jq -r '.[]?.name? // empty')
        if [ -z "${PAGE_TAG_NAMES}" ]; then
            break
        fi
        ALL_TAG_NAMES="${ALL_TAG_NAMES}${PAGE_TAG_NAMES}"$'\n'
        TAGS_PAGE=$((TAGS_PAGE + 1))
    done
    LSST_SURVEY_SIM_REFERENCE=$(printf '%s' "${ALL_TAG_NAMES}" \
        | { grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' || true; } \
        | sort -V \
        | tail -1)
    LSST_SURVEY_SIM_REFERENCE="${LSST_SURVEY_SIM_REFERENCE:-}"
fi

# If the user has packages installed locally, this can keep pip
# from installing them, but if this is run with cron, shell
# executables like fetch_lsst_visits will not be in the path.
# Be sure to get all packages into the environment in PRENIGHT_VENV.
pip install --progress-bar off --force-reinstall \
    "git+https://github.com/lsst-sims/lsst_survey_sim.git@${LSST_SURVEY_SIM_REFERENCE}"

# Commands expected from the newly-created environment.
require_commands \
    fetch_lsst_visits \
    make_lsst_scheduler \
    make_model_observatory \
    make_band_scheduler \
    run_lsst_sim \
    vseqarchive || {
    echo "ERROR: Environment dependency preflight failed after activation/install." >&2
    exit 1
}

##############################################################################
# Scheduler configuration
##############################################################################

# Get the scheduler configuration script.
# It lives in ts_config_scheduler.
echo "Using ts_config_scheduler ${SCHED_CONFIG_FNAME} from ${TS_CONFIG_SCHEDULER_REFERENCE}"
git clone --depth 1 "${SCHED_CONFIG_REPO_BASE}"
cd ts_config_scheduler
git fetch --depth 1 origin "${TS_CONFIG_SCHEDULER_REFERENCE}"
git checkout FETCH_HEAD

# Get the URL for archiving
SCHED_CONFIG_HASH=$(git rev-parse HEAD)
SCHED_CONFIG_URL="${SCHED_CONFIG_REPO_BASE}/blob/${SCHED_CONFIG_HASH}/${RELATIVE_SCHED_CONFIG_FNAME}"

cd "${WORK_DIR}"

##############################################################################
# Environment
##############################################################################

export RUBIN_SCHEDULER_VERSION="$(conda list rubin-scheduler --json | jq -r '.[0].version')"

# Save the conda environment specification (once; the environment does
# not change during this job).
export CONDA_ENV_HASH=$(vseqarchive record-conda-env)

##############################################################################
# Completed visits
##############################################################################

log "Fetching completed visits"
fetch_lsst_visits "${DAYOBS}" completed_visits.db ~/.lsst/usdf_access_token
grant_group_access completed_visits.db

# Record hash of fetched visits
COMPLETED=$(vseqarchive record-visitseq-metadata \
    completed \
    completed_visits.db \
    "Consdb query through ${LASTNIGHTISO}" \
    --first_day_obs 20250620 \
    --last_day_obs "${LASTNIGHTISO}")

if ! is_valid_uuid "${COMPLETED}"; then
    echo "ERROR: Invalid UUID returned by vseqarchive: '${COMPLETED}'" >&2
    exit 1
fi


##############################################################################
# Simulation inputs
##############################################################################

log "Creating scheduler pickle"
make_lsst_scheduler scheduler.p --opsim completed_visits.db --config_script "${SCHED_CONFIG_FNAME}"
grant_group_access scheduler.p

log "Creating model observatory"
# Use the seeing value from line 2 of table 9 of LPM-017
make_model_observatory observatory.p --seeing 0.6
grant_group_access observatory.p

log "Creating the band scheduler"
make_band_scheduler band_scheduler.p
grant_group_access band_scheduler.p

# Make dir for output
OPSIM_RESULT_DIR="${WORK_DIR}/opsim_results"
mkdir -p "${OPSIM_RESULT_DIR}"
grant_group_access "${OPSIM_RESULT_DIR}"

##############################################################################
# Simulations
##############################################################################

run_and_archive_sim observatory.p \
    "prenight_nominal_noreward" \
    "Nominal start and overhead, ideal conditions" \
    false false \
    "prenight ideal nominal" \
    --delay 0 --anom_overhead_scale 0

# Update the index here to make sure it has at least
# the completed nominal simulation, even if something in the rest
# of this job fails.
update_prenight_index

run_and_archive_sim observatory.p \
    "prenight_nominal" \
    "Nominal start and overhead, ideal conditions" \
    true true \
    "prenight ideal nominal rewards" \
    --delay 0 --anom_overhead_scale 0

# Update the index here so that if the prenight report gets updated,
# it can see this simulation even if the ones that follow fail.
update_prenight_index

DELAY=240
run_and_archive_sim observatory.p \
    "prenight_delay${DELAY}" \
    "Start time delayed by ${DELAY} minutes, nominal slew and visit overhead, ideal conditions" \
    false true \
    "prenight ideal delay_${DELAY}" \
    --delay "${DELAY}" --anom_overhead_scale 0

ANOM_SCALE="10.0"
ANOM_SEED=101
run_and_archive_sim observatory.p \
    "prenight_anom${ANOM_SEED}" \
    "Anomalous overhead (${ANOM_SEED}, ${ANOM_SCALE}), nominal start, ideal conditions" \
    false true \
    "prenight ideal anomalous_overhead" \
    --delay 0 \
    --anom_overhead_scale "${ANOM_SCALE}" \
    --anom_overhead_seed "${ANOM_SEED}"

rm -f observatory.p

# Run a simulation with good seeing.
# Use the value from line 1 of table 9 of LPM-017:
# an observatory with zenith 500nm seeing of 0.44.
make_model_observatory observatory_seeing044.p --seeing 0.44
grant_group_access observatory_seeing044.p

run_and_archive_sim observatory_seeing044.p \
    "prenight_seeing044" \
    "Nominal start and overhead, seeing=0.44" \
    true true \
    "prenight seeing044 nominal rewards" \
    --delay 0 --anom_overhead_scale 0

rm -f observatory_seeing044.p

# Run a simulation with poor seeing.
# Make it high enough that the template tier is never triggered:
# an observatory with zenith 500nm seeing of 1.3.
make_model_observatory observatory_seeing130.p --seeing 1.3
grant_group_access observatory_seeing130.p

run_and_archive_sim observatory_seeing130.p \
    "prenight_seeing130" \
    "Nominal start and overhead, seeing=1.3" \
    true true \
    "prenight seeing130 nominal rewards" \
    --delay 0 --anom_overhead_scale 0

rm -f observatory_seeing130.p scheduler.p

##############################################################################
# Final index update
##############################################################################

update_prenight_index

##############################################################################
# Mark this work directory as complete for cleanup_prenight.sh
##############################################################################

touch "${WORK_DIR}/.done"
