#!/usr/bin/env bash
#SBATCH --account=rubin:developers      # Account name
#SBATCH --job-name=auxtel_prenight_daily   # Job name
#SBATCH --output=/sdf/data/rubin/shared/scheduler/prenight/sbatch/run_auxtel_prenight_sims_%A_%a.out # Output file (stdout)
#SBATCH --error=/sdf/data/rubin/shared/scheduler/prenight/sbatch/run_auxtel_prenight_sims_%A_%a.out  # Error file (stderr)
#SBATCH --partition=milano              # Partition (queue) names
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks run in parallel
#SBATCH --cpus-per-task=1               # Number of CPUs per task
#SBATCH --mem=8G                        # Requested memory
#SBATCH --time=1:30:00                  # Wall time (hh:mm:ss)

# Design documentation: batch/design.md in the lsst_survey_sim repository
# https://github.com/lsst-sims/lsst_survey_sim/blob/main/batch/design.md

echo "******** START of run_auxtel_prenight_sims.sh **********"
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
CRONGATE="/sdf/data/rubin/shared/scheduler/cron_gates/run_auxtel_prenight_sims/${USER}"
if [ ! -e "${CRONGATE}" ]; then
    echo "Aborting because ${CRONGATE} does not exist."
    echo "See /sdf/data/rubin/shared/scheduler/cron_gates/README.txt"
    exit 1
fi

# Re-execute under the rubin_users group if we are not already in it.
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

##############################################################################
# Constants
##############################################################################

LSST_SURVEY_SIM_REFERENCE="main"

readonly SIM_NIGHTS=3
readonly TS_CONFIG_SCHEDULER_REFERENCE="develop"
readonly SCHEDULER_GROUP_USERS="lynnej neilsen yoachim"
readonly SCHED_CONFIG_REPO_BASE="https://github.com/lsst-ts/ts_config_scheduler"
readonly RELATIVE_SCHED_CONFIG_FNAME="Scheduler/feature_scheduler/auxtel/fbs_spec_flex_survey.py"
readonly SCHED_CONFIG_FNAME="ts_config_scheduler/${RELATIVE_SCHED_CONFIG_FNAME}"
readonly ARCHIVE="s3://rubin:rubin-scheduler-prenight/opsim/vseq/"

readonly PRENIGHT_WORK_ROOT="/sdf/data/rubin/shared/scheduler/prenight/work/run_auxtel_prenight_sims"
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

check_min_free_space_kb() {
    local TARGET_PATH="$1"
    local MIN_KB="$2"
    local LABEL="$3"

    local AVAIL_KB
    AVAIL_KB=$(df -Pk "${TARGET_PATH}" | awk 'NR==2 {print $4}')

    if [ -z "${AVAIL_KB}" ]; then
        echo "ERROR: Could not determine free disk space for ${LABEL} path '${TARGET_PATH}'" >&2
        return 1
    fi

    if [ "${AVAIL_KB}" -lt "${MIN_KB}" ]; then
        echo "ERROR: Insufficient disk space for ${LABEL} at '${TARGET_PATH}': ${AVAIL_KB} KB available, ${MIN_KB} KB required." >&2
        return 1
    fi

    log "Disk space check passed for ${LABEL} at '${TARGET_PATH}': ${AVAIL_KB} KB available (required ${MIN_KB} KB)."
}

preflight_check() {
    log "Running preflight dependency checks"

    require_commands \
        date id sg ls find cat mktemp mkdir ln rm chmod setfacl \
        git curl jq df awk || {
        echo "ERROR: Base dependency preflight failed." >&2
        exit 1
    }

    if [ ! -f /sdf/group/rubin/sw/w_latest/loadLSST.sh ]; then
        echo "ERROR: Missing /sdf/group/rubin/sw/w_latest/loadLSST.sh" >&2
        exit 1
    fi

    if [ ! -f ~/.lsst/usdf_access_token ]; then
        echo "ERROR: Missing token file ~/.lsst/usdf_access_token" >&2
        exit 1
    fi

    check_min_free_space_kb "${PRENIGHT_WORK_ROOT}" 5242880 "WORK_DIR" || exit 1
    check_min_free_space_kb "${PRENIGHT_VENV_ROOT}" 2097152 "PRENIGHT_VENV" || exit 1
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
            setfacl -m "${SCHEDULER_GROUP_USER}:rwX" "${PATH_ITEM}"
        done
    done
}

on_exit() {
    local STATUS=$?
    if [ "${STATUS}" -ne 0 ]; then
        echo "run_auxtel_prenight_sims.sh FAILED with exit status ${STATUS}" >&2
    fi
    echo "Design docs: https://github.com/lsst-sims/lsst_survey_sim/blob/main/batch/design.md"
    echo "******** END of run_auxtel_prenight_sims.sh (status ${STATUS}) **********"
    date --iso=s
}
trap on_exit EXIT

preflight_check

##############################################################################
# Working directory
##############################################################################

while true; do
    WORK_DATE=$(date '+%Y-%m-%dT%H%M%S' --utc)
    WORK_DIR="${PRENIGHT_WORK_ROOT}/${WORK_DATE}"
    if [ -e "${WORK_DIR}" ]; then
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

pip install --progress-bar off --force-reinstall \
    "git+https://github.com/lsst-sims/lsst_survey_sim.git@${LSST_SURVEY_SIM_REFERENCE}"

# Commands expected from the newly-created environment.
require_commands \
    fetch_lsst_visits \
    make_lsst_scheduler \
    ideal_model_observatory \
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
# Environment metadata
##############################################################################

export RUBIN_SCHEDULER_VERSION="$(conda list rubin-scheduler --json | jq -r '.[0].version')"
export CONDA_ENV_HASH=$(vseqarchive record-conda-env)

##############################################################################
# Simulation inputs
##############################################################################

# Get an empty set of completed visits so we have something
# to pass make_lsst_scheduler (auxtel sims start from scratch).
log "Fetching placeholder completed visits"
fetch_lsst_visits 20000101 completed_visits.db ~/.lsst/usdf_access_token
grant_group_access completed_visits.db

log "Creating scheduler pickle"
make_lsst_scheduler scheduler.p --opsim completed_visits.db --config_script "${SCHED_CONFIG_FNAME}" --config_ddf_script ""
grant_group_access scheduler.p

log "Creating model observatory"
ideal_model_observatory scheduler.p observatory.p
grant_group_access observatory.p

# Make dir for output
OPSIM_RESULT_DIR="${WORK_DIR}/opsim_results"
mkdir -p "${OPSIM_RESULT_DIR}"
grant_group_access "${OPSIM_RESULT_DIR}"

##############################################################################
# Simulation
##############################################################################

TIMESTAMP=$(date --iso=s)
OPSIMRUN="prenight_auxtel_nominal_${TIMESTAMP}"
LABEL="Nominal start and overhead, ideal conditions, run at ${TIMESTAMP}"

log "Running auxtel simulation ${OPSIMRUN}"
run_lsst_sim scheduler.p observatory.p "" "${DAYOBS}" "${SIM_NIGHTS}" "${OPSIMRUN}" \
    --keep_rewards \
    --label "${LABEL}" \
    --delay 0 --anom_overhead_scale 0 \
    --results "${OPSIM_RESULT_DIR}"

log "Creating entry in metadata database"
SIM_UUID=$(vseqarchive record-visitseq-metadata \
    simulations \
    "${OPSIM_RESULT_DIR}/opsim.db" \
    "${LABEL}" \
    --telescope auxtel \
    --first_day_obs "${DAYOBS}" \
    --last_day_obs "${LAST_DAYOBS}")

if ! is_valid_uuid "${SIM_UUID}"; then
    echo "ERROR: Invalid SIM_UUID returned by vseqarchive: '${SIM_UUID}'" >&2
    exit 1
fi

log "SIM_SUMMARY LABEL=\"${LABEL}\" SIM_UUID=\"${SIM_UUID}\""

vseqarchive update-visitseq-metadata "${SIM_UUID}" scheduler_version "${RUBIN_SCHEDULER_VERSION}"
vseqarchive update-visitseq-metadata "${SIM_UUID}" conda_env_sha256 "${CONDA_ENV_HASH}"
vseqarchive update-visitseq-metadata "${SIM_UUID}" config_url "${SCHED_CONFIG_URL}"

vseqarchive archive-file "${SIM_UUID}" "${OPSIM_RESULT_DIR}/opsim.db" visits --archive-base "${ARCHIVE}"
vseqarchive archive-file "${SIM_UUID}" "${OPSIM_RESULT_DIR}/rewards.h5" rewards --archive-base "${ARCHIVE}"

vseqarchive tag "${SIM_UUID}" prenight ideal nominal rewards

vseqarchive get-file "${SIM_UUID}" visits visits.h5
vseqarchive add-nightly-stats "${SIM_UUID}" visits.h5 azimuth altitude
rm -f visits.h5

# Clean up simulation outputs.
rm -f \
    "${OPSIM_RESULT_DIR}/opsim.db" \
    "${OPSIM_RESULT_DIR}/rewards.h5" \
    "${OPSIM_RESULT_DIR}/obs_stats.txt" \
    "${OPSIM_RESULT_DIR}/sim_metadata.yaml" \
    "${OPSIM_RESULT_DIR}"/*.p

##############################################################################
# Update prenight index
##############################################################################

for DAYOBS_TO_INDEX in ${DAYOBS_SIMULATED}; do
    vseqarchive make-prenight-index "${DAYOBS_TO_INDEX}" auxtel
done

##############################################################################
# Mark this work directory as complete for cleanup_prenight.sh
##############################################################################

touch "${WORK_DIR}/.done"
