#!/usr/bin/env bash
#SBATCH --account=rubin:developers      # Account name
#SBATCH --job-name=write_prenight_cleanup  # Job name
#SBATCH --output=/sdf/data/rubin/shared/scheduler/prenight/sbatch/write_cleanup_prenight_%A.out # Output file (stdout)
#SBATCH --error=/sdf/data/rubin/shared/scheduler/prenight/sbatch/write_cleanup_prenight_%A.out  # Error file (stderr)
#SBATCH --partition=milano              # Partition (queue) names
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks run in parallel
#SBATCH --cpus-per-task=1               # Number of CPUs per task
#SBATCH --mem=2G                        # Requested memory
#SBATCH --time=0:30:00                  # Wall time (hh:mm:ss)

# Design documentation: batch/design.md in the lsst_survey_sim repository
# https://github.com/lsst-sims/lsst_survey_sim/blob/main/batch/design.md


echo "******** START of write_cleanup_prenight.sh **********"
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
# Skip the gate check when running interactively (stdin is a terminal).
if [ ! -t 0 ]; then
    CRONGATE="/sdf/data/rubin/shared/scheduler/cron_gates/cleanup_prenight/${USER}"
    if [ ! -e "${CRONGATE}" ]; then
        echo "Aborting because ${CRONGATE} does not exist."
        echo "See /sdf/data/rubin/shared/scheduler/cron_gates/README.txt"
        exit 1
    fi
fi

# Re-execute under the rubin_users group if we are not already in it.
if [ "$(id -gn)" != "rubin_users" ]; then
    exec sg rubin_users -c "$(printf '%q ' "$0" "$@")"
fi

set -x
set -euo pipefail

##############################################################################
# Constants
##############################################################################

readonly SIMONYI_WORK_ROOT="/sdf/data/rubin/shared/scheduler/prenight/work/run_prenight_sims"
readonly AUXTEL_WORK_ROOT="/sdf/data/rubin/shared/scheduler/prenight/work/run_auxtel_prenight_sims"
readonly PRENIGHT_VENV_ROOT="/sdf/scratch/users/${USER:0:1}/${USER}/prenight_venvs"
readonly ARCHIVE_OFFLOAD_DIR="/sdf/data/rubin/user/neilsen/data/run_prenight_sims"
readonly MIN_WORK_FREE_KB=10485760  # 10 GiB
readonly MIN_VENV_FREE_KB=10485760  # 10 GiB

readonly OUTPUT_DIR="/sdf/data/rubin/shared/scheduler/prenight/cleanup_scripts"

##############################################################################
# Helper functions
##############################################################################

log() {
    echo "$*"
    date --iso=s
}

on_exit() {
    local STATUS=$?
    if [ "${STATUS}" -ne 0 ]; then
        echo "write_cleanup_prenight.sh FAILED with exit status ${STATUS}" >&2
    fi
    echo "Design docs: https://github.com/lsst-sims/lsst_survey_sim/blob/main/batch/design.md"
    echo "******** END of write_cleanup_prenight.sh (status ${STATUS}) **********"
    date --iso=s
}
trap on_exit EXIT

##############################################################################
# Create the output script
##############################################################################

mkdir -p "${OUTPUT_DIR}"
TIMESTAMP=$(date '+%Y-%m-%dT%H%M%S' --utc)
OUTPUT_SCRIPT="${OUTPUT_DIR}/cleanup_prenight_${TIMESTAMP}.sh"

cat > "${OUTPUT_SCRIPT}" << 'INNER_HEADER'
#!/usr/bin/env bash
# Auto-generated cleanup script.
# Review each command below, then execute this script to perform cleanup.

set -x
set -euo pipefail

INNER_HEADER

##############################################################################
# Emit commands to archive and remove completed work directories
##############################################################################

emit_work_dir_cleanup() {
    local WORK_ROOT="$1"

    if [ ! -d "${WORK_ROOT}" ]; then
        echo "# Work root does not exist, skipping: ${WORK_ROOT}" >> "${OUTPUT_SCRIPT}"
        echo "" >> "${OUTPUT_SCRIPT}"
        return 0
    fi

    local WORK_DIR WORK_DIR_NAME VENV_TARGET VENV_NAME
    for WORK_DIR in "${WORK_ROOT}"/*/; do
        # Skip if glob didn't match anything
        [ -d "${WORK_DIR}" ] || continue

        # Only process directories with a .done marker
        if [ ! -f "${WORK_DIR}/.done" ]; then
            continue
        fi

        WORK_DIR_NAME="$(basename "${WORK_DIR}")"

        # Validate the directory name is in the expected timestamp format
        if [[ ! "${WORK_DIR_NAME}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{6}$ ]]; then
            echo "# WARNING: Skipping directory with unexpected name format: '${WORK_DIR_NAME}'" >> "${OUTPUT_SCRIPT}"
            continue
        fi

        {
            echo "# Archive and remove work directory: ${WORK_ROOT}/${WORK_DIR_NAME}"
            echo "cd ${WORK_ROOT}"
            echo "tar -czf ${WORK_ROOT}/${WORK_DIR_NAME}.tgz ${WORK_DIR_NAME}"
            echo ""
        } >> "${OUTPUT_SCRIPT}"

        # Emit venv archive/removal if the symlink exists
        if [ -L "${WORK_DIR}/venv" ]; then
            VENV_TARGET="$(readlink -f "${WORK_DIR}/venv")"
            if [ -d "${VENV_TARGET}" ]; then
                # Guard: only operate on venvs whose absolute path is
                # directly under PRENIGHT_VENV_ROOT.
                if [[ "${VENV_TARGET}" != "${PRENIGHT_VENV_ROOT}/"* ]]; then
                    echo "# WARNING: Venv target '${VENV_TARGET}' is not under '${PRENIGHT_VENV_ROOT}'; skipping." >> "${OUTPUT_SCRIPT}"
                else
                    VENV_NAME="$(basename "${VENV_TARGET}")"
                    if [[ "${VENV_NAME}" =~ ^prenight-[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{6}-[A-Za-z0-9]{6}$ ]]; then
                        {
                            echo "# Archive and remove venv: ${PRENIGHT_VENV_ROOT}/${VENV_NAME}"
                            echo "cd ${PRENIGHT_VENV_ROOT}"
                            echo "tar -czf ${PRENIGHT_VENV_ROOT}/${VENV_NAME}.tgz ${VENV_NAME}"
                            echo "rm -r --one-file-system ${PRENIGHT_VENV_ROOT}/${VENV_NAME}"
                            echo ""
                        } >> "${OUTPUT_SCRIPT}"
                    else
                        echo "# WARNING: Venv name does not match expected pattern, skipping: '${VENV_NAME}'" >> "${OUTPUT_SCRIPT}"
                    fi
                fi
            fi
        fi

        # Emit removal of the work directory
        {
            echo "# Remove work directory: ${WORK_ROOT}/${WORK_DIR_NAME}"
            echo "rm -f --one-file-system ${WORK_ROOT}/${WORK_DIR_NAME}/ts_config_scheduler/.git/objects/pack/pack-*.pack"
            echo "rm -f --one-file-system ${WORK_ROOT}/${WORK_DIR_NAME}/ts_config_scheduler/.git/objects/pack/pack-*.idx"
            echo "rm -r --one-file-system ${WORK_ROOT}/${WORK_DIR_NAME}"
            echo ""
        } >> "${OUTPUT_SCRIPT}"
    done
}

log "Scanning simonyi prenight work directories"
echo "##############################################################################" >> "${OUTPUT_SCRIPT}"
echo "# Simonyi prenight work directories" >> "${OUTPUT_SCRIPT}"
echo "##############################################################################" >> "${OUTPUT_SCRIPT}"
echo "" >> "${OUTPUT_SCRIPT}"
emit_work_dir_cleanup "${SIMONYI_WORK_ROOT}"

log "Scanning auxtel prenight work directories"
echo "##############################################################################" >> "${OUTPUT_SCRIPT}"
echo "# Auxtel prenight work directories" >> "${OUTPUT_SCRIPT}"
echo "##############################################################################" >> "${OUTPUT_SCRIPT}"
echo "" >> "${OUTPUT_SCRIPT}"
emit_work_dir_cleanup "${AUXTEL_WORK_ROOT}"

##############################################################################
# Emit conditional archive offload commands
##############################################################################

echo "##############################################################################" >> "${OUTPUT_SCRIPT}"
echo "# Conditional archive offload" >> "${OUTPUT_SCRIPT}"
echo "##############################################################################" >> "${OUTPUT_SCRIPT}"
echo "" >> "${OUTPUT_SCRIPT}"

WORK_AVAIL_KB=$(df -Pk "${SIMONYI_WORK_ROOT}" | awk 'NR==2 {print $4}')
if [ -z "${WORK_AVAIL_KB}" ]; then
    echo "# WARNING: Could not determine free space; skipping archive offload." >> "${OUTPUT_SCRIPT}"
elif [ "${WORK_AVAIL_KB}" -lt "${MIN_WORK_FREE_KB}" ]; then
    {
        echo "# Free space is ${WORK_AVAIL_KB} KB (below ${MIN_WORK_FREE_KB} KB threshold)."
        echo "# Moving .tgz archives older than 30 days to offload directory."
        echo "mkdir -p ${ARCHIVE_OFFLOAD_DIR}"
        echo ""
    } >> "${OUTPUT_SCRIPT}"

    # Find actual files and emit explicit mv commands
    find "${SIMONYI_WORK_ROOT}" -maxdepth 1 -type f -name "*.tgz" -mtime +30 -print0 \
        | while IFS= read -r -d '' OLD_TGZ; do
            echo "mv -n ${OLD_TGZ} ${ARCHIVE_OFFLOAD_DIR}/" >> "${OUTPUT_SCRIPT}"
        done

    find "${AUXTEL_WORK_ROOT}" -maxdepth 1 -type f -name "*.tgz" -mtime +30 -print0 \
        | while IFS= read -r -d '' OLD_TGZ; do
            echo "mv -n ${OLD_TGZ} ${ARCHIVE_OFFLOAD_DIR}/" >> "${OUTPUT_SCRIPT}"
        done

    echo "" >> "${OUTPUT_SCRIPT}"
else
    echo "# Free space is ${WORK_AVAIL_KB} KB (above ${MIN_WORK_FREE_KB} KB threshold); no archive offload needed." >> "${OUTPUT_SCRIPT}"
fi

echo "" >> "${OUTPUT_SCRIPT}"

##############################################################################
# Emit conditional venv archive cleanup commands
##############################################################################

echo "##############################################################################" >> "${OUTPUT_SCRIPT}"
echo "# Conditional prenight venv archive cleanup" >> "${OUTPUT_SCRIPT}"
echo "##############################################################################" >> "${OUTPUT_SCRIPT}"
echo "" >> "${OUTPUT_SCRIPT}"

if [ ! -d "${PRENIGHT_VENV_ROOT}" ]; then
    echo "# Prenight venv root does not exist: ${PRENIGHT_VENV_ROOT}" >> "${OUTPUT_SCRIPT}"
else
    VENV_AVAIL_KB=$(df -Pk "${PRENIGHT_VENV_ROOT}" | awk 'NR==2 {print $4}')
    if [ -z "${VENV_AVAIL_KB}" ]; then
        echo "# WARNING: Could not determine free space for '${PRENIGHT_VENV_ROOT}'; skipping." >> "${OUTPUT_SCRIPT}"
    elif [ "${VENV_AVAIL_KB}" -lt "${MIN_VENV_FREE_KB}" ]; then
        {
            echo "# Free space is ${VENV_AVAIL_KB} KB (below ${MIN_VENV_FREE_KB} KB threshold)."
            echo "# Removing prenight venv archives older than 30 days."
            echo ""
        } >> "${OUTPUT_SCRIPT}"

        find "${PRENIGHT_VENV_ROOT}" \
            -mindepth 1 -maxdepth 1 \
            -type f \
            -name "prenight-*.tgz" \
            -mtime +30 \
            -print0 \
            | while IFS= read -r -d '' OLD_VENV_TGZ; do
                echo "rm -f --one-file-system ${OLD_VENV_TGZ}" >> "${OUTPUT_SCRIPT}"
            done

        echo "" >> "${OUTPUT_SCRIPT}"
    else
        echo "# Free space is ${VENV_AVAIL_KB} KB (above ${MIN_VENV_FREE_KB} KB threshold); no venv archive cleanup needed." >> "${OUTPUT_SCRIPT}"
    fi
fi

##############################################################################
# Done
##############################################################################

chmod +x "${OUTPUT_SCRIPT}"
log "Cleanup script written to: ${OUTPUT_SCRIPT}"
echo "Review it with: cat ${OUTPUT_SCRIPT}"
echo "Execute it with: bash ${OUTPUT_SCRIPT}"
