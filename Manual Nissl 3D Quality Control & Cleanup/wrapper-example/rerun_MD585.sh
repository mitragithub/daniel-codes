#! /bin/bash

source ./nissl/bin/activate

# Add emlddmm to PYTHONPATH
export PYTHONPATH="/home/mitralab/nissl-nissl:$PYTHONPATH"

# Function to display usage
usage() {
    echo "Usage: $0 -b BASE_DIR [-r RANGE] [-d] [-h]"
    echo "  -b: Base directory path"
    echo "  -r: Range to process (1, 2, or 3)"
    echo "      1: slices 0000-0099"
    echo "      2: slices 0100-0199"
    echo "      3: slices 0200-0299"
    echo "  -d: Dry run (only show what would be done)"
    echo "  -h: Display this help message"
    exit 1
}

# Function to echo debug messages
debug() {
    echo "[DEBUG] $1"
}

# Function to echo info messages
info() {
    echo "[INFO] $1"
}

# Function to echo error messages
error() {
    echo "[ERROR] $1 >&2"
}

# Parse command line arguments
DRY_RUN=false
while getopts "b:r:dh" opt; do
    case $opt in
        b) BASE_DIR="$OPTARG" ;;
        r) RANGE="$OPTARG" ;;
        d) DRY_RUN=true ;;
        h) usage ;;
        ?) usage ;;
    esac
done

# Validate BASE_DIR
if [ -z "$BASE_DIR" ]; then
    error "Missing required base directory argument"
    usage
fi

# Validate RANGE
if [ -z "$RANGE" ] || ! [[ "$RANGE" =~ ^[1-3]$ ]]; then
    error "Missing or invalid range argument (must be 1, 2, or 3)"
    usage
fi


# Set up directory structure
NISSL_DIR="${BASE_DIR}MD585_registration/MD585-nissl-registered/MD585-nissl_to_MD585-nissl-registered"
MRI_DIR="${BASE_DIR}MD585_registration/MD585-nissl-registered/mri_to_MD585-nissl-registered"
BASE_OUTPUT="/nfs/data/main/M38/nissl_manual_correction_output/MD585"
OUTPUT_DIR="${BASE_OUTPUT}/points"
DONE_DIR="${BASE_OUTPUT}/rerun_done"
LOCK_DIR="${BASE_OUTPUT}/rerun_lock"

debug "Directory structure:"
debug "NISSL_DIR: $NISSL_DIR"
debug "MRI_DIR: $MRI_DIR"
debug "OUTPUT_DIR: $OUTPUT_DIR"
debug "DONE_DIR: $DONE_DIR"
debug "LOCK_DIR: $LOCK_DIR"

# Create necessary directories
if [ "$DRY_RUN" = false ]; then
    mkdir -p "$OUTPUT_DIR" "$DONE_DIR" "$LOCK_DIR"
fi

# Function to check if slice is locked
is_slice_locked() {
    local slice_num="$1"
    [ -f "${LOCK_DIR}/slice_${slice_num}.lock" ]
}

# Function to check if slice is done
is_slice_done() {
    local slice_num="$1"
    [ -f "${DONE_DIR}/slice_${slice_num}.done" ]
}

# Function to lock slice
lock_slice() {
    local slice_num="$1"
    touch "${LOCK_DIR}/slice_${slice_num}.lock"
}

# Function to unlock slice
unlock_slice() {
    local slice_num="$1"
    rm -f "${LOCK_DIR}/slice_${slice_num}.lock"
}

# Function to mark slice as done
mark_slice_done() {
    local slice_num="$1"
    touch "${DONE_DIR}/slice_${slice_num}.done"
}

# Function to cleanup on exit
cleanup() {
    local slice_num="$1"
    info "Cleaning up..."
    unlock_slice "$slice_num"
    exit 1
}

# Set up trap for Ctrl+C
trap 'cleanup "$CURRENT_SLICE"' INT

# Function to process single slice
process_slice() {
    local slice_num="$1"
    local current_type="$2"   
    CURRENT_SLICE="$slice_num"  # For trap handler
    
    # Skip if already done
    if is_slice_done "$slice_num"; then
        info "Skipping slice ${slice_num} - already processed"
        return 0
    fi

    # Skip if locked by another process
    if is_slice_locked "$slice_num"; then
        info "Skipping slice ${slice_num} - locked by another process"
        return 0
    fi

    info "Processing slice: $slice_num"
    
    # Lock the slice
    if [ "$DRY_RUN" = false ]; then
        lock_slice "$slice_num"
    fi
    
    # Find target file based on explicit type
    if [ "$current_type" = "N" ]; then
     TARGET_FILE=$(find "$NISSL_DIR/images" -type f -name "MD585-N*_${slice_num}_to_MD585-N*_${slice_num}-registered.vtk" 2>/dev/null | head -n 1)
    else
     TARGET_FILE=$(find "$NISSL_DIR/images" -type f -name "MD585-IHC*_${slice_num}_to_MD585-IHC*_${slice_num}-registered.vtk" 2>/dev/null | head -n 1)
    fi
    debug "Found target file: $TARGET_FILE"

    # Skip if target file doesn't exist
    if [ -z "$TARGET_FILE" ]; then
     info "Skipping slice ${slice_num} - target file not found"
     unlock_slice "$slice_num"
     return 0
    fi

    # Determine neighbor type (opposite of current type)
    if [ "$current_type" = "N" ]; then
     neighbor_type="IHC"
    else
     neighbor_type="N"
    fi

    # Find neighbor file (of opposite type)
    local prev_slice=$(printf "%04d" $((10#$slice_num - 1)))
    NEIGHBOR_NISSL=$(find "$NISSL_DIR/images" -type f -name "MD585-${neighbor_type}*_${prev_slice}_to_MD585-${neighbor_type}*_${prev_slice}-registered.vtk" 2>/dev/null | head -n 1)
    debug "Found neighbor nissl: $NEIGHBOR_NISSL"

    # MRI should be of the same type as target
    MRI_FILE=$(find "$MRI_DIR/images" -type f -name "mri_to_MD585-${current_type}*_${slice_num}-registered.vtk" 2>/dev/null | head -n 1)
    debug "Found MRI file: $MRI_FILE"

    # Ensure only one path per variable
     TARGET_FILE=$(echo "$TARGET_FILE" | head -n 1)
     NEIGHBOR_NISSL=$(echo "$NEIGHBOR_NISSL" | head -n 1)
     MRI_FILE=$(echo "$MRI_FILE" | head -n 1)



    # Build command
    CMD="python manual_nissl_3d_QC_v01_save_points.py"
    CMD+=" -T \"$TARGET_FILE\""
    CMD+=" -N \"$NEIGHBOR_NISSL\""
    CMD+=" -M \"$MRI_FILE\""
    CMD+=" -O \"$OUTPUT_DIR\""

    # Print command in a more readable way
    info "Command to execute:"
    printf "%s\n" "$CMD"

    if [ "$DRY_RUN" = false ]; then
        info "Executing command..."
        if eval "$CMD"; then
            info "Command executed successfully"
            mark_slice_done "$slice_num"
            unlock_slice "$slice_num"
            info "Successfully processed slice ${slice_num}"
            return 0
        else
            error "Error processing slice ${slice_num}"
            unlock_slice "$slice_num"
            return 1
        fi
    else
        info "DRY RUN: Would execute above command"
        return 0
    fi
}

# Set slice range based on input
case "$RANGE" in
    1) START_SLICE="0007"; END_SLICE="0085" ;;  # First third: 0-137
    2) START_SLICE="0086"; END_SLICE="0163" ;;  # Second third: 138-275
    3) START_SLICE="0164"; END_SLICE="0230" ;;  # Final third: 276-413
esac

info "Processing range $RANGE (slices $START_SLICE to $END_SLICE)"

# Process all slices in the range
current_slice="$START_SLICE"
current_type="N"
while [ "$current_slice" -le "$END_SLICE" ]; do
    process_slice "$current_slice" "$current_type"

    # Alternate type for next slice
    if [ "$current_type" = "N" ]; then
        current_type="IHC"
    else
        current_slice=$(printf "%04d" $((10#$current_slice + 1)))
        current_type="N"
    fi
done

info "Script completed"
