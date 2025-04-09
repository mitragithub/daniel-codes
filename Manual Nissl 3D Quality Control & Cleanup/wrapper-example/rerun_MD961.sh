#! /bin/bash

source ./nissl/bin/activate

# Add emlddmm to PYTHONPATH
export PYTHONPATH="/home/mitralab/nissl-nissl:$PYTHONPATH"

# Function to display usage
usage() {
    echo "Usage: $0 -b BASE_DIR [-r RANGE | -s SLICE] [-d] [-h]"
    echo "  -b: Base directory path"
    echo "  -r: Range to process (1, 2, or 3)"
    echo "      1: slices 0000-0099"
    echo "      2: slices 0100-0199"
    echo "      3: slices 0200-0299"
    echo "  -s: Single slice number to process (e.g., 0042)"
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
while getopts "b:r:s:dh" opt; do
    case $opt in
        b) BASE_DIR="$OPTARG" ;;
        r) RANGE="$OPTARG" ;;
        s) SINGLE_SLICE="$OPTARG" ;;
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

# Validate RANGE and SINGLE_SLICE
if [ -n "$RANGE" ] && [ -n "$SINGLE_SLICE" ]; then
    error "Cannot specify both range and single slice"
    usage
fi

if [ -z "$RANGE" ] && [ -z "$SINGLE_SLICE" ]; then
    error "Must specify either range (-r) or single slice (-s)"
    usage
fi

if [ -n "$RANGE" ] && ! [[ "$RANGE" =~ ^[1-3]$ ]]; then
    error "Invalid range argument (must be 1, 2, or 3)"
    usage
fi

if [ -n "$SINGLE_SLICE" ] && ! [[ "$SINGLE_SLICE" =~ ^[0-9]{4}$ ]]; then
    error "Invalid slice number (must be a 4-digit number like 0042)"
    usage
fi

# Set up directory structure
NISSL_DIR="${BASE_DIR}/MD961-nissl-registered/MD961-nissl_to_MD961-nissl-registered"
MRI_DIR="${BASE_DIR}/MD961-nissl-registered/mri_to_MD961-nissl-registered"
BASE_OUTPUT="/nfs/data/main/M38/nissl_manual_correction_output/MD961"
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
    CURRENT_SLICE="$slice_num"  # For trap handler
    local bridge_slice=""  # Variable to track which slice we're bridging with
    
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
    
    # Find target file
    TARGET_FILE=$(find "$NISSL_DIR/images" -type f -name "MD961-N*_MD961_[1-9]_${slice_num}_to_MD961-N*_MD961_[1-9]_${slice_num}-registered.vtk" 2>/dev/null)
    debug "Found target file: $TARGET_FILE"

    # If target file doesn't exist, try to find closest neighbors for bridging
    if [ -z "$TARGET_FILE" ]; then
        info "Target file for slice ${slice_num} not found, attempting to bridge with neighbors..."
        
        # Find the closest previous slice
        local prev_slice=$((10#$slice_num - 1))
        local prev_file=""
        while [ $prev_slice -ge 0 ]; do
            local prev_slice_padded=$(printf "%04d" $prev_slice)
            prev_file=$(find "$NISSL_DIR/images" -type f -name "MD961-N*_MD961_[1-9]_${prev_slice_padded}_to_MD961-N*_MD961_[1-9]_${prev_slice_padded}-registered.vtk" 2>/dev/null | head -1)
            if [ -n "$prev_file" ]; then
                debug "Found previous neighbor: slice $prev_slice_padded"
                break
            fi
            prev_slice=$((prev_slice - 1))
        done
        
        # Find the closest next slice
        local next_slice=$((10#$slice_num + 1))
        local next_file=""
        while [ $next_slice -le 9999 ]; do
            local next_slice_padded=$(printf "%04d" $next_slice)
            next_file=$(find "$NISSL_DIR/images" -type f -name "MD961-N*_MD961_[1-9]_${next_slice_padded}_to_MD961-N*_MD961_[1-9]_${next_slice_padded}-registered.vtk" 2>/dev/null | head -1)
            if [ -n "$next_file" ]; then
                debug "Found next neighbor: slice $next_slice_padded"
                break
            fi
            next_slice=$((next_slice + 1))
        done
        
        # If we found at least one neighbor, use it as the target
        if [ -n "$prev_file" ] || [ -n "$next_file" ]; then
            # Choose the closest neighbor as target
            if [ -n "$prev_file" ] && [ -n "$next_file" ]; then
                # Both neighbors found, use the closest one
                if [ $((10#$slice_num - prev_slice)) -le $((next_slice - 10#$slice_num)) ]; then
                    TARGET_FILE="$prev_file"
                    bridge_slice=$(printf "%04d" $prev_slice)
                    info "Bridging with previous slice ${bridge_slice} (closer neighbor)"
                else
                    TARGET_FILE="$next_file"
                    bridge_slice=$(printf "%04d" $next_slice)
                    info "Bridging with next slice ${bridge_slice} (closer neighbor)"
                fi
            elif [ -n "$prev_file" ]; then
                TARGET_FILE="$prev_file"
                bridge_slice=$(printf "%04d" $prev_slice)
                info "Bridging with previous slice ${bridge_slice} (only previous found)"
            else
                TARGET_FILE="$next_file"
                bridge_slice=$(printf "%04d" $next_slice)
                info "Bridging with next slice ${bridge_slice} (only next found)"
            fi
        else
            info "Skipping slice ${slice_num} - no suitable neighbors found for bridging"
            unlock_slice "$slice_num"
            return 0
        fi
    fi

    # Set the slice number to use for neighbor and MRI files
    local lookup_slice=${bridge_slice:-$slice_num}
    info "Using slice ${lookup_slice} for file lookups"

    # Find neighbor files - try both previous and next slice
    local prev_slice=$(printf "%04d" $((10#$lookup_slice - 1)))
    local next_slice=$(printf "%04d" $((10#$lookup_slice + 1)))
    
    NEIGHBOR_NISSL=$(find "$NISSL_DIR/images" -type f -name "MD961-N*_MD961_[1-9]_${prev_slice}_to_MD961-N*_MD961_[1-9]_${prev_slice}-registered.vtk" 2>/dev/null)
    
    # If previous slice neighbor not found, try next slice
    if [ -z "$NEIGHBOR_NISSL" ]; then
        debug "Previous slice neighbor not found, trying next slice..."
        NEIGHBOR_NISSL=$(find "$NISSL_DIR/images" -type f -name "MD961-N*_MD961_[1-9]_${next_slice}_to_MD961-N*_MD961_[1-9]_${next_slice}-registered.vtk" 2>/dev/null)
    fi
    
    debug "Found neighbor nissl: $NEIGHBOR_NISSL"
    
    # Find MRI file - use the same slice number as the target file we're using
    MRI_FILE=$(find "$MRI_DIR/images" -type f -name "mri_to_MD961-N*_MD961_[1-9]_${lookup_slice}-registered.vtk" 2>/dev/null)
    debug "Found MRI file: $MRI_FILE"

    # Build command
    CMD="python manual_nissl_3d_QC_v01_save_points.py"
    CMD+=" -T \"$TARGET_FILE\""
    
    # Only add neighbor parameter if we found a neighbor
    if [ -n "$NEIGHBOR_NISSL" ]; then
        CMD+=" -N \"$NEIGHBOR_NISSL\""
    else
        info "Warning: No neighbor slice found for ${lookup_slice}"
        # If your Python script requires -N parameter even if empty, keep this line
        CMD+=" -N \"\""
    fi
    
    # Only add MRI parameter if we found an MRI file
    if [ -n "$MRI_FILE" ]; then
        CMD+=" -M \"$MRI_FILE\""
    else
        info "Warning: No MRI file found for ${lookup_slice}"
        CMD+=" -M \"\""
    fi
    
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
if [ -n "$RANGE" ]; then
    case "$RANGE" in
        1) START_SLICE="0001"; END_SLICE="0125" ;;  # First third: 0-137
        2) START_SLICE="0126"; END_SLICE="0250" ;;  # Second third: 138-275
        3) START_SLICE="0251"; END_SLICE="400" ;;  # Final third: 276-413
    esac
    info "Processing range $RANGE (slices $START_SLICE to $END_SLICE)"
    
    # Process all slices in the range
    current_slice="$START_SLICE"
    while [ "$current_slice" -le "$END_SLICE" ]; do
        process_slice "$current_slice"
        current_slice=$(printf "%04d" $((10#$current_slice + 1)))
    done
else
    # Process single slice
    info "Processing single slice $SINGLE_SLICE"
    process_slice "$SINGLE_SLICE"
fi

info "Script completed"
