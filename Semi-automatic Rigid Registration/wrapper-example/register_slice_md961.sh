#!/bin/bash
base_dir=/nfs/data/main/M38/mba_converted_imaging_data/MD961/MD961/
registration_base=/nfs/data/main/M38/RegistrationData/Data/MD961
lock_dir=${registration_base}/locks
done_dir=${registration_base}/done
corrected_dir=${registration_base}/Corrected

# Create directories without disturbing symlinks
for dir in "$lock_dir" "$done_dir" "$corrected_dir"; do
    if [ ! -e "$dir" ]; then
        mkdir -p "$dir"
        chmod 775 "$dir" 
    fi
done

# Function to find least loaded GPU
get_available_gpu() {
    # Get GPU utilization and sort by usage
    local gpu_id=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | \
        sort -t',' -k2,2n -k3,3n | \
        head -n1 | \
        cut -d',' -f1)
    if [ -z "$gpu_id" ]; then
        echo "No GPU available"
        exit 1
    fi
    echo "cuda:$gpu_id"
}

# Function to extract section number from filename
get_section() {
    echo $(basename "$1" | grep -o 'MD961_[0-9]_[0-9]\+' | grep -o '[0-9]\+$')
}

# Find the next available section
for nissl in $(find $base_dir -name "MD961-N*_MD961_*_*.tif"); do
    section=$(get_section "$nissl")
    myelin=$(find $base_dir -name "MD961-My*_MD961_*_${section}.tif")
    
    # Skip if no matching myelin file
    if [ -z "$myelin" ]; then
        echo "Warning: Found Nissl for section $section but no matching Myelin file"
        continue
    fi

    # Skip if already being processed or done
    if [ -f "$lock_dir/${section}.lock" ] || [ -f "$done_dir/${section}.done" ]; then
        continue
    fi
    
    # Get least loaded GPU
    device=$(get_available_gpu)

    # Found an available section, process it
    echo "Processing section $section..."
    echo "Nissl:  $nissl"
    echo "Myelin: $myelin"

    # Create lock file
    touch "$lock_dir/${section}.lock"

    # Run registration with selected GPU
    if [ -z "$device" ]; then
        echo "No GPU available, running on CPU"
        bash run.sh "$nissl" "$myelin" "cpu"
    else
        echo "Using GPU device: $device"
        bash run.sh "$nissl" "$myelin" "$device"
    fi

    # If registration was successful
    if [ $? -eq 0 ]; then
        # Move lock to done
        mv "$lock_dir/${section}.lock" "$done_dir/${section}.done"
        echo "Successfully completed section $section"
        
        # Ask if user wants to continue with next section
        read -p "Continue with next section? (y/n) " answer
        if [ "$answer" != "y" ]; then
            exit 0
        fi
    else
        # Remove lock if failed
        rm "$lock_dir/${section}.lock"
        echo "Failed to process section $section"
        exit 1
    fi
done

echo "No more sections available for processing" 