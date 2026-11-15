
#!/bin/bash

# Existing process to wait for
WAIT_PID=233115

# Log file
LOG_FILE="sequential_attacks.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "Waiting for existing process: $WAIT_PID" | tee -a "$LOG_FILE"

# Wait until the existing process terminates
while kill -0 "$WAIT_PID" 2>/dev/null; do
    echo "Process $WAIT_PID is still running. Checking again in 10 seconds..."
    sleep 10
done

echo "Process $WAIT_PID has terminated." | tee -a "$LOG_FILE"
echo "Starting sequential attack execution..." | tee -a "$LOG_FILE"

# Dataset folders
FOLDERS=(
    "Data_SMOL"
    "Data_Llama"
    "data-code"
)

# Execute each dataset sequentially
for FOLDER in "${FOLDERS[@]}"; do

    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Starting: $FOLDER" | tee -a "$LOG_FILE"
    echo "Time: $(date)" | tee -a "$LOG_FILE"

    python attacks/attackmode.py \
        --input "$FOLDER/watermarked" \
        --output "$FOLDER/attacked" \
        --donor-data "$FOLDER/unwatermarked" \
        2>&1 | tee -a "$LOG_FILE"

    STATUS=${PIPESTATUS[0]}

    if [ "$STATUS" -eq 0 ]; then
        echo "$FOLDER completed successfully." | tee -a "$LOG_FILE"
    else
        echo "ERROR: $FOLDER exited with status $STATUS." | tee -a "$LOG_FILE"
    fi

done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "All dataset jobs have been processed." | tee -a "$LOG_FILE"
echo "Finished at: $(date)" | tee -a "$LOG_FILE"
