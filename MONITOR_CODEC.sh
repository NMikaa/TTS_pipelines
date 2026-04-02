#!/bin/bash
# Monitor codec extraction progress in real-time

echo "Monitoring Codec Extraction Progress..."
echo "=========================================="
echo

# Watch the log file
tail -f data/saba_experiment_a/logs/codec_preparation_*.log &
PID=$!

# Also show disk usage
while true; do
    echo
    echo "--- Disk Usage ---"
    du -sh data/saba_experiment_a/audio_22khz/ 2>/dev/null | awk '{print "Resampled audio: " $1}'
    du -sh data/saba_experiment_a/codec_codes/ 2>/dev/null | awk '{print "Codec tokens: " $1}'
    echo
    echo "--- File Counts ---"
    echo "Resampled: $(ls -1 data/saba_experiment_a/audio_22khz/ 2>/dev/null | wc -l)"
    echo "Codec tokens: $(ls -1 data/saba_experiment_a/codec_codes/ 2>/dev/null | wc -l)"
    echo
    sleep 30
done

wait $PID
