#!/bin/zsh
# Overnight cat-howl detector launcher (no coreaudiod restart).
# Use fix_audio_and_start.sh instead if audio is wedged.

cd "$(dirname "$0")"

echo "=== Cat-Howl Detector ==="
date
echo

echo "Step 1: Requesting microphone access for python..."
.venv/bin/cat-howl-request-mic
if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Mic permission for python was denied."
    echo "Press Return to close..."
    read
    exit 1
fi
echo "  ✓ python mic permission granted"

pkill -f "cat_howl_deterrent|cat-howl-deterrent" 2>/dev/null
sleep 1
mkdir -p logs howl_recordings

export SSL_CERT_FILE="$(.venv/bin/python -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
export PYTHONUNBUFFERED=1
export BACKEND=cpu
export LOG_ONLY=1
export RECORD_HOWLS=1
export MIC_DEVICE=Yeti
export MIC_BACKEND=sd

echo
echo "Step 2: Launching detector..."
nohup .venv/bin/cat-howl-deterrent >> logs/detector.log 2>&1 &
DETECTOR_PID=$!
disown

echo "  Watching for first audio callback (up to 25s)..."
for i in $(seq 1 25); do
    sleep 1
    if ! kill -0 "$DETECTOR_PID" 2>/dev/null; then
        echo "✗ Detector died. Last 30 log lines:"
        tail -30 logs/detector.log | sed 's/^/    /'
        echo
        echo "Press Return to close..."
        read
        exit 1
    fi
    if grep -q "audio is flowing\|first mic callback\|mic: rms" logs/detector.log 2>/dev/null; then
        break
    fi
done

if kill -0 "$DETECTOR_PID" 2>/dev/null && grep -q "audio is flowing\|first mic callback\|mic: rms" logs/detector.log; then
    echo "  ✓ Detector running, PID $DETECTOR_PID"
    echo "  Log:           ./logs/detector.log"
    echo "  Howl audio:    ./howl_recordings/"
    echo "  Stop:          kill $DETECTOR_PID"
    echo
    echo "Last 12 log lines:"
    tail -12 logs/detector.log | sed 's/^/    /'
    echo
    echo "Close this Terminal window — the detector keeps running."
    echo "Press Return to close, or Ctrl+C to leave open."
    read
else
    echo
    echo "✗ Detector did not produce audio in 25 seconds."
    tail -30 logs/detector.log | sed 's/^/    /'
    echo "Press Return to close..."
    read
    exit 1
fi
