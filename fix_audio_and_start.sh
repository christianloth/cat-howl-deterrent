#!/bin/zsh
# One-shot: restart coreaudiod to clear a HAL hang, then launch the cat-howl
# detector. You will be prompted for your admin password (once).

cd "$(dirname "$0")"

echo "==============================================================="
echo "  Cat-Howl Detector — Recovery + Launch"
echo "==============================================================="
echo
echo "Your Mac's coreaudiod may be in a stuck state (e.g. Loopback's"
echo "AudioServerPlugin locked up). This script will:"
echo "  1. Restart coreaudiod (1-2 seconds; all audio briefly stops)"
echo "  2. Launch the detector"
echo
echo "Enter your admin password when prompted."
echo

# launchctl kickstart is blocked by SIP on macOS system daemons. Use killall
# instead — launchd auto-respawns coreaudiod in ~1 second with fresh state.
sudo /usr/bin/killall coreaudiod
if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Could not signal coreaudiod (maybe wrong password)."
    echo "Re-run this script."
    echo "Press Return to close..."
    read
    exit 1
fi
echo "  ✓ coreaudiod killed; waiting for launchd to respawn..."
for i in 1 2 3 4 5 6 7 8; do
    sleep 1
    if pgrep -x coreaudiod >/dev/null; then
        echo "  ✓ coreaudiod respawned (PID $(pgrep -x coreaudiod), uptime fresh)"
        break
    fi
done
sleep 1

echo
echo "Granting python mic access (dialog may appear — click Allow)..."
.venv/bin/cat-howl-request-mic
if [ $? -ne 0 ]; then
    echo "Mic permission denied. Press Return to close..."
    read
    exit 1
fi

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
echo "Launching detector..."
nohup .venv/bin/cat-howl-deterrent >> logs/detector.log 2>&1 &
DETECTOR_PID=$!
disown

echo "  PID: $DETECTOR_PID  (detached, will survive this window closing)"
echo "  Log: ./logs/detector.log"
echo "  Howl audio: ./howl_recordings/"
echo "  Stop: kill $DETECTOR_PID"
echo
echo "Waiting up to 20 seconds for first audio callback..."
for i in $(seq 1 20); do
    sleep 1
    if grep -q "audio is flowing\|first mic callback\|mic: rms" logs/detector.log 2>/dev/null; then
        echo "  ✓ Audio flowing!"
        echo
        tail -12 logs/detector.log | sed 's/^/    /'
        echo
        echo "All set. You may close this window — detector keeps running."
        echo "Press Return to close..."
        read
        exit 0
    fi
done

echo
echo "✗ No audio after 20s. coreaudiod or the mic device is still stuck."
echo "Last log lines:"
tail -25 logs/detector.log | sed 's/^/    /'
echo
echo "Try unplugging and re-plugging the Yeti USB cable, then re-run this script."
echo "Press Return to close..."
read
