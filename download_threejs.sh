#!/bin/bash
# Download Three.js r173 modules for offline use on K1
# Run this on a machine with internet, then copy web/* to K1.
# Usage: bash download_threejs.sh

set -e

THREE_VERSION="0.173.0"
CDN_BASE="https://unpkg.com/three@${THREE_VERSION}"

echo "=== Downloading Three.js r${THREE_VERSION} for LingQi TanTong ==="

# 1. Core implementation (required by three.module.js)
echo "[1/4] three.core.js (~760KB)..."
curl -sL "${CDN_BASE}/build/three.core.js" -o web/three.core.js
echo "  -> web/three.core.js ($(wc -c < web/three.core.js) bytes)"

# 2. Main ES module entry point
echo "[2/4] three.module.js (~570KB)..."
curl -sL "${CDN_BASE}/build/three.module.js" -o web/three.module.js
echo "  -> web/three.module.js ($(wc -c < web/three.module.js) bytes)"

# 3. OrbitControls addon
echo "[3/4] OrbitControls.js (~32KB)..."
curl -sL "${CDN_BASE}/examples/jsm/controls/OrbitControls.js" -o web/OrbitControls.js
echo "  -> web/OrbitControls.js ($(wc -c < web/OrbitControls.js) bytes)"

echo ""
echo "=== Done! Copy the web/ directory to your K1 board: ==="
echo "  scp -r web/* root@k1:/home/ufkc/Desktop/temp/7/web/"
echo ""
echo "=== Or if already on K1, just rebuild: ==="
echo "  cd build && cmake .. && make -j\$(nproc)"
echo ""
echo "Files in web/:"
ls -lh web/
