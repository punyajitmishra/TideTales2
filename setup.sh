#!/bin/bash
# One-shot: commit everything and wire up the GitHub remote.
set -e
echo "🌊 Initializing TideTales repository..."
[ -d .git ] || git init -b main
git add .
git commit -m "feat: initial commit for TideTales coastal storytelling platform"
git branch -M main
git remote add origin https://github.com/punyajitmishra/TideTales.git 2>/dev/null || true
echo "Ready to push! Run: git push -u origin main"
