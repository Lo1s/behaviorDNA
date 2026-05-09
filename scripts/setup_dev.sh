#!/usr/bin/env bash
# scripts/setup_dev.sh — run once after cloning on WSL
set -e

echo "==> Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "==> Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Installing pre-commit hooks..."
pre-commit install

echo "==> Initialising DVC..."
dvc init --no-scm 2>/dev/null || true   # no-scm if git already init'd separately

echo ""
echo "✅ Dev environment ready."
echo ""
echo "Next steps:"
echo "  1. Create a DagsHub repo at https://dagshub.com"
echo "  2. Update configs/training.yaml with your DagsHub username"
echo "  3. Run: dvc remote add origin https://dagshub.com/YOUR_USERNAME/behaviorDNA.dvc"
echo "  4. On Windows host: pip install -r requirements-collector.txt"
echo "  5. Start collecting sessions: python collector/record_session.py"
