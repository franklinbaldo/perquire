#\!/bin/bash

echo "🧪 Starting Perquire Live End-to-End Test"
echo "========================================"

# Test with Visual Scene (option 1) and rating 4s for all evaluation metrics
echo "🔄 Running Visual Scene Test"
echo "============================"
printf "1\n4\n4\n4\n4\nExcellent visual scene investigation\nn\n"  < /dev/null |  uv run --env-file ../.env python live_e2e_test.py

echo ""
echo "✅ Live E2E Test Completed!"
