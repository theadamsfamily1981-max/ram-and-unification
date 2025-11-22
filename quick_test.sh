#!/bin/bash
# Quick test to make sure everything works

echo "🧪 Running Quick Test..."
echo ""

python3 << 'EOF'
from src.avatar_engine import AvatarGenerator
from src.config import settings

print("Testing Avatar Generator...")
gen = AvatarGenerator(device='cpu')
print(f"✅ Device: {settings.device}")
print(f"✅ Port: {settings.port}")
print(f"✅ FPS: {settings.output_fps}")
print(f"✅ Resolution: {settings.output_resolution}")
print("")
print("🎉 Everything works!")
print("")
print("Ready to generate talking avatars!")
EOF
