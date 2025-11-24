#!/bin/bash
# Start the T-FAN Cockpit (Spaceship HUD)

COCKPIT_DIR="/home/user/Quanta-meis-nib-cis"

echo "🚀 Starting T-FAN Cockpit..."
echo ""

# Check if T-FAN is downloaded
if [ ! -d "$COCKPIT_DIR" ]; then
    echo "❌ T-FAN Cockpit not found!"
    echo ""
    echo "Please download it first:"
    echo "  1. Go to: https://github.com/theadamsfamily1981-max/Quanta-meis-nib-cis"
    echo "  2. Click 'Code' → 'Download ZIP'"
    echo "  3. Unzip to: $COCKPIT_DIR"
    echo ""
    exit 1
fi

cd "$COCKPIT_DIR"

# Check if there's a start script or requirements
if [ -f "start.sh" ]; then
    echo "Running T-FAN start script..."
    ./start.sh
elif [ -f "main.py" ]; then
    echo "Starting T-FAN Python app..."
    python main.py
elif [ -f "app.py" ]; then
    echo "Starting T-FAN app..."
    python app.py
elif [ -f "server.py" ]; then
    echo "Starting T-FAN server..."
    python server.py
else
    echo "⚠️  T-FAN is downloaded but I'm not sure how to start it."
    echo ""
    echo "Check the README in: $COCKPIT_DIR"
    echo ""
    echo "Look for files like:"
    echo "  - README.md"
    echo "  - start.sh"
    echo "  - main.py"
    echo "  - app.py"
    echo ""
    ls -la "$COCKPIT_DIR"
fi
