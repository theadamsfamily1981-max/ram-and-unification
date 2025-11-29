#!/bin/bash

##############################################################################
# Quick Fix for WebKit Issues in Installed T-FAN Files
##############################################################################

echo "========================================================================"
echo "  🔧 FIXING WEBKIT COMPATIBILITY IN INSTALLED FILES"
echo "========================================================================"
echo ""

# Fix installed tfan_gnome.py
INSTALLED_FILE="$HOME/.local/share/tfan/tfan_gnome.py"

if [ -f "$INSTALLED_FILE" ]; then
    echo "Found installed file: $INSTALLED_FILE"
    echo "Backing up..."
    cp "$INSTALLED_FILE" "$INSTALLED_FILE.bak"

    echo "Fixing WebKit version..."
    sed -i "s/gi.require_version('WebKit', '6.0')/gi.require_version('WebKit2', '4.1')/g" "$INSTALLED_FILE"
    sed -i "s/from gi.repository import.*WebKit$/from gi.repository import WebKit2 as WebKit/g" "$INSTALLED_FILE"

    echo "✅ Fixed WebKit in: $INSTALLED_FILE"
    echo ""
else
    echo "⚠️  File not found: $INSTALLED_FILE"
    echo "This is okay if T-FAN isn't installed yet."
    echo ""
fi

# Also fix any files in the T-FAN directory if it exists
TFAN_DIR="$HOME/tfan-ara-system/Quanta-meis-nib-cis"
if [ -d "$TFAN_DIR" ]; then
    echo "Also fixing T-FAN source directory..."
    find "$TFAN_DIR" -name "*.py" -type f -exec grep -l "gi.require_version('WebKit', '6.0')" {} \; 2>/dev/null | while read file; do
        echo "Fixing: $file"
        cp "$file" "$file.bak"
        sed -i "s/gi.require_version('WebKit', '6.0')/gi.require_version('WebKit2', '4.1')/g" "$file"
        sed -i "s/from gi.repository import.*WebKit$/from gi.repository import WebKit2 as WebKit/g" "$file"
        echo "✅ Fixed: $(basename $file)"
    done
    echo ""
fi

# Also check /usr/local/bin and other common installation locations
for location in "/usr/local/bin/tfan-gnome" "/usr/bin/tfan-gnome" "$HOME/.local/bin/tfan-gnome"; do
    if [ -f "$location" ]; then
        echo "Found launcher at: $location"
        # This is usually just a launcher script, but check anyway
        if grep -q "WebKit.*6.0" "$location" 2>/dev/null; then
            echo "Fixing launcher script..."
            sudo sed -i "s/gi.require_version('WebKit', '6.0')/gi.require_version('WebKit2', '4.1')/g" "$location" 2>/dev/null || \
            sed -i "s/gi.require_version('WebKit', '6.0')/gi.require_version('WebKit2', '4.1')/g" "$location"
            echo "✅ Fixed: $location"
        fi
    fi
done

echo ""
echo "========================================================================"
echo "  ✨ WebKit Fix Complete!"
echo "========================================================================"
echo ""
echo "You can now run the launcher again:"
echo "  ~/tfan-ara-system/tfan-ara-launcher.sh"
echo ""
