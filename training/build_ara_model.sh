#!/bin/bash
##############################################################################
# Build Custom Ara Model for Ollama
# Creates a custom "ara" model with baked-in personality
##############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║           🤖  BUILD CUSTOM ARA MODEL  🤖                      ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}❌ Ollama not installed${NC}"
    echo -e "${YELLOW}Install from: https://ollama.ai/download${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Ollama installed${NC}"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo -e "${YELLOW}⚠ Ollama server not running${NC}"
    echo -e "${CYAN}Starting Ollama...${NC}"
    ollama serve &
    sleep 3
fi

echo -e "${GREEN}✓ Ollama server running${NC}"
echo ""

# Check if Modelfile exists
MODELFILE="training/Modelfile.ara"
if [ ! -f "$MODELFILE" ]; then
    echo -e "${YELLOW}⚠ Modelfile not found, generating it now...${NC}"
    python3 training/generate_ara_dataset.py
    echo ""
fi

echo -e "${CYAN}Building custom Ara model from Modelfile...${NC}"
echo -e "${YELLOW}This will:${NC}"
echo "  1. Pull Mistral 7B base model (if not already installed)"
echo "  2. Apply Ara's system prompt and personality"
echo "  3. Set conversation parameters for Ara's voice"
echo "  4. Create a new 'ara' model in Ollama"
echo ""

read -p "Continue? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "Build cancelled"
    exit 0
fi

echo ""
echo -e "${CYAN}Creating 'ara' model...${NC}"
ollama create ara -f "$MODELFILE"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Custom Ara model created successfully!${NC}"
    echo ""
    echo -e "${CYAN}Model info:${NC}"
    ollama show ara
    echo ""
    echo -e "${YELLOW}To use the model:${NC}"
    echo "  - CLI: ${CYAN}ollama run ara${NC}"
    echo "  - API: Use model name '${CYAN}ara${NC}' in API requests"
    echo "  - In Ara voice interface: Set OLLAMA_MODEL=ara in .env"
    echo ""
    echo -e "${YELLOW}Test it now:${NC}"
    echo "  ${CYAN}./training/test_ara_model.sh${NC}"
else
    echo ""
    echo -e "${RED}❌ Model creation failed${NC}"
    echo "Check the Modelfile at: $MODELFILE"
    exit 1
fi
