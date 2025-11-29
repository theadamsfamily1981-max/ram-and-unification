#!/bin/bash
##############################################################################
# Test Custom Ara Model
# Sends test prompts to verify Ara's personality is working
##############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
MAGENTA='\033[0;35m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║            🧪  TEST CUSTOM ARA MODEL  🧪                      ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""

# Check if ara model exists
if ! ollama list | grep -q "^ara "; then
    echo -e "${RED}❌ Custom 'ara' model not found${NC}"
    echo -e "${YELLOW}Build it first: ./training/build_ara_model.sh${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Custom 'ara' model found${NC}"
echo ""

# Test prompts that should trigger Ara's personality
test_prompts=(
    "Hey Ara, how are you doing?"
    "What can you help me with?"
    "Show me GPU stats"
    "I'm stressed out, nothing is working"
    "We hit 98% accuracy!"
    "What's the difference between transformers and RNNs?"
    "How's training going?"
)

echo -e "${CYAN}Running personality tests...${NC}"
echo -e "${YELLOW}These prompts test different aspects of Ara's personality${NC}"
echo ""

for prompt in "${test_prompts[@]}"; do
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}User:${NC} $prompt"
    echo -e "${GREEN}Ara:${NC}"

    # Send prompt to ara model
    ollama run ara "$prompt" 2>/dev/null

    echo ""
done

echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${GREEN}✅ Personality tests complete!${NC}"
echo ""
echo -e "${CYAN}How did Ara sound?${NC}"
echo ""
echo -e "${YELLOW}Expected characteristics:${NC}"
echo "  ✓ Warm, intimate tone"
echo "  ✓ Slightly playful and affectionate"
echo "  ✓ Competent and protective"
echo "  ✓ Natural pauses and conversational flow"
echo "  ✓ Technical explanations that are clear and accessible"
echo ""
echo -e "${CYAN}To chat interactively:${NC}"
echo "  ${YELLOW}ollama run ara${NC}"
echo ""
echo -e "${CYAN}To use in Ara voice interface:${NC}"
echo "  Edit ${YELLOW}.env${NC} and set: ${YELLOW}OLLAMA_MODEL=ara${NC}"
echo "  Then run: ${YELLOW}./start_ara.sh${NC}"
