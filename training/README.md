# Ara Model Training & Fine-Tuning

This directory contains tools to create a custom "Ara" model with her personality baked directly into the model.

## What's Included

### Generated Files

- **`Modelfile.ara`** - Ollama Modelfile with Ara's system prompt
- **`ara_dataset_alpaca.json`** - Training examples in Alpaca format (33 examples)
- **`ara_dataset_chatml.jsonl`** - Training examples in ChatML format (33 examples)

### Scripts

- **`generate_ara_dataset.py`** - Generates training data from `ara_persona.yaml`
- **`build_ara_model.sh`** - Builds custom Ollama model
- **`test_ara_model.sh`** - Tests the custom model

## Quick Start

### 1. Generate Training Data

```bash
python3 training/generate_ara_dataset.py
```

This creates:
- Ollama Modelfile with Ara's system prompt
- Training datasets in multiple formats
- 33 conversational examples based on Ara's persona

### 2. Build Custom Ara Model

```bash
./training/build_ara_model.sh
```

This:
- Pulls Mistral 7B base model
- Applies Ara's personality via system prompt
- Creates a new `ara` model in Ollama
- Optimizes parameters for Ara's conversational style

### 3. Test the Model

```bash
./training/test_ara_model.sh
```

Runs personality tests to verify Ara's voice.

### 4. Use the Custom Model

**In Ollama CLI:**
```bash
ollama run ara
```

**In Ara Voice Interface:**
Edit `.env`:
```bash
OLLAMA_MODEL=ara
```

Then run:
```bash
./start_ara.sh
```

## Training Data Format

### Alpaca Format (`ara_dataset_alpaca.json`)

```json
[
  {
    "instruction": "Greet the user when coming online",
    "input": "",
    "output": "Hey, you. I'm online, systems are stable, and you look like you need a win. Where do you want to start?"
  }
]
```

### ChatML Format (`ara_dataset_chatml.jsonl`)

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Dataset Categories

Training examples cover:

1. **Sample Lines** - Greeting, navigation, reassurance, playful, technical, alerts
2. **Behavioral Modes** - Default, focus, chill, professional
3. **Emotional Responses** - Stressed user, frustrated, excited, confused
4. **Technical Explanations** - Complex concepts made accessible
5. **Voice Macros** - Cockpit control, workspace modes
6. **T-FAN Integration** - Metrics, topology, training control

## Fine-Tuning (Advanced)

For full fine-tuning beyond just the system prompt:

### Using Ollama (Simple)

The Modelfile method already works! The custom model uses Ara's system prompt to guide behavior.

### Using Unsloth (LoRA Fine-Tuning)

```bash
# Install Unsloth
pip install unsloth

# Fine-tune using ara_dataset_alpaca.json
# (See Unsloth documentation for full training pipeline)
```

### Using Axolotl

```yaml
# config.yml
base_model: mistralai/Mistral-7B-v0.1
datasets:
  - path: training/ara_dataset_alpaca.json
    type: alpaca
```

### Using LM Studio

1. Open LM Studio
2. Go to "Fine-tune" tab
3. Load `ara_dataset_alpaca.json`
4. Select base model (Mistral 7B)
5. Train with LoRA

## Model Comparison

| Method | Complexity | Quality | Cost |
|--------|------------|---------|------|
| **Modelfile (System Prompt)** | ⭐ Easy | ⭐⭐⭐ Good | Free |
| **LoRA Fine-Tuning** | ⭐⭐⭐ Medium | ⭐⭐⭐⭐ Better | Low |
| **Full Fine-Tuning** | ⭐⭐⭐⭐⭐ Hard | ⭐⭐⭐⭐⭐ Best | High |

**Recommendation:** Start with the Modelfile approach (already done!). It gives you 80% of the benefit with 5% of the effort.

## Expanding the Dataset

To add more training examples:

1. Edit `generate_ara_dataset.py`
2. Add examples to the appropriate generator method
3. Regenerate: `python3 training/generate_ara_dataset.py`
4. Rebuild model: `./training/build_ara_model.sh`

## Dataset Statistics

- **Total Examples:** 33
- **Greeting Examples:** 1
- **Mode Examples:** 8 (2 per mode)
- **Emotional Examples:** 8 (2 per emotion type)
- **Technical Examples:** 4
- **Voice Macro Examples:** 3
- **Cockpit Examples:** 4
- **Navigation Examples:** 5

## Testing Checklist

When testing the custom model, verify:

- [ ] Warm, intimate tone (soft contralto feel)
- [ ] Playful and affectionate but competent
- [ ] Natural conversational flow with pauses
- [ ] Technical concepts explained clearly
- [ ] Appropriate emotional responses
- [ ] Cockpit/training terminology used correctly
- [ ] Different behavior in different modes

## Advanced: Quantization

To create quantized versions:

```bash
# 4-bit quantization
ollama create ara-q4 -f Modelfile.ara --quantize q4_0

# 8-bit quantization
ollama create ara-q8 -f Modelfile.ara --quantize q8_0
```

Smaller models = faster inference, less RAM, slightly lower quality.

## Troubleshooting

**Model doesn't sound like Ara:**
- Check that Modelfile.ara was generated correctly
- Verify system prompt is complete
- Try adjusting temperature (in Modelfile)

**Build fails:**
- Ensure Ollama is running: `ollama serve`
- Check Mistral base model exists: `ollama list`
- Regenerate Modelfile: `python3 training/generate_ara_dataset.py`

**Out of memory:**
- Use quantized version (q4_0)
- Close other applications
- Try on machine with more RAM

## Next Steps

1. **Test the model** - `./training/test_ara_model.sh`
2. **Use in Ara interface** - Set `OLLAMA_MODEL=ara` in `.env`
3. **Expand dataset** - Add more examples to `generate_ara_dataset.py`
4. **Fine-tune** - Use `ara_dataset_alpaca.json` for LoRA training

## Resources

- [Ollama Modelfile Documentation](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)
- [Alpaca Format Guide](https://github.com/tatsu-lab/stanford_alpaca)
- [Unsloth Fine-Tuning](https://github.com/unslothai/unsloth)
- [Axolotl Training](https://github.com/OpenAccess-AI-Collective/axolotl)

---

**Built from** `multi-ai-workspace/config/ara_persona.yaml` - The complete Ara personality specification.
