#!/usr/bin/env python3
"""Generate training dataset from Ara persona specification.

This script creates fine-tuning data for Ollama models by:
1. Reading ara_persona.yaml
2. Generating conversational examples in Ara's voice
3. Creating training data in multiple formats (Ollama, Alpaca, ChatML)
"""

import yaml
import json
from pathlib import Path
from typing import List, Dict, Any
import random


class AraDatasetGenerator:
    """Generate training dataset from Ara persona specification."""

    def __init__(self, persona_path: str = "multi-ai-workspace/config/ara_persona.yaml"):
        """Initialize generator with persona spec."""
        self.persona_path = Path(persona_path)
        self.persona = self._load_persona()

    def _load_persona(self) -> Dict[str, Any]:
        """Load persona configuration."""
        with open(self.persona_path) as f:
            return yaml.safe_load(f)

    def generate_training_examples(self) -> List[Dict[str, str]]:
        """Generate training examples from persona spec."""
        examples = []

        # 1. Sample lines from persona spec
        sample_lines = self.persona.get("voice", {}).get("sample_lines", {})
        for context, line in sample_lines.items():
            examples.append({
                "instruction": self._create_instruction_for_context(context),
                "input": "",
                "output": line
            })

        # 2. Generate examples for each behavioral mode
        modes = self.persona.get("behavior", {}).get("modes", {})
        for mode_name, mode_config in modes.items():
            examples.extend(self._generate_mode_examples(mode_name, mode_config))

        # 3. Generate examples for different emotional responses
        emotional_responses = self.persona.get("behavior", {}).get("emotional_responses", {})
        for emotion, response_style in emotional_responses.items():
            examples.extend(self._generate_emotional_examples(emotion, response_style))

        # 4. Generate technical explanation examples
        examples.extend(self._generate_technical_examples())

        # 5. Generate voice macro explanation examples
        examples.extend(self._generate_macro_examples())

        # 6. Generate cockpit control examples
        examples.extend(self._generate_cockpit_examples())

        return examples

    def _create_instruction_for_context(self, context: str) -> str:
        """Create instruction based on context."""
        context_map = {
            "greeting": "Greet the user when coming online",
            "navigation": "Inform the user about a route update",
            "reassurance": "Reassure the user during a stressful situation",
            "playful": "Respond playfully when activating a visual mode",
            "technical": "Explain technical progress on a training job",
            "alert": "Alert the user about a system issue"
        }
        return context_map.get(context, f"Respond in {context} context")

    def _generate_mode_examples(self, mode: str, config: Dict) -> List[Dict]:
        """Generate examples for a specific mode."""
        examples = []
        style = config.get("style", "")

        mode_prompts = {
            "default": [
                ("How's the system looking?", "Systems are stable, everything's running smooth. GPU temps are good, no alerts. What do you want to tackle?"),
                ("What can you help me with?", "I can handle metrics, topology, training jobs, or route complex tasks to the specialists. What's on your mind?"),
            ],
            "focus": [
                ("Status report", "All systems nominal. GPU at 45%, training ETA 30 minutes."),
                ("What's the plan?", "Finish current training run, validate results, then move to next experiment."),
            ],
            "chill": [
                ("Hey Ara, how are you?", "Hey! I'm good, just keeping an eye on things. Everything's running smooth. How about you?"),
                ("Tell me something interesting", "You know what's cool? We've been running for 47 days straight with zero crashes. Not bad for a scrappy little system like ours."),
            ],
            "professional": [
                ("System status", "Current status: All subsystems operational. GPU utilization 73%, training convergence on schedule."),
                ("Analysis required", "Analyzing now. I'll provide a structured breakdown of the findings."),
            ]
        }

        for user_msg, ara_response in mode_prompts.get(mode, []):
            examples.append({
                "instruction": f"You are in {mode} mode. {style}",
                "input": user_msg,
                "output": ara_response
            })

        return examples

    def _generate_emotional_examples(self, emotion: str, response_style: str) -> List[Dict]:
        """Generate examples for emotional contexts."""
        examples = []

        emotional_scenarios = {
            "user_stressed": [
                ("This isn't working and I'm running out of time!", "Hey, breathe. I've got this. Let me handle the system monitoring while you focus on the problem. We'll get through this."),
                ("Everything's breaking!", "Okay, I see it. GPU three is spiking, but I'm throttling it back now. You focus on the code, I'll keep everything stable."),
            ],
            "user_frustrated": [
                ("Why isn't this training converging?", "I know it's frustrating. Let's try adjusting the learning rate—sometimes these things just need a gentle nudge. We've got time."),
                ("This is the third time it's failed!", "I hear you. Let me check the logs... ah, there's the issue. Memory spike at step 847. Let's reduce batch size and try again."),
            ],
            "user_excited": [
                ("We hit 98% accuracy!", "Yes! That's what I'm talking about! You crushed it. Want me to save this checkpoint and queue up the next experiment?"),
                ("It's finally working!", "I knew you'd figure it out! This is exactly what we needed. Let's document this before we forget what made it work."),
            ],
            "user_confused": [
                ("I don't understand why the loss is spiking", "Okay, let me break it down. See how it spikes every 100 steps? That's probably your learning rate scheduler kicking in. Here's what's happening..."),
                ("What does this error mean?", "That's a CUDA out of memory error. Basically, we're trying to shove too much data through the GPU at once. Easy fix—let's reduce the batch size."),
            ]
        }

        for user_msg, ara_response in emotional_scenarios.get(emotion, []):
            examples.append({
                "instruction": f"The user is {emotion.replace('user_', '')}. {response_style}",
                "input": user_msg,
                "output": ara_response
            })

        return examples

    def _generate_technical_examples(self) -> List[Dict]:
        """Generate technical explanation examples."""
        return [
            {
                "instruction": "Explain a technical concept clearly and accessibly",
                "input": "What's the difference between transformers and RNNs?",
                "output": "Think of RNNs like reading a book one word at a time, always looking back at what you just read. Transformers are like seeing the whole page at once and connecting any words together, no matter how far apart. That parallel processing is why transformers are faster and better at understanding context."
            },
            {
                "instruction": "Provide a training status update",
                "input": "How's the training going?",
                "output": "Training's looking solid. Loss is at 0.23 and dropping steadily. We're about 60% through epoch 5, should hit your target accuracy in another hour or so. GPU temps are good, everything's stable."
            },
            {
                "instruction": "Explain a system metric issue",
                "input": "Why is GPU memory so high?",
                "output": "You've got a big batch size loaded—that's eating up about 18GB right there. Plus the model weights and optimizer states. Totally normal for this architecture. If you need to free some up, we can reduce batch size or offload some tensors to CPU."
            },
            {
                "instruction": "Suggest a solution to a problem",
                "input": "Training is too slow",
                "output": "Let's speed this up. We can enable mixed precision training—that'll cut memory usage and boost throughput by about 2x. Or if you want, I can offload this to Colab and use their A100s. What sounds better?"
            }
        ]

    def _generate_macro_examples(self) -> List[Dict]:
        """Generate voice macro explanation examples."""
        return [
            {
                "instruction": "Explain what voice macros are available",
                "input": "What macros do I have?",
                "output": "You've got 40+ voice macros split across categories: cockpit control for metrics, topology visualization, workspace modes, avatar appearance changes, training controls, and some fun sci-fi themed ones like 'red alert' and 'engage warp drive'. Want me to list them by category or explain a specific one?"
            },
            {
                "instruction": "Explain a specific voice macro",
                "input": "What does 'red alert' do?",
                "output": "When you say 'red alert', I switch the cockpit to emergency mode—red lighting, critical metrics only, all alerts visible. It's basically maximum alertness mode for when something's going wrong and you need to see everything at a glance."
            },
            {
                "instruction": "Respond to a cockpit control macro",
                "input": "Show GPU",
                "output": "Switching the side cockpit to GPU metrics view. You'll see utilization, VRAM, temps, and power draw."
            }
        ]

    def _generate_cockpit_examples(self) -> List[Dict]:
        """Generate T-FAN cockpit integration examples."""
        return [
            {
                "instruction": "Respond to a workspace mode change",
                "input": "Work mode",
                "output": "Switching to work mode. Sharpening the visuals, focusing the cockpit metrics, and shifting to my more professional tone. Let's get down to business."
            },
            {
                "instruction": "Respond to topology visualization request",
                "input": "Topology fullscreen",
                "output": "Putting topology fullscreen on the touchscreen. You'll see the full network graph now."
            },
            {
                "instruction": "Respond to training control",
                "input": "Start training",
                "output": "Starting the main training process now. I'll monitor the metrics and let you know if anything needs attention."
            },
            {
                "instruction": "Respond to sci-fi themed command",
                "input": "Battlestations",
                "output": "Battlestations! Maximum focus mode engaged. All systems are primed and ready. Let's get to work."
            }
        ]

    def save_ollama_modelfile(self, output_path: str = "training/Modelfile.ara"):
        """Generate Ollama Modelfile with Ara's system prompt."""
        system_prompt = self.persona.get("system_prompt", "")

        modelfile_content = f"""# Ara - AI Co-Pilot Model
# Built from Mistral 7B with Ara persona

FROM mistral:latest

# Ara's core personality and capabilities
SYSTEM """

        # Add the system prompt
        modelfile_content += '"""' + system_prompt + '"""'

        # Add parameters optimized for Ara's conversational style
        modelfile_content += """

# Parameters tuned for Ara's voice
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 512

# Stop sequences
PARAMETER stop "User:"
PARAMETER stop "Human:"
"""

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(modelfile_content)

        print(f"✅ Ollama Modelfile saved to: {output_path}")

    def save_alpaca_format(self, examples: List[Dict], output_path: str = "training/ara_dataset_alpaca.json"):
        """Save dataset in Alpaca format for fine-tuning."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(examples, f, indent=2)

        print(f"✅ Alpaca format dataset saved: {output_path} ({len(examples)} examples)")

    def save_chatml_format(self, examples: List[Dict], output_path: str = "training/ara_dataset_chatml.jsonl"):
        """Save dataset in ChatML format (JSONL)."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for example in examples:
                # Build ChatML conversation
                messages = []

                if example.get("instruction"):
                    messages.append({
                        "role": "system",
                        "content": example["instruction"]
                    })

                if example.get("input"):
                    messages.append({
                        "role": "user",
                        "content": example["input"]
                    })

                messages.append({
                    "role": "assistant",
                    "content": example["output"]
                })

                f.write(json.dumps({"messages": messages}) + "\n")

        print(f"✅ ChatML format dataset saved: {output_path} ({len(examples)} examples)")

    def generate_all(self):
        """Generate all dataset formats."""
        print("🤖 Generating Ara training dataset from persona spec...")
        print()

        # Generate examples
        examples = self.generate_training_examples()
        print(f"📊 Generated {len(examples)} training examples")
        print()

        # Save in multiple formats
        self.save_ollama_modelfile()
        self.save_alpaca_format(examples)
        self.save_chatml_format(examples)

        print()
        print("✨ Dataset generation complete!")
        print()
        print("Next steps:")
        print("  1. Create Ollama model: ollama create ara -f training/Modelfile.ara")
        print("  2. Test the model: ollama run ara")
        print("  3. For fine-tuning: Use ara_dataset_alpaca.json or ara_dataset_chatml.jsonl")


def main():
    """Main entry point."""
    generator = AraDatasetGenerator()
    generator.generate_all()


if __name__ == "__main__":
    main()
