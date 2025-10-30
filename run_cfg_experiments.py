"""
Run experiments with different CFG configurations for JASCO
Tests 5 different CFG settings to find optimal balance
"""

import subprocess
import json
import shutil
from pathlib import Path

# Define guidance scale configurations to test for MusicGen-Melody
CFG_CONFIGS = [
    {
        "name": "guidance_low",
        "guidance_scale": 2.0,
        "description": "Low guidance scale - more creative/diverse output"
    },
    {
        "name": "guidance_medium",
        "guidance_scale": 3.0,
        "description": "Medium guidance scale - balanced"
    },
    {
        "name": "guidance_high",
        "guidance_scale": 5.0,
        "description": "High guidance scale - stronger adherence to prompts"
    },
    {
        "name": "guidance_very_high",
        "guidance_scale": 7.0,
        "description": "Very high guidance scale - maximum prompt adherence"
    }
]

def update_guidance_scale(guidance_scale):
    """Update guidance_scale in music_generator.py for MusicGen-Melody"""
    file_path = Path("src/generation/music_generator.py")

    with open(file_path, 'r') as f:
        content = f.read()

    # Replace the guidance_scale in the default parameter
    import re
    # Find and replace in _generate_musicgen method signature
    content = re.sub(
        r'cfg_scale\s*=\s*[\d.]+',
        f'cfg_scale={guidance_scale}',
        content
    )

    with open(file_path, 'w') as f:
        f.write(content)

    print(f"✓ Updated guidance_scale: {guidance_scale}")

def run_experiment(config):
    """Run pipeline with specific guidance scale configuration"""
    name = config['name']
    guidance_scale = config['guidance_scale']

    print("\n" + "="*80)
    print(f"EXPERIMENT: {name}")
    print(f"Description: {config['description']}")
    print(f"Guidance Scale: {guidance_scale}")
    print("="*80)

    # Update guidance scale in code
    update_guidance_scale(guidance_scale)

    # Run pipeline
    output_dir = f"outputs_cfg_{name}"

    # Clean output directory
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)

    # Create config file for this experiment
    base_config = {
        "target_dir": "home/fundwotsai/Deep_MIR_hw2/target_music_list_60s",
        "reference_dir": "home/fundwotsai/Deep_MIR_hw2/referecne_music_list_60s",
        "output_dir": output_dir,
        "caption_model": "qwen2-audio-natural",
        "generators": ["musicgen-melody"],
        "num_files": 2  # Run only 2 songs per experiment for faster comparison
    }

    config_path = f"config_cfg_{name}.json"
    with open(config_path, 'w') as f:
        json.dump(base_config, f, indent=2)

    print(f"\nRunning pipeline with config: {config_path}")

    # Run the pipeline
    result = subprocess.run(
        ["uv", "run", "python", "main_pipeline.py", "--config", config_path],
        capture_output=False,
        text=True
    )

    if result.returncode == 0:
        print(f"✓ Experiment {name} completed successfully!")

        # Copy report with descriptive name
        report_src = Path(output_dir) / "REPORT-musicgen-melody.md"
        report_dst = Path(f"REPORT_cfg_musicgen_{name}.md")
        if report_src.exists():
            shutil.copy(report_src, report_dst)
            print(f"✓ Report saved to: {report_dst}")
    else:
        print(f"✗ Experiment {name} failed!")

    return result.returncode == 0

def create_comparison_report():
    """Create a summary comparing all experiments"""
    print("\n" + "="*80)
    print("Creating comparison report...")
    print("="*80)

    results = []

    for config in CFG_CONFIGS:
        name = config['name']
        report_path = Path(f"REPORT_cfg_musicgen_{name}.md")

        if not report_path.exists():
            print(f"⚠️  Report not found for {name}")
            continue

        # Parse the report to extract average scores
        with open(report_path, 'r') as f:
            content = f.read()

        # Extract average scores from the table
        import re
        match = re.search(
            r'\| \*\*musicgen-melody\*\* \| \d+ \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|',
            content
        )

        if match:
            results.append({
                'name': name,
                'guidance_scale': config['guidance_scale'],
                'description': config['description'],
                'gen_target': float(match.group(1)),
                'text_gen': float(match.group(2)),
                'text_target': float(match.group(3)),
                'aesthetics_ce': float(match.group(4)),
                'aesthetics_pq': float(match.group(5)),
                'melody_acc': float(match.group(6))
            })

    # Create comparison report
    with open('REPORT_CFG_MUSICGEN_COMPARISON.md', 'w') as f:
        f.write("# CFG Configuration Comparison\n\n")
        f.write("## Summary\n\n")
        f.write("Comparison of different CFG (Classifier-Free Guidance) configurations for MusicGen-Melody.\n\n")

        f.write("## Configurations Tested\n\n")
        for config in CFG_CONFIGS:
            f.write(f"### {config['name']}\n")
            f.write(f"- **guidance_scale**: {config['guidance_scale']}\n")
            f.write(f"- **Description**: {config['description']}\n\n")

        f.write("## Results Comparison\n\n")
        f.write("| Config | Guidance Scale | Gen↔Target | Text↔Gen | Text↔Target | Melody Acc | Aesthetics CE |\n")
        f.write("|--------|----------------|------------|----------|-------------|------------|---------------|\n")

        for r in results:
            f.write(f"| **{r['name']}** | {r['guidance_scale']:.1f} | "
                   f"{r['gen_target']:.4f} | {r['text_gen']:.4f} | {r['text_target']:.4f} | "
                   f"{r['melody_acc']:.4f} | {r['aesthetics_ce']:.3f} |\n")

        f.write("\n## Analysis\n\n")

        # Find best for each metric
        if results:
            best_gen_target = max(results, key=lambda x: x['gen_target'])
            best_text_gen = max(results, key=lambda x: x['text_gen'])
            best_melody = max(results, key=lambda x: x['melody_acc'])

            f.write(f"### Best Configurations\n\n")
            f.write(f"- **Best Gen↔Target**: `{best_gen_target['name']}` ({best_gen_target['gen_target']:.4f})\n")
            f.write(f"- **Best Text↔Gen**: `{best_text_gen['name']}` ({best_text_gen['text_gen']:.4f})\n")
            f.write(f"- **Best Melody Accuracy**: `{best_melody['name']}` ({best_melody['melody_acc']:.4f})\n\n")

            f.write("### Observations\n\n")
            f.write("- **Higher guidance_scale** → Stronger adherence to text and melody conditioning\n")
            f.write("- **Lower guidance_scale** → More creative/diverse output\n")
            f.write("- Optimal guidance scale balances prompt adherence with generation quality\n")

    print("✓ Comparison report saved to: REPORT_CFG_MUSICGEN_COMPARISON.md")

def main():
    print("CFG Experiment Runner")
    print("=" * 80)
    print(f"Will run {len(CFG_CONFIGS)} experiments with different CFG configurations")
    print("Starting experiments...")

    successful = 0
    failed = 0

    for i, config in enumerate(CFG_CONFIGS, 1):
        print(f"\n[{i}/{len(CFG_CONFIGS)}] Starting experiment: {config['name']}")

        if run_experiment(config):
            successful += 1
        else:
            failed += 1

    print("\n" + "="*80)
    print(f"All experiments completed!")
    print(f"Successful: {successful}/{len(CFG_CONFIGS)}")
    print(f"Failed: {failed}/{len(CFG_CONFIGS)}")
    print("="*80)

    # Create comparison report
    if successful > 0:
        create_comparison_report()

if __name__ == "__main__":
    main()
