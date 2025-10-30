"""
Main Pipeline for Homework 2: Controllable Text-to-Music Generation
Integrates retrieval, captioning, feature extraction, generation, and evaluation
"""

import os
# Suppress xFormers warnings
os.environ['XFORMERS_DISABLED'] = '1'

import sys
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from retrieval.audio_encoder import AudioEncoder
from retrieval.similarity import MusicRetrieval
from captioning.captioner_factory import create_captioner
from extraction.melody_extractor import MelodyExtractor
from extraction.rhythm_extractor import RhythmExtractor
from extraction.dynamics_extractor import DynamicsExtractor
from generation.music_generator import MusicGenerator
from evaluation.clap_similarity import CLAPSimilarity
from evaluation.audiobox_aesthetics import AudioboxAesthetics
from evaluation.melody_accuracy import MelodyAccuracy


class HW2Pipeline:
    """
    Complete pipeline for HW2
    """

    def __init__(self, config: dict):
        """
        Initialize pipeline

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Paths
        self.target_dir = Path(config['target_dir'])
        self.reference_dir = Path(config['reference_dir'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)


        # Initialize components
        print("=" * 80)
        print("Initializing HW2 Pipeline")
        print("=" * 80)

        # Generators - will be loaded one at a time to save GPU memory
        self.generator_names = config.get('generators', ['musicgen-melody'])
        if isinstance(self.generator_names, str):
            self.generator_names = [self.generator_names]

        print(f"\nWill process with generators: {', '.join(self.generator_names)}")
        print("Note: Generators will be loaded one at a time to conserve GPU memory")

        # Initialize caption model using factory pattern
        caption_model_name = config.get('caption_model', 'qwen2-audio')

        print("\n" + "-" * 80)

        # Prepare config for caption model
        caption_config = {}
        if caption_model_name == 'lpmusiccaps':
            # Calculate optimal caption length for LP-MusicCaps
            caption_config['max_caption_length'] = self._calculate_optimal_caption_length(
                self.generator_names,
                target_audio_duration=60.0
            )

        # Create unified captioner
        self.captioner = create_captioner(caption_model_name, caption_config)
        print("-" * 80)

        print("\n" + "-" * 80)
        print("🎵 Initializing Audio Encoder & Retrieval (CLAP)")
        print("-" * 80)
        self.audio_encoder = AudioEncoder()
        self.retrieval = MusicRetrieval(self.audio_encoder)

        print("\n" + "-" * 80)
        print("🎸 Initializing Feature Extractors")
        print("-" * 80)
        self.melody_extractor = MelodyExtractor()
        self.rhythm_extractor = RhythmExtractor()
        self.dynamics_extractor = DynamicsExtractor()

        print("\n" + "-" * 80)
        print("📊 Initializing Evaluators")
        print("-" * 80)
        self.clap_eval = CLAPSimilarity()
        self.aesthetics_eval = AudioboxAesthetics()
        self.melody_eval = MelodyAccuracy()

        print("\n" + "=" * 80)
        print("✅ Pipeline initialized successfully!")
        print("=" * 80)

    def _calculate_optimal_caption_length(self, generator_names: list, target_audio_duration: float = 60.0) -> int:
        """
        Calculate optimal caption length based on the most restrictive model

        Args:
            generator_names: List of generator model names
            target_audio_duration: Duration of target audio files in seconds

        Returns:
            Optimal max_caption_length per 10s segment
        """
        # Model text capacity when using conditioning (tokens)
        model_limits = {
            'musicgen-melody': 256,      # MusicGen-Melody with melody conditioning
            'musicgen': 512,              # MusicGen text-only (more capacity)
            'coco-mulla': 512,            # Coco-Mulla text encoding capacity
            'jasco': 512,                 # JASCO text encoding capacity
            'musecontrollite': 512,       # MuseControlLite text encoding capacity
        }

        # Find the most restrictive model
        min_limit = 512  # Default to most permissive
        most_restrictive_model = None

        for gen_name in generator_names:
            gen_name_lower = gen_name.lower()
            if gen_name_lower in model_limits:
                model_limit = model_limits[gen_name_lower]
                if model_limit < min_limit:
                    min_limit = model_limit
                    most_restrictive_model = gen_name

        # Calculate segments (assuming LP-MusicCaps 10s per segment)
        num_segments = int(target_audio_duration / 10.0)

        # Reserve 10% safety margin for tokenization variations
        safe_limit = int(min_limit * 0.9)

        # Calculate tokens per segment
        tokens_per_segment = safe_limit // num_segments

        print("\nCaption Length Optimization:")
        print(f"  Target audio duration: {target_audio_duration}s")
        print(f"  Number of 10s segments: {num_segments}")
        print(f"  Most restrictive model: {most_restrictive_model} (limit: {min_limit} tokens)")
        print(f"  Safe limit (90%): {safe_limit} tokens")
        print(f"  Calculated tokens per segment: {tokens_per_segment}")
        print(f"  Total caption length: ~{tokens_per_segment * num_segments} tokens")

        return tokens_per_segment

    def run_retrieval(self):
        """
        Run retrieval task
        """
        print("\n" + "=" * 80)
        print("TASK 1: RETRIEVAL")
        print("=" * 80)

        # Build reference database
        cache_file = self.output_dir / "reference_embeddings.pkl"
        self.retrieval.build_reference_database(
            str(self.reference_dir),
            cache_file=str(cache_file)
        )

        # Retrieve for all targets
        retrieval_results = self.retrieval.retrieve_for_targets(
            str(self.target_dir),
            top_k=5
        )

        # Save results
        self.retrieval.save_results(
            retrieval_results,
            str(self.output_dir / "retrieval_results.pkl")
        )

        # Evaluate retrieval
        retrieval_eval = {}
        for target_path, matches in retrieval_results.items():
            best_match = matches[0][0]  # Top match
            clap_score = self.clap_eval.audio_to_audio_similarity(target_path, best_match)

            target_name = Path(target_path).name
            retrieval_eval[target_name] = {
                'best_match': Path(best_match).name,
                'clap_score': clap_score,
                'all_matches': [(Path(m[0]).name, m[1]) for m in matches]
            }

        # Save evaluation
        with open(self.output_dir / "retrieval_evaluation.json", 'w') as f:
            json.dump(retrieval_eval, f, indent=2)

        return retrieval_results, retrieval_eval

    def run_generation(self):
        """
        Run generation task with all configured models (one at a time)
        """
        print("\n" + "=" * 80)
        print("TASK 2: CONTROLLABLE GENERATION")
        print("=" * 80)

        # Get target files
        target_files = list(self.target_dir.glob("*.wav")) + \
                      list(self.target_dir.glob("*.mp3"))

        # Pre-generate all captions to cache them before loading heavy generators
        print("\n" + "=" * 80)
        print("Pre-generating captions for all target files")
        print("=" * 80)
        caption_cache = {}
        for target_file in target_files:
            print(f"\nGenerating caption for: {target_file.name}")
            # Use a default caption length for caching
            try:
                caption = self.captioner.caption_audio(
                    str(target_file),
                    target_duration=60.0,
                    target_total_length=300,
                    temperature=0.7,
                    max_new_tokens=256
                )
                caption_cache[target_file.name] = caption
                print(f"Cached caption ({len(caption)} chars): {caption[:100]}...")
            except Exception as e:
                print(f"Error generating caption for {target_file.name}: {e}")
                caption_cache[target_file.name] = "Instrumental music"

        # Now unload captioner to free GPU memory before generation
        print("\n" + "=" * 80)
        print("Unloading caption model to free GPU memory for generation")
        print("=" * 80)
        del self.captioner
        self.captioner = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU Memory cleared")

        # Store caption cache in instance for use during generation
        self.caption_cache = caption_cache

        # Results per model
        all_generation_results = {gen_name: {} for gen_name in self.generator_names}

        # Process each generator separately to save GPU memory
        for gen_name in self.generator_names:
            print(f"\n{'#' * 80}")
            print(f"# Loading Generator: {gen_name}")
            print(f"{'#' * 80}")

            # Clear GPU memory before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"GPU Memory cleared")

            # Load generator
            try:
                generator = MusicGenerator(model_name=gen_name)
            except Exception as e:
                print(f"ERROR: Failed to initialize {gen_name}: {e}")
                print(f"Skipping {gen_name} entirely")
                continue

            # Process all targets with this generator
            for target_file in tqdm(target_files, desc=f"Generating with {gen_name}"):
                print(f"\n{'=' * 80}")
                print(f"Target: {target_file.name} | Generator: {gen_name}")
                print(f"{'=' * 80}")

                try:
                    result = self._process_single_target(target_file, gen_name, generator)
                    all_generation_results[gen_name][target_file.name] = result
                except RuntimeError as e:
                    error_msg = str(e)
                    if "CUDA" in error_msg:
                        print(f"CUDA ERROR generating {target_file.name} with {gen_name}")
                        print(f"Error: {error_msg[:200]}...")  # Print first 200 chars
                        print(f"Skipping this file - CUDA device is in error state")
                        print(f"Note: Continuing may cause instability. Consider restarting after this model.")
                        # Don't try to synchronize or clear cache - CUDA is already in error state
                    else:
                        print(f"ERROR generating {target_file.name} with {gen_name}: {e}")
                        print(f"Skipping this file")
                except Exception as e:
                    print(f"ERROR generating {target_file.name} with {gen_name}: {e}")
                    print(f"Skipping this file")

            # Unload generator to free memory
            print(f"\n{'#' * 80}")
            print(f"# Unloading {gen_name}")
            print(f"{'#' * 80}")
            del generator
            if torch.cuda.is_available():
                try:
                    # Only try to clear cache - avoid synchronize which can fail on error state
                    torch.cuda.empty_cache()
                    print("GPU Memory cleared")
                except Exception as e:
                    print(f"Warning: Could not clear GPU memory: {e}")
                    # Continue anyway

            # Save results for this model
            if all_generation_results[gen_name]:
                result_file = self.output_dir / f"generation_results_{gen_name}.json"
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(all_generation_results[gen_name], f, indent=2, ensure_ascii=False)
                print(f"Results saved for {gen_name}")

        return all_generation_results

    def _process_single_target(self, target_path: Path, generator_name: str, generator: MusicGenerator) -> dict:
        """
        Process single target file through complete pipeline

        Args:
            target_path: Path to target audio file

        Returns:
            Dictionary with all results
        """
        target_name = target_path.stem

        # Step 1: Get caption from cache
        print(f"\n[1/5] Getting caption from cache...")

        # Use cached caption generated before model loading
        caption = self.caption_cache.get(target_path.name, "Instrumental music")
        print(f"Caption ({len(caption)} chars): {caption}")

        # Step 2: Extract features
        print(f"\n[2/5] Extracting features...")

        melody_features = self.melody_extractor.extract_melody_contour(str(target_path))
        self.melody_extractor.save_melody_features(
            str(target_path),
            str(self.output_dir / "features" / "melody")
        )

        rhythm_features = self.rhythm_extractor.extract_rhythm_features(str(target_path))
        self.rhythm_extractor.save_rhythm_features(
            str(target_path),
            str(self.output_dir / "features" / "rhythm")
        )

        dynamics_features = self.dynamics_extractor.extract_dynamics_features(str(target_path))
        self.dynamics_extractor.save_dynamics_features(
            str(target_path),
            str(self.output_dir / "features" / "dynamics")
        )
        print(f"Features extracted and saved")

        # Step 3: Generate music
        print(f"\n[3/5] Generating music with {generator_name}...")

        output_path = self.output_dir / "generated" / generator_name / f"{target_name}_generated.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine generation strategy based on file type and model
        generation_config = self._get_generation_config(target_name, generator_name, generator)

        generated_audio = generator.generate(
            text_prompt=caption,
            melody=melody_features.get('cqt_top4') if generation_config['use_melody'] else None,
            rhythm=rhythm_features.get('beat_mask') if generation_config['use_rhythm'] else None,
            dynamics=dynamics_features.get('dynamics_curve') if generation_config['use_dynamics'] else None,
            audio_condition=str(target_path),  # Pass the target audio file for models that need it
            duration=generation_config['duration'],
            cfg_scale=generation_config['cfg_scale']
        )

        generator.save_audio(generated_audio, str(output_path))

        # Step 4: Evaluate
        print(f"\n[4/5] Evaluating...")

        # Trim target to match generated length
        import librosa
        import soundfile as sf

        target_trimmed_path = self.output_dir / "targets_trimmed" / generator_name / f"{target_name}_trimmed.wav"
        target_trimmed_path.parent.mkdir(parents=True, exist_ok=True)

        y_target, sr_target = librosa.load(
            str(target_path),
            sr=generator.sample_rate,
            duration=generation_config['duration']
        )
        sf.write(target_trimmed_path, y_target, sr_target)

        # CLAP evaluation
        clap_results = self.clap_eval.evaluate_generation(
            str(target_trimmed_path),
            str(output_path),
            caption
        )

        # Aesthetics evaluation
        aesthetics_target = self.aesthetics_eval.evaluate(str(target_trimmed_path))
        aesthetics_generated = self.aesthetics_eval.evaluate(str(output_path))

        # Melody accuracy
        melody_accuracy = self.melody_eval.melody_accuracy_comprehensive(
            str(target_trimmed_path),
            str(output_path),
            duration=generation_config['duration']
        )

        # Step 5: Compile results
        print(f"\n[5/5] Compiling results...")

        results = {
            'target_file': str(target_path.name),
            'generated_file': str(output_path.name),
            'generator': generator_name,
            'caption': caption,
            'generation_config': generation_config,
            'evaluation': {
                'clap': clap_results,
                'aesthetics_target': aesthetics_target,
                'aesthetics_generated': aesthetics_generated,
                'melody_accuracy': melody_accuracy
            }
        }

        # Save individual result
        result_file = self.output_dir / "results" / generator_name / f"{target_name}_result.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nCompleted processing {target_name} with {generator_name}")

        return results

    def _load_melody_features(self, target_name: str) -> dict:
        """Load pre-extracted melody features"""
        import pickle
        feature_path = self.output_dir / "features" / "melody" / f"{target_name}_melody.pkl"
        if feature_path.exists():
            with open(feature_path, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"Warning: Melody features not found for {target_name}, returning empty dict")
            return {}

    def _load_rhythm_features(self, target_name: str) -> dict:
        """Load pre-extracted rhythm features"""
        import pickle
        feature_path = self.output_dir / "features" / "rhythm" / f"{target_name}_rhythm.pkl"
        if feature_path.exists():
            with open(feature_path, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"Warning: Rhythm features not found for {target_name}, returning empty dict")
            return {}

    def _load_dynamics_features(self, target_name: str) -> dict:
        """Load pre-extracted dynamics features"""
        import pickle
        feature_path = self.output_dir / "features" / "dynamics" / f"{target_name}_dynamics.pkl"
        if feature_path.exists():
            with open(feature_path, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"Warning: Dynamics features not found for {target_name}, returning empty dict")
            return {}

    def _get_generation_config(self, target_name: str, generator_name: str, generator: MusicGenerator) -> dict:
        """
        Get generation configuration based on target type

        Args:
            target_name: Target file name

        Returns:
            Configuration dictionary
        """
        # Default config based on model
        # Set duration based on model capabilities
        if generator_name == 'jasco':
            duration = 10.0
        elif generator_name == 'coco-mulla':
            duration = 20.0
        elif generator_name == 'musecontrollite':
            duration = 47.0
        else:  # musicgen variants
            duration = 30.0

        config = {
            'use_melody': True,
            'use_rhythm': True,
            'use_dynamics': True,
            'duration': duration,
            'cfg_scale': 3.0  # Default guidance scale
        }

        # Customize based on content
        name_lower = target_name.lower()

        # Piano pieces - emphasize melody and dynamics
        if any(word in name_lower for word in ['piano', 'mussorgsky', 'spirited', 'pletnev']):
            config['use_melody'] = True
            config['use_rhythm'] = False
            config['use_dynamics'] = True
            config['cfg_scale'] = 3.5

        # Drum pieces - emphasize rhythm
        elif any(word in name_lower for word in ['jazz', 'rock', 'country', 'beat']):
            config['use_melody'] = False
            config['use_rhythm'] = True
            config['use_dynamics'] = True
            config['cfg_scale'] = 3.0

        # Flute pieces - emphasize melody
        elif any(word in name_lower for word in ['dizi', '竹笛', 'flute']):
            config['use_melody'] = True
            config['use_rhythm'] = False
            config['use_dynamics'] = True
            config['cfg_scale'] = 3.5

        # Duet pieces - use all conditions
        elif any(word in name_lower for word in ['菊花台', '伴奏']):
            config['use_melody'] = True
            config['use_rhythm'] = True
            config['use_dynamics'] = True
            config['cfg_scale'] = 3.0

        return config

    def generate_report(self, all_generation_results: dict):
        """
        Generate comprehensive report with model comparisons

        Args:
            all_generation_results: Results from all generators {model_name: {target_name: result}}
        """
        print("\n" + "=" * 80)
        print("GENERATING REPORT")
        print("=" * 80)

        # Create report filename based on models used
        models_with_results = [model for model, results in all_generation_results.items() if results]
        if len(models_with_results) == 1:
            report_filename = f"REPORT-{models_with_results[0]}.md"
        else:
            report_filename = f"REPORT-{'-'.join(models_with_results)}.md"

        report_path = self.output_dir / report_filename

        # Get list of models that have results
        models_with_results = [model for model, results in all_generation_results.items() if results]

        if not models_with_results:
            print("No results to report!")
            return

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Homework 2: Controllable Text-to-Music Generation\n\n")
            f.write("## Summary\n\n")
            f.write(f"- Models evaluated: {', '.join(models_with_results)}\n")
            f.write(f"- Caption model: {self.config.get('caption_model', 'simple')}\n\n")

            # Model Comparison Table
            f.write("## Model Comparison\n\n")
            f.write("Average performance across all target files:\n\n")

            # Calculate averages per model
            model_stats = {}
            for model_name in models_with_results:
                results = all_generation_results[model_name]
                if not results:
                    continue

                stats = {
                    'clap_audio_audio': [],
                    'clap_text_audio': [],
                    'clap_text_target': [],
                    'aesthetics_ce': [],
                    'aesthetics_pq': [],
                    'melody_acc': [],
                    'count': len(results)
                }

                for target_name, result in results.items():
                    eval_data = result['evaluation']
                    # FIXED: Use correct keys from CLAP evaluation
                    stats['clap_audio_audio'].append(eval_data['clap'].get('generated_target_similarity', 0))
                    stats['clap_text_audio'].append(eval_data['clap'].get('text_generated_similarity', 0))
                    stats['clap_text_target'].append(eval_data['clap'].get('target_text_similarity', 0))
                    # FIXED: Use correct keys from aesthetics evaluation
                    stats['aesthetics_ce'].append(eval_data['aesthetics_generated'].get('CE', 0))
                    stats['aesthetics_pq'].append(eval_data['aesthetics_generated'].get('PQ', 0))
                    # FIXED: Use correct key from melody evaluation
                    stats['melody_acc'].append(eval_data['melody_accuracy'].get('overall_melody_accuracy', 0))

                model_stats[model_name] = {
                    'clap_audio_audio': np.mean(stats['clap_audio_audio']),
                    'clap_text_audio': np.mean(stats['clap_text_audio']),
                    'clap_text_target': np.mean(stats['clap_text_target']),
                    'aesthetics_ce': np.mean(stats['aesthetics_ce']),
                    'aesthetics_pq': np.mean(stats['aesthetics_pq']),
                    'melody_acc': np.mean(stats['melody_acc']),
                    'count': stats['count']
                }

            # Write comparison table
            f.write("| Model | Files | CLAP (Gen↔Target) | CLAP (Text↔Gen) | CLAP (Text↔Target) | Aesthetics (CE) | Aesthetics (PQ) | Melody Acc |\n")
            f.write("|-------|-------|-------------------|-----------------|--------------------|-----------------|-----------------|-----------|\n")

            for model_name in models_with_results:
                if model_name not in model_stats:
                    continue
                stats = model_stats[model_name]
                f.write(f"| **{model_name}** | {stats['count']} | "
                       f"{stats['clap_audio_audio']:.4f} | "
                       f"{stats['clap_text_audio']:.4f} | "
                       f"{stats['clap_text_target']:.4f} | "
                       f"{stats['aesthetics_ce']:.3f} | "
                       f"{stats['aesthetics_pq']:.3f} | "
                       f"{stats['melody_acc']:.4f} |\n")

            f.write("\n")

            # Detailed Results by Model and Target
            f.write("## Detailed Results by Model\n\n")

            for model_name in models_with_results:
                results = all_generation_results[model_name]
                if not results:
                    continue

                f.write(f"### Model: {model_name}\n\n")

                for target_name, result in results.items():
                    f.write(f"#### Target: {target_name}\n\n")
                    f.write(f"**Caption:** {result['caption']}\n\n")
                    f.write(f"**Generated File:** `{result['generated_file']}`\n\n")

                    eval_data = result['evaluation']

                    f.write("**CLAP Similarity:**\n")
                    for key, value in eval_data['clap'].items():
                        f.write(f"- {key}: {value:.4f}\n")

                    f.write("\n**Audiobox Aesthetics (Target):**\n")
                    for key, value in eval_data['aesthetics_target'].items():
                        f.write(f"- {key}: {value:.3f}\n")

                    f.write("\n**Audiobox Aesthetics (Generated):**\n")
                    for key, value in eval_data['aesthetics_generated'].items():
                        f.write(f"- {key}: {value:.3f}\n")

                    f.write("\n**Melody Accuracy:**\n")
                    for key, value in eval_data['melody_accuracy'].items():
                        f.write(f"- {key}: {value:.4f}\n")

                    f.write("\n")

                f.write("\n---\n\n")

        print(f"Report saved to {report_path}")

    def run(self):
        """
        Run complete pipeline
        """
        # Task 1: Retrieval
        retrieval_results, retrieval_eval = self.run_retrieval()

        # Task 2: Generation
        generation_results = self.run_generation()

        # Generate report
        self.generate_report(generation_results)

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nResults saved to: {self.output_dir}")


def main():
    """
    Main entry point
    """
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='HW2 Pipeline for Text-to-Music Generation')
    parser.add_argument('--config', type=str, help='Path to JSON config file')
    args = parser.parse_args()

    # Load config from JSON file if provided, otherwise use default
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"📝 Loaded config from: {args.config}")
    else:
        # Default configuration
        config = {
            'target_dir': './home/fundwotsai/Deep_MIR_hw2/target_music_list_60s',
            'reference_dir': './home/fundwotsai/Deep_MIR_hw2/referecne_music_list_60s',
            'output_dir': './outputs',
            'caption_model': 'lpmusiccaps',  # or 'qwen2-audio'
            'generators': ['jucos', 'musicgen-melody'],  # Use MusicGen-Melody (works!)
            # Note: coco-mulla has architectural incompatibility with checkpoint
        }
        print("📝 Using default config")

    # Initialize and run pipeline
    pipeline = HW2Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
