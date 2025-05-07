"""
Calculate metrics for OCR recognition using CER, WER, and BLEU.
Compares the OCR text output with corrected_text as ground truth,
and provides metrics for the entire dataset and broken down by text block class.
"""

import os
import json
import logging
import numpy as np
import difflib
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

try:
    from jiwer import wer, cer
except ImportError:
    logger.warning("jiwer not installed. Install with 'pip install jiwer' for CER and WER metrics.")

    def cer(reference, hypothesis):
        """Fallback CER implementation using difflib"""
        matcher = difflib.SequenceMatcher(None, hypothesis, reference)
        edits = matcher.get_opcodes()
        edit_count = sum(max(i2-i1, j2-j1) for tag, i1, i2, j1, j2 in edits if tag != 'equal')
        return edit_count / len(reference) if reference else 0
    
    def wer(reference, hypothesis):
        """Fallback WER implementation using difflib"""
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        matcher = difflib.SequenceMatcher(None, hyp_words, ref_words)
        edits = matcher.get_opcodes()
        edit_count = sum(max(i2-i1, j2-j1) for tag, i1, i2, j1, j2 in edits if tag != 'equal')
        return edit_count / len(ref_words) if ref_words else 0

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
    logger.warning("nltk not installed. Install with 'pip install nltk' for BLEU metrics.")

    def sentence_bleu(references, hypothesis, smoothing_function=None):
        """Fallback BLEU implementation (simplified)"""
        if not hypothesis or not references:
            return 0.0

        ref_words = set()
        for ref in references:
            ref_words.update(ref)
        
        matches = sum(1 for word in hypothesis if word in ref_words)
        if not matches:
            return 0.0

        precision = matches / len(hypothesis) if hypothesis else 0
        return precision

    class SmoothingFunction:
        def method1(self):
            return lambda p: p or 0.01


def load_data(file_path: str) -> Dict[str, Any]:
    """
    Load JSON data from the given file path
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise


def calculate_metrics_by_class(blocks: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate CER, WER, and BLEU metrics for each class_name
    
    Args:
        blocks: List of text blocks containing OCR results
        
    Returns:
        Dictionary of metrics by class
    """
    metrics_by_class = {}

    class_counters = {}

    for block in blocks:
        class_name = block.get('class_name', 'unknown')
        text = block.get('text', '')
        corrected_text = block.get('corrected_text', '')

        if not text or not corrected_text:
            continue
        
        if class_name not in metrics_by_class:
            metrics_by_class[class_name] = {
                'cer_sum': 0.0,
                'wer_sum': 0.0,
                'bleu_sum': 0.0,
                'count': 0
            }
            class_counters[class_name] = 0

        try:
            block_cer = cer(corrected_text, text)
            block_wer = wer(corrected_text, text)

            smoothie = SmoothingFunction().method1
            reference = [corrected_text.split()]
            hypothesis = text.split()
            block_bleu = sentence_bleu(reference, hypothesis, smoothing_function=smoothie)

            metrics_by_class[class_name]['cer_sum'] += block_cer
            metrics_by_class[class_name]['wer_sum'] += block_wer
            metrics_by_class[class_name]['bleu_sum'] += block_bleu
            metrics_by_class[class_name]['count'] += 1
            
        except Exception as e:
            logger.warning(f"Error calculating metrics for block {block.get('id')}: {e}")

    results = {}
    for class_name, metrics in metrics_by_class.items():
        count = metrics['count']
        if count > 0:
            results[class_name] = {
                'cer': metrics['cer_sum'] / count,
                'wer': metrics['wer_sum'] / count,
                'bleu': metrics['bleu_sum'] / count,
                'count': count
            }
        else:
            results[class_name] = {
                'cer': 0.0,
                'wer': 0.0,
                'bleu': 0.0,
                'count': 0
            }
    
    return results


def calculate_overall_metrics(blocks: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate overall CER, WER, and BLEU metrics for all blocks
    
    Args:
        blocks: List of text blocks containing OCR results
        
    Returns:
        Dictionary of overall metrics
    """
    all_references = []
    all_hypotheses = []
    all_references_char = []
    all_hypotheses_char = []
    
    valid_block_count = 0
    
    for block in blocks:
        text = block.get('text', '')
        corrected_text = block.get('corrected_text', '')

        if not text or not corrected_text:
            continue
        
        all_references.append(corrected_text)
        all_hypotheses.append(text)
        all_references_char.extend(list(corrected_text))
        all_hypotheses_char.extend(list(text))
        valid_block_count += 1

    try:
        overall_cer = cer(' '.join(all_references), ' '.join(all_hypotheses))
        overall_wer = wer(' '.join(all_references), ' '.join(all_hypotheses))

        total_bleu = 0.0
        smoothie = SmoothingFunction().method1
        
        for ref, hyp in zip(all_references, all_hypotheses):
            reference = [ref.split()]
            hypothesis = hyp.split()
            try:
                total_bleu += sentence_bleu(reference, hypothesis, smoothing_function=smoothie)
            except Exception as e:
                logger.warning(f"Error calculating BLEU: {e}")
        
        overall_bleu = total_bleu / valid_block_count if valid_block_count > 0 else 0.0
        
    except Exception as e:
        logger.error(f"Error calculating overall metrics: {e}")
        overall_cer = overall_wer = overall_bleu = 0.0
    
    return {
        'cer': overall_cer,
        'wer': overall_wer,
        'bleu': overall_bleu,
        'count': valid_block_count
    }


def analyze_character_errors(blocks: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    """
    Analyze character-level errors in OCR output
    
    Args:
        blocks: List of text blocks containing OCR results
        
    Returns:
        Tuple containing (error_counts, confusion_matrix)
    """
    error_counts = {"insertions": 0, "deletions": 0, "substitutions": 0}
    confusion_matrix = {}  # Maps correct_char -> {predicted_char -> count}
    
    for block in blocks:
        text = block.get('text', '')
        corrected_text = block.get('corrected_text', '')

        if not text or not corrected_text:
            continue

        matcher = difflib.SequenceMatcher(None, text, corrected_text)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                # Substitution
                error_counts["substitutions"] += max(i2 - i1, j2 - j1)

                for k in range(min(i2 - i1, j2 - j1)):
                    correct_char = corrected_text[j1 + k]
                    predicted_char = text[i1 + k]
                    
                    if correct_char not in confusion_matrix:
                        confusion_matrix[correct_char] = {}
                    
                    if predicted_char not in confusion_matrix[correct_char]:
                        confusion_matrix[correct_char][predicted_char] = 0
                    
                    confusion_matrix[correct_char][predicted_char] += 1
                    
            elif tag == 'delete':
                # Characters in text but not in corrected_text (insertions)
                error_counts["insertions"] += i2 - i1
                
            elif tag == 'insert':
                # Characters in corrected_text but not in text (deletions)
                error_counts["deletions"] += j2 - j1
    
    return error_counts, confusion_matrix


def find_problematic_blocks(blocks: List[Dict[str, Any]], threshold: float = 0.3) -> List[Dict[str, Any]]:
    """
    Find blocks with high error rates
    
    Args:
        blocks: List of text blocks containing OCR results
        threshold: CER threshold for identifying problematic blocks
        
    Returns:
        List of problematic blocks with error metrics
    """
    problematic_blocks = []
    
    for block in blocks:
        text = block.get('text', '')
        corrected_text = block.get('corrected_text', '')

        if not text or not corrected_text:
            continue
        
        try:
            block_cer = cer(corrected_text, text)
            
            if block_cer > threshold:
                problematic_blocks.append({
                    'id': block.get('id'),
                    'class_name': block.get('class_name'),
                    'text': text,
                    'corrected_text': corrected_text,
                    'cer': block_cer
                })
        except Exception as e:
            logger.warning(f"Error calculating CER for block {block.get('id')}: {e}")

    problematic_blocks.sort(key=lambda x: x['cer'], reverse=True)
    
    return problematic_blocks


def analyze_by_length(blocks: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Analyze recognition performance by text length
    
    Args:
        blocks: List of text blocks containing OCR results
        
    Returns:
        Dictionary of metrics by length category
    """
    length_categories = {
        'very_short': (0, 5),
        'short': (6, 20),
        'medium': (21, 50),
        'long': (51, 100),
        'very_long': (101, float('inf'))
    }

    metrics_by_length = {category: {'cer_sum': 0.0, 'wer_sum': 0.0, 'count': 0} 
                         for category in length_categories}
    
    for block in blocks:
        text = block.get('text', '')
        corrected_text = block.get('corrected_text', '')

        if not text or not corrected_text:
            continue
        
        text_length = len(corrected_text)
        category = None
        
        for cat, (min_len, max_len) in length_categories.items():
            if min_len <= text_length <= max_len:
                category = cat
                break
        
        if category is None:
            continue
        
        try:
            block_cer = cer(corrected_text, text)
            block_wer = wer(corrected_text, text)
            
            metrics_by_length[category]['cer_sum'] += block_cer
            metrics_by_length[category]['wer_sum'] += block_wer
            metrics_by_length[category]['count'] += 1
            
        except Exception as e:
            logger.warning(f"Error calculating metrics for block {block.get('id')}: {e}")

    results = {}
    for category, metrics in metrics_by_length.items():
        count = metrics['count']
        if count > 0:
            results[category] = {
                'cer': metrics['cer_sum'] / count,
                'wer': metrics['wer_sum'] / count,
                'count': count
            }
        else:
            results[category] = {
                'cer': 0.0,
                'wer': 0.0,
                'count': 0
            }
    
    return results


def generate_report(blocks: List[Dict[str, Any]], output_file: str, error_threshold: float = 0.3):
    """
    Generate comprehensive report with all metrics and analyses
    
    Args:
        blocks: List of text blocks containing OCR results
        output_file: Path to save the report
        error_threshold: CER threshold for identifying problematic blocks
    """
    overall_metrics = calculate_overall_metrics(blocks)
    metrics_by_class = calculate_metrics_by_class(blocks)
    error_counts, confusion_matrix = analyze_character_errors(blocks)
    problematic_blocks = find_problematic_blocks(blocks, error_threshold)
    metrics_by_length = analyze_by_length(blocks)

    report = {
        "report_timestamp": datetime.now().isoformat(),
        "overall_metrics": overall_metrics,
        "metrics_by_class": metrics_by_class,
        "metrics_by_length": metrics_by_length,
        "error_analysis": {
            "error_counts": error_counts,
            "top_confusions": []
        },
        "problematic_blocks": {
            "count": len(problematic_blocks),
            "threshold": error_threshold,
            "examples": problematic_blocks[:10]  # Only include top 10 examples
        }
    }

    confusion_list = []
    for correct_char, predictions in confusion_matrix.items():
        for predicted_char, count in predictions.items():
            if correct_char != predicted_char:  # Only include errors
                confusion_list.append({
                    "correct": correct_char,
                    "predicted": predicted_char,
                    "count": count
                })

    confusion_list.sort(key=lambda x: x["count"], reverse=True)
    report["error_analysis"]["top_confusions"] = confusion_list[:20]  # Top 20 confusions

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report


def print_report_summary(report: Dict[str, Any]):
    """
    Print a summary of the report to console
    
    Args:
        report: Generated report dictionary
    """
    print("\n" + "="*80)
    print(f"OCR RECOGNITION METRICS SUMMARY")
    print("="*80)
    
    om = report["overall_metrics"]
    print(f"\nOVERALL METRICS (across {om['count']} text blocks):")
    print(f"  Character Error Rate (CER): {om['cer']:.4f}")
    print(f"  Word Error Rate (WER): {om['wer']:.4f}")
    print(f"  BLEU Score: {om['bleu']:.4f}")

    print("\nMETRICS BY TEXT BLOCK CLASS:")
    for class_name, metrics in report["metrics_by_class"].items():
        print(f"  {class_name} (n={metrics['count']}):")
        print(f"    CER: {metrics['cer']:.4f}")
        print(f"    WER: {metrics['wer']:.4f}")
        print(f"    BLEU: {metrics['bleu']:.4f}")

    print("\nMETRICS BY TEXT LENGTH:")
    for category, metrics in report["metrics_by_length"].items():
        if metrics['count'] > 0:
            print(f"  {category} (n={metrics['count']}):")
            print(f"    CER: {metrics['cer']:.4f}")
            print(f"    WER: {metrics['wer']:.4f}")

    error_counts = report["error_analysis"]["error_counts"]
    total_errors = sum(error_counts.values())
    print("\nERROR ANALYSIS:")
    print(f"  Total errors: {total_errors}")
    for error_type, count in error_counts.items():
        percentage = count / total_errors * 100 if total_errors > 0 else 0
        print(f"  {error_type.capitalize()}: {count} ({percentage:.1f}%)")

    print("\nTOP 5 CHARACTER CONFUSIONS:")
    for i, conf in enumerate(report["error_analysis"]["top_confusions"][:5], 1):
        print(f"  {i}. '{conf['correct']}' → '{conf['predicted']}': {conf['count']} occurrences")

    pb = report["problematic_blocks"]
    print(f"\nPROBLEMATIC BLOCKS (CER > {pb['threshold']}):")
    print(f"  Found {pb['count']} blocks with high error rates")
    if pb["examples"]:
        print("\nEXAMPLE PROBLEMATIC BLOCK:")
        example = pb["examples"][0]
        print(f"  ID: {example['id']}, Class: {example['class_name']}, CER: {example['cer']:.4f}")
        print(f"  Text: {example['text']}")
        print(f"  Corrected: {example['corrected_text']}")
    
    print("\n" + "="*80)
    print(f"Full report saved to: {output_file}")
    print("="*80 + "\n")


def main():
    """
    Main function to calculate recognition metrics
    """

    input_file = os.environ.get('OCR_METRICS_INPUT', 'paste.txt')
    output_file = os.environ.get('OCR_METRICS_OUTPUT', 'ocr_metrics_report.json')
    error_threshold = 0.3
    quiet_mode = False

    logger.info(f"Loading data from {input_file}")
    data = load_data(input_file)

    blocks = []
    if 'frames' in data:
        for frame in data['frames']:
            if 'blocks' in frame:
                blocks.extend(frame['blocks'])
    else:
        blocks = data.get('blocks', [])


    logger.info(f"Found {len(blocks)} blocks in the data")

    logger.info("Generating comprehensive metrics report")
    report = generate_report(blocks, output_file, error_threshold)

    if not quiet_mode:
        print_report_summary(report)


if __name__ == "__main__":
    main()