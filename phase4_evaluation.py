#!/usr/bin/env python
"""
Phase 4: Model Evaluation & Inference Pipeline
Comprehensive evaluation of the Kaggle-trained EfficientNet model.
Tests on local dataset if available and generates inference report.
"""

import json
import cv2
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from src.core.pipeline import ProcessingPipeline
from src.config import RAW_DATA_DIR, LOGS_DIR, DEVICE

def get_local_dataset():
    """Load images from local dataset"""
    print("\n" + "="*60)
    print("STEP 1: Loading Local Dataset")
    print("="*60)
    
    dataset = {"attentive": [], "distracted": []}
    
    for label_dir in ["attentive", "distracted"]:
        label_path = RAW_DATA_DIR / label_dir
        if label_path.exists():
            images = list(label_path.glob("*.[jJ][pP][gG]")) + list(label_path.glob("*.[pP][nN][gG]"))
            dataset[label_dir] = images
            print(f"  ✓ {label_dir.upper()}: {len(images)} images")
        else:
            print(f"  ⚠ {label_dir.upper()}: folder not found at {label_path}")
    
    total = len(dataset["attentive"]) + len(dataset["distracted"])
    print(f"\n  Total images: {total}")
    return dataset

def run_inference_on_image(pipeline, image_path):
    """Run inference on a single image"""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return None, "Failed to load image"
        
        results = pipeline.process_frame(image)
        return results, None
    except Exception as e:
        return None, str(e)

def evaluate_dataset(pipeline, dataset):
    """Evaluate model on local dataset"""
    print("\n" + "="*60)
    print("STEP 2: Inference Evaluation")
    print("="*60)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": "EfficientNet-B0 (Kaggle-trained, 85.88% test acc)",
        "device": DEVICE,
        "evaluation": {
            "attentive": {"total": 0, "processed": 0, "errors": 0, "classifications": []},
            "distracted": {"total": 0, "processed": 0, "errors": 0, "classifications": []}
        },
        "inference_samples": []
    }
    
    for label in ["attentive", "distracted"]:
        images = dataset[label]
        results["evaluation"][label]["total"] = len(images)
        
        print(f"\n  Evaluating {label.upper()} ({len(images)} images):")
        
        for i, img_path in enumerate(images):
            inferences, error = run_inference_on_image(pipeline, img_path)
            
            if error:
                results["evaluation"][label]["errors"] += 1
                print(f"    [{i+1}/{len(images)}] ✗ Error: {error}")
            else:
                results["evaluation"][label]["processed"] += 1
                
                # Log inference
                sample = {
                    "image": str(img_path.name),
                    "ground_truth": label,
                    "detections_count": len(inferences),
                    "detections": []
                }
                
                for det in inferences:
                    sample["detections"].append({
                        "type": det.get("type"),
                        "status": det.get("status"),
                        "confidence": float(det.get("confidence", 0.0))
                    })
                
                results["evaluation"][label]["classifications"].append(sample)
                
                # Print progress
                if (i + 1) % max(1, len(images) // 5) == 0 or (i + 1) == len(images):
                    print(f"    [{i+1}/{len(images)}] ✓ Processed")
    
    return results

def compute_statistics(eval_results):
    """Compute evaluation statistics"""
    print("\n" + "="*60)
    print("STEP 3: Statistics & Analysis")
    print("="*60)
    
    stats = {
        "total_images": 0,
        "total_processed": 0,
        "total_errors": 0,
        "processing_rate": 0.0,
        "by_label": {}
    }
    
    for label in ["attentive", "distracted"]:
        label_data = eval_results["evaluation"][label]
        stats["total_images"] += label_data["total"]
        stats["total_processed"] += label_data["processed"]
        stats["total_errors"] += label_data["errors"]
        
        if label_data["total"] > 0:
            rate = (label_data["processed"] / label_data["total"]) * 100
        else:
            rate = 0.0
        
        stats["by_label"][label] = {
            "total": label_data["total"],
            "processed": label_data["processed"],
            "errors": label_data["errors"],
            "success_rate": rate
        }
        
        print(f"  {label.upper()}:")
        print(f"    Total: {label_data['total']}")
        print(f"    Processed: {label_data['processed']}")
        print(f"    Errors: {label_data['errors']}")
        print(f"    Success rate: {rate:.1f}%")
    
    if stats["total_images"] > 0:
        stats["processing_rate"] = (stats["total_processed"] / stats["total_images"]) * 100
    
    print(f"\n  OVERALL:")
    print(f"    Total images: {stats['total_images']}")
    print(f"    Successfully processed: {stats['total_processed']}")
    print(f"    Errors: {stats['total_errors']}")
    print(f"    Overall success rate: {stats['processing_rate']:.1f}%")
    
    return stats

def save_report(eval_results, stats):
    """Save evaluation report to JSON"""
    print("\n" + "="*60)
    print("STEP 4: Saving Report")
    print("="*60)
    
    report = {
        "phase": 4,
        "title": "Model Evaluation & Inference Report",
        "timestamp": eval_results["timestamp"],
        "model": eval_results["model"],
        "device": eval_results["device"],
        "statistics": stats,
        "detailed_results": eval_results["evaluation"]
    }
    
    # Create report file
    report_dir = LOGS_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"phase4_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"  ✓ Report saved to: {report_path}")
    return report_path

def main():
    print("\n" + "█"*60)
    print("█ PHASE 4: MODEL EVALUATION & INFERENCE PIPELINE")
    print("█ Comprehensive Evaluation of Kaggle-Trained Model")
    print("█"*60)
    
    # Initialize pipeline
    print("\n" + "="*60)
    print("INITIALIZATION")
    print("="*60)
    try:
        pipeline = ProcessingPipeline()
        print(f"✓ Pipeline initialized")
        print(f"  Model: EfficientNet-B0 (Kaggle-trained, 85.88% test accuracy)")
        print(f"  Device: {DEVICE}")
    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        return False
    
    # Load dataset
    dataset = get_local_dataset()
    
    if sum(len(v) for v in dataset.values()) == 0:
        print("\n⚠ No local images found. Skipping evaluation.")
        print("  To add images: place .jpg or .png files in:")
        print(f"    - {RAW_DATA_DIR / 'attentive'}")
        print(f"    - {RAW_DATA_DIR / 'distracted'}")
        return True
    
    # Run evaluation
    eval_results = evaluate_dataset(pipeline, dataset)
    
    # Compute statistics
    stats = compute_statistics(eval_results)
    
    # Save report
    report_path = save_report(eval_results, stats)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Phase 4 Evaluation Complete!")
    print(f"\n  Processed: {stats['total_processed']}/{stats['total_images']} images")
    print(f"  Success rate: {stats['processing_rate']:.1f}%")
    print(f"  Report location: {report_path}")
    print(f"\nNext steps:")
    print(f"  - Deploy API: python src/api/app.py")
    print(f"  - Run CLI: python src/cli.py")
    print(f"  - Check report: {report_path}")
    print("█"*60 + "\n")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
