#!/usr/bin/env python3
"""
PHASE 5: Pipeline Integration Test
Tests YOLO detection + EfficientNet classification + Attention scoring
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.pipeline import ProcessingPipeline
from src.core.scorer import CogniVisionScorer
from src.config import MODELS_DIR, DATA_DIR


def test_pipeline_initialization():
    """Test 1: Pipeline initialization"""
    print("\n" + "="*60)
    print("TEST 1: Pipeline Initialization")
    print("="*60)
    
    try:
        pipeline = ProcessingPipeline()
        print("✓ Pipeline initialized successfully")
        print(f"  - Detector model: {type(pipeline.detector).__name__}")
        print(f"  - Classifier: EfficientNet-B0")
        print(f"  - Input size: {pipeline.transform.transforms[0].size}")
        print(f"  - Labels: {pipeline.labels}")
        return pipeline
    except Exception as e:
        print(f"✗ Pipeline initialization failed: {e}")
        return None


def test_synthetic_frame_inference(pipeline):
    """Test 2: Inference on synthetic frame"""
    print("\n" + "="*60)
    print("TEST 2: Synthetic Frame Inference")
    print("="*60)
    
    if pipeline is None:
        print("⚠ Skipping - pipeline not initialized")
        return None
    
    try:
        # Create synthetic frame (640x480)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        results = pipeline.process_frame(frame)
        
        print(f"✓ Frame processed successfully")
        print(f"  - Frame size: {frame.shape}")
        print(f"  - Detections: {len(results)}")
        
        for i, det in enumerate(results):
            print(f"\n  Detection {i+1}:")
            print(f"    - Type: {det['type']}")
            print(f"    - Status: {det['status']}")
            print(f"    - Confidence: {det['confidence']:.4f}")
            if 'error' in det:
                print(f"    - Error: {det['error']}")
        
        return results
    except Exception as e:
        print(f"✗ Synthetic frame inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_real_dataset_inference(pipeline):
    """Test 3: Inference on real dataset images"""
    print("\n" + "="*60)
    print("TEST 3: Real Dataset Inference")
    print("="*60)
    
    if pipeline is None:
        print("⚠ Skipping - pipeline not initialized")
        return None
    
    # Check if dataset exists
    dataset_path = DATA_DIR / "raw"
    if not dataset_path.exists():
        print(f"⚠ Dataset not found at {dataset_path}")
        return None
    
    try:
        # Count images in each category
        attentive_path = dataset_path / "attentive"
        distracted_path = dataset_path / "distracted"
        
        attentive_images = list(attentive_path.glob("*.jpg")) + list(attentive_path.glob("*.png"))
        distracted_images = list(distracted_path.glob("*.jpg")) + list(distracted_path.glob("*.png"))
        
        print(f"✓ Found {len(attentive_images)} attentive images")
        print(f"✓ Found {len(distracted_images)} distracted images")
        
        # Test inference on first image from each category
        results_summary = {"attentive": 0, "distracted": 0, "phone": 0, "unknown": 0}
        
        for category, images in [("attentive", attentive_images), ("distracted", distracted_images)]:
            if not images:
                continue
            
            image_path = images[0]
            frame = cv2.imread(str(image_path))
            
            if frame is None:
                print(f"⚠ Could not read {image_path}")
                continue
            
            results = pipeline.process_frame(frame)
            
            print(f"\n  Sample from '{category}': {image_path.name}")
            print(f"    - Frame shape: {frame.shape}")
            print(f"    - Detections: {len(results)}")
            
            for det in results:
                if det['type'] == 'student':
                    results_summary[det['status']] += 1
                    print(f"    - Student: {det['status']} (conf: {det['confidence']:.4f})")
                elif det['type'] == 'distraction':
                    results_summary['phone'] += 1
                    print(f"    - Phone detected (conf: {det['detection_confidence']:.4f})")
        
        print(f"\n  Results Summary:")
        print(f"    - Attentive: {results_summary['attentive']}")
        print(f"    - Distracted: {results_summary['distracted']}")
        print(f"    - Phones: {results_summary['phone']}")
        print(f"    - Unknown: {results_summary['unknown']}")
        
        return results_summary
    
    except Exception as e:
        print(f"✗ Real dataset inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_scorer_integration(pipeline):
    """Test 4: Scorer integration with pipeline output"""
    print("\n" + "="*60)
    print("TEST 4: Scorer Integration")
    print("="*60)
    
    if pipeline is None:
        print("⚠ Skipping - pipeline not initialized")
        return None
    
    try:
        scorer = CogniVisionScorer(phone_penalty=20)
        print("✓ Scorer initialized")
        
        # Create mock engine results
        mock_results = [
            {'type': 'student', 'status': 'attentive', 'confidence': 0.92},
            {'type': 'student', 'status': 'attentive', 'confidence': 0.88},
            {'type': 'student', 'status': 'distracted', 'confidence': 0.75},
            {'type': 'student', 'status': 'unknown', 'confidence': 0.0},
            {'type': 'distraction', 'status': 'phone_detected', 'confidence': 0.85}
        ]
        
        # Calculate class score
        class_score = scorer.calculate_class_score(mock_results)
        print(f"✓ Class score calculated: {class_score}")
        
        # Generate report
        report = scorer.get_individual_report(mock_results)
        print(f"✓ Report generated:")
        print(f"  - Total students: {report['total_students']}")
        print(f"  - Attentive: {report['attentive']}")
        print(f"  - Distracted: {report['distracted']}")
        print(f"  - Unknown: {report.get('unknown', 0)}")
        print(f"  - Phones detected: {report.get('phones_detected', 0)}")
        print(f"  - Class score: {report['class_score']}")
        
        return report
    
    except Exception as e:
        print(f"✗ Scorer integration failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_end_to_end():
    """Test 5: Full end-to-end pipeline"""
    print("\n" + "="*60)
    print("TEST 5: End-to-End Pipeline")
    print("="*60)
    
    try:
        # Initialize pipeline and scorer
        pipeline = ProcessingPipeline()
        scorer = CogniVisionScorer(phone_penalty=20)
        
        print("✓ Pipeline and scorer initialized")
        
        # Create synthetic video frames
        num_frames = 3
        results_log = []
        
        for frame_idx in range(num_frames):
            # Create random frame
            frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
            
            # Process frame
            detections = pipeline.process_frame(frame)
            
            # Calculate score
            score = scorer.calculate_class_score(detections)
            report = scorer.get_individual_report(detections)
            
            results_log.append({
                'frame': frame_idx,
                'detections': len(detections),
                'score': score,
                'report': report
            })
            
            print(f"\n  Frame {frame_idx}:")
            print(f"    - Detections: {len(detections)}")
            print(f"    - Class score: {score}")
        
        print(f"\n✓ Processed {num_frames} frames successfully")
        return results_log
    
    except Exception as e:
        print(f"✗ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all Phase 5 integration tests"""
    print("\n" + "█"*60)
    print("█ PHASE 5: PIPELINE INTEGRATION TEST SUITE")
    print("█"*60)
    
    # Test 1: Initialization
    pipeline = test_pipeline_initialization()
    
    # Test 2: Synthetic inference
    synthetic_results = test_synthetic_frame_inference(pipeline)
    
    # Test 3: Real dataset
    dataset_results = test_real_dataset_inference(pipeline)
    
    # Test 4: Scorer
    scorer_report = test_scorer_integration(pipeline)
    
    # Test 5: End-to-end
    e2e_results = test_end_to_end()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    tests = [
        ("Pipeline Initialization", pipeline is not None),
        ("Synthetic Frame Inference", synthetic_results is not None),
        ("Real Dataset Inference", dataset_results is not None),
        ("Scorer Integration", scorer_report is not None),
        ("End-to-End Pipeline", e2e_results is not None)
    ]
    
    passed = sum(1 for _, status in tests if status)
    total = len(tests)
    
    for test_name, status in tests:
        symbol = "✓" if status else "✗"
        print(f"{symbol} {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 PHASE 5 COMPLETE: All integration tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Review output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
