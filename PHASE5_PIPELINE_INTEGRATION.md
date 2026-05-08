# PHASE 5: PIPELINE INTEGRATION REPORT

**Status**: ✓ Complete  
**Date**: May 8, 2026  
**Test Results**: 5/5 Passed

---

## SUMMARY

Phase 5 successfully integrated YOLO detection + EfficientNet-B0 classification + attention scoring into a unified pipeline. All components work correctly together.

---

## WORK COMPLETED

### 1. ✓ YOLO + Classifier Connection
**File**: `src/core/pipeline.py`

- Unified `ProcessingPipeline` class orchestrates detection → classification
- YOLO detector identifies persons and cell phones
- EfficientNet classifier processes face crops to determine attention status
- Proper error handling for edge cases (invalid crops, classification errors)

**Key Code**:
```python
detections = self.detector.detect_students(frame)  # YOLO detection
for det in detections:
    face_crop = self._extract_face_region(frame, bbox)
    classification = self.classifier(tensor_image)  # EfficientNet
    status = self.labels[predicted_class]  # attentive/distracted
```

---

### 2. ✓ Bounding Box Logic Improvements
**File**: `src/core/pipeline.py`

**New Method**: `_extract_face_region(frame, bbox, face_height_ratio=0.45)`

**Improvements**:
- ✓ Bounds checking: Ensures crop coordinates stay within frame dimensions
- ✓ Validation: Checks crop size (minimum 10×10 pixels)
- ✓ Configurable: Face height ratio can be adjusted (currently 45% - optimal for eyes/mouth)
- ✓ Error handling: Returns None for invalid crops instead of crashing

**Before**:
```python
x1, y1, x2, y2 = bbox
face_crop = frame[y1:y1+int(height*0.45), x1:x2]
if face_crop.size > 0:  # Only checks if empty
```

**After**:
```python
face_crop = self._extract_face_region(frame, bbox, face_height_ratio=0.45)
# Bounds checking, validation, error handling
if face_crop is not None:
```

---

### 3. ✓ Attention Scoring Implementation
**File**: `src/core/scorer.py`

**Fixed Issues**:

1. **Pipeline Output Mismatch** (Bug #1)
   - Was looking for `'status': 'distraction (phone)'`
   - Pipeline outputs `'status': 'phone_detected'`
   - **Fixed**: Updated to match actual pipeline output format

2. **Incomplete Report** (Bug #2)
   - Missing 'unknown' and 'phones_detected' counts
   - Only returned minimal data
   - **Fixed**: Enhanced `get_individual_report()` to include full metrics

3. **Score Calculation** (Bug #3)
   - Didn't properly detect phone distractions
   - **Fixed**: Now correctly identifies and penalizes phone detections

**Current Implementation**:
```python
class CogniVisionScorer:
    def calculate_class_score(self, engine_results):
        # Formula: (Attentive / Total) * 100 - (Phones * 20)
        students = [r for r in engine_results if r['type'] == 'student']
        phones = [r for r in engine_results if r['type'] == 'distraction']
        
        attentive_count = sum(1 for s in students if s['status'] == 'attentive')
        base_score = (attentive_count / len(students)) * 100
        final_score = max(0.0, base_score - (len(phones) * 20))
        return final_score

    def get_individual_report(self, engine_results):
        # Returns comprehensive report with all metrics
        return {
            "total_students": len(students),
            "attentive": count_attentive,
            "distracted": count_distracted,
            "unknown": count_unknown,
            "phones_detected": count_phones,
            "class_score": score
        }
```

**Scoring Formula**:
$$\text{Class Score} = \max(0, \frac{\text{Attentive Students}}{\text{Total Students}} \times 100 - (\text{Phones} \times 20))$$

---

## TEST RESULTS

### Pipeline Initialization Test
```
✓ Detector model: CogniVisionDetector (YOLOv8n)
✓ Classifier: EfficientNet-B0
✓ Input size: (224, 224)
✓ Labels: ['distracted', 'attentive']
```

### Real Dataset Inference
```
✓ Found 16 attentive images
✓ Found 22 distracted images
✓ Successfully processed samples from both categories

Sample Results:
  - Attentive image: 2 students detected, both classified as attentive
  - Distracted image: 1 student detected

Performance: 3/3 students successfully classified
```

### Scorer Integration
```
Mock Results (5 objects):
  - 2 attentive students
  - 1 distracted student
  - 1 unknown student
  - 1 phone detection

✓ Class score: 30.0
  Calculation: (2/4) * 100 - (1 * 20) = 50 - 20 = 30

✓ Report generated with all metrics:
  - total_students: 4
  - attentive: 2
  - distracted: 1
  - unknown: 1
  - phones_detected: 1
  - class_score: 30.0
```

### End-to-End Pipeline
```
✓ Pipeline initialization: Success
✓ Scorer initialization: Success
✓ 3 frames processed: Success
✓ Scores calculated: Success
```

---

## OUTPUT FORMAT

### Pipeline Output (per frame)
```python
{
    'type': 'student',              # or 'distraction'
    'bbox': [x1, y1, x2, y2],      # Bounding box coordinates
    'status': 'attentive',          # or 'distracted', 'unknown', 'phone_detected'
    'confidence': 0.92,             # Classification confidence (0-1)
    'detection_confidence': 0.85,   # YOLO detection confidence
    'error': None                   # Error message if occurred
}
```

### Scorer Output (per frame)
```python
{
    'total_students': 4,
    'attentive': 2,
    'distracted': 1,
    'unknown': 1,
    'phones_detected': 1,
    'class_score': 30.0             # Overall class attention score
}
```

---

## ARCHITECTURE

```
Video Frame
    ↓
[YOLO Detection]  ← YOLOv8n detects persons & phones
    ↓
    ├─ Person Detection
    │   ↓
    │ [Face Region Extraction]  ← Top 45% of bbox (eyes/mouth area)
    │   ↓
    │ [Resize to 224×224]
    │   ↓
    │ [EfficientNet Classification]  ← Emotion → Attention mapping
    │   ↓
    │ Label: attentive/distracted/unknown
    │
    └─ Phone Detection
        ↓
        Label: phone_detected (distraction)

[Scoring Engine]
    ↓
    ├─ Count attentive students
    ├─ Count phones
    ├─ Calculate: (attentive/total) × 100 - (phones × 20)
    ↓
Class Score (0-100)
```

---

## KNOWN BEHAVIORS

### 1. Emotion → Attention Mapping
The classifier maps FER2013 emotions to attention:
- **Attentive**: Happy, Neutral (engaged, paying attention)
- **Distracted**: Angry, Disgust, Fear, Sad, Surprise (disengaged)

**Note**: This is emotion-based, not true attention. Real attention detection would require gaze tracking or other eye-gaze metrics.

### 2. Face Crop Strategy
- **Top 45% of person bbox**: Focuses on eyes and mouth region
- **Validated**: Checks minimum 10×10 pixel size
- **Bounded**: Ensures crop stays within frame dimensions
- **Fallback**: Returns 'unknown' status if crop fails

### 3. Phone Detection
- **YOLO Class 67**: Cell phone from COCO dataset
- **Penalty**: -20 points per phone detected
- **Purpose**: Flags obvious distraction sources

### 4. Scoring Edge Cases
- **No students**: Class score = 0.0
- **All attentive**: Score = 100.0 (before phone penalties)
- **All distracted + phones**: Score = 0.0 (capped at minimum)

---

## FILES MODIFIED

| File | Changes |
|------|---------|
| `src/core/pipeline.py` | Added `_extract_face_region()` method, improved `process_frame()` |
| `src/core/scorer.py` | Fixed phone detection filter, enhanced report generation |
| **New**: `test_phase5_integration.py` | Comprehensive integration test suite (5 tests) |

---

## VALIDATION

✓ All integration tests passed (5/5)  
✓ YOLO detection working  
✓ EfficientNet classification working  
✓ Bounding box logic correct  
✓ Scorer integration complete  
✓ End-to-end pipeline functional  

---

## READY FOR PHASE 6

**Phase 6: Testing Phase**
- System is ready for user testing
- Can run CLI inference: `python src/cli.py --video <video_file>`
- Can run API server: `python src/api/app.py`
- Can test with real classroom video

---

**PHASE 5 COMPLETE**  
**Next: PHASE 6 - Testing Phase**

All pipeline components integrated and validated.
