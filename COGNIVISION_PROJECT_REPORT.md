# CogniVision: Automated Student Attention and Cheating Behavior Detection

## Project Members

| Name | Student ID |
| --- | --- |
| [Member 1 Name] | [Student ID] |
| [Member 2 Name] | [Student ID] |
| [Member 3 Name] | [Student ID] |

## Executive Summary

CogniVision is a computer vision project for live student monitoring in exam or classroom settings. The system combines two complementary detection tracks: an RGB attention classifier for student face crops and a YOLO-based behavior detector for cheating-related actions and objects. The final design uses a dual-detector pipeline with no cross-detector suppression so that valid phone and behavior detections are not removed by unrelated person detections.

The strongest recorded results are from the RGB attention classifier, which achieved a best validation accuracy of 90.10% and a test accuracy of 89.05%. The classifier also reached 89.2% precision, 88.8% recall, and an F1-score of 0.890. The Track 1 YOLO detector was trained on 656 images across four behavior classes and achieved 37.3 mAP@0.5 on the preserved documentation snapshot.

## I. Problem Definition and Domain Research

### I.1 Problem Statement

Manual invigilation is difficult to scale in crowded exam halls, hybrid classrooms, and remote proctoring scenarios. A human supervisor may miss suspicious behavior when multiple students appear in the frame, when the camera angle is limited, or when attention must be shared across many learners. CogniVision was designed to provide an objective monitoring layer that can highlight risky attention patterns and cheating-like behavior in real time.

The project was not intended to replace human judgment. Its purpose is to provide visible risk signals so that an instructor or proctor can inspect the frame faster and more consistently.

### I.2 Datasets Used

#### Dataset 1: RGB Attention Classification

The attention classifier was trained on FER2013-derived facial data mapped to a binary task: attentive versus distracted.

| Attribute | Value |
| --- | --- |
| Dataset source | FER2013-derived facial emotion data |
| Task | Binary attention classification |
| Input type | RGB face crops |
| Input size | 224 x 224 |
| Classes | 2: attentive, distracted |
| Split | Train/validation/test recorded in the training logs |
| Recorded test accuracy | 89.05% |

The key design decision was to treat face cues as a lightweight classification problem rather than a full emotion taxonomy. That made the model more suitable for real-time monitoring.

#### Dataset 2: Track 1 Cheating Behavior Detection

The behavior detector was trained on a custom four-class YOLO dataset with 656 labeled images.

| Attribute | Value |
| --- | --- |
| Dataset size | 656 images |
| Train split | 480 images |
| Validation split | 176 images |
| Test split | Not separately preserved in the available snapshot |
| Annotation format | YOLO bounding boxes |
| Classes | leaning_to_copy, looking_around, sharing_answers, using_mobile |
| Model size | About 6.2 MB for the selected best checkpoint |

This dataset was designed for domain-specific classroom monitoring rather than general object detection. The `using_mobile` class was especially important because phone usage is a direct exam-proctoring concern.

### I.3 Preprocessing and Feature Engineering

#### RGB Attention Track

The facial attention pipeline used the following preprocessing steps:

1. Face crops were resized to 224 x 224.
2. ImageNet normalization was applied for transfer learning.
3. Data augmentation was used to improve robustness to lighting and pose changes.
4. EfficientNet-B2 pretrained weights were fine-tuned for the two-class target.

The model choice was driven by the need for a compact but accurate classifier that could run alongside a live detection pipeline.

#### Track 1 Behavior Track

The YOLO dataset was converted into standard object-detection format with normalized bounding boxes. Each annotation file stored class ID, center coordinates, width, and height. The Track 1 detector was trained with transfer learning from a COCO-pretrained YOLOv8n backbone.

## II. Methodology and Design Choices

### II.1 Model Selection

#### EfficientNet-B2 for Attention Classification

EfficientNet-B2 was selected because it provides a strong balance of accuracy, compute cost, and deployment friendliness. It performed well on the binary attentive/distracted task and reached 90.10% validation accuracy.

Key reasons for the choice:

1. It is compact enough for near real-time inference.
2. It transfers well from ImageNet pretrained weights.
3. It offers better efficiency than larger classification backbones.

#### YOLOv8n for Behavior Detection

YOLOv8n was selected for the cheating-behavior track because the project needed real-time bounding-box detection. The nano variant was preferred over larger YOLO models because it delivered an acceptable balance between speed and accuracy for classroom monitoring.

Key reasons for the choice:

1. It supports fast live inference.
2. It handles multi-object detection in a single pass.
3. It is easier to deploy than heavier detection backbones.

### II.2 Training Configuration

#### RGB Classifier Configuration

| Hyperparameter | Value |
| --- | --- |
| Backbone | EfficientNet-B2 |
| Input size | 224 x 224 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Epochs | 30 |
| Dropout | 0.3 |
| Scheduler | ReduceLROnPlateau |
| Best validation accuracy | 90.10% |
| Test accuracy | 89.05% |

The recorded precision, recall, and F1-score values were 89.2%, 88.8%, and 0.890 respectively.

#### Track 1 YOLO Configuration

| Hyperparameter | Value |
| --- | --- |
| Model | YOLOv8n |
| Input size | 960 |
| Batch size | 16 |
| Epochs | 100 |
| Patience | 15 |
| Confidence threshold | 0.25 |
| Record of best validation score | 37.3 mAP@0.5 on the preserved snapshot |

The confidence threshold was kept low because exam proctoring prioritizes recall over aggressive filtering. Missing a phone or suspicious action is more costly than keeping an extra candidate box.

### II.3 System Architecture

The final runtime pipeline uses three logical passes:

1. Detect persons with the COCO YOLO model and classify their face crops with EfficientNet-B2.
2. Detect general classroom objects such as phones with the COCO detector.
3. Detect cheating behaviors with the custom Track 1 YOLO model.

The important architectural change was removing suppression between detectors. Independent outputs are now kept separate so one detector does not erase another detector’s valid evidence.

## III. Failure Analysis

### III.1 Suppression Bug in the Multi-Detector Pipeline

The most important failure in the project came after the Track 1 detector was integrated into the live pipeline. The earlier design used three YOLO instances and applied strict IoU-based suppression across outputs. That caused phone detections to disappear even when the phone was visible in the frame.

The user-facing symptom was clear: a student box appeared, but the phone box did not. Lowering the confidence threshold from 0.4 to 0.25 did not fix the issue, which showed that the root cause was not thresholding. The real problem was the suppression logic.

### III.2 Architectural Fix

The pipeline was restructured into a dual-detector design:

1. A COCO-based detector handles persons and generic objects.
2. A Track 1 detector handles cheating behaviors.
3. The outputs are merged without cross-detector suppression.

This change preserved valid detections from both tracks and removed the false-negative failure mode that had hidden the phone box.

### III.3 Model Loading and Deployment Issues

The project also encountered model-loading and runtime issues during development. A corrupted or unavailable `yolov8m.pt` path caused fallback behavior to be needed, after which `yolov8n.pt` was used as the stable detector backbone. A local port conflict on port 8000 also appeared during dashboard launch and had to be cleared before the FastAPI app could start successfully.

These failures reinforced a practical lesson: a simpler and more predictable runtime configuration is often more reliable than a heavier one.

## IV. Results and Hyperparameter Tuning

### IV.1 RGB Attention Classifier Results

| Metric | Value |
| --- | --- |
| Best validation accuracy | 90.10% |
| Test accuracy | 89.05% |
| Precision | 89.2% |
| Recall | 88.8% |
| F1-score | 0.890 |

The classifier performed well enough for binary attention monitoring and was stable in the final inference pipeline.

### IV.2 YOLO Behavior Detector Results

| Metric | Value |
| --- | --- |
| Dataset size | 656 images |
| Train / validation split | 480 / 176 |
| Selected model | YOLOv8n best checkpoint |
| Recorded mAP@0.5 | 37.3% |
| Recorded mAP@0.5:0.95 | Not preserved in the current snapshot |

The documented performance was acceptable for a real-time monitoring use case where recall and responsiveness matter more than perfect box overlap.

### IV.3 Interpretation of the Results

The project demonstrates a useful trade-off between speed and accuracy. The RGB classifier is strong enough for classroom attention scoring, while the YOLO detector provides domain-specific behavior signals that support live proctoring.

The final architecture is intentionally modular:

1. It can be retrained track by track.
2. It can be debugged track by track.
3. It can be deployed track by track.

## V. Ethical Reflection and Limitations

### V.1 Ethical Considerations

The system should be treated as decision support, not automatic judgment. A computer vision model can identify suspicious visual patterns, but it cannot determine intent.

Important ethical points:

1. False positives can unfairly flag legitimate behavior.
2. Students with different movement patterns may be misread by the model.
3. Camera angle, lighting, and occlusion can change the output.
4. Continuous monitoring raises privacy concerns.

### V.2 Technical Limitations

1. The behavior dataset is relatively small compared with the RGB dataset.
2. Exact per-class YOLO metrics were not preserved in the current workspace snapshot.
3. The pipeline is frame-based and does not yet model long temporal context.
4. Real classroom conditions may differ from the training distribution.

## Conclusion

CogniVision combines attention classification, object detection, and behavior detection into one practical monitoring pipeline. The project’s central technical lesson is that detector outputs should not suppress one another when they serve different purposes. Once the pipeline was simplified into two independent detectors and a face classifier, the system became stable, explainable, and suitable for live demonstration.

The final result is a working classroom monitoring prototype with recorded RGB classifier performance of 90.10% validation accuracy and 89.05% test accuracy, plus a Track 1 YOLO detector trained on 656 images and documented at 37.3 mAP@0.5.
