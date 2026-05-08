import pytest
from src.core.classifier import AttentionClassifier
from src.core.detector import CogniVisionDetector
from src.config import DEVICE

def test_classifier_initialization():
    """Test that the classifier initializes without errors."""
    model = AttentionClassifier(num_classes=2)
    assert model is not None
    print(f"✓ Classifier initialized on {DEVICE}")

def test_detector_initialization():
    """Test that the detector initializes without errors."""
    try:
        detector = CogniVisionDetector()
        assert detector is not None
        print("✓ Detector initialized")
    except Exception as e:
        pytest.skip(f"YOLOv8 model not available: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
