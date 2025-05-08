import pytest
from model.classifier import ContentSafetyClassifier

@pytest.fixture
def classifier():
    return ContentSafetyClassifier()

def test_classifier_initialization(classifier):
    assert classifier.model_name == "distilbert-base-uncased"
    assert len(classifier.categories) > 0
    assert classifier.max_length == 512

def test_preprocess_text(classifier):
    text = "This is a test text"
    processed = classifier.preprocess_text(text)
    assert 'input_ids' in processed
    assert 'attention_mask' in processed

def test_classify_output_format(classifier):
    text = "This is a test text"
    result = classifier.classify(text)
    
    # Check if all categories are present
    for category in classifier.categories:
        assert category in result
    
    # Check if metadata is present
    assert 'text' in result
    assert 'timestamp' in result
    
    # Check if predictions are between 0 and 1
    for category in classifier.categories:
        assert 0 <= result[category] <= 1 