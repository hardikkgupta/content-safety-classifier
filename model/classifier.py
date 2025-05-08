import os
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import logging

logger = logging.getLogger(__name__)

class ContentSafetyClassifier:
    def __init__(self):
        self.model_name = "distilbert-base-uncased"
        self.max_length = 512
        self.categories = [
            "hate_speech",
            "harassment",
            "violence",
            "sexual_content",
            "self_harm",
            "misinformation"
        ]
        
        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = TFAutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.categories)
            )
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_text(self, text):
        """Preprocess the input text."""
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
    
    def classify(self, text):
        """
        Classify the input text and return predictions for each category.
        
        Args:
            text (str): Input text to classify
            
        Returns:
            dict: Dictionary containing predictions for each category
        """
        try:
            # Preprocess text
            inputs = self.preprocess_text(text)
            
            # Get model predictions
            outputs = self.model(inputs)
            predictions = tf.nn.sigmoid(outputs.logits)
            
            # Convert predictions to dictionary
            results = {
                category: float(pred)
                for category, pred in zip(self.categories, predictions[0])
            }
            
            # Add metadata
            results['text'] = text
            results['timestamp'] = tf.timestamp().numpy()
            
            return results
            
        except Exception as e:
            logger.error(f"Error during classification: {str(e)}")
            raise 