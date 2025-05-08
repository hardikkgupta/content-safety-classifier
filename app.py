import os
import json
import redis
import logging
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram
import time
from model.classifier import ContentSafetyClassifier

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Redis client
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=0
)

# Initialize model
classifier = ContentSafetyClassifier()

# Prometheus metrics
REQUEST_COUNT = Counter('request_count', 'Total requests received')
CLASSIFICATION_TIME = Histogram('classification_time_seconds', 'Time spent classifying text')
CACHE_HITS = Counter('cache_hits', 'Number of cache hits')
CACHE_MISSES = Counter('cache_misses', 'Number of cache misses')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/classify', methods=['POST'])
def classify_text():
    REQUEST_COUNT.inc()
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        # Check cache
        cache_key = f"text:{hash(text)}"
        cached_result = redis_client.get(cache_key)
        
        if cached_result:
            CACHE_HITS.inc()
            return jsonify(json.loads(cached_result))
        
        CACHE_MISSES.inc()
        
        # Classify text
        start_time = time.time()
        result = classifier.classify(text)
        CLASSIFICATION_TIME.observe(time.time() - start_time)
        
        # Cache result (expire after 1 hour)
        redis_client.setex(cache_key, 3600, json.dumps(result))
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 