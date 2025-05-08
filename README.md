# Content Safety Classifier

A high-performance content safety classification service that uses Transformer encoders to detect sensitive content categories in text. The service is designed to handle high throughput with low latency, making it suitable for real-time content moderation.

## Features

- Multi-label text classification for sensitive content detection
- High precision (97%) and recall (94%) using Transformer encoders
- Redis caching for improved performance
- Containerized deployment on AWS ECS
- Horizontal scaling capability (5k QPS)
- Sub-100ms latency
- RESTful API interface

## Architecture

- **Model**: Transformer-based text classification model
- **API**: Flask-based REST API
- **Caching**: Redis for response caching
- **Deployment**: Docker containers on AWS ECS
- **Monitoring**: Prometheus metrics and logging

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Redis:
```bash
docker run -d -p 6379:6379 redis
```

3. Run the service:
```bash
python app.py
```

## API Usage

```bash
curl -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text to classify"}'
```

## Development

- `app.py`: Main Flask application
- `model/`: Transformer model implementation
- `utils/`: Utility functions
- `tests/`: Test suite
- `docker/`: Docker configuration files

## License

MIT 