# API Documentation

## POST /predict

Classifies a news text.

### Request

```json
{
  "text": "Your news article text..."
}
```

### Response

```json
{
  "prediction": "FAKE",
  "confidence": 91.23
}
```

## GET /history

Returns up to 20 most recent predictions.

## GET /health

Returns:

```json
{
  "status": "ok",
  "model_loaded": true
}
```
