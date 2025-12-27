# Getting Your Gemini API Key

## Step 1: Get API Key
1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Click "Create API Key"
3. Copy the key

## Step 2: Set Environment Variable

### On Mac/Linux:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or add to your `~/.zshrc` or `~/.bashrc`:
```bash
echo 'export GEMINI_API_KEY="your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### On Windows:
```cmd
set GEMINI_API_KEY=your-api-key-here
```

## Step 3: Verify
```bash
echo $GEMINI_API_KEY
```

You should see your API key printed.

## Step 4: Run the App
```bash
./run.sh
```

The AI features will now be enabled!

## Testing AI Features

### Via API:
```bash
# Interpret a reading
curl -X POST http://localhost:5000/interpret \
  -H "Content-Type: application/json" \
  -d '{"reading": "95", "device_type": "ventilator", "device_id": "room_301"}'

# Ask a question
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the highest reading?", "device_id": "room_301"}'
```

### Via Python:
```python
import requests

# Interpret
response = requests.post('http://localhost:5000/interpret', json={
    'reading': '95',
    'device_type': 'ventilator',
    'device_id': 'room_301'
})
print(response.json())

# Ask
response = requests.post('http://localhost:5000/ask', json={
    'question': 'What was the highest reading?',
    'device_id': 'room_301'
})
print(response.json())
```
