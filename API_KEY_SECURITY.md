# API Key Security Notice

⚠️ **IMPORTANT**: You accidentally shared an OpenAI API key in the chat.

## Immediate Actions Required

### 1. Revoke the Exposed Key
1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Find the key starting with `sk-proj-ibFaqav...`
3. Click "Revoke" or "Delete"

### 2. For LegacyLens (Gemini)
This project uses **Google Gemini**, not OpenAI.

**Get Gemini API Key**:
1. Visit [Google AI Studio](https://aistudio.google.com/apikey)
2. Click "Create API Key"
3. Copy the key (starts with `AI...`)

**Set it securely**:
```bash
# In terminal (not in code/chat)
export GEMINI_API_KEY="AIza..."
```

### 3. Optional: Use OpenAI Instead
If you prefer OpenAI over Gemini, I can modify the code to support it. Let me know!

---

## Best Practices
- ❌ Never share API keys in chat, code, or public repos
- ✅ Use environment variables
- ✅ Add `.env` to `.gitignore`
- ✅ Rotate keys regularly
