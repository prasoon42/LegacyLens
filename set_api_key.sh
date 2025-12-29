#!/bin/bash

# Check if API key is provided
if [ -z "$1" ]; then
    echo "Usage: ./set_api_key.sh YOUR_GEMINI_API_KEY"
    echo "Get a key from: https://aistudio.google.com/app/apikey"
    exit 1
fi

NEW_KEY=$1

# Set Gemini API Key permanently in .zshrc
echo "export GEMINI_API_KEY=\"$NEW_KEY\"" >> ~/.zshrc

# Also create/update .env file in the current directory
if [ -f ".env" ]; then
    # Update existing key
    if grep -q "GEMINI_API_KEY=" .env; then
        sed -i '' "s/GEMINI_API_KEY=.*/GEMINI_API_KEY=$NEW_KEY/" .env
    else
        echo "GEMINI_API_KEY=$NEW_KEY" >> .env
    fi
else
    echo "GEMINI_API_KEY=$NEW_KEY" > .env
fi

# Also set for current session
export GEMINI_API_KEY="$NEW_KEY"

echo "✅ Gemini API key configured!"
echo "1. The key is now set permanently in your ~/.zshrc"
echo "2. A .env file has been created/updated in this directory"
echo ""
echo "To apply to current terminal, run: source ~/.zshrc"
echo "To run the app now: ./run.sh"
