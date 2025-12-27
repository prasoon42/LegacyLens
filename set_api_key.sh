#!/bin/bash

# Set Gemini API Key permanently
echo 'export GEMINI_API_KEY="AIzaSyBovou4goliOQ8d_kvrHG99lA07JSpW44M"' >> ~/.zshrc

# Also set for current session
export GEMINI_API_KEY="AIzaSyBovou4goliOQ8d_kvrHG99lA07JSpW44M"

echo "✅ Gemini API key configured!"
echo "The key is now set permanently in your ~/.zshrc"
echo ""
echo "To verify, run: echo \$GEMINI_API_KEY"
