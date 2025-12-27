"""
GenAI Interpreter Module - Gemini Version
Interprets OCR readings using Google Gemini API
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import google.generativeai as genai

# Configure API key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print(f"✅ AI Interpreter initialized")
else:
    print("❌ No GEMINI_API_KEY found! Please run: source ~/.zshrc")

# Initialize model
# Using gemini-flash-latest as it is explicitly listed in available models
model = genai.GenerativeModel('gemini-flash-latest')

class DeviceContext:
    """Stores context about a monitored device"""
    def __init__(self, device_id: str, device_type: str = "unknown"):
        self.device_id = device_id
        self.device_type = device_type
        self.readings_history: List[Dict] = []
        self.normal_range: Optional[Dict] = None
        
    def add_reading(self, value: str, timestamp: Optional[str] = None):
        """Add a reading to history"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        self.readings_history.append({
            'value': value,
            'timestamp': timestamp
        })
        
        # Keep only last 50 readings
        if len(self.readings_history) > 50:
            self.readings_history = self.readings_history[-50:]
    
    def get_recent_readings(self, count: int = 10) -> List[str]:
        """Get recent reading values"""
        return [r['value'] for r in self.readings_history[-count:]]

class LegacyLensAI:
    """Main AI interpreter for legacy device readings"""
    
    def __init__(self):
        self.devices: Dict[str, DeviceContext] = {}
        self.enabled = bool(GEMINI_API_KEY)
        
    def register_device(self, device_id: str, device_type: str = "unknown", 
                       normal_range: Optional[Dict] = None):
        """Register a new device for monitoring"""
        device = DeviceContext(device_id, device_type)
        device.normal_range = normal_range
        self.devices[device_id] = device
        return device
    
    def interpret_reading(self, device_id: str, reading: str, 
                         device_type: str = "7-segment display") -> Dict:
        """
        Interpret a reading using Gemini
        
        Args:
            device_id: Unique identifier for the device
            reading: The OCR-extracted value
            device_type: Type of device (for context)
            
        Returns:
            Dict with status, message, and recommended action
        """
        if not self.enabled:
            return {
                'status': 'info',
                'message': f'Reading: {reading}',
                'action': 'Set GEMINI_API_KEY to enable AI interpretation',
                'raw_reading': reading
            }
        
        # Get or create device context
        if device_id not in self.devices:
            self.register_device(device_id, device_type)
        
        device = self.devices[device_id]
        device.add_reading(reading)
        
        # Build prompt with context
        prompt = self._build_prompt(device, reading)
        
        try:
            # Call Gemini API
            response = model.generate_content(prompt)
            
            # Parse response
            result = self._parse_response(response.text, reading)
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'AI interpretation failed: {str(e)}',
                'action': 'Check API key and connection',
                'raw_reading': reading
            }
    
    def _build_prompt(self, device: DeviceContext, current_reading: str) -> str:
        """Build context-aware prompt for Gemini"""
        
        recent = device.get_recent_readings(10)
        history_str = ', '.join(recent) if recent else 'No history'
        
        prompt = f"""You are an AI assistant monitoring a {device.device_type}.

Current Reading: {current_reading}
Recent History (last 10): {history_str}

Analyze this reading and provide:
1. Status: normal, warning, or critical
2. Brief explanation (1-2 sentences max)
3. Recommended action (if any, otherwise "None")

Respond in JSON format:
{{
  "status": "normal|warning|critical",
  "message": "brief explanation",
  "action": "recommended action or None"
}}

Be concise and practical. Focus on actionable insights."""

        if device.normal_range:
            prompt += f"\n\nNormal Range: {device.normal_range}"
        
        return prompt
    
    def _parse_response(self, response_text: str, raw_reading: str) -> Dict:
        """Parse Gemini response into structured format"""
        try:
            # Try to extract JSON from response
            if '```json' in response_text:
                json_str = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                json_str = response_text.split('```')[1].split('```')[0].strip()
            else:
                json_str = response_text.strip()
            
            result = json.loads(json_str)
            result['raw_reading'] = raw_reading
            return result
            
        except (json.JSONDecodeError, IndexError):
            # Fallback if JSON parsing fails
            return {
                'status': 'info',
                'message': response_text[:200],
                'action': 'None',
                'raw_reading': raw_reading
            }
    
    def ask_question(self, device_id: str, question: str) -> str:
        """
        Ask a conversational question about a device
        
        Args:
            device_id: Device to ask about
            question: Natural language question
            
        Returns:
            AI-generated answer
        """
        if not self.enabled:
            return "AI assistant is disabled. Set GEMINI_API_KEY to enable."
        
        if device_id not in self.devices:
            return f"Device '{device_id}' not found. Register it first."
        
        device = self.devices[device_id]
        recent = device.get_recent_readings(20)
        
        prompt = f"""You are monitoring a {device.device_type} (ID: {device_id}).

Recent readings: {', '.join(recent)}

User question: {question}

Provide a helpful, concise answer based on the available data."""

        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

# Global instance
ai_interpreter = LegacyLensAI()
