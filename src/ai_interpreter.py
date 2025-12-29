"""
GenAI Interpreter Module - Gemini Version
Interprets OCR readings using Google Gemini API
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure API key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print(f"✅ AI Interpreter initialized")
else:
    print("❌ No GEMINI_API_KEY found!")
    print("👉 Please run: ./set_api_key.sh YOUR_API_KEY")
    print("   Or create a .env file with GEMINI_API_KEY=your_key")

# Initialize model
# Using gemini-flash-latest as it is confirmed available for this key
model = genai.GenerativeModel('gemini-flash-latest')

class DeviceContext:
    """Stores context about a monitored device"""
    def __init__(self, device_id: str, device_type: str = "unknown"):
        self.device_id = device_id
        self.device_type = device_type
        self.readings_history: List[Dict] = []
        self.config: Dict = {
            'min_val': None,
            'max_val': None,
            'unit': '',
            'name': device_id
        }
        
    def update_config(self, config: Dict):
        """Update device configuration"""
        self.config.update(config)
        if 'name' in config:
            self.device_type = config['name']  # Update type/name for context
        
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
    
    def update_device_config(self, device_id: str, config: Dict):
        """Update configuration for a specific device"""
        if device_id not in self.devices:
            self.register_device(device_id)
        
        self.devices[device_id].update_config(config)
        return True
    
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
                'action': 'Run ./set_api_key.sh to enable AI interpretation',
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
            
            # STRICT OVERRIDE: If we have explicit thresholds, enforce them
            try:
                import re
                val = None
                
                # 1. Try to use the AI's identified primary reading
                primary_val_str = result.get('primary_reading')
                if primary_val_str:
                    num_match = re.search(r"[-+]?\d*\.\d+|\d+", str(primary_val_str))
                    if num_match:
                        val = float(num_match.group())
                
                # 2. Fallback: Extract the first numeric-looking value from the raw reading string
                if val is None:
                    numeric_matches = re.findall(r"[-+]?\d*\.\d+|\d+", reading)
                    if numeric_matches:
                        val = float(numeric_matches[0])
                
                if val is not None:
                    min_val = device.config.get('min_val')
                    max_val = device.config.get('max_val')
                    
                    if min_val is not None and val < float(min_val):
                        result['status'] = 'critical'
                        result['message'] = f"Reading {val} is below minimum threshold of {min_val}"
                        result['action'] = "Check system immediately"
                    elif max_val is not None and val > float(max_val):
                        result['status'] = 'critical'
                        result['message'] = f"Reading {val} is above maximum threshold of {max_val}"
                        result['action'] = "Check system immediately"
            except (ValueError, TypeError):
                pass  # Reading wasn't a number or thresholds weren't set
                
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
        
        prompt = f"""You are an expert industrial systems analyst monitoring a {device.config.get('name', device.device_type)}.

Current Detected Items (with spatial context): {current_reading}
Recent History (last 10): {history_str}

Configuration Context:
- Min Safe Value: {device.config.get('min_val', 'Not set')}
- Max Safe Value: {device.config.get('max_val', 'Not set')}
- Unit: {device.config.get('unit', 'Not set')}

Your task is to provide a professional, high-confidence analysis of this equipment.
The 'Current Detected Items' is a list of text found on the display, including their 'center' (x, y from 0 to 1), 'area' (relative size), and any 'alternatives' (other possible readings for the same area).

INTELLIGENT FOCUS & ARBITRATION:
1. Identify the 'primary_reading'. This is usually the item with the LARGEST 'area' and located near the CENTER (0.5, 0.5).
2. 7-SEGMENT CONFUSION RULES: 7-segment displays often confuse '2', '3', and '5'. 
   - If the OCR says '135' but '120' is an alternative or fits the history/context better, it is likely '120'.
   - Look at the 'Recent History' to see if the value has been stable around a certain number.
3. If an item has 'alternatives', check if any of the alternatives make more technical sense for this device than the main 'text'.
4. Use smaller text items as 'labels' or 'units' (like PSI, BAR, TEMP, RPM, V, A) to provide context.

Analyze if the reading is within safe operating limits, identify any trends, and suggest technical actions.
If units are detected in the text, use them to provide a more specific analysis.

Respond ONLY in valid JSON format:
{{
  "identified_device": "The type of device identified (e.g., Pressure Gauge, Ventilator, etc.)",
  "primary_reading": "The specific numeric value you identified as the main instrument reading",
  "status": "normal|warning|critical",
  "message": "A professional technical explanation of the current state, mentioning the specific units if found.",
  "action": "Specific technical recommendation or 'Continue monitoring'."
}}

Ensure the JSON is perfectly formatted and contains no other text."""

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
            return "AI assistant is disabled. Run ./set_api_key.sh YOUR_API_KEY to enable."
        
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
