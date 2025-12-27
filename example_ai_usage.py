"""
Example: How to use the AI Interpreter
"""
import os
from src.ai_interpreter import ai_interpreter

# Set your OpenAI API key (get it from https://platform.openai.com/api-keys)
os.environ['OPENAI_API_KEY'] = 'sk-proj-your-api-key-here'

# Example 1: Register a device
ai_interpreter.register_device(
    device_id='ventilator_room_301',
    device_type='medical ventilator',
    normal_range={'O2': (95, 100), 'pressure': (10, 20)}
)

# Example 2: Interpret a reading
result = ai_interpreter.interpret_reading(
    device_id='ventilator_room_301',
    reading='95',
    device_type='medical ventilator'
)

print("Interpretation:", result)
# Output: {
#   'status': 'normal',
#   'message': 'O2 saturation at 95% is within safe range',
#   'action': 'Continue monitoring',
#   'raw_reading': '95'
# }

# Example 3: Ask a question
answer = ai_interpreter.ask_question(
    device_id='ventilator_room_301',
    question='What was the lowest reading today?'
)

print("Answer:", answer)
