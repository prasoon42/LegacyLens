# LegacyLens - 7-Segment OCR App

LegacyLens is a web application designed to read numbers from 7-segment digital displays (like multimeters, digital clocks, and industrial meters) **and interpret them using AI**.

## 🚀 New: GenAI Integration
LegacyLens now includes **OpenAI GPT-4** to:
*   📊 **Interpret readings** with context and history
*   ⚠️ **Detect anomalies** and provide alerts
*   💬 **Answer questions** about your devices
*   🤖 **Act as an AI assistant** for legacy equipment

## 📸 New: Live Camera
You can now use your webcam or phone camera directly!
*   Click **"Live Camera"** in the app
*   Point at the device
*   Click **"Capture & Analyze"**

See [OPENAI_SETUP.md](OPENAI_SETUP.md) for setup instructions.

## Supported Images
This app is optimized for:
*   **Red LED Displays**: (Best Performance) Digital numbers glowing red on a black background.
*   **LCD Displays**: Black numbers on a grey/green background.
*   **VFD / OLED**: Light numbers on a dark background.

### Tips for Best Results
*   **Contrast**: Ensure the digits stand out clearly from the background.
*   **Lighting**: Avoid heavy glare or reflections on the screen.
*   **Orientation**: Keep the text roughly horizontal.
*   **Focus**: Ensure the numbers are in focus.

## How to Run
1.  **Setup**: Run `./setup.sh` (First time only)
2.  **Start**: Run `./run.sh`
3.  **Open**: Go to [http://localhost:5000](http://localhost:5000)

## Troubleshooting
If detection fails:
*   Try moving closer to the display.
*   Reduce glare.
*   Ensure the image isn't blurry.
