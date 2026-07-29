"""
Set up ngrok tunnel + Twilio WhatsApp webhook.

Usage: python run_ngrok.py

Requires the FastAPI server to already be running on port 8000.
"""

import os
import time
import logging
import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ngrok")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")


def check_server():
    """Make sure the FastAPI server is running."""
    for _ in range(10):
        try:
            r = httpx.get("http://127.0.0.1:8000/health", timeout=3)
            if r.status_code == 200:
                log.info("Server is running ✅")
                return True
        except Exception:
            time.sleep(1)
    log.error("Server is not running on port 8000. Start it first: cd backend && uvicorn main:app --reload")
    return False


def start_ngrok():
    """Start ngrok tunnel and return the public URL."""
    from pyngrok import ngrok, conf

    # Get ngrok binary path from pyngrok config
    config = conf.get_default()
    installed_path = config.ngrok_path
    log.info(f"ngrok binary path: {installed_path}")

    # Install if missing
    if not os.path.exists(str(installed_path)):
        log.info("Downloading ngrok binary …")
        from pyngrok.installer import install_ngrok
        install_ngrok(str(installed_path))

    # Set auth token if provided
    if NGROK_AUTH_TOKEN:
        log.info("Configuring ngrok auth token …")
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    else:
        log.warning("No NGROK_AUTH_TOKEN set. Free ngrok may show a warning page instead of tunneling.")

    public_url = ngrok.connect(8000, bind_tls=True).public_url
    return public_url.rstrip("/")


def set_twilio_webhook(webhook_url: str):
    """Set the Twilio WhatsApp Sandbox incoming message webhook."""
    if not TWILIO_ACCOUNT_SID or "your_twilio" in TWILIO_ACCOUNT_SID:
        log.warning("Twilio credentials are placeholders — skipping webhook config.")
        log.info(f"\n👉  Manually set your webhook in Twilio Console:")
        log.info(f"    {webhook_url}")
        return False

    log.info(f"Updating Twilio WhatsApp Sandbox webhook → {webhook_url}")

    base_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}"
    sandbox_url = f"{base_url}/SMS/WhatsApp/Sandbox.json"

    resp = httpx.post(
        sandbox_url,
        auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
        data={"SmsUrl": webhook_url, "SmsMethod": "POST"},
    )

    if resp.status_code in (200, 201, 204):
        log.info("Twilio WhatsApp Sandbox webhook updated successfully ✅")
        return True
    else:
        log.warning(f"Twilio API returned {resp.status_code}: {resp.text}")
        log.info("\n⚠️  Auto-config failed. Please set the webhook manually:")
        log.info(f"   1. Go to: https://console.twilio.com")
        log.info(f"   2. Messaging → Try it Out → WhatsApp Sandbox")
        log.info(f"   3. Set 'When a message comes in' to: {webhook_url}")
        log.info(f"   4. Method: HTTP POST")
        return False


if __name__ == "__main__":
    if not check_server():
        exit(1)

    webhook_url = None
    try:
        public_url = start_ngrok()
        webhook_url = f"{public_url}/api/citizen/whatsapp"
        log.info(f"\n🌍  Public URL: {public_url}")
        log.info(f"📱  Webhook URL: {webhook_url}")

        set_twilio_webhook(webhook_url)

        log.info("\n✨  WhatsApp bot is live!")
        log.info(f"    Message your Twilio sandbox WhatsApp number to test it.")
        log.info(f"\n    📋  Quick test:")
        log.info(f"    curl -X POST {webhook_url} \\")
        log.info(f'      -d "Body=Hello" -d "From=whatsapp:+919999999999"')
        log.info(f"\n    Press Ctrl+C to stop the tunnel.")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        log.info("Shutting down …")
    except Exception as e:
        log.error(f"Error: {e}")
    finally:
        try:
            from pyngrok import ngrok
            ngrok.kill()
        except Exception:
            pass
