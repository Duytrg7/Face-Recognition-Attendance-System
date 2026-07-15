import json
import os
import urllib.parse
import urllib.request
from datetime import datetime


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "combros_config.json")


def load_combros_config():
    if not os.path.exists(CONFIG_PATH):
        return {
            "enabled": False,
            "error": "Không tìm thấy combros_config.json"
        }

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)

        return config

    except Exception as e:
        return {
            "enabled": False,
            "error": str(e)
        }


def send_combros_fields(
    field1=None,
    field2=None,
    field3=None,
    field4=None,
    field5=None,
    field6=None,
    field7=None,
    field8=None,
    timeout=3
):
    config = load_combros_config()

    if not config.get("enabled", False):
        return False, "Combros disabled or config missing"

    server = config.get("server", "").rstrip("/")
    write_token = config.get("write_token", "")
    device_key = config.get("device_key", "")

    if not server or not write_token or not device_key:
        return False, "Thiếu server/write_token/device_key"

    params = {
        "write_token": write_token,
        "device": device_key,
    }

    fields = {
        "field1": field1,
        "field2": field2,
        "field3": field3,
        "field4": field4,
        "field5": field5,
        "field6": field6,
        "field7": field7,
        "field8": field8,
    }

    for key, value in fields.items():
        if value is not None:
            params[key] = value

    url = f"{server}/channels/update?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return True, body

    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    ok, response = send_combros_fields(
        field1=1,
        field2=1,
        field3=0,
        field4=1,
        field5=7.5,
        field6=1,
        field7=1,
        field8=1
    )

    print("OK:", ok)
    print("Response:", response)
    print("Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))