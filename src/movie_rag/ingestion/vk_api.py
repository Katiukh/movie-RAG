import os

import requests
from dotenv import load_dotenv

load_dotenv() 
VK_TOKEN = os.environ["VK_ACCESS_TOKEN"]

VK_API_URL = "https://api.vk.com/method/wall.get"
VK_API_VERSION = "5.199"

def fetch_posts(domain: str, count: int = 100) -> list[dict]:
    response = requests.get(
        VK_API_URL,
        params = {
            "domain": domain,
            "count": count,
            "access_token": VK_TOKEN,
            "v": VK_API_VERSION
        },
        timeout=30
    )

    data = response.json()

    if "error" in data:
        raise RuntimeError(data["error"])

    return data["response"]["items"]

if __name__ == "__main__":
    posts = fetch_posts("cinemafromivan", count=10)

    print(f"Получено постов: {len(posts)}")

    for post in posts[:3]:
        print("=" * 80)
        print(post["text"][:1000])