from pathlib import Path

import requests


URL = "https://vk.ru/cinemafromivan"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Safari/537.36"
    )
}


def fetch_page(url: str) -> requests.Response:
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    return response


if __name__ == "__main__":
    response = fetch_page(URL)

    print("Status:", response.status_code)
    print("HTML length:", len(response.text))

    print("'ЧТО:' найдено:", "ЧТО:" in response.text)
    print("'ЗАЧЕМ:' найдено:", "ЗАЧЕМ:" in response.text)

    Path("data/raw").mkdir(parents=True, exist_ok=True)

    Path("data/raw/vk_page.html").write_text(
        response.text,
        encoding="utf-8",
    )