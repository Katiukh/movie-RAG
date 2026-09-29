from playwright.sync_api import Page, sync_playwright

# Shared by extraction and progress detection: comments are not posts.
_CANONICAL_URL_JS = r"""
const canonicalPostUrl = href => {
    try {
        const url = new URL(href, location.href);
        if (!/^(www\.|m\.)?vk\.(ru|com)$/.test(url.hostname) ||
            !/^\/wall-\d+_\d+$/.test(url.pathname) ||
            url.searchParams.has("reply") ||
            url.searchParams.has("thread")) return null;
        return "https://vk.ru" + url.pathname;
    } catch { return null; }
};
"""


URL = "https://vk.ru/cinemafromivan"


def extract_visible_posts(page: Page) -> list[dict]:
    """Достаёт movie-посты, которые сейчас присутствуют в DOM."""

    return page.evaluate(
        "() => {"
        + _CANONICAL_URL_JS
        + r"""
            const result = [];
            const seen = new Set();
            for (const link of document.querySelectorAll("a[href]")) {
                const postUrl = canonicalPostUrl(link.href);
                if (!postUrl || seen.has(postUrl)) continue;
                let element = link;
                while (element && element !== document.body) {
                    const wallUrls = new Set(
                        [...element.querySelectorAll("a[href]")]
                            .map(a => canonicalPostUrl(a.href)).filter(Boolean)
                    );
                    // Never climb into the whole feed or a different post.
                    if (wallUrls.size > 1) break;
                    const text = (element.innerText || "").trim();
                    if (text.includes("ЧТО:") &&
                        (text.includes("О ЧЁМ:") || text.includes("О ЧЕМ:")) &&
                        text.includes("ЗАЧЕМ:")) {
                        // A thumbnail may link to the same post before its
                        // displayed date. Prefer a text-bearing permalink.
                        const dateLink = [...element.querySelectorAll("a[href]")]
                            .find(a => canonicalPostUrl(a.href) === postUrl &&
                                (a.innerText || "").trim()) || link;
                        result.push({url: postUrl,
                            date: (dateLink.innerText || "").trim(), text});
                        seen.add(postUrl);
                        break;
                    }
                    // VK's explicit post identity is a safer boundary than
                    // obfuscated classes, including when a card has no text.
                    if (element.getAttribute("data-testid") === "post" ||
                        element.tagName === "ARTICLE") break;
                    element = element.parentElement;
                }
            }
            return result;
        }
        """
    )


def diagnose_dom(page: Page) -> dict:
    """Snapshot of real permalink progress, independent of virtual spacers."""
    return page.evaluate(
        "() => {"
        + _CANONICAL_URL_JS
        + """
            const root = document.scrollingElement;
            return {
                wall_urls: [...new Set([...document.querySelectorAll('a[href]')]
                    .map(a => canonicalPostUrl(a.href)).filter(Boolean))],
                scroll_y: root.scrollTop,
                scroll_height: root.scrollHeight,
                viewport_height: root.clientHeight,
            };
        }
        """
    )


def fetch_posts(
    url: str,
    target_movies: int = 500,
    max_scrolls: int = 1000,
    scroll_pause_ms: int = 1200,
    *,
    debug: bool = False,
) -> list[dict]:
    """Collect incrementally from VK's virtualized public feed.

    max_scrolls bounds the run even if VK keeps growing an empty spacer.
    debug adds DOM counts, viewport position and the last permalink URLs.
    Lack of new movies alone is not an end condition: ordinary posts count
    as feed progress too. A stopped feed may also mean a network/login limit;
    a browser cannot prove that the community's entire history was returned.
    """
    if target_movies < 0 or max_scrolls < 0 or scroll_pause_ms < 0:
        raise ValueError("target_movies, max_scrolls and scroll_pause_ms must be >= 0")
    if target_movies == 0:
        return []

    posts_by_url: dict[str, dict] = {}
    seen_wall_urls: set[str] = set()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            # networkidle is unreliable with VK's background requests. Wait
            # for an actual post instead of reading the initial skeletons.
            page.wait_for_function(
                "() => {"
                + _CANONICAL_URL_JS
                + """
                    return [...document.querySelectorAll('a[href]')]
                        .some(a => canonicalPostUrl(a.href));
                }""",
                timeout=60_000,
            )
            idle_scrolls = 0
            bottom_retries = 0
            for step in range(max_scrolls + 1):
                # The public page offers a dismissible sign-in invitation.
                close = page.get_by_role("button", name="Закрыть", exact=True)
                if close.count() and close.first.is_visible():
                    close.first.click(timeout=2000)

                visible_posts = extract_visible_posts(page)
                before = len(posts_by_url)
                for post in visible_posts:
                    posts_by_url[post["url"]] = post
                    if len(posts_by_url) >= target_movies:
                        break
                state = diagnose_dom(page)
                new_wall_urls = set(state["wall_urls"]) - seen_wall_urls
                seen_wall_urls.update(state["wall_urls"])
                new_movies = len(posts_by_url) - before
                idle_scrolls = 0 if new_wall_urls or new_movies else idle_scrolls + 1
                at_bottom = (
                    state["scroll_y"] + state["viewport_height"]
                    >= state["scroll_height"] - 2
                )
                bottom_retries = (
                    bottom_retries + 1
                    if at_bottom and not new_wall_urls and not new_movies
                    else 0
                )
                label = "Initial" if step == 0 else f"Scroll {step}/{max_scrolls}"
                print(
                    f"{label}: +{new_movies} new, total={len(posts_by_url)}, "
                    f"new wall URLs={len(new_wall_urls)}",
                    flush=True,
                )
                if debug:
                    print(
                        f"  scrollY={state['scroll_y']}, "
                        f"scrollHeight={state['scroll_height']}, "
                        f"unique wall links={len(state['wall_urls'])}, "
                        f"movie posts in DOM={len(visible_posts)}, "
                        f"last URLs={state['wall_urls'][-3:]}",
                        flush=True,
                    )
                if len(posts_by_url) >= target_movies:
                    print(f"Reached target_movies={target_movies}", flush=True)
                    break
                if bottom_retries >= 6 or idle_scrolls >= 30:
                    print(
                        "Stopped: no new real post URLs after repeated waits "
                        "(end of available feed or loading stalled).",
                        flush=True,
                    )
                    break
                if step == max_scrolls:
                    print(f"Stopped: max_scrolls={max_scrolls}", flush=True)
                    break

                # VK uses virtual article positions and estimated blank space.
                # Jumping to scrollHeight skips the rendering window and can
                # grow the spacer forever without mounting any new posts.
                page.mouse.move(600, 450)
                page.mouse.wheel(0, min(600, state["viewport_height"] * 0.8))
                page.wait_for_timeout(scroll_pause_ms)
                if at_bottom:
                    # Give a slow final batch time to arrive before retrying.
                    page.wait_for_timeout(max(0, 1000 - scroll_pause_ms))
        finally:
            browser.close()

    return list(posts_by_url.values())


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(description="Collect movie posts from public VK DOM")
    cli.add_argument("--target-movies", type=int, default=500)
    cli.add_argument("--max-scrolls", type=int, default=1000)
    cli.add_argument("--scroll-pause-ms", type=int, default=1200)
    cli.add_argument("--debug", action="store_true")
    args = cli.parse_args()
    posts = fetch_posts(
        URL,
        target_movies=args.target_movies,
        max_scrolls=args.max_scrolls,
        scroll_pause_ms=args.scroll_pause_ms,
        debug=args.debug,
    )
    print(f"\nВсего найдено movie posts: {len(posts)}")
    for post in posts[:5]:
        print("=" * 80)
        print("URL:", post["url"])
        print("DATE:", post["date"])
        print(post["text"][:1000])
