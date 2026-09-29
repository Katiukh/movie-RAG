"""Browser regression tests; no VK network requests are needed."""

import pytest
from playwright.sync_api import sync_playwright

from movie_rag.ingestion.vk_browser import extract_visible_posts, fetch_posts


@pytest.fixture
def page():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        yield browser.new_page()
        browser.close()


def test_comment_link_cannot_supply_post_date(page):
    page.set_content("""<article>
      <a href="https://vk.ru/wall-1_2?reply=3">comment date</a>
      <p>Film (2000) ЧТО: drama О ЧЁМ: life ЗАЧЕМ: art</p>
      <a href="https://vk.com/wall-1_2">28 сентября</a>
    </article>""")
    assert extract_visible_posts(page) == [
        {
            "url": "https://vk.ru/wall-1_2",
            "date": "28 сентября",
            "text": "comment date\n\nFilm (2000) ЧТО: drama О ЧЁМ: life ЗАЧЕМ: art\n\n28 сентября",
        }
    ]


def test_does_not_merge_feed_or_extract_comment_only_post(page):
    page.set_content("""<main>
      <article><a href="https://vk.ru/wall-1_1">today</a>ЧТО: drama</article>
      <article><a href="https://vk.ru/wall-1_2">yesterday</a>О ЧЁМ: life ЗАЧЕМ: art</article>
      <article><a href="https://vk.ru/wall-1_3?reply=4">comment</a>
      ЧТО: drama О ЧЁМ: life ЗАЧЕМ: art</article>
    </main>""")
    assert extract_visible_posts(page) == []


# A virtual feed: jumping below the rendered range extends only its spacer.
# Traversing it replaces old cards, including a stretch of non-movie posts.
FEED = """<body style="margin:0"><main style="height:15000px"></main>
<script>
const feed = document.querySelector('main');
let highest = 0;
function render() {
 const index = Math.floor(scrollY / 600);
 if (index > highest + 2) { feed.style.height = (scrollY + 15000) + 'px'; return; }
 highest = Math.max(highest, index);
 const start = Math.max(0, index - 1);
 feed.innerHTML = '';
 for (let i = start; i < Math.min(18, index + 3); i++) {
  const a = document.createElement('article');
  a.style = `position:absolute;top:${i*600}px;height:600px`;
  a.innerHTML = `<a href="https://vk.ru/wall-1_${i+1}">date ${i}</a><p>` +
    ((i>=3 && i<=12) ? 'Community news' : `Movie ${i} (2000) ЧТО: drama О ЧЁМ: life ЗАЧЕМ: art`) + '</p>';
  feed.append(a);
 }
 if(index >= 15) feed.style.height = '10800px';
}
addEventListener('scroll', render); render();
</script>"""


def test_collects_virtualized_feed_and_stops_at_target(tmp_path):
    fixture = tmp_path / "feed.html"
    fixture.write_text(FEED)
    posts = fetch_posts(
        fixture.as_uri(), target_movies=7, max_scrolls=60, scroll_pause_ms=50
    )
    assert [p["url"] for p in posts] == [
        "https://vk.ru/wall-1_1",
        "https://vk.ru/wall-1_2",
        "https://vk.ru/wall-1_3",
        "https://vk.ru/wall-1_14",
        "https://vk.ru/wall-1_15",
        "https://vk.ru/wall-1_16",
        "https://vk.ru/wall-1_17",
    ]


def test_stops_at_end_without_reaching_scroll_limit(tmp_path, capsys):
    fixture = tmp_path / "feed.html"
    fixture.write_text(FEED)
    posts = fetch_posts(
        fixture.as_uri(), target_movies=50, max_scrolls=60, scroll_pause_ms=50
    )
    assert len(posts) == 8
    assert "Scroll 60/" not in capsys.readouterr().out


def test_initial_batch_respects_target(tmp_path):
    fixture = tmp_path / "feed.html"
    fixture.write_text(FEED)
    posts = fetch_posts(fixture.as_uri(), target_movies=1, max_scrolls=0)
    assert len(posts) == 1
    assert posts[0]["url"] == "https://vk.ru/wall-1_1"


def test_growing_spacer_does_not_count_as_new_posts(tmp_path, capsys):
    fixture = tmp_path / "stalled.html"
    fixture.write_text("""<main style="height:15000px"><article>
        <a href="https://vk.ru/wall-1_1">today</a>
        ЧТО: drama О ЧЁМ: life ЗАЧЕМ: art</article></main>
        <script>addEventListener('scroll', () => {
          document.querySelector('main').style.height = (scrollY+15000)+'px';
        });</script>""")
    posts = fetch_posts(
        fixture.as_uri(), target_movies=50, max_scrolls=60, scroll_pause_ms=50
    )
    assert len(posts) == 1
    assert "Scroll 60/" not in capsys.readouterr().out


def test_zero_target_never_opens_url():
    assert fetch_posts("not a URL", target_movies=0) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_movies": -1},
        {"max_scrolls": -1},
        {"scroll_pause_ms": -1},
    ],
)
def test_negative_limits_are_rejected(kwargs):
    with pytest.raises(ValueError):
        fetch_posts("not a URL", **kwargs)


def test_thumbnail_and_host_aliases_keep_one_post_with_displayed_date(page):
    page.set_content("""<article>
      <a href="https://vk.com/wall-1_2"><img alt="poster"></a>
      <p>Film (2000) ЧТО: drama О ЧЁМ: life ЗАЧЕМ: art</p>
      <a href="https://vk.ru/wall-1_2?from=feed">28 сентября</a>
      <a href="https://m.vk.com/wall-1_2">28 сентября</a>
    </article>""")
    posts = extract_visible_posts(page)
    assert len(posts) == 1
    assert posts[0]["url"] == "https://vk.ru/wall-1_2"
    assert posts[0]["date"] == "28 сентября"
