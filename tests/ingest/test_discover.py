"""find_tax_answer_links: picks the right Tax Answer pages out of index and page HTML."""

from zeimu.ingest.discover import CORPUS_CATEGORIES, find_tax_answer_links

BASE = "https://www.nta.go.jp/taxes/shiraberu/taxanswer/code/index.htm"

# Synthetic HTML shaped like the NTA code index and a Tax Answer page.
HTML = """
<ul>
  <li><a href="/taxes/shiraberu/taxanswer/sozoku/4152.htm">No.4152</a></li>
  <li><a href="/taxes/shiraberu/taxanswer/sozoku/4152.htm">duplicate</a></li>
  <li><a href="../zoyo/4402.htm">relative gift link</a></li>
  <li><a href="https://www.nta.go.jp/taxes/shiraberu/taxanswer/hyoka/4602.htm">absolute</a></li>
  <li><a href="/taxes/shiraberu/taxanswer/shotoku/1100.htm">income tax: out of scope</a></li>
  <li><a href="/taxes/shiraberu/taxanswer/sozoku/4152_qa.htm#q1">Q1</a></li>
  <li><a href="/taxes/shiraberu/taxanswer/sozoku/4152_qa.htm#q2">Q2</a></li>
  <li><a href="https://example.com/taxes/shiraberu/taxanswer/sozoku/4999.htm">other host</a></li>
  <li><a href="/taxes/shiraberu/taxanswer/sozoku/souzoku.htm">not a numbered page</a></li>
  <li><a>no href</a></li>
</ul>
"""


def test_main_pages() -> None:
    urls = find_tax_answer_links(HTML, BASE, categories=CORPUS_CATEGORIES, kind="main")

    assert urls == [
        "https://www.nta.go.jp/taxes/shiraberu/taxanswer/hyoka/4602.htm",
        "https://www.nta.go.jp/taxes/shiraberu/taxanswer/sozoku/4152.htm",
        "https://www.nta.go.jp/taxes/shiraberu/taxanswer/zoyo/4402.htm",
    ]


def test_qa_pages_are_deduplicated_without_fragments() -> None:
    urls = find_tax_answer_links(HTML, BASE, categories=CORPUS_CATEGORIES, kind="qa")

    assert urls == ["https://www.nta.go.jp/taxes/shiraberu/taxanswer/sozoku/4152_qa.htm"]


def test_category_filter() -> None:
    urls = find_tax_answer_links(HTML, BASE, categories=["zoyo"], kind="main")

    assert urls == ["https://www.nta.go.jp/taxes/shiraberu/taxanswer/zoyo/4402.htm"]
