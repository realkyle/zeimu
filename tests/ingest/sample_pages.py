"""Small synthetic pages shaped like NTA Tax Answer pages (main and Q&A)."""

MAIN_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><script>var x = 1;</script></head>
<body>
<nav><ul><li>ホーム</li></ul></nav>
<div id="bodyArea">
  <div class="page-header"><h1>No.9999　テストの計算</h1></div>
  <p>[令和８年４月１日現在法令等]</p>
  <h2>対象税目</h2>
  <p>相続税</p>
  <h2>概要</h2>
  <p>計算は<a href="/x.htm">次の順序</a>で行います。</p>
  <h3>総額の計算</h3>
  <p>基礎控除額（3,000万円 ＋ 600万円 × 法定相続人の数）</p>
  <p><img src="/taxes/shiraberu/taxanswer/images/sho/9999.gif" alt="課税価格 － 基礎控除額 ＝ 課税遺産総額"></p>
  <img src="/template/img/navi.png" alt="">
  <h3>税率</h3>
  <table>
    <caption>速算表</caption>
    <tr><th>取得金額</th><th>税率</th><th>控除額</th></tr>
    <tr><td>1,000万円以下</td><td>10％</td><td>－</td></tr>
    <tr><td>3,000万円以下</td><td>15％</td><td>50万円</td></tr>
  </table>
  <!-- a comment that must not appear -->
  <h2>根拠法令等</h2>
  <p>相法15、16</p>
  <h2>関連コード</h2>
  <ul><li>4155 相続税の税率</li><li>4102 相続税がかかる場合</li></ul>
  <h2>お問い合わせ先</h2>
  <p>国税に関するご相談は、税についての相談窓口をご覧ください。</p>
</div>
</body></html>
"""

QA_PAGE = """<html><head><meta http-equiv="Content-Type" content="text/html; charset=Shift_JIS"></head>
<body><div id="bodyArea">
  <h1>No.9999　テストの計算</h1>
  <p>[令和8年4月1日現在法令等]</p>
  <h2>未分割の場合の計算例</h2>
  <h3>Ｑ1</h3>
  <p>各相続人の課税価格はいくらですか。①の場合。</p>
  <h3>Ａ1</h3>
  <p>{long_answer}</p>
  <h4>1 課税価格</h4>
  <p>{long_answer}</p>
  <h2>お問い合わせ先</h2>
  <p>相談窓口</p>
</div></body></html>
"""
