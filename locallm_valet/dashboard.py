"""Usage dashboard: a self-contained HTML page (inline CSS/JS, no CDN).

Served at ``GET /gateway/dashboard``; it fetches ``/gateway/usage`` and
``/gateway/models`` / ``/gateway/status`` for data.
"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>locallm-valet · 用量看板</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --panel2:#1e222c; --fg:#e6e8ee; --muted:#8b93a3;
          --accent:#4f8cff; --ok:#3ecf8e; --warn:#ffb454; --err:#ff6b6b; --border:#262b36; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--fg); font:14px/1.5 "SF Mono", Consolas, "Microsoft YaHei", monospace; padding:20px; }
  header { display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:16px; }
  h1 { font-size:18px; }
  .chip { padding:2px 10px; border-radius:99px; font-size:12px; background:var(--panel2); color:var(--muted); }
  .chip.ok { color:var(--ok); } .chip.err { color:var(--err); }
  .spacer { flex:1; }
  button, select { background:var(--panel2); color:var(--fg); border:1px solid var(--border);
    border-radius:6px; padding:5px 10px; font-size:13px; cursor:pointer; }
  button:hover { border-color:var(--accent); }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-bottom:16px; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }
  .card .label { color:var(--muted); font-size:12px; }
  .card .value { font-size:22px; font-weight:700; margin-top:4px; }
  .card .sub { color:var(--muted); font-size:12px; margin-top:2px; }
  .panel { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:14px 16px; margin-bottom:16px; }
  .panel h2 { font-size:14px; margin-bottom:10px; color:var(--muted); font-weight:600; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--border); white-space:nowrap; }
  th { color:var(--muted); font-weight:600; }
  td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
  .bar-wrap { display:flex; align-items:center; gap:8px; }
  .bar-bg { flex:1; height:8px; background:var(--panel2); border-radius:4px; overflow:hidden; min-width:80px; }
  .bar { height:100%; background:linear-gradient(90deg, var(--accent), #7fb0ff); border-radius:4px; }
  .trend { display:flex; align-items:flex-end; gap:2px; height:120px; padding-top:8px; }
  .trend .col { flex:1; display:flex; flex-direction:column; justify-content:flex-end; align-items:center; gap:2px; min-width:0; height:100%; }
  .trend .bar2 { width:70%; background:linear-gradient(180deg,#7fb0ff,var(--accent)); border-radius:3px 3px 0 0; }
  .trend .lbl { font-size:10px; color:var(--muted); transform:rotate(-45deg); transform-origin:top center; }
  .muted { color:var(--muted); }
  .err-text { color:var(--err); }
  #refreshAt { color:var(--muted); font-size:12px; }
  a { color:var(--accent); text-decoration:none; }
</style>
</head>
<body>
<header>
  <h1>📊 locallm-valet 用量看板</h1>
  <span class="chip" id="stateChip">…</span>
  <span class="spacer"></span>
  <select id="rangeSel">
    <option value="3600">最近 1 小时</option>
    <option value="86400" selected>最近 24 小时</option>
    <option value="604800">最近 7 天</option>
    <option value="2592000">最近 30 天</option>
    <option value="0">全部</option>
  </select>
  <select id="modelSel"><option value="">全部模型</option></select>
  <select id="groupSel">
    <option value="hour" selected>按小时</option>
    <option value="day">按天</option>
  </select>
  <button id="refreshBtn">刷新</button>
  <a href="/gateway/benchmark" style="font-size:13px;background:var(--panel2);border:1px solid var(--border);border-radius:6px;padding:5px 10px;color:var(--accent);text-decoration:none;">📊 Benchmark</a>
  <span id="refreshAt"></span>
</header>

<div class="cards" id="cards"></div>

<div class="panel">
  <h2>趋势（输出 tokens）</h2>
  <div class="trend" id="trend"></div>
</div>

<div class="panel">
  <h2>按模型</h2>
  <table id="modelTable">
    <thead><tr><th>模型</th><th class="num">请求数</th><th class="num">输入 tokens</th>
      <th class="num">输出 tokens</th><th class="num">总 tokens</th><th>占比</th></tr></thead>
    <tbody></tbody>
  </table>
</div>

<div class="panel">
  <h2>最近请求</h2>
  <table id="recentTable">
    <thead><tr><th>时间 (UTC)</th><th>模型</th><th>接口</th><th>流式</th><th class="num">状态</th>
      <th class="num">输入</th><th class="num">输出</th><th class="num">耗时 ms</th></tr></thead>
    <tbody></tbody>
  </table>
</div>

<script>
const $ = id => document.getElementById(id);
let autoTimer = null;
let apiKey = sessionStorage.getItem('sgm_api_key') || '';

async function authedFetch(url) {
  const headers = {};
  if (apiKey) headers['Authorization'] = 'Bearer ' + apiKey;
  const r = await fetch(url, { headers });
  if (r.status === 401) {
    const k = prompt('请输入 locallm-valet API key（取消则跳过）：');
    if (k === null) return r;
    apiKey = k.trim();
    sessionStorage.setItem('sgm_api_key', apiKey);
    return authedFetch(url);
  }
  return r;
}

function fmt(n) { return (n ?? 0).toLocaleString('zh-CN'); }
function pct(a, b) { return b > 0 ? (100 * a / b).toFixed(1) + '%' : '0%'; }

async function loadModels() {
  try {
    const r = await authedFetch('/gateway/models');
    const data = await r.json();
    const sel = $('modelSel');
    for (const m of data.models) {
      const opt = document.createElement('option');
      opt.value = m.name; opt.textContent = m.name;
      sel.appendChild(opt);
    }
    const s = await (await authedFetch('/gateway/status')).json();
    const chip = $('stateChip');
    chip.textContent = s.state === 'running'
      ? '🟢 ' + s.state + ' · ' + s.model + (s.max_context_tokens ? ' · ctx ' + Number(s.max_context_tokens).toLocaleString('zh-CN') : '')
      : '⚪ ' + s.state;
    chip.className = 'chip ' + (s.state === 'running' ? 'ok' : '');
  } catch (e) { /* manager endpoints always up */ }
}

function render(data) {
  const s = data.summary;
  $('cards').innerHTML = [
    ['请求数', fmt(s.requests), ''],
    ['输入 tokens', fmt(s.prompt_tokens), ''],
    ['输出 tokens', fmt(s.completion_tokens), ''],
    ['总 tokens', fmt(s.total_tokens), s.requests ? '均 ' + s.avg_duration_ms + ' ms/请求' : ''],
  ].map(([l, v, sub]) =>
    `<div class="card"><div class="label">${l}</div><div class="value">${v}</div><div class="sub">${sub}</div></div>`
  ).join('');

  // trend
  const series = data.series;
  const trend = $('trend');
  if (!series.length) { trend.innerHTML = '<span class="muted">暂无数据</span>'; }
  else {
    const max = Math.max(...series.map(x => x.completion_tokens), 1);
    trend.innerHTML = series.map(x => {
      const h = Math.max(2, Math.round(100 * x.completion_tokens / max));
      const t = new Date(x.bucket_epoch * 1000);
      const pad = n => String(n).padStart(2, '0');
      const lbl = (t.getMonth() + 1) + '-' + pad(t.getDate()) + ' ' + pad(t.getHours()) + ':00';
      return `<div class="col" title="${lbl} · 出 ${fmt(x.completion_tokens)} / 入 ${fmt(x.prompt_tokens)} / ${fmt(x.requests)} 请求">
        <span class="bar2" style="height:${h}%"></span><span class="lbl">${lbl}</span></div>`;
    }).join('');
  }

  // by model
  const total = s.total_tokens || 1;
  $('modelTable').querySelector('tbody').innerHTML = data.by_model.map(m =>
    `<tr><td>${m.model}</td><td class="num">${fmt(m.requests)}</td><td class="num">${fmt(m.prompt_tokens)}</td>
     <td class="num">${fmt(m.completion_tokens)}</td><td class="num">${fmt(m.total_tokens)}</td>
     <td><div class="bar-wrap"><div class="bar-bg"><div class="bar" style="width:${pct(m.total_tokens, total)}"></div></div>
     <span class="muted">${pct(m.total_tokens, total)}</span></div></td></tr>`
  ).join('') || '<tr><td colspan="6" class="muted">暂无数据</td></tr>';

  // recent
  $('recentTable').querySelector('tbody').innerHTML = data.recent.map(r =>
    `<tr><td>${r.ts.replace('T', ' ').slice(0, 19)}</td><td>${r.model}</td><td>${r.endpoint}</td>
     <td>${r.stream ? '✅' : '—'}</td><td class="num">${r.status ?? '—'}</td>
     <td class="num">${fmt(r.prompt_tokens)}</td><td class="num">${fmt(r.completion_tokens)}</td>
     <td class="num">${fmt(r.duration_ms)}</td></tr>`
  ).join('') || '<tr><td colspan="8" class="muted">暂无数据</td></tr>';
}

async function load() {
  const range = parseInt($('rangeSel').value, 10);
  const since = range > 0 ? Math.floor(Date.now() / 1000) - range : '';
  const q = new URLSearchParams();
  if (since) q.set('since', since);
  if ($('modelSel').value) q.set('model', $('modelSel').value);
  q.set('group_by', $('groupSel').value);
  q.set('limit', '50');
  try {
    const r = await authedFetch('/gateway/usage?' + q.toString());
    if (!r.ok) throw new Error('HTTP ' + r.status);
    render(await r.json());
    $('refreshAt').textContent = '更新于 ' + new Date().toLocaleTimeString('zh-CN');
  } catch (e) {
    $('cards').innerHTML = `<div class="err-text">加载失败: ${e.message}</div>`;
  }
}

$('refreshBtn').onclick = load;
$('rangeSel').onchange = load;
$('modelSel').onchange = load;
$('groupSel').onchange = load;
setInterval(load, 15000);   // 自动刷新
loadModels();
load();
</script>
</body>
</html>
"""
