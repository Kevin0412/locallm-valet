"""Usage dashboard — bilingual, dark/light, rendered from the shared design
system (see frontend.py). Served at ``GET /gateway/dashboard``; it fetches
``/gateway/usage``, ``/gateway/models`` and ``/gateway/status``.
"""

from __future__ import annotations

from .frontend import page

DASHBOARD_HTML = page(
    "用量总览",
    "Usage Overview",
    active="dashboard",
    body="""<main>
  <h1 class="page-title" data-i18n="dash_title">用量总览</h1>
  <p class="page-sub" data-i18n="dash_sub">模型服务状态与 token 用量</p>

  <div class="controls">
    <span class="tag" id="stateChip">-</span>
    <span class="spacer"></span>
    <select id="rangeSel">
      <option value="3600" data-i18n="1h">最近 1 小时</option>
      <option value="86400" selected data-i18n="24h">最近 24 小时</option>
      <option value="604800" data-i18n="7d">最近 7 天</option>
      <option value="2592000" data-i18n="30d">最近 30 天</option>
      <option value="0" data-i18n="all">全部</option>
    </select>
    <select id="modelSel"><option value="" data-i18n="all_models">全部模型</option></select>
    <select id="groupSel">
      <option value="hour" selected data-i18n="by_hour">按小时</option>
      <option value="day" data-i18n="by_day">按天</option>
    </select>
    <button id="refreshBtn" class="icon-btn" data-i18n="refresh">刷新</button>
    <span id="refreshAt" class="muted"></span>
  </div>

  <div class="cards" id="cards"></div>

  <div id="slotsPanel"></div>

  <div class="panel">
    <h2 data-i18n="trend_title">输出趋势</h2>
    <div class="trend" id="trend"></div>
  </div>

  <div class="panel">
    <h2 data-i18n="by_model_title">按模型</h2>
    <table id="modelTable">
      <thead><tr>
        <th data-i18n="model">模型</th>
        <th class="num" data-i18n="requests">请求数</th>
        <th class="num" data-i18n="in_tokens">输入 tokens</th>
        <th class="num" data-i18n="out_tokens">输出 tokens</th>
        <th class="num" data-i18n="total_tokens">总 tokens</th>
        <th data-i18n="share">占比</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>

  <div class="panel">
    <h2 data-i18n="recent_title">最近请求</h2>
    <table id="recentTable">
      <thead><tr>
        <th data-i18n="time">时间</th><th data-i18n="model">模型</th><th data-i18n="endpoint">接口</th>
        <th class="num" data-i18n="status">状态</th><th class="num" data-i18n="in_tokens">输入</th>
        <th class="num" data-i18n="out_tokens">输出</th><th class="num" data-i18n="duration">耗时 ms</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>
</main>
<style>
.controls { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 18px; }
.controls select, .controls button { height: 30px; }
</style>
""",
    extra_js=r"""
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
    // Multi-slot: aggregate running slots; pools shown in the slots panel.
    const slots = s.slots || {};
    const running = Object.entries(slots).filter(([, v]) => v.state === 'running');
    const chip = $('stateChip');
    if (running.length) {
      chip.textContent = running.map(([n, v]) => n + ':' + v.model).join('  ');
      chip.className = 'tag ok';
    } else {
      chip.textContent = 'stopped';
      chip.className = 'tag';
    }
    renderSlots(s);
  } catch (e) {}
}

function renderSlots(s) {
  const slots = s.slots || {};
  const pools = s.pools || {};
  const wrap = $('slotsPanel');
  if (!wrap) return;
  let rows = '';
  for (const [name, v] of Object.entries(slots)) {
    const state = v.state === 'running' ? 'ok' : '';
    rows += `<tr><td>${name}</td>
      <td><span class="tag ${state}">${v.state}</span></td>
      <td>${v.model || '—'}</td>
      <td class="num">${fmt(v.active_requests)}</td>
      <td class="num">${v.max_context_tokens ? fmt(v.max_context_tokens) : '—'}</td></tr>`;
  }
  let poolRows = '';
  for (const [name, p] of Object.entries(pools)) {
    const pct = p.total_gib > 0 ? (100 * (1 - p.available_gib / p.total_gib)).toFixed(0) : 0;
    poolRows += `<tr><td>${name}${p.probeable ? '' : ' (未探测)'}</td>
      <td class="num">${p.available_gib} / ${p.total_gib} GiB</td>
      <td><div class="bar-wrap"><div class="bar-bg"><div class="bar" style="width:${pct}%"></div></div>
      <span class="muted">${pct}%</span></div></td></tr>`;
  }
  wrap.innerHTML = rows
    ? `<div class="panel"><h2 data-i18n="slots_title">设备槽位</h2><table>
         <thead><tr><th>Slot</th><th data-i18n="status">状态</th><th data-i18n="model">模型</th>
         <th class="num" data-i18n="requests">请求数</th><th class="num">Ctx</th></tr></thead>
         <tbody>${rows}</tbody></table></div>
       <div class="panel"><h2 data-i18n="pools_title">资源池</h2><table>
         <thead><tr><th>Pool</th><th class="num">可用/总量</th><th>占用</th></tr></thead>
         <tbody>${poolRows}</tbody></table></div>`
    : '<div class="panel"><h2 data-i18n="slots_title">设备槽位</h2><p class="empty">—</p></div>';
}

function pct(a, b) { return b > 0 ? (100 * a / b).toFixed(1) + '%' : '0%'; }

function render(data) {
  const s = data.summary;
  const cards = [
    ['Requests', fmt(s.requests), ''],
    ['In tokens', fmt(s.prompt_tokens), ''],
    ['Out tokens', fmt(s.completion_tokens), ''],
    ['Total tokens', fmt(s.total_tokens), s.requests ? (s.avg_duration_ms + ' ms/req') : ''],
  ];
  $('cards').innerHTML = cards.map(([k, v, sub]) =>
    `<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${sub}</div></div>`
  ).join('');

  // continuous trend
  const series = data.series || [];
  const trend = $('trend');
  if (!series.length) {
    trend.innerHTML = '<span class="muted">' + i18n('empty') + '</span>';
  } else {
    const first = new Date(series[0].bucket_epoch * 1000);
    const last = new Date(series[series.length - 1].bucket_epoch * 1000);
    const buckets = [];
    if ($('groupSel').value === 'hour') {
      const start = new Date(first); start.setMinutes(0, 0, 0);
      const end = new Date(last); end.setMinutes(0, 0, 0);
      for (let t = new Date(start); t <= end; t.setHours(t.getHours() + 1)) buckets.push(t.getTime() / 1000);
    } else {
      const start = new Date(first); start.setHours(0, 0, 0, 0);
      const end = new Date(last); end.setHours(0, 0, 0, 0);
      for (let t = new Date(start); t <= end; t.setDate(t.getDate() + 1)) buckets.push(t.getTime() / 1000);
    }
    const map = {};
    for (const x of series) map[x.bucket_epoch] = x;
    const all = buckets.map(e => map[e] || { bucket_epoch: e, completion_tokens: 0, prompt_tokens: 0, requests: 0 });
    const max = Math.max(...all.map(x => x.completion_tokens), 1);
    const pad = n => String(n).padStart(2, '0');
    trend.innerHTML = all.map(x => {
      const h = Math.max(3, Math.round(100 * x.completion_tokens / max));
      const t = new Date(x.bucket_epoch * 1000);
      const lbl = $('groupSel').value === 'hour'
        ? pad(t.getHours()) + ':00'
        : (t.getMonth() + 1) + '-' + pad(t.getDate());
      const title = lbl + ' · out ' + fmt(x.completion_tokens) + ' / in ' + fmt(x.prompt_tokens) + ' / ' + fmt(x.requests) + ' req';
      const cls = x.requests > 0 ? 'col' : 'col empty-bar';
      return `<div class="${cls}" title="${title}"><span class="bar2" style="height:${h}%"></span><span class="lbl">${lbl}</span></div>`;
    }).join('');
  }

  const total = s.total_tokens || 1;
  $('modelTable').querySelector('tbody').innerHTML = data.by_model.map(m =>
    `<tr><td>${m.model}</td><td class="num">${fmt(m.requests)}</td><td class="num">${fmt(m.prompt_tokens)}</td>
     <td class="num">${fmt(m.completion_tokens)}</td><td class="num">${fmt(m.total_tokens)}</td>
     <td><div class="bar-wrap"><div class="bar-bg"><div class="bar" style="width:${pct(m.total_tokens, total)}"></div></div>
     <span class="muted">${pct(m.total_tokens, total)}</span></div></td></tr>`
  ).join('') || `<tr><td colspan="6" class="empty">${i18n('empty')}</td></tr>`;

  $('recentTable').querySelector('tbody').innerHTML = data.recent.map(r =>
    `<tr><td>${(r.ts || '').replace('T', ' ').slice(0, 19)}</td><td>${r.model}</td><td>${r.endpoint}</td>
     <td class="num">${r.status ?? '—'}</td><td class="num">${fmt(r.prompt_tokens)}</td>
     <td class="num">${fmt(r.completion_tokens)}</td><td class="num">${fmt(r.duration_ms)}</td></tr>`
  ).join('') || `<tr><td colspan="7" class="empty">${i18n('empty')}</td></tr>`;
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
    $('refreshAt').textContent = i18n('updated') + ' ' + new Date().toLocaleTimeString();
  } catch (e) {
    $('cards').innerHTML = `<div class="err-text">${i18n('failed')}: ${e.message}</div>`;
  }
}

$('refreshBtn').onclick = load;
$('rangeSel').onchange = load;
$('modelSel').onchange = load;
$('groupSel').onchange = load;
setInterval(load, 15000);
loadModels();
load();
""",
)
