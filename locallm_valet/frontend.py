"""Shared frontend design system for locallm-valet pages.

All web pages (dashboard, benchmark) render from one design system so they
look and feel identical: bilingual (zh/en), dark/light theme toggle, clean
typography, no emoji-as-icon placeholders. Pages are fully self-contained
(inline CSS/JS, no CDN) and read-only GET endpoints stay auth-free while
data endpoints are gated by the API middleware.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

CSS = r"""
:root {
  /* dark theme (default) */
  color-scheme: dark;
  --bg: #101216; --bg-soft: #16191f; --panel: #1a1e26; --panel-2: #222733;
  --fg: #e7eaf0; --fg-2: #aab2c0; --fg-3: #7d8694;
  --accent: #5b8cff; --accent-soft: rgba(91, 140, 255, 0.14);
  --ok: #3fbf7f; --warn: #d9a441; --err: #e06c6c;
  --border: #2a2f3a; --border-soft: #232833;
  --shadow: 0 1px 2px rgba(0,0,0,.25);
}
html[data-theme="light"] {
  color-scheme: light;
  --bg: #f5f6f8; --bg-soft: #eceef2; --panel: #ffffff; --panel-2: #f1f3f6;
  --fg: #1d2129; --fg-2: #4a5260; --fg-3: #7b8494;
  --accent: #2f6bff; --accent-soft: rgba(47, 107, 255, 0.10);
  --ok: #1f9d63; --warn: #b97f16; --err: #d14a4a;
  --border: #dfe3ea; --border-soft: #e8ebf0;
  --shadow: 0 1px 2px rgba(20,30,50,.08);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; }
body {
  background: var(--bg); color: var(--fg);
  font: 14px/1.55 -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  -webkit-font-smoothing: antialiased;
}
body.mono-nums { font-variant-numeric: tabular-nums; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

/* top bar */
.topbar {
  display: flex; align-items: center; gap: 18px; flex-wrap: wrap;
  padding: 12px 22px; border-bottom: 1px solid var(--border-soft);
  background: var(--bg-soft); position: sticky; top: 0; z-index: 20;
}
.brand { font-size: 15px; font-weight: 650; letter-spacing: .2px; }
.brand .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  background: var(--accent); margin-right: 9px; vertical-align: 1px; }
.nav { display: flex; gap: 4px; }
.nav a {
  color: var(--fg-2); padding: 5px 12px; border-radius: 6px; font-size: 13px;
}
.nav a.active { color: var(--fg); background: var(--accent-soft); }
.spacer { flex: 1; }
.icon-btn, select, button, input {
  background: var(--panel); color: var(--fg); border: 1px solid var(--border);
  border-radius: 6px; padding: 5px 11px; font-size: 13px; cursor: pointer;
  font-family: inherit;
}
/* native dropdown list & multi-select items follow the theme too */
select option { background: var(--panel); color: var(--fg); }
.icon-btn:hover, button:hover, select:hover, input:hover { border-color: var(--accent); }
button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
button.primary:hover { filter: brightness(1.08); }
button.danger { color: var(--err); border-color: var(--err); }
button:disabled { opacity: .45; cursor: not-allowed; }
.seg { display: flex; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
.seg button { border: 0; border-radius: 0; background: transparent; }
.seg button.on { background: var(--accent-soft); color: var(--accent); }

main { max-width: 1180px; margin: 0 auto; padding: 22px; }
.page-title { font-size: 20px; font-weight: 650; margin-bottom: 4px; }
.page-sub { color: var(--fg-3); font-size: 13px; margin-bottom: 20px; }

/* cards */
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-bottom: 20px; }
.card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; box-shadow: var(--shadow); }
.card .k { color: var(--fg-3); font-size: 12px; }
.card .v { font-size: 24px; font-weight: 650; margin-top: 3px; font-variant-numeric: tabular-nums; }
.card .s { color: var(--fg-3); font-size: 12px; margin-top: 2px; }

/* panels */
.panel { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; margin-bottom: 18px; box-shadow: var(--shadow); }
.panel > h2 { font-size: 13px; font-weight: 650; color: var(--fg-2); text-transform: uppercase; letter-spacing: .4px; margin-bottom: 14px; }

/* tables */
.table-scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--border-soft); white-space: nowrap; }
/* long cells (model names, category tags, ground truth) wrap instead of
   blowing the table box; numeric cells stay nowrap */
td.wrap, th.wrap { white-space: normal; word-break: break-word; }
th { color: var(--fg-3); font-weight: 600; font-size: 12px; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
tbody tr:hover { background: var(--bg-soft); }
.empty { color: var(--fg-3); text-align: center; padding: 22px 0; }

/* bars */
.bar-wrap { display: flex; align-items: center; gap: 8px; }
.bar-bg { flex: 1; height: 6px; background: var(--panel-2); border-radius: 3px; overflow: hidden; min-width: 60px; }
.bar { height: 100%; background: var(--accent); border-radius: 3px; }
.tag { display: inline-block; padding: 1px 8px; border-radius: 4px; font-size: 12px; margin-right: 4px; }
.tag.ok { background: rgba(63,191,127,.14); color: var(--ok); }
.tag.warn { background: rgba(217,164,65,.14); color: var(--warn); }
.tag.err { background: rgba(224,108,108,.14); color: var(--err); }

/* trend */
.trend { display: flex; align-items: flex-end; gap: 2px; height: 150px; padding-top: 10px; }
.trend .col { flex: 1; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; gap: 3px; min-width: 0; height: 100%; }
.trend .bar2 { width: 72%; background: linear-gradient(180deg, var(--accent), transparent 140%); border-radius: 3px 3px 0 0; min-height: 2px; }
.trend .lbl { font-size: 10px; color: var(--fg-3); }
/* zero-data slots keep a faint but VISIBLE baseline so the timeline reads
   as continuous instead of disconnected gaps */
.trend .col.empty-bar { opacity: 1; }
.trend .col.empty-bar .bar2 { background: var(--panel-2); border: 1px dashed var(--border); min-height: 14px; }

/* progress */
.progress { height: 8px; background: var(--panel-2); border-radius: 4px; overflow: hidden; margin: 8px 0 4px; }
.progress > i { display: block; height: 100%; background: var(--accent); width: 0; transition: width .4s ease; }
.muted { color: var(--fg-3); }
.err-text { color: var(--err); }
.ok-text { color: var(--ok); }
#refreshAt { color: var(--fg-3); font-size: 12px; }
"""


def _base_js() -> str:
    return r"""
const $ = id => document.getElementById(id);
const T = {
  zh: {
    dashboard: '总览', benchmark: '评测', theme_dark: '深色', theme_light: '浅色',
    refresh: '刷新', updated: '更新于', loading: '加载中…', empty: '暂无数据', failed: '加载失败',
    all_models: '全部模型', by_hour: '按小时', by_day: '按天', all: '全部',
    model: '模型', requests: '请求数', in_tokens: '输入 tokens', out_tokens: '输出 tokens',
    total_tokens: '总 tokens', share: '占比', time: '时间', endpoint: '接口', status: '状态',
    duration: '耗时 ms', dash_title: '用量总览', dash_sub: '模型服务状态与 token 用量',
    trend_title: '输出趋势', by_model_title: '按模型', recent_title: '最近请求',
    bm_title: '模型评测', bm_sub: '基于公认题库（MMLU / MMLU-Pro / BFCL / MMStar / OCRBench）评估模型能力',
    run_title: '运行评测', model_hint: 'Ctrl/Shift 多选，留空 = 全部模型',
    start: '开始', pause: '暂停', resume: '继续', stop: '停止',
    results_title: '评测结果', accuracy: '准确率', correct_total: '正确/总数',
    category: '分项', avg_lat: '平均耗时 ms', avg_tps: '吞吐 tok/s',
    running: '运行中', paused: '已暂停', select_all: '全选', select_none: '清空',
    slots_title: '设备槽位', pools_title: '资源池',
    thinking: '思考', non_thinking: '不思考', mode: '模式', avg_tok: '平均输出 tokens',
    dataset: '数据集',
    settings: '设置', login: '登录', logout: '登出',
    username: '用户名', password: '密码', login_failed: '登录失败',
    settings_title: '系统设置', account: '账号',
    api_keys: 'API 密钥', model_config: '模型后端配置',
    save: '保存', cancel: '取消',
    generate_key: '生成新密钥', delete_key: '删除',
    key_masked: '密钥（掩码）', copy_hint: '请立即复制，此密钥仅显示一次',
    change_password: '修改密码', current_password: '当前密码',
    new_password: '新密码', confirm_password: '确认新密码', new_username: '新用户名',
    command_template: '命令模板', extra_args: '额外参数（每行一个）',
    health_path: '健康检查路径', restart_hint: '修改已保存到配置文件。需停止并重新加载模型才能生效。',
    saved_ok: '已保存', select_model: '选择模型',
  },
  en: {
    dashboard: 'Dashboard', benchmark: 'Benchmark', theme_dark: 'Dark', theme_light: 'Light',
    refresh: 'Refresh', updated: 'Updated', loading: 'Loading…', empty: 'No data', failed: 'Failed to load',
    all_models: 'All models', by_hour: 'By hour', by_day: 'By day', all: 'All',
    model: 'Model', requests: 'Requests', in_tokens: 'Input tokens', out_tokens: 'Output tokens',
    total_tokens: 'Total tokens', share: 'Share', time: 'Time', endpoint: 'Endpoint', status: 'Status',
    duration: 'Duration ms', dash_title: 'Usage Overview', dash_sub: 'Model service status & token usage',
    trend_title: 'Output Trend', by_model_title: 'By Model', recent_title: 'Recent Requests',
    bm_title: 'Model Benchmark', bm_sub: 'Standard datasets (MMLU / MMLU-Pro / BFCL / MMStar / OCRBench)',
    run_title: 'Run Benchmark', model_hint: 'Ctrl/Shift to select multiple; empty = all models',
    start: 'Start', pause: 'Pause', resume: 'Resume', stop: 'Stop',
    results_title: 'Results', accuracy: 'Accuracy', correct_total: 'Correct/Total',
    category: 'Breakdown', avg_lat: 'Avg Latency ms', avg_tps: 'Throughput tok/s',
    running: 'Running', paused: 'Paused', select_all: 'Select All', select_none: 'Clear',
    slots_title: 'Device Slots', pools_title: 'Resource Pools',
    thinking: 'Thinking', non_thinking: 'Non-thinking', mode: 'Mode', avg_tok: 'Avg output tokens',
    dataset: 'Dataset',
    settings: 'Settings', login: 'Log in', logout: 'Log out',
    username: 'Username', password: 'Password', login_failed: 'Login failed',
    settings_title: 'Settings', account: 'Account',
    api_keys: 'API Keys', model_config: 'Model Backend Config',
    save: 'Save', cancel: 'Cancel',
    generate_key: 'Generate new key', delete_key: 'Delete',
    key_masked: 'Key (masked)', copy_hint: 'Copy it now — shown only once',
    change_password: 'Change password', current_password: 'Current password',
    new_password: 'New password', confirm_password: 'Confirm new password',
    new_username: 'New username',
    command_template: 'Command template', extra_args: 'Extra args (one per line)',
    health_path: 'Health path', restart_hint: 'Saved to the config file. Stop and reload the model for changes to take effect.',
    saved_ok: 'Saved', select_model: 'Select a model',
  },
};
const i18n = key => (T[lang()] && T[lang()][key]) || key;
let lang = () => localStorage.getItem('valet_lang') || 'zh';
let theme = () => localStorage.getItem('valet_theme') || 'dark';

function applyTheme() {
  document.documentElement.setAttribute('data-theme', theme());
  document.querySelectorAll('[data-theme-btn]').forEach(b => {
    b.textContent = theme() === 'dark' ? i18n('theme_light') : i18n('theme_dark');
  });
}
function toggleTheme() {
  localStorage.setItem('valet_theme', theme() === 'dark' ? 'light' : 'dark');
  applyTheme();
}
function toggleLang() {
  localStorage.setItem('valet_lang', lang() === 'zh' ? 'en' : 'zh');
  location.reload();
}
function applyLangText() {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    el.textContent = i18n(el.getAttribute('data-i18n'));
  });
}
function fmt(n) { return (n ?? 0).toLocaleString(lang() === 'zh' ? 'zh-CN' : 'en-US'); }

// ---------------------------------------------------------------------------
// Auth: username/password (Basic auth) instead of the old prompt()-driven
// Bearer key. Credentials live in localStorage; a 401 pops the login modal
// and the caller sees the failed response (no retry loop).
// ---------------------------------------------------------------------------

let credentials = JSON.parse(localStorage.getItem('valet_credentials') || 'null');

function basicAuthHeader() {
  if (!credentials || !credentials.u || credentials.p === undefined) return {};
  // btoa on non-Latin1 passwords throws; encodeURIComponent keeps it safe.
  return { 'Authorization': 'Basic ' + btoa(unescape(encodeURIComponent(credentials.u + ':' + credentials.p))) };
}

async function authedFetch(url, opts) {
  const headers = { ...((opts && opts.headers) || {}), ...basicAuthHeader() };
  const r = await fetch(url, { ...opts, headers });
  if (r.status === 401) showLoginModal();
  return r;
}

function logout() {
  localStorage.removeItem('valet_credentials');
  location.reload();
}

function ensureLoginModal() {
  if ($('loginWrap')) return $('loginWrap');
  const wrap = document.createElement('div');
  wrap.id = 'loginWrap';
  wrap.style.cssText =
    'position:fixed;inset:0;background:rgba(0,0,0,.55);display:flex;align-items:center;' +
    'justify-content:center;z-index:100';
  wrap.innerHTML = `
    <form id="loginForm" style="background:var(--panel);border:1px solid var(--border);
      border-radius:10px;padding:22px;width:300px;box-shadow:var(--shadow);color:var(--fg)">
      <h2 style="font-size:16px;margin-bottom:14px">locallm-valet · <span data-i18n="login">${i18n('login')}</span></h2>
      <label style="display:block;font-size:12px;color:var(--fg-3);margin:8px 0 3px" data-i18n="username">${i18n('username')}</label>
      <input type="text" id="loginUser" autocomplete="username" style="width:100%">
      <label style="display:block;font-size:12px;color:var(--fg-3);margin:8px 0 3px" data-i18n="password">${i18n('password')}</label>
      <input type="password" id="loginPass" autocomplete="current-password" style="width:100%">
      <div id="loginErr" class="err-text" style="font-size:12px;min-height:18px;margin-top:6px"></div>
      <button type="submit" class="primary" style="width:100%;margin-top:4px" data-i18n="login">${i18n('login')}</button>
    </form>`;
  document.body.appendChild(wrap);
  wrap.querySelector('#loginForm').onsubmit = async e => {
    e.preventDefault();
    const u = wrap.querySelector('#loginUser').value.trim();
    const p = wrap.querySelector('#loginPass').value;
    try {
      const r = await fetch('/gateway/settings/auth-check', {
        method: 'POST',
        headers: { 'Authorization': 'Basic ' + btoa(unescape(encodeURIComponent(u + ':' + p))) },
      });
      if (r.ok) {
        credentials = { u, p };
        localStorage.setItem('valet_credentials', JSON.stringify(credentials));
        location.reload();
        return;
      }
    } catch (err) {}
    wrap.querySelector('#loginErr').textContent = i18n('login_failed');
  };
  return wrap;
}

function showLoginModal() {
  ensureLoginModal().querySelector('#loginUser').focus();
}

async function autoCheckCredentials() {
  try {
    const r = await authedFetch('/gateway/settings/auth-check', { method: 'POST' });
    if (r.status !== 401 && !r.ok) throw new Error();
  } catch (e) {
    if (credentials) {
      localStorage.removeItem('valet_credentials'); // stale saved login
      credentials = null;
    }
    showLoginModal();
  }
}

"""


def _topbar(active: str) -> str:
    """Nav bar shared by all pages (dashboard ↔ benchmark)."""
    def _nav(link: str, key: str, is_active: bool) -> str:
        cls = ' class="active"' if is_active else ""
        return f'<a href="{link}"{cls} data-i18n="{key}">{key}</a>'

    return f"""
<div class="topbar">
  <span class="brand"><span class="dot"></span>locallm-valet</span>
  <nav class="nav">
    {_nav('/gateway/dashboard', 'dashboard', active == 'dashboard')}
    {_nav('/gateway/benchmark', 'benchmark', active == 'benchmark')}
  </nav>
  <span class="spacer"></span>
  <a class="icon-btn" href="/gateway/settings" data-i18n="settings">⚙ 设置</a>
  <button class="icon-btn" data-theme-btn onclick="toggleTheme()"></button>
  <button class="icon-btn" onclick="toggleLang()">EN / 中文</button>
  <button class="icon-btn" onclick="logout()" data-i18n="logout">登出</button>
</div>
"""


def page(title_zh: str, title_en: str, body: str, active: str, extra_js: str = "") -> str:
    """Render a full page from the shared design system."""
    title = title_zh if lang_default() == "zh" else title_en
    return f"""<!DOCTYPE html>
<html lang="zh-CN" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>locallm-valet · {title}</title>
<style>{CSS}</style>
</head>
<body>
{_topbar(active)}
{body}
<script>
{_base_js()}
applyTheme();
document.addEventListener('DOMContentLoaded', () => {{
  applyLangText();
  document.querySelectorAll('[data-theme-btn]').forEach(b => b.onclick = toggleTheme);
  {extra_js}
}});
autoCheckCredentials();
</script>
</body>
</html>"""


def lang_default() -> str:
    return "zh"
