"""Settings page — served at ``GET /gateway/settings`` (auth-exempt shell so
the shared login modal can render; its data endpoints stay auth-gated).

Three panels rendered from the shared design system (frontend.py):

- Account — current username, change password / username.
- API Keys — masked list, generate (full value shown exactly once), delete.
- Model Backend Config — per-model command_template / extra_args /
  health_path, persisted straight into the YAML config; changes need a
  model stop+reload to take effect.
"""

from __future__ import annotations

from .frontend import page

_BODY = """
<main>
  <h1 class="page-title" data-i18n="settings_title">系统设置</h1>
  <p class="page-sub" data-i18n="dash_sub">账号 · API 密钥 · 模型后端配置</p>

  <div class="panel">
    <h2 data-i18n="account">账号</h2>
    <p style="font-size:13px;margin-bottom:14px">
      <span style="color:var(--fg-3)">用户名 / <span data-i18n="username">用户名</span>:</span>
      <b id="userDisplay">—</b>
    </p>
    <div style="display:flex;gap:26px;flex-wrap:wrap">
      <form id="pwForm" style="min-width:260px">
        <div style="font-size:13px;font-weight:650;margin-bottom:8px" data-i18n="change_password">修改密码</div>
        <label class="f-label" data-i18n="current_password">当前密码</label>
        <input type="password" id="curPw" autocomplete="current-password" style="width:100%">
        <label class="f-label" data-i18n="new_password">新密码</label>
        <input type="password" id="newPw" autocomplete="new-password" style="width:100%">
        <label class="f-label" data-i18n="confirm_password">确认新密码</label>
        <input type="password" id="newPw2" autocomplete="new-password" style="width:100%">
        <div id="pwMsg" class="err-text" style="font-size:12px;min-height:16px;margin-top:5px"></div>
        <button type="submit" class="primary" data-i18n="save">保存</button>
      </form>
      <form id="userForm" style="min-width:220px">
        <div style="font-size:13px;font-weight:650;margin-bottom:8px" data-i18n="username">用户名</div>
        <label class="f-label" data-i18n="current_password">当前密码</label>
        <input type="password" id="userCurPw" autocomplete="current-password" style="width:100%">
        <label class="f-label" data-i18n="new_username">新用户名</label>
        <input type="text" id="newUser" autocomplete="off" style="width:100%">
        <div id="userMsg" class="err-text" style="font-size:12px;min-height:16px;margin-top:5px"></div>
        <button type="submit" class="primary" data-i18n="save">保存</button>
      </form>
    </div>
    <p class="muted" style="font-size:12px;margin-top:10px">
      <span id="pwSaved" class="ok-text"></span><span data-i18n="restart_hint"></span>
    </p>
  </div>

  <div class="panel">
    <h2 data-i18n="api_keys">API 密钥</h2>
    <div style="margin-bottom:10px">
      <button id="genKeyBtn" class="primary" data-i18n="generate_key">生成新密钥</button>
    </div>
    <div id="newKeyBox" style="display:none;border:1px solid var(--accent);border-radius:8px;
         padding:10px 12px;margin-bottom:12px;background:var(--accent-soft)">
      <code id="newKeyVal" style="word-break:break-all;font-size:13px"></code>
      <div style="display:flex;gap:8px;margin-top:6px;align-items:center">
        <span class="muted" style="font-size:12px" data-i18n="copy_hint">请立即复制，此密钥仅显示一次</span>
        <button id="copyKeyBtn" class="icon-btn" style="margin-left:auto">⧉ copy</button>
      </div>
    </div>
    <div class="table-scroll">
      <table id="keyTable">
        <thead><tr><th data-i18n="key_masked">密钥（掩码）</th><th></th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <div class="panel">
    <h2 data-i18n="model_config">模型后端配置</h2>
    <label class="f-label" data-i18n="model">模型</label>
    <select id="modelSel" style="min-width:240px"></select>

    <label class="f-label" data-i18n="command_template">命令模板</label>
    <textarea id="tplInput" rows="3" style="width:100%;font-family:ui-monospace,monospace;
      font-size:12px;background:var(--bg-soft);color:var(--fg);border:1px solid var(--border);
      border-radius:6px;padding:7px 9px"
      placeholder="{python} -m sglang.launch_server --model-path {model_path} --host {host} --port {port} {extra_args}"></textarea>
    <div class="muted" style="font-size:11px;margin-top:2px">{python} {model_path} {model_name} {host} {port} {device} {extra_args}</div>
    <div class="muted" style="font-size:11px" id="globalTplHint"></div>

    <label class="f-label" data-i18n="extra_args">额外参数（每行一个）</label>
    <textarea id="argsInput" rows="4" style="width:100%;font-family:ui-monospace,monospace;
      font-size:12px;background:var(--bg-soft);color:var(--fg);border:1px solid var(--border);
      border-radius:6px;padding:7px 9px" placeholder="--context-length&#10;262144"></textarea>

    <label class="f-label" data-i18n="health_path">健康检查路径</label>
    <input type="text" id="hpInput" style="width:280px" placeholder="/health">

    <div id="modelMsg" class="err-text" style="font-size:12px;min-height:16px;margin-top:8px"></div>
    <div style="display:flex;gap:10px;align-items:center;margin-top:4px">
      <button id="saveModelBtn" class="primary" data-i18n="save">保存</button>
      <span id="modelOk" class="ok-text" style="font-size:12px"></span>
    </div>
    <p class="muted warn-hint" style="font-size:12px;margin-top:8px" data-i18n="restart_hint">
      修改已保存到配置文件。需停止并重新加载模型才能生效。
    </p>
  </div>
</main>
<style>
.f-label { display:block; font-size:12px; color:var(--fg-3); margin:10px 0 3px; }
.panel button.primary { margin-top:4px; }
td code { font-size:12px; }
</style>
"""

_SETTINGS_JS = r"""
async function errOf(r) {
  const d = await r.json().catch(() => ({}));
  return (d.error && d.error.message) || 'HTTP ' + r.status;
}

async function loadUser() {
  const r = await authedFetch('/gateway/settings/credentials');
  if (!r.ok) return;
  const j = await r.json();
  $('userDisplay').textContent = j.username || '(none)';
}

$('pwForm').onsubmit = async e => {
  e.preventDefault();
  const msg = $('pwMsg');
  msg.textContent = ''; $('pwSaved').textContent = '';
  const np = $('newPw').value;
  if (np !== $('newPw2').value) {
    msg.textContent = i18n('confirm_password') + ' ≠ ' + i18n('new_password');
    return;
  }
  const r = await authedFetch('/gateway/settings/credentials', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ current_password: $('curPw').value, new_password: np }),
  });
  if (!r.ok) { msg.textContent = await errOf(r); return; }
  msg.textContent = '';
  $('pwSaved').textContent = i18n('saved_ok') + ' — ';
  // Saved password invalidates the browser-stored credentials: force relogin.
  localStorage.removeItem('valet_credentials');
  showLoginModal();
};

$('userForm').onsubmit = async e => {
  e.preventDefault();
  const msg = $('userMsg');
  msg.textContent = ''; $('pwSaved').textContent = '';
  const r = await authedFetch('/gateway/settings/credentials', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      current_password: $('userCurPw').value,
      new_username: $('newUser').value.trim(),
    }),
  });
  if (!r.ok) { msg.textContent = await errOf(r); return; }
  // Username changed too — stored credentials are stale now.
  credentials = { u: $('newUser').value.trim(), p: $('userCurPw').value };
  localStorage.setItem('valet_credentials', JSON.stringify(credentials));
  $('pwSaved').textContent = i18n('saved_ok') + ' — ';
  loadUser();
};

async function loadKeys() {
  const r = await authedFetch('/gateway/settings/api-keys');
  if (!r.ok) return;
  const j = await r.json();
  const tb = $('keyTable').querySelector('tbody');
  tb.innerHTML = (j.keys.length ? j.keys : [{ masked: '' }]).map(k => k.masked
    ? `<tr><td><code>${k.masked}</code></td><td><button class="danger del-key" data-p="${k.masked.slice(0, 8)}">${i18n('delete_key')}</button></td></tr>`
    : '<tr><td colspan="2" class="empty">' + i18n('empty') + '</td></tr>').join('');
  tb.querySelectorAll('.del-key').forEach(btn => {
    btn.onclick = async () => {
      const rr = await authedFetch('/gateway/settings/api-keys/' + btn.dataset.p, { method: 'DELETE' });
      if (rr.ok) loadKeys();
    };
  });
}

$('genKeyBtn').onclick = async () => {
  const r = await authedFetch('/gateway/settings/api-keys', { method: 'POST' });
  if (!r.ok) return;
  const j = await r.json();
  $('newKeyBox').style.display = 'block';
  $('newKeyVal').textContent = j.key;
  $('copyKeyBtn').onclick = async () => {
    try {
      await navigator.clipboard.writeText(j.key);
      $('copyKeyBtn').textContent = '✓';
      setTimeout(() => { $('copyKeyBtn').textContent = '⧉ copy'; }, 1500);
    } catch (e) {}
  };
  loadKeys();
};

// ------------------------------------------------------------- model backends

let modelData = null;

async function loadModels() {
  const r = await authedFetch('/gateway/settings/models');
  if (!r.ok) return;
  modelData = await r.json();
  $('globalTplHint').textContent =
    'global: ' + (modelData.global_command_template || '');
  const sel = $('modelSel');
  sel.innerHTML = '<option value="">— ' + i18n('select_model') + ' —</option>' +
    modelData.models.map(m => `<option value="${m.name}">${m.name}</option>`).join('');
  sel.onchange = () => {
    $('modelOk').textContent = ''; $('modelMsg').textContent = '';
    const m = modelData.models.find(x => x.name === sel.value);
    if (!m) return;
    $('tplInput').value = m.command_template || '';
    $('argsInput').value = m.extra_args.join('\n');
    $('hpInput').value = m.health_path || '';
  };
}

$('saveModelBtn').onclick = async () => {
  const name = $('modelSel').value;
  if (!name) return;
  $('modelMsg').textContent = ''; $('modelOk').textContent = '';
  const r = await authedFetch('/gateway/settings/models/' + encodeURIComponent(name), {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      command_template: $('tplInput').value.trim() || null,
      extra_args: $('argsInput').value.split('\n').map(s => s.trim()).filter(Boolean),
      health_path: $('hpInput').value.trim() || null,
    }),
  });
  if (!r.ok) { $('modelMsg').textContent = await errOf(r); return; }
  $('modelOk').textContent = '✓ ' + i18n('saved_ok');
  loadModels();
};

loadUser();
loadKeys();
loadModels();
"""


def render_settings_page() -> str:
    """Full settings page HTML built fresh on every request."""
    return page(
        "系统设置",
        "Settings",
        active="settings",
        body=_BODY,
        extra_js=_SETTINGS_JS,
    )
