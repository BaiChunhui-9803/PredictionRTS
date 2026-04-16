def _build_live_game_html(port: int = 8000) -> str:
    _html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
:root{--bg:#0e1117;--sf:#262730;--sf2:#1e1f26;--bd:#3d3f4a;--tx:#fafafa;--tx2:#a3a8b8;
--ac:#4fc3f7;--grn:#4caf50;--ylw:#ffc107;--red:#f44336;--rd:8px}
.light{--bg:#f5f5f5;--sf:#ffffff;--sf2:#eeeeee;--bd:#d0d0d0;--tx:#1a1a1a;--tx2:#666666;
--ac:#0288d1;--grn:#2e7d32;--ylw:#f57f17;--red:#d32f2f;--rd:8px}
.light .console-body{background:#fff;color:#333}
.light .log-line{border-bottom-color:rgba(0,0,0,.08)}
.light .log-ts{color:#999}
.light .lv-info{color:#666}.light .lv-success{color:#2e7d32}.light .lv-warn{color:#e65100}.light .lv-error{color:#c62828}
.light .src-api{background:#e3f2fd;color:#1565c0}
.light .src-game{background:#eceff1;color:#546e7a}
.light .badge-gray{background:#e0e0e0;color:#616161}
.light .badge-green{background:#e8f5e9;color:#2e7d32}
.light .badge-yellow{background:#fff8e1;color:#f57f17}
.light .btn-primary{background:#0288d1;color:#fff}
.light .btn-ghost{background:#eee;color:#333;border-color:#ccc}
.light .toggle .slider{background:#bbb}
.light .toast-ok{background:#e8f5e9;color:#1b5e20}
.light .lv-debug{color:#00796b}
.light .src-debug{background:#00796b;color:#b2dfdb}
.light .toast-err{background:#ffebee;color:#b71c1c}
.toast-wrap{position:fixed;top:12px;left:50%;transform:translateX(-50%);z-index:9999;display:flex;flex-direction:column;gap:6px;align-items:center;pointer-events:none}
.toast{padding:8px 20px;border-radius:6px;font-size:13px;font-weight:600;pointer-events:auto;animation:toast-in .2s ease;box-shadow:0 2px 8px rgba(0,0,0,.3)}
.toast-ok{background:#1b5e20;color:#a5d6a7}
.toast-err{background:#b71c1c;color:#ffcdd2}
@keyframes toast-in{from{opacity:0;transform:translateY(-10px)}to{opacity:1;transform:translateY(0)}}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Source Sans Pro',-apple-system,sans-serif;background:var(--bg);color:var(--tx);font-size:14px;line-height:1.6;overflow:hidden;height:100vh}
.root{display:flex;height:100vh}
.left{flex:1;overflow-y:auto;padding:16px;min-width:0;border-right:1px solid var(--bd)}
.right{width:480px;display:flex;flex-direction:column;flex-shrink:0}
.console-header{padding:10px 14px;background:var(--sf);border-bottom:1px solid var(--bd);display:flex;justify-content:space-between;align-items:center;flex-shrink:0;position:relative}
.console-header span{font-size:13px;font-weight:600;color:var(--tx2);text-transform:uppercase;letter-spacing:.5px}
.filter-panel{position:absolute;top:100%;right:0;z-index:100;background:var(--sf);border:1px solid var(--bd);border-radius:var(--rd);padding:12px;min-width:200px;display:none;box-shadow:0 4px 12px rgba(0,0,0,.3)}
.filter-panel.show{display:block}
.filter-panel h4{font-size:12px;color:var(--tx2);margin:0 0 8px;font-weight:600}
.filter-group{margin-bottom:10px}
.filter-group:last-child{margin-bottom:0}
.filter-group label{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--tx);padding:2px 0;cursor:pointer}
.filter-group label:hover{color:var(--ac)}
.filter-group input[type=checkbox]{accent-color:var(--ac)}
.console-body{flex:1;overflow-y:auto;padding:8px 10px;font-family:'Cascadia Code','Fira Code','Consolas',monospace;font-size:12px;line-height:1.8}
.log-line{display:flex;gap:8px;padding:1px 0;border-bottom:1px solid rgba(61,63,74,.3)}
.log-ts{color:#666;flex-shrink:0;width:60px;text-align:right}
.log-src{flex-shrink:0;min-width:36px;padding:0 8px;font-weight:700;text-align:center;border-radius:3px;white-space:nowrap}
.src-api{background:#1565c0;color:#90caf9}
.src-game{background:#37474f;color:#b0bec5}
.log-msg{flex:1;word-break:break-all}
.lv-info{color:var(--tx2)}.lv-success{color:#a5d6a7}.lv-warn{color:#fff9c4}.lv-error{color:#ef9a9a}.lv-debug{color:#80cbc4}
.console-footer{padding:6px 14px;background:var(--sf);border-top:1px solid var(--bd);display:flex;justify-content:space-between;align-items:center;flex-shrink:0;font-size:12px;color:var(--tx2)}
.card{background:var(--sf);border:1px solid var(--bd);border-radius:var(--rd);padding:14px;margin-bottom:10px}
.card-title{font-size:13px;font-weight:600;color:var(--tx2);text-transform:uppercase;letter-spacing:.5px;margin-bottom:10px}
.metrics{display:flex;gap:10px;flex-wrap:wrap}
.metric{flex:1;min-width:70px;background:var(--sf2);border-radius:6px;padding:8px 10px}
.metric-label{font-size:11px;color:var(--tx2);margin-bottom:2px}
.metric-value{font-size:18px;font-weight:700}
.badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600}
.badge-green{background:#1b5e20;color:#a5d6a7}
.badge-yellow{background:#f57f17;color:#fff9c4}
.badge-gray{background:#424242;color:#9e9e9e}
.btn{border:none;border-radius:6px;padding:7px 14px;font-size:13px;font-weight:600;cursor:pointer;transition:opacity .15s}
.btn:hover{opacity:.85}.btn:active{opacity:.7}
.btn-primary{background:#4fc3f7;color:#fff}
.btn-blue-dark{background:#0288d1;color:#fff}
.btn-step{background:#4fc3f7;color:#fff}
.btn-danger{background:#f44336;color:#fff}
.btn-warn{background:#ff9800;color:#fff}
.btn-ok{background:#4caf50;color:#fff}
.btn-ghost{background:var(--sf2);color:var(--tx);border:1px solid var(--bd)}
.btn-sm{padding:5px 16px;font-size:12px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.row-between{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}
.card{background:var(--sf2);border-radius:8px;padding:12px;margin-bottom:10px}
.card-title{font-size:13px;font-weight:600;color:var(--tx2);margin-bottom:8px}
.toggle-wrap{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--tx2)}
.toggle{position:relative;width:36px;height:20px;cursor:pointer}
.toggle input{opacity:0;width:0;height:0}
.toggle .slider{position:absolute;inset:0;background:#424242;border-radius:10px;transition:.2s}
.toggle .slider:before{content:"";position:absolute;width:16px;height:16px;left:2px;top:2px;background:#fff;border-radius:50%;transition:.2s}
.toggle input:checked+.slider{background:var(--ac)}
.toggle input:checked+.slider:before{transform:translateX(16px)}
.ep-item{border:1px solid var(--bd);border-radius:6px;padding:8px 10px;margin-bottom:6px;background:var(--sf)}
.ep-item:hover{border-color:var(--ac)}
.ep-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}
.ep-title{font-size:13px;font-weight:600;color:var(--tx)}
.ep-badge{padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600}
.ep-badge-win{background:#1b5e20;color:#a5d6a7}
.ep-badge-loss{background:#b71c1c;color:#ef9a9a}
.ep-badge-interrupted{background:#424242;color:#9e9e9e}
.ep-badge-dogfall{background:#e65100;color:#ffe0b2}
.ep-meta{font-size:12px;color:var(--tx2);margin-bottom:4px}
.ep-export-btns{margin-left:auto;display:flex;gap:3px;flex-shrink:0}
.ep-export-btns button{font-size:10px;padding:1px 6px;border-radius:3px;border:1px solid var(--bd);background:var(--sf2);color:var(--tx2);cursor:pointer;font-weight:600;line-height:1.4}
.ep-export-btns button:hover{background:var(--ac);color:#fff;border-color:var(--ac)}
.ep-flow{display:flex;flex-wrap:wrap;gap:2px;font-size:11px;margin-top:4px}
.ep-flow-item{padding:2px 5px;border-radius:3px;background:var(--sf2);color:var(--tx2);white-space:nowrap;font-size:11px}
.ep-flow-arrow{color:var(--ac);padding:0 1px}
.ep-events{font-size:10px;color:var(--tx2);margin-top:3px;max-height:60px;overflow-y:auto}
.ep-page-btn{padding:2px 8px;border-radius:4px;border:1px solid var(--bd);background:var(--sf);color:var(--tx);cursor:pointer;font-size:11px}
.ep-page-btn:hover{border-color:var(--ac)}
.ep-page-btn.active{background:var(--ac);color:#fff;border-color:var(--ac)}
.ep-details{margin-top:6px;border:1px solid var(--bd);border-radius:4px;overflow:hidden}
.ep-details summary{padding:5px 8px;font-size:11px;color:var(--tx2);cursor:pointer;background:var(--sf);user-select:none}
.ep-details summary:hover{color:var(--ac)}
.ep-plan-block{padding:6px 8px;border-bottom:1px solid var(--bd);font-size:11px}
.ep-plan-block:last-child{border-bottom:none}
.ep-plan-title{font-weight:600;color:var(--ac);margin-bottom:4px}
.ep-plan-details{border-bottom:1px solid var(--bd)}
.ep-plan-details:last-child{border-bottom:none}
.ep-plan-details summary{padding:5px 8px;font-size:11px;color:var(--ac);cursor:pointer;background:var(--sf);user-select:none;font-weight:600}
.ep-plan-details summary:hover{opacity:.85}
.ep-plan-details .ep-plan-content{padding:6px 8px}
.ep-beam-path{margin-bottom:3px;padding:3px 5px;border-radius:3px;border:1px solid var(--bd);font-size:10px}
.ep-beam-path.chosen{border-color:rgba(13,71,161,.45);background:rgba(13,71,161,.06)}
.ep-beam-path .path-label{font-weight:600;margin-right:4px}
.ep-beam-path.chosen .path-label{color:#1565c0}
.ep-beam-path:not(.chosen) .path-label{color:var(--tx2)}
.ep-beam-path .path-metrics{float:right;color:var(--tx2);font-size:9px}
.ep-beam-path .path-metrics span{margin-left:6px}
.ep-beam-step{display:inline-block;padding:0 3px;border-radius:2px;background:var(--sf2);color:var(--tx2);white-space:nowrap}
.ep-beam-path .path-arrow{color:var(--ac);opacity:.6;margin:0 1px}
.ep-beam-table{width:100%;border-collapse:collapse;font-size:10px;margin-top:4px}
.ep-beam-table th{text-align:left;padding:2px 4px;color:var(--tx2);font-weight:600;border-bottom:1px solid var(--bd)}
.ep-beam-table td{padding:2px 4px;color:var(--tx)}
</style></head>
<body>

<div id="toast-container" class="toast-wrap"></div>

<div class="root">
<div class="left">

<div style="display:flex;align-items:stretch;gap:10px;margin-bottom:10px">
  <button class="btn btn-ghost btn-sm" style="padding:5px 16px;font-size:15px;border-radius:var(--rd)" onclick="toggleTheme()" id="theme-btn" title="切换主题">&#9728;</button>
  <div class="card" style="margin-bottom:0;display:flex;align-items:center;gap:4px;padding:4px 8px">
    <span style="font-size:10px;color:var(--tx2);white-space:nowrap">窗口</span>
    <input type="number" id="win-x" value="2600" style="width:50px;padding:2px 3px;font-size:10px;border:1px solid var(--bd);border-radius:var(--rd);background:var(--sf);color:var(--tx);text-align:center" title="X">
    <input type="number" id="win-y" value="50" style="width:48px;padding:2px 3px;font-size:10px;border:1px solid var(--bd);border-radius:var(--rd);background:var(--sf);color:var(--tx);text-align:center" title="Y">
    <span style="font-size:10px;color:var(--tx2)">&times;</span>
    <input type="number" id="win-w" value="640" style="width:52px;padding:2px 3px;font-size:10px;border:1px solid var(--bd);border-radius:var(--rd);background:var(--sf);color:var(--tx);text-align:center" title="宽">
    <input type="number" id="win-h" value="480" style="width:52px;padding:2px 3px;font-size:10px;border:1px solid var(--bd);border-radius:var(--rd);background:var(--sf);color:var(--tx);text-align:center" title="高">
    <button class="btn btn-ghost btn-sm" style="padding:2px 8px;font-size:10px;margin-left:2px" onclick="applyWindowPos()">应用</button>
  </div>
  <div class="card" style="margin-bottom:0;flex:1">
    <div class="row-between">
      <div style="display:flex;align-items:center;gap:12px">
        <span class="conn-dot" id="conn-dot"></span>
        <span style="font-size:12px;color:var(--tx2)" id="conn-text">检测中...</span>
        <span style="font-size:11px;color:var(--tx2)">|</span>
        <span style="font-size:12px;color:var(--tx2)">端口: __PORT__</span>
      </div>
      <div class="toggle-wrap">
        <label class="toggle"><input type="checkbox" id="auto-refresh" checked><span class="slider"></span></label>
        <select class="input" id="refresh-interval" style="width:55px;padding:3px 5px;font-size:11px">
          <option value="1000">1s</option>
          <option value="2000" selected>2s</option>
          <option value="3000">3s</option>
          <option value="5000">5s</option>
        </select>
      </div>
    </div>
  </div>
  <div class="card" style="margin-bottom:0">
    <div style="display:flex;align-items:center;gap:6px;justify-content:center">
      <button class="btn btn-warn btn-sm" onclick="ctrl('pause')">暂停</button>
      <button class="btn btn-ok btn-sm" onclick="ctrl('resume')">恢复</button>
      <span style="color:var(--bd);font-size:16px;margin:0 2px">|</span>
      <button class="btn btn-step btn-sm" onclick="ctrl('step')">步进</button>
      <button class="btn btn-blue-dark btn-sm" onclick="ctrl('run_episode')">单局运行</button>
      <span style="color:var(--bd);font-size:16px;margin:0 2px">|</span>
      <button class="btn btn-danger btn-sm" onclick="shutdownService()">停止服务</button>
    </div>
  </div>
</div>

<div class="card">
  <div class="card-title">游戏状态</div>
  <div class="metrics" style="margin-bottom:10px">
    <div class="metric"><div class="metric-label">状态</div><div class="metric-value"><span class="badge badge-gray" id="state">--</span></div></div>
    <div class="metric"><div class="metric-label">帧</div><div class="metric-value" id="frame">--</div></div>
    <div class="metric"><div class="metric-label">Episode</div><div class="metric-value" id="episode">--</div></div>
    <div class="metric"><div class="metric-label">我方</div><div class="metric-value" id="my-count">--</div></div>
    <div class="metric"><div class="metric-label">敌方</div><div class="metric-value" id="enemy-count">--</div></div>
    <div class="metric"><div class="metric-label">聚类</div><div class="metric-value" id="cluster" style="font-size:13px">--</div></div>
  </div>
  <div class="metrics">
    <div class="metric"><div class="metric-label">我方 HP</div><div class="metric-value" id="my-hp">--</div></div>
    <div class="metric"><div class="metric-label">敌方 HP</div><div class="metric-value" id="enemy-hp">--</div></div>
    <div class="metric" style="flex:2"><div class="metric-label">上一步动作</div><div class="metric-value" id="last-action" style="font-size:13px;color:var(--ac)">--</div></div>
    <div class="metric"><div class="metric-label">KG</div><div class="metric-value" id="kg-status" style="font-size:11px">--</div></div>
    <div class="metric"><div class="metric-label">Buffer</div><div class="metric-value" id="history-buffer" style="font-size:11px;color:var(--ac)">--</div></div>
  </div>
</div>

<div class="card" id="episodes-card">
  <div class="card-title" style="display:flex;justify-content:space-between;align-items:center">
    <span>对局记录</span>
    <div style="display:flex;gap:6px;align-items:center">
      <select id="ep-sort" class="input" style="padding:3px 6px;font-size:11px;border-radius:4px;border:1px solid var(--bd);background:var(--sf);color:var(--tx)" onchange="epCurrentPage=1;renderLocalEpisodes()">
        <option value="id_desc">最新优先</option>
        <option value="id_asc">最早优先</option>
        <option value="score_desc">得分降序</option>
        <option value="score_asc">得分升序</option>
      </select>
      <input id="ep-search" type="text" placeholder="搜索..." style="padding:3px 8px;font-size:11px;border-radius:4px;border:1px solid var(--bd);background:var(--sf);color:var(--tx);width:80px" oninput="debounceSearch()">
      <button class="btn btn-ghost btn-sm" style="padding:3px 8px;font-size:11px" onclick="loadEpisodes()">刷新</button>
      <button class="btn btn-ghost btn-sm" style="padding:3px 8px;font-size:11px;color:#f44336" onclick="clearEpisodes()">清空</button>
    </div>
  </div>
  <div id="episodes-list" style="max-height:540px;overflow-y:auto"></div>
  <div id="ep-pagination" style="display:flex;justify-content:center;gap:4px;margin-top:8px;font-size:12px"></div>
  <div style="text-align:center;font-size:11px;color:var(--tx2);margin-top:4px"><span id="ep-count">0</span> 条记录</div>
</div>

</div>

<div class="right">
  <div class="console-header">
    <span>控制台日志</span>
    <div style="display:flex;gap:6px">
      <div style="position:relative">
        <button class="btn btn-ghost btn-sm" onclick="toggleFilter()">过滤</button>
        <div class="filter-panel" id="filter-panel">
          <div class="filter-group">
            <h4>日志级别</h4>
            <label><input type="checkbox" data-filter="level" value="info" checked onchange="applyFilters()"> Info</label>
            <label><input type="checkbox" data-filter="level" value="success" checked onchange="applyFilters()"> Success</label>
            <label><input type="checkbox" data-filter="level" value="warn" checked onchange="applyFilters()"> Warn</label>
            <label><input type="checkbox" data-filter="level" value="error" checked onchange="applyFilters()"> Error</label>
            <label><input type="checkbox" data-filter="level" value="debug" onchange="applyFilters()"> Debug</label>
          </div>
          <div class="filter-group">
            <h4>来源</h4>
            <label><input type="checkbox" data-filter="source" value="game" checked onchange="applyFilters()"> GAME</label>
            <label><input type="checkbox" data-filter="source" value="api" checked onchange="applyFilters()"> API</label>
            <label><input type="checkbox" data-filter="source" value="autopilot" checked onchange="applyFilters()"> Autopilot</label>
          </div>
          <div class="filter-group">
            <h4>消息类型</h4>
            <label><input type="checkbox" data-filter="type" value="info" checked onchange="applyFilters()"> Info</label>
            <label><input type="checkbox" data-filter="type" value="action" checked onchange="applyFilters()"> Action</label>
            <label><input type="checkbox" data-filter="type" value="fallback" checked onchange="applyFilters()"> Fallback</label>
            <label><input type="checkbox" data-filter="type" value="result" checked onchange="applyFilters()"> Result</label>
            <label><input type="checkbox" data-filter="type" value="episode" checked onchange="applyFilters()"> Episode</label>
            <label><input type="checkbox" data-filter="type" value="control" checked onchange="applyFilters()"> Control</label>
          </div>
          <div style="display:flex;gap:6px;margin-top:10px;border-top:1px solid var(--bd);padding-top:8px">
            <button class="btn btn-primary btn-sm" style="flex:1;padding:4px 0;font-size:11px" onclick="saveFilterConfig()">保存配置</button>
            <button class="btn btn-ghost btn-sm" style="flex:1;padding:4px 0;font-size:11px" onclick="resetFilterConfig()">重置默认</button>
          </div>
        </div>
      </div>
      <button class="btn btn-ghost btn-sm" onclick="clearLogs()">清空</button>
    </div>
  </div>
  <div class="console-body" id="console-body">
    <div style="text-align:center;color:#666;padding:40px 0">等待日志...</div>
  </div>
  <div class="console-footer">
    <span id="log-count">0 条</span>
    <span id="log-scroll-hint" style="cursor:pointer;color:var(--ac)" onclick="scrollConsoleBottom()">↓ 滚动到底部</span>
  </div>
</div>
</div>

<script>
const API='http://localhost:__PORT__';
let timer=null, logTimer=null, latestSeq=0, renderedMaxSeq=0, recommendedAction='';
let userScrolled=false;

const LV_COLORS={info:'lv-info',success:'lv-success',warn:'lv-warn',error:'lv-error',debug:'lv-debug'};
const SRC_CLS={api:'src-api',game:'src-game'};
let _filterLevels=new Set(['info','success','warn','error']);
let _filterSources=new Set(['game','api','autopilot']);
let _filterTypes=new Set(['info','action','fallback','result','episode','control']);
function _extractType(msg){
  if(!msg)return 'info';
  if(msg.startsWith('执行动作'))return 'action';
  if(msg.startsWith('回退策略'))return 'fallback';
  if(msg.startsWith('判定'))return 'result';
  if(msg.startsWith('Episode'))return 'episode';
  if(msg.startsWith('步进')||msg.startsWith('游戏暂停')||msg.startsWith('游戏恢复')||msg.startsWith('游戏停止'))return 'control';
  return 'info';
}
function _isFiltered(log){if(log.level==='debug')return !_filterLevels.has('debug');return !_filterLevels.has(log.level||'info')||!_filterSources.has(log.source||'game')||!_filterTypes.has(_extractType(log.message))}
function toggleFilter(){document.getElementById('filter-panel').classList.toggle('show')}
function _syncCheckboxesFromSets(){
  document.querySelectorAll('#filter-panel input[type=checkbox]').forEach(function(cb){
    var t=cb.getAttribute('data-filter'),v=cb.value;
    if(t==='level')cb.checked=_filterLevels.has(v);
    else if(t==='source')cb.checked=_filterSources.has(v);
    else cb.checked=_filterTypes.has(v);
  });
}
function _syncSetsFromCheckboxes(){
  _filterLevels=new Set();_filterSources=new Set();_filterTypes=new Set();
  document.querySelectorAll('#filter-panel input[type=checkbox]').forEach(function(cb){
    if(cb.checked){var t=cb.getAttribute('data-filter'),v=cb.value;if(t==='level')_filterLevels.add(v);else if(t==='source')_filterSources.add(v);else _filterTypes.add(v)}
  });
}
var FILTER_STORAGE_KEY='live_filter_cfg';
var FILTER_DEFAULTS={levels:['info','success','warn','error'],sources:['game','api','autopilot'],types:['info','action','fallback','result','episode','control']};
function loadFilterConfig(){
  var raw=localStorage.getItem(FILTER_STORAGE_KEY);
  if(!raw)return;
  try{
    var cfg=JSON.parse(raw);
    _filterLevels=new Set(cfg.levels||FILTER_DEFAULTS.levels);
    _filterSources=new Set(cfg.sources||FILTER_DEFAULTS.sources);
    _filterTypes=new Set(cfg.types||FILTER_DEFAULTS.types);
    _syncCheckboxesFromSets();
    document.querySelectorAll('#console-body .log-line').forEach(function(el){
      el.style.display=(_filterLevels.has(el.getAttribute('data-level'))&&_filterSources.has(el.getAttribute('data-source'))&&_filterTypes.has(el.getAttribute('data-type')))?'':'none';
    });
  }catch(e){}
}
function saveFilterConfig(){
  _syncSetsFromCheckboxes();
  localStorage.setItem(FILTER_STORAGE_KEY,JSON.stringify({levels:Array.from(_filterLevels),sources:Array.from(_filterSources),types:Array.from(_filterTypes)}));
  toast('过滤配置已保存');
}
function resetFilterConfig(){
  localStorage.removeItem(FILTER_STORAGE_KEY);
  _filterLevels=new Set(FILTER_DEFAULTS.levels);
  _filterSources=new Set(FILTER_DEFAULTS.sources);
  _filterTypes=new Set(FILTER_DEFAULTS.types);
  _syncCheckboxesFromSets();
  document.querySelectorAll('#console-body .log-line').forEach(function(el){el.style.display=''});
  toast('已重置为默认');
}
function applyFilters(){
  _syncSetsFromCheckboxes();
  document.querySelectorAll('#console-body .log-line').forEach(function(el){
    el.style.display=(_filterLevels.has(el.getAttribute('data-level'))&&_filterSources.has(el.getAttribute('data-source'))&&_filterTypes.has(el.getAttribute('data-type')))?'':'none';
  });
}
document.addEventListener('click',function(e){if(!e.target.closest('.filter-panel')&&!e.target.closest('[onclick*="toggleFilter"]')){document.getElementById('filter-panel').classList.remove('show')}});

async function apiGet(p){const r=await fetch(API+p);if(!r.ok)throw new Error(r.status);return r.json()}
async function apiPost(p,b){const r=await fetch(API+p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});if(!r.ok){const t=await r.text().catch(()=>'');throw new Error(t||r.status)}return r.json()}

function toast(m,ok=true){const c=document.getElementById('toast-container'),d=document.createElement('div');d.className='toast '+(ok?'toast-ok':'toast-err');d.textContent=m;c.appendChild(d);setTimeout(()=>d.remove(),3000)}

function setConn(ok){const d=document.getElementById('conn-dot'),t=document.getElementById('conn-text');d.className='conn-dot '+(ok?'conn-ok':'conn-err');t.textContent=ok?'已连接':'连接失败'}
function setStatus(s){const el=document.getElementById('state');if(s.running&&!s.paused){el.textContent='运行中';el.className='badge badge-green'}else if(s.paused){el.textContent='暂停';el.className='badge badge-yellow'}else{el.textContent='已停止';el.className='badge badge-gray'}document.getElementById('frame').textContent=s.frame||0;document.getElementById('episode').textContent=s.episode||0;document.getElementById('my-count').textContent=s.my_count!=null?s.my_count:'-';document.getElementById('enemy-count').textContent=s.enemy_count!=null?s.enemy_count:'-';document.getElementById('cluster').textContent=s.state_cluster||'-';const kgEl=document.getElementById('kg-status');if(s.kg_loaded){const f=s.kg_file||'';kgEl.textContent=f.split('/').pop().replace('.pkl','');kgEl.style.color='#4caf50'}else{kgEl.textContent='未加载';kgEl.style.color='#f44336'}const bufEl=document.getElementById('history-buffer');if(s.history_episodes!=null){bufEl.textContent=s.history_episodes+' eps / '+s.history_frames+' frames / '+s.history_capacity+' cap'}else{bufEl.textContent='--'}}
function setObs(o){if(o.error)return;document.getElementById('my-hp').textContent=o.my_total_hp||0;document.getElementById('enemy-hp').textContent=o.enemy_total_hp||0;document.getElementById('last-action').textContent=o.last_action||'-'}

async function refresh(){try{const s=await apiGet('/game/status');setStatus(s);setConn(true);if(!logTimer)startLogTimer();if(s.running){try{const o=await apiGet('/game/observation');setObs(o)}catch(e){}}}catch(e){setConn(false)}}

function startTimer(){stopTimer();const ms=parseInt(document.getElementById('refresh-interval').value)||2000;timer=setInterval(refresh,ms)}
function stopTimer(){if(timer){clearInterval(timer);timer=null}}
document.getElementById('auto-refresh').addEventListener('change',function(){if(this.checked)startTimer();else stopTimer()});
document.getElementById('refresh-interval').addEventListener('change',function(){if(document.getElementById('auto-refresh').checked)startTimer()});

const consoleBody=document.getElementById('console-body');
consoleBody.addEventListener('scroll',function(){const el=this;userScrolled=(el.scrollTop+el.clientHeight)<el.scrollHeight-30});
function scrollConsoleBottom(){consoleBody.scrollTop=consoleBody.scrollHeight;userScrolled=false}

function appendLogLine(log){
  if(log.seq!==undefined&&log.seq<=renderedMaxSeq)return;
  if(log.seq!==undefined&&log.seq>renderedMaxSeq)renderedMaxSeq=log.seq;
  const div=document.createElement('div');
  div.className='log-line';
  div.setAttribute('data-level',log.level||'info');
  div.setAttribute('data-source',log.source||'game');
  div.setAttribute('data-type',_extractType(log.message));
  if(_isFiltered(log)){div.style.display='none'}
  const lvCls=LV_COLORS[log.level]||'lv-info';
  const srcCls=SRC_CLS[log.source]||'src-game';
  div.innerHTML=`<span class="log-ts">${log.ts||''}</span><span class="log-src ${srcCls}">${(log.source||'').toUpperCase()}</span><span class="log-msg ${lvCls}">${escHtml(log.message||'')}</span>`;
  consoleBody.appendChild(div);
  while(consoleBody.childElementCount>200){
    var first=consoleBody.firstChild;
    if(first.style.display==='none'){consoleBody.removeChild(first)}
    else{var vc=0;for(var c=consoleBody.firstChild;c;c=c.nextSibling){if(c.style.display!=='none')vc++}if(vc<=200)break;consoleBody.removeChild(first)}
  }
  if(!userScrolled)scrollConsoleBottom();
}
function escHtml(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}

async function fetchLogs(){
  try{
    const r=await apiGet('/game/logs?after_seq='+latestSeq);
    const logs=r.logs||[];
    if(logs.length===0)return;
    if(consoleBody.querySelector('[style*="text-align:center"]'))consoleBody.innerHTML='';
    logs.forEach(l=>appendLogLine(l));
    if(r.latest_seq>latestSeq)latestSeq=r.latest_seq;
    document.getElementById('log-count').textContent=consoleBody.childElementCount+' 条';
  }catch(e){}
}

function startLogTimer(){stopLogTimer();logTimer=setInterval(fetchLogs,1500)}
function stopLogTimer(){if(logTimer){clearInterval(logTimer);logTimer=null}}

async function clearLogs(){try{await apiPost('/game/logs/clear',{})}catch(e){}consoleBody.innerHTML='<div style="text-align:center;color:#666;padding:40px 0">已清空</div>';latestSeq=0;renderedMaxSeq=0;document.getElementById('log-count').textContent='0 条'}

async function ctrl(cmd){try{await apiPost('/game/control',{command:cmd});toast('已发送: '+cmd);await refresh()}catch(e){toast('失败: '+e.message,false)}}
function toggleTheme(){const d=document.documentElement;d.classList.toggle('light');const isLight=d.classList.contains('light');document.getElementById('theme-btn').innerHTML=isLight?'&#9728;':'&#9790;';localStorage.setItem('live_theme',isLight?'light':'dark')}
function loadWindowPos(){var s=localStorage.getItem('live_win_pos');if(!s)return;try{var p=JSON.parse(s);document.getElementById('win-x').value=p.x||2600;document.getElementById('win-y').value=p.y||50;document.getElementById('win-w').value=p.w||640;document.getElementById('win-h').value=p.h||480}catch(e){}}
async function applyWindowPos(){var x=parseInt(document.getElementById('win-x').value)||50;var y=parseInt(document.getElementById('win-y').value)||50;var w=parseInt(document.getElementById('win-w').value)||640;var h=parseInt(document.getElementById('win-h').value)||480;localStorage.setItem('live_win_pos',JSON.stringify({x:x,y:y,w:w,h:h}));try{var r=await apiPost('/game/window_pos',{x:x,y:y,w:w,h:h});if(r.ok)toast('窗口位置已更新');else toast('更新失败: '+(r.error||''),false)}catch(e){toast('请求失败: '+e.message,false)}}
async function shutdownService(){stopTimer();stopLogTimer();setConn(false);latestSeq=0;renderedMaxSeq=0;consoleBody.innerHTML='<div style="text-align:center;color:#666;padding:40px 0">等待日志...</div>';document.getElementById('log-count').textContent='0 条';document.getElementById('conn-text').textContent='已停止';try{await fetch(API+'/game/autopilot',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:false})})}catch(e){}try{await fetch(API+'/game/shutdown',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})}catch(e){}toast('服务已停止')}

var epCurrentPage=1,epPerPage=10,epSearchTimer=null,allEpisodes=[];
function debounceSearch(){clearTimeout(epSearchTimer);epSearchTimer=setTimeout(function(){epCurrentPage=1;renderLocalEpisodes()},400)}
async function loadEpisodes(){
  try{
    var data=await apiGet('/game/episodes?page=1&per_page=999');
    var newEps=data.episodes||[];
    newEps.forEach(function(ep){
      var idx=allEpisodes.findIndex(function(e){return e.id===ep.id});
      if(idx>=0){
        if((!ep.plans||ep.plans.length===0)&&allEpisodes[idx].plans&&allEpisodes[idx].plans.length>0){
          ep.plans=allEpisodes[idx].plans;
        }
        allEpisodes[idx]=ep
      }else{allEpisodes.push(ep)}
    });
    var agentIds=[];
    newEps.forEach(function(ep){agentIds.push(ep.id)});
    if(agentIds.length>0){try{await apiPost('/game/episodes/ack',{ids:agentIds})}catch(e){}}
    renderLocalEpisodes();
  }catch(e){}
}
function exportEpData(epId,type){
  var ep=allEpisodes.find(function(e){return e.id===epId});
  if(!ep){toast('未找到对局 #'+epId,false);return}
  var data,filename;
  if(type==='frames'){
    data={id:ep.id,result:ep.result,score:ep.score,steps:ep.steps,timestamp:ep.timestamp,markov_flow:ep.markov_flow,events:ep.events};
    filename='episode_'+epId+'_frames.json';
  }else if(type==='plans'){
    data={id:ep.id,result:ep.result,plans:ep.plans||[]};
    filename='episode_'+epId+'_plans.json';
  }else{
    data=ep;filename='episode_'+epId+'_full.json';
  }
  var blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});
  var a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=filename;a.click();URL.revokeObjectURL(a.href);
  toast('已导出 '+filename);
}
function renderLocalEpisodes(){
  var sort=document.getElementById('ep-sort').value;
  var search=document.getElementById('ep-search').value.trim().toLowerCase();
  var filtered=allEpisodes.slice();
  if(search){filtered=filtered.filter(function(ep){return ep.result.toLowerCase().indexOf(search)>=0||ep.mode.toLowerCase().indexOf(search)>=0})}
  if(sort==='id_desc')filtered.sort(function(a,b){return b.id-a.id});
  else if(sort==='id_asc')filtered.sort(function(a,b){return a.id-b.id});
  else if(sort==='score_desc')filtered.sort(function(a,b){return(b.score||0)-(a.score||0)});
  else if(sort==='score_asc')filtered.sort(function(a,b){return(a.score||0)-(b.score||0)});
  var total=filtered.length;
  var start=(epCurrentPage-1)*epPerPage;
  var pageItems=filtered.slice(start,start+epPerPage);
  renderEpisodes({episodes:pageItems,total:total,page:epCurrentPage,per_page:epPerPage,total_pages:Math.max(1,Math.ceil(total/epPerPage))});
}
var epAutoTimer=null;
function startEpAutoRefresh(){stopEpAutoRefresh();epAutoTimer=setInterval(loadEpisodes,5000)}
function stopEpAutoRefresh(){if(epAutoTimer){clearInterval(epAutoTimer);epAutoTimer=null}}
async function clearEpisodes(){try{await apiPost('/game/episodes/clear');allEpisodes=[];renderEpisodes({episodes:[],total:0,page:1,per_page:epPerPage,total_pages:1});toast('对局记录已清空')}catch(e){toast('清空失败: '+e.message,false)}}

function renderEpisodes(data){
  var list=document.getElementById('episodes-list');
  var pgn=document.getElementById('ep-pagination');
  var cnt=document.getElementById('ep-count');
  var eps=data.episodes||[];
  cnt.textContent=data.total||0;
  if(eps.length===0){list.innerHTML='<div style="text-align:center;color:var(--tx2);padding:20px 0;font-size:12px">暂无对局记录</div>';pgn.innerHTML='';return}
  var html='';
  for(var i=0;i<eps.length;i++){
    var ep=eps[i];
    var isMulti=ep.mode==='multi_step';
    var badgeCls='ep-badge-interrupted';
    if(ep.result==='Win')badgeCls='ep-badge-win';
    else if(ep.result==='Loss')badgeCls='ep-badge-loss';
    else if(ep.result==='Dogfall')badgeCls='ep-badge-dogfall';
    var flowHtml='';
    if(ep.markov_flow&&ep.markov_flow.length>0){
      var maxShow=Math.min(ep.markov_flow.length,20);
      for(var j=0;j<maxShow;j++){
        var item=ep.markov_flow[j];
        var ev=(ep.events&&ep.events[j])?ep.events[j]:{event_type:'no_action'};
        var et=ev.event_type||'no_action';
        var bg,bd,tc;
        if(et==='kg_plan'){
          if(isMulti){bg='rgba(13,71,161,.18)';bd='2px dashed rgba(13,71,161,.5)';tc='#42a5f5'}
          else{bg='rgba(13,71,161,.12)';bd='1px solid rgba(13,71,161,.3)';tc='#64b5f6'}
        }else if(et==='kg_follow'){
          bg='rgba(27,94,32,.10)';bd='1px solid rgba(27,94,32,.25)';tc='#81c784';
        }else if(et==='diverge'){
          bg='rgba(245,127,23,.15)';bd='1px solid rgba(245,127,23,.3)';tc='#ffb74d';
        }else if(et==='fallback'){
          bg='rgba(183,28,28,.15)';bd='1px solid rgba(183,28,28,.3)';tc='#ef9a9a';
        }else if(et==='backup_switch'){
          bg='rgba(123,31,162,.15)';bd='1px solid rgba(123,31,162,.3)';tc='#ce93d8';
        }else{
          bg='var(--sf2)';bd='1px solid var(--bd)';tc='var(--tx2)';
        }
        var lbl={kg_plan:'规划',kg_follow:'跟随',diverge:'偏离',fallback:'回退',backup_switch:'备选',external:'外部',no_action:'-'}[et]||et;
        if(j>0)flowHtml+='<span class="ep-flow-arrow">&rarr;</span>';
        var sid=item[0]!=null?(Array.isArray(item[0])?'M('+item[0].join(',')+')':'S'+item[0]):'S?';
        var act=item[1]||'';
        if(act.startsWith('action_'))act=act.replace(/^action_/,'').substring(0,6);
        flowHtml+='<span class="ep-flow-item" style="background:'+bg+';border:'+bd+';color:'+tc+'" title="'+lbl+'">'+sid+'<span style="opacity:.7;margin-left:2px">'+escHtml(act)+'</span></span>';
      }
      if(ep.markov_flow.length>20)flowHtml+='<span class="ep-flow-item" style="color:var(--ylw);background:none;border:none">+'+(ep.markov_flow.length-20)+'</span>';
    }
    var scoreStr=ep.score!==undefined&&ep.score!==0?(ep.score>0?'+':'')+ep.score.toFixed(0):'-';
    html+='<div class="ep-item">';
    html+='<div class="ep-header"><span class="ep-title">#'+ep.id+'</span><span class="ep-badge '+badgeCls+'">'+(ep.result||'?')+'</span>';
    html+='<span class="ep-export-btns">';
    html+='<button onclick="exportEpData('+ep.id+',&#39;frames&#39;)" title="导出帧流">F</button>';
    html+='<button onclick="exportEpData('+ep.id+',&#39;plans&#39;)" title="导出推演详情">P</button>';
    html+='<button onclick="exportEpData('+ep.id+',&#39;all&#39;)" title="导出全部">A</button>';
    html+='</span></div>';
    html+='<div class="ep-meta">得分: '+scoreStr+' | 步数: '+ep.steps+' | '+(ep.mode==='multi_step'?'多步':'单步')+' '+(ep.match_mode?'('+ep.match_mode+')':'')+' | '+(ep.timestamp||'')+'</div>';
    if(flowHtml)html+='<div class="ep-flow">'+flowHtml+'</div>';
    if(ep.plans&&ep.plans.length>0){
      html+='<details class="ep-details"><summary>推演详情 ('+ep.plans.length+' 次规划)</summary>';
      for(var p=0;p<ep.plans.length;p++){
        var pl=ep.plans[p];
        var trigger=pl.trigger||'';
        var trigLbl={diverge:'偏离触发',exhausted:'用尽重规划',single_step:'单步规划'}[trigger]||trigger||pl.mode;
        html+='<details class="ep-plan-details"><summary>规划 #'+(p+1)+' — S'+pl.state_id+' ('+trigLbl+')</summary>';
        html+='<div class="ep-plan-content">';
        if(pl.beam_paths&&pl.beam_paths.length>0){
          for(var pi=0;pi<pl.beam_paths.length;pi++){
            var path=pl.beam_paths[pi];
            var chosen=path.chosen?'chosen':'';
            html+='<div class="ep-beam-path '+chosen+'">';
            html+='<span class="path-label">'+(path.chosen?'[选中] ':'')+path.rank+'</span>';
            html+='<span class="path-metrics"><span>CumP:'+(path.cum_prob*100).toFixed(1)+'%</span><span>'+(path.steps.length-1)+'步</span></span>';
            html+='<br>';
            for(var si=0;si<path.steps.length;si++){
              var st=path.steps[si];
              if(si===0){
                html+='<span class="ep-beam-step" style="font-weight:600">S'+st.state+'</span>';
              }else{
                html+='<span class="path-arrow">&rarr;</span>';
                if(st.action&&st.action!=='')html+='<span class="ep-beam-step" style="color:var(--ac)">'+st.action+'</span>';
                html+='<span class="ep-beam-step">S'+st.state+'</span>';
              }
            }
            html+='</div>';
          }
        }else if(pl.action_plan&&pl.action_plan.length>0){
          html+='<div class="ep-beam-path chosen">';
          html+='<span class="path-label">[选中] 1</span>';
          html+='<br>';
          for(var a=0;a<pl.action_plan.length;a++){
            if(a===0){
              html+='<span class="ep-beam-step" style="font-weight:600">S'+pl.planned_states[a]+'</span>';
            }else{
              html+='<span class="path-arrow">&rarr;</span>';
              html+='<span class="ep-beam-step" style="color:var(--ac)">'+pl.action_plan[a]+'</span>';
              html+='<span class="ep-beam-step">S'+pl.planned_states[a]+'</span>';
            }
          }
          html+='</div>';
        }
        if(pl.beam_results&&pl.beam_results.length>0){
          html+='<details style="margin-top:4px"><summary style="font-size:10px;color:var(--tx2);cursor:pointer">Beam Search ('+pl.beam_results.length+' 节点)</summary>';
          html+='<table class="ep-beam-table"><tr><th>Step</th><th>State</th><th>Action</th><th>Beam</th><th>WR</th><th>Quality</th><th>CumP</th></tr>';
          for(var b=0;b<pl.beam_results.length;b++){
            var br=pl.beam_results[b];
            var wrStr=(br.win_rate*100).toFixed(1)+'%';
            var qsStr=br.quality_score.toFixed(1);
            var cpStr=br.cumulative_probability.toFixed(4);
            html+='<tr><td>'+br.step+'</td><td>S'+br.state+'</td><td>'+(br.action||'-')+'</td><td>B'+br.beam_id+'</td><td>'+wrStr+'</td><td>'+qsStr+'</td><td>'+cpStr+'</td></tr>';
          }
          html+='</table></details>';
        }
        html+='</div>';
        html+='</details>';
      }
      html+='</details>';
    }
    html+='</div>';
  }
  list.innerHTML=html;
  var totalPages=data.total_pages||1;
  if(totalPages<=1){pgn.innerHTML='';return}
  var phtml='';
  for(var p=1;p<=totalPages;p++){
    if(p===epCurrentPage)phtml+='<button class="ep-page-btn active">'+p+'</button>';
    else phtml+='<button class="ep-page-btn" onclick="epCurrentPage='+p+';renderLocalEpisodes()">'+p+'</button>';
  }
  pgn.innerHTML=phtml;
}

async function init(){
  if(localStorage.getItem('live_theme')!=='dark'){document.documentElement.classList.add('light');document.getElementById('theme-btn').innerHTML='&#9728;'}
  try{await refreshAutopilotStatus()}catch(e){}
  loadFilterConfig();
  loadWindowPos();
  await refresh();
  startTimer();
  startLogTimer();
  fetchLogs();
  loadEpisodes();
}
init();
</script>
</body></html>"""
    return _html.replace("__PORT__", str(port))
