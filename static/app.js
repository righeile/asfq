/* Tile assembler front end.
 *
 * The canvas holds the whole layout in *mosaic pixel* coordinates and draws
 * tiles from their thumbnails, so dragging stays smooth with a hundred fields
 * open.  Only decisions travel to the server: after a drag, the tile's new
 * position is sent and the correlation refines it there.
 */

const $ = (sel) => document.querySelector(sel);
const api = async (url, body) => {
  const opt = body === undefined
    ? {}
    : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  const r = await fetch(url, opt);
  let j;
  try {
    j = await r.json();
  } catch (parseError) {
    // Say what actually went wrong rather than inventing a status code: a
    // malformed body is a different bug from a failed request.
    const body = await r.text().catch(() => '');
    throw new Error(r.ok
      ? `${url}: the server sent a malformed reply — ${parseError.message}\n${body.slice(0, 300)}`
      : `${url}: HTTP ${r.status}\n${body.slice(0, 300)}`);
  }
  if (!r.ok || j.error) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
};

const S = {
  sid: null, tiles: [], poses: [], thumbs: [], qc: null, stats: {},
  sel: new Set(), view: { ox: 0, oy: 0, z: 0.05 },
  drag: null, hover: -1, pxum: null, conflictPairs: [],
  undo: [], redo: [], lastPush: 0,
  tool: 'move', marquee: null, spaceHeld: false,
  ghost: 1,   // opacity of whatever is being moved; 1 = off
};

/* ------------------------------------------------------------------ undo
 * Every operation that moves a tile snapshots the layout first.  Nothing else
 * is recorded: the view, the selection and the parameters are not part of the
 * result, and putting them in the history would make ctrl-Z do something
 * different from "put the pieces back where they were".
 */
const HISTORY_LIMIT = 80;
const snapshotPoses = () => S.poses.map(p => ({ ...p }));

function pushHistory(label) {
  if (!S.poses.length) return;
  const now = Date.now();
  const top = S.undo[S.undo.length - 1];
  // Repeated nudges of the same kind collapse into one step, so a held arrow
  // key does not fill the history with single pixels.
  if (top && top.label === label && now - S.lastPush < 700) {
    S.lastPush = now;
  } else {
    S.undo.push({ poses: snapshotPoses(), label });
    if (S.undo.length > HISTORY_LIMIT) S.undo.shift();
    S.lastPush = now;
  }
  S.redo.length = 0;
  updateHistoryUI();
}

async function stepHistory(from, to, verb, noun) {
  if (!from.length) { $('#status').textContent = `nothing to ${noun}`; return; }
  const entry = from.pop();
  to.push({ poses: snapshotPoses(), label: entry.label });
  if (to.length > HISTORY_LIMIT) to.shift();
  S.poses = entry.poses.map(p => ({ ...p }));
  S.sel.clear();
  refreshIslandSelects();
  draw();
  updateHistoryUI();
  $('#status').textContent = `${verb}: ${entry.label}`;
  const r = await pushPoses(true);
  if (r && r.qc) { applyQC(r.qc); draw(); }
}

const undo = () => stepHistory(S.undo, S.redo, 'undid', 'undo');
const redo = () => stepHistory(S.redo, S.undo, 'redid', 'redo');

function updateHistoryUI() {
  const u = $('#btn-undo'), r = $('#btn-redo');
  if (!u) return;
  u.disabled = !S.undo.length;
  r.disabled = !S.redo.length;
  u.title = S.undo.length ? `undo ${S.undo[S.undo.length - 1].label}` : 'nothing to undo';
  r.title = S.redo.length ? `redo ${S.redo[S.redo.length - 1].label}` : 'nothing to redo';
}

/* ------------------------------------------------------------------ tabs */
document.querySelectorAll('.tab').forEach(b => b.onclick = () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
  document.querySelectorAll('.tabpane').forEach(x => x.classList.remove('active'));
  b.classList.add('active');
  $('#tab-' + b.dataset.tab).classList.add('active');
  if (b.dataset.tab === 'assemble') resize();
updateHistoryUI();
updateToolUI();
});

/* -------------------------------------------------------------- progress */
function showProgress(el, frac, msg) {
  el.classList.remove('hidden');
  el.querySelector('.bar').style.width = (100 * frac).toFixed(1) + '%';
  el.querySelector('span').textContent = msg || '';
}
function hideProgress(el) { el.classList.add('hidden'); }

async function runJob(jobId, el, label) {
  for (;;) {
    const j = await api('/api/job/' + jobId);
    showProgress(el, j.progress, `${label}: ${j.message}`);
    if (j.done) {
      hideProgress(el);
      if (j.error) throw new Error(j.error + (j.traceback ? '\n' + j.traceback.split('\n').slice(-4).join('\n') : ''));
      return j.result;
    }
    await new Promise(r => setTimeout(r, 350));
  }
}

/* ------------------------------------------------------------- filesystem
   The dialog is labkit's, so choosing a folder here works exactly as it does
   in the vesicle detector and the figure builder -- folders on the left, the
   images actually in them on the right. */
const IMAGE_EXTS = ['.czi', '.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'];

$('#btn-browse').onclick = async () => {
  const picked = await LK.browse({
    mode: 'dir', exts: IMAGE_EXTS, path: $('#folder').value,
    title: 'Folder of tiles', hint: 'One folder of overlapping tiles from one section.',
  });
  if (picked) $('#folder').value = picked;
};

/* ------------------------------------------------------------------ load */
$('#btn-load').onclick = async () => {
  const folder = $('#folder').value.trim();
  if (!folder) return alert('Pick a folder first.');
  try {
    const { job } = await api('/api/session/create', { folder, params: stitchParams() });
    const res = await runJob(job, $('#progress'), 'loading');
    S.sid = res.sid;
    await loadSession();
    tilesAreLoaded(res, folder);
  } catch (e) { hideProgress($('#progress')); alert(e.message); }
};

/* Everything that has to become true once a session has tiles in it.  Reopening
   a saved session arrives here by a different road and needs the same things
   said: without it the header still read "no folder loaded" over a restored
   mosaic, and every button that acts on tiles stayed disabled. */
function tilesAreLoaded(res, folder) {
  $('#status').textContent = `${res.n_tiles} tiles · `
    + `${res.px_um ? res.px_um.toFixed(3) + ' µm/px' : 'scale unknown'} · ${folder}`;
  ['btn-auto', 'btn-snap', 'btn-snapall', 'btn-snapislands', 'btn-mergeislands', 'btn-export', 'btn-savelayout',
   'btn-loadlayout', 'btn-suggest', 'btn-resolve'].forEach(id => { const b = $('#' + id); if (b) b.disabled = false; });
}

function stitchParams() {
  return {
    coarse_max_dim: +$('#p-coarse').value,
    min_ncc: +$('#p-ncc').value,
    min_peak_ratio: +$('#p-pr').value,
    min_overlap_frac: +$('#p-ov').value,
    rotation_search_deg: +$('#p-rot').value,
    neighbour_window: +$('#p-win').value,
  };
}

async function loadSession() {
  const j = await api('/api/session/' + S.sid);
  S.tiles = j.tiles; S.poses = j.poses; S.pxum = j.px_um; S.stats = j.stats || {};
  S.thumbs = S.tiles.map(t => {
    const im = new Image();
    im.onload = draw;
    im.src = `/api/session/${S.sid}/thumb/${t.index}.jpg`;
    return im;
  });
  S.sel.clear();
  S.undo.length = 0; S.redo.length = 0;
  updateHistoryUI();
  updateToolUI();
  refreshIslandSelects();
  fitView();
  draw();
}

/* ------------------------------------------------------------ auto-stitch */
$('#btn-auto').onclick = () => autoStitch(false);
$('#btn-resolve').onclick = () => autoStitch(true);

async function autoStitch(keepLocked) {
  try {
    pushHistory('auto-stitch');
    const { job } = await api(`/api/session/${S.sid}/autostitch`,
      { params: stitchParams(), keep_locked: keepLocked, rematch: !keepLocked });
    const res = await runJob(job, $('#progress'), 'auto-stitch');
    S.poses = res.poses; S.stats = res.stats;
    applyQC(res.qc);
    refreshIslandSelects();
    fitView(); draw();
    const st = res.stats;
    $('#status').textContent =
      `${st.island_sizes[0]} of ${S.tiles.length} tiles placed · ${st.n_islands} island(s) · ` +
      `${st.n_edges_used} usable pairs, ${st.n_proposals_rejected} placements refused`;
    if (st.merge_suggestions && st.merge_suggestions.length) showSuggestions(st.merge_suggestions);
  } catch (e) { hideProgress($('#progress')); alert(e.message); }
}

function applyQC(qc) {
  S.qc = {};
  S.conflictPairs = qc.conflicts || [];
  (qc.tiles || []).forEach(t => S.qc[t.index] = t);
  const bad = (qc.tiles || []).filter(t => t.n_conflicts > 0).length;
  const scores = (qc.tiles || []).map(t => t.score).filter(v => v !== null);
  scores.sort((a, b) => a - b);
  const med = scores.length ? scores[Math.floor(scores.length / 2)] : null;
  const islands = {};
  (qc.tiles || []).forEach(t => islands[t.island] = (islands[t.island] || 0) + 1);
  $('#qc-summary').innerHTML =
    `<b>${med === null ? '—' : med.toFixed(3)}</b> median agreement with neighbours\n` +
    `<b>${bad}</b> tile(s) contradicting a neighbour\n` +
    `islands: ${Object.entries(islands).map(([k, v]) => `${k}:${v}`).join('  ')}`;
}

/* ----------------------------------------------------------- suggestions */
$('#btn-suggest').onclick = async () => {
  try {
    const j = await api(`/api/session/${S.sid}/suggestions`, {});
    showSuggestions(j.suggestions);
  } catch (e) { alert(e.message); }
};

function showSuggestions(list) {
  const box = $('#suggestions');
  box.innerHTML = '';
  if (!list || !list.length) { box.innerHTML = '<div class="hint">No further merges proposed.</div>'; return; }
  list.forEach(s => {
    const el = document.createElement('div');
    el.className = 'suggestion';
    el.innerHTML =
      `island <b>${s.moving_island}</b> → island <b>${s.into_island}</b><br>` +
      `${Math.round(s.support_px)} px of agreeing tissue over ${s.n_agreeing_pairs} pair(s)<br>` +
      `proposed by ${s.proposed_by}<br>` +
      (s.n_conflicts
        ? `<span class="bad">would collide with ${s.n_conflicts} placement(s): ${(s.conflicting_tiles || []).slice(0, 4).join(', ')}${(s.conflicting_tiles || []).length > 4 ? '…' : ''}</span>`
        : '<span>no conflicts</span>');
    const b = document.createElement('button');
    b.textContent = s.n_conflicts ? 'Apply anyway' : 'Apply';
    b.onclick = async () => {
      pushHistory(`merge island ${s.moving_island}`);
      const r = await api(`/api/session/${S.sid}/apply_shift`, {
        island: s.moving_island, dx: s.dx, dy: s.dy, dtheta: s.dtheta, into_island: s.into_island,
      });
      S.poses = r.poses; applyQC(r.qc); refreshIslandSelects(); draw();
    };
    el.appendChild(b);
    box.appendChild(el);
  });
}

/* ------------------------------------------------------------- alignment */
$('#btn-snap').onclick = () => alignSelection();
$('#btn-snapall').onclick = () => snap(null);
$('#btn-snapislands').onclick = () => snapIslands();
$('#btn-mergeislands').onclick = () => mergeSelectedIslands();

/* Aligning a hand-dropped *island* is a different operation from aligning a
 * tile: the group's internal geometry is already solved, so it must move as one
 * rigid body rather than letting each member drift to its own local optimum. */
function alignSelection() {
  const sel = [...S.sel];
  if (!sel.length) return alert('Select a tile first (click it).');
  const islands = new Set(sel.map(i => S.poses[i].island));
  if ($('#chk-island').checked && islands.size === 1) return snapIsland([...islands][0]);
  return snap(sel);
}

async function snapIsland(island) {
  try {
    pushHistory(`align island ${island}`);
    await pushPoses();
    const { job } = await api(`/api/session/${S.sid}/snap_island`,
      { island, search_frac: +$('#p-search').value });
    const res = await runJob(job, $('#progress'), 'aligning island');
    S.poses = res.poses; applyQC(res.qc); refreshIslandSelects(); draw();
    const d = res.info;
    $('#status').textContent = d.ok
      ? `island ${island} moved by ${d.dx.toFixed(1)}, ${d.dy.toFixed(1)} px on ${d.n_voters} tile vote(s)` +
        (d.merged_into !== null && d.merged_into !== undefined
          ? ` — joined island ${d.merged_into}`
          : (d.n_conflicts ? ` — ${d.n_conflicts} conflict(s), left separate` : ''))
      : `could not align island: ${d.reason}`;
  } catch (e) { hideProgress($('#progress')); alert(e.message); }
}

/* The pass for after the hand work: every island keeps its own solved
 * geometry and moves as a whole, repeatedly, until nothing shifts. "Align
 * all" is the wrong tool once pieces have been placed by hand -- it nudges
 * tiles one at a time and pulls a correct island apart. */
async function snapIslands() {
  if (!S.sid) return;
  try {
    pushHistory('settle islands');
    await pushPoses();
    const { job } = await api(`/api/session/${S.sid}/snap_islands`,
      { search_frac: +$('#p-search').value, passes: 4 });
    const res = await runJob(job, $('#progress'), 'settling islands');
    S.poses = res.poses; applyQC(res.qc); refreshIslandSelects(); draw();
    const d = res.info || {};
    if (!d.ok) {
      $('#status').textContent = `nothing to settle: ${d.reason || 'no islands moved'}`;
      return;
    }
    const moved = Object.entries(d.moves || {})
      .map(([isl, m]) => `${isl}: ${m.dx.toFixed(1)}, ${m.dy.toFixed(1)} px`).join(' · ');
    const skipped = Object.keys(d.skipped || {}).length;
    $('#status').textContent =
      `settled ${d.n_settled} of ${d.n_islands} island(s) in ${d.passes_used} pass(es)` +
      `${d.converged ? '' : ' — still moving, run it again'}` +
      `${moved ? ' — ' + moved : ''}` +
      `${skipped ? ` — ${skipped} had nothing to align against` : ''}`;
  } catch (e) { hideProgress($('#progress')); alert(e.message); }
}

/* The override for when you can see where a piece goes and the evidence
 * cannot. Nothing moves -- the tiles stay exactly where you put them and only
 * their island label changes, so conflicts stay visible in the quality check
 * rather than being argued away by the decision. */
async function mergeSelectedIslands() {
  if (!S.sid) return;
  const sel = [...S.sel];
  const islands = [...new Set(sel.map(i => S.poses[i].island))];
  if (islands.length < 2) {
    return alert('Select a tile in each piece first — at least two islands.\n' +
                 'Click one tile, then shift-click a tile in the other island.');
  }
  try {
    pushHistory(`merge islands ${islands.join(', ')}`);
    await pushPoses();
    const r = await api(`/api/session/${S.sid}/merge_islands`, { islands });
    S.poses = r.poses; applyQC(r.qc); refreshIslandSelects(); draw();
    const d = r.info;
    $('#status').textContent =
      `merged island(s) ${d.merged.join(', ')} into ${d.target} — ` +
      `${d.n_tiles_moved} tile(s) relabelled, none moved`;
  } catch (e) { alert(e.message); }
}

async function snap(indices) {
  if (!S.sid) return;
  if (indices && !indices.length) return alert('Select a tile first (click it).');
  try {
    pushHistory(indices ? `align ${indices.length} tile(s)` : 'align all');
    await pushPoses();
    const { job } = await api(`/api/session/${S.sid}/snap`,
      { indices, search_frac: +$('#p-search').value, passes: 2 });
    const res = await runJob(job, $('#progress'), 'aligning');
    S.poses = res.poses; applyQC(res.qc); draw();
    if (res.detail && res.detail.length) {
      const ok = res.detail.filter(d => d.ok);
      const bad = res.detail.filter(d => !d.ok);
      if (res.detail.length === 1) {
        const d = res.detail[0];
        $('#status').textContent = d.ok
          ? `moved by ${d.dx.toFixed(1)}, ${d.dy.toFixed(1)} px to agree with ${d.votes.length} neighbour(s)`
          : `could not align: ${d.reason}`;
      } else {
        const shift = ok.length
          ? Math.max(...ok.map(d => Math.hypot(d.dx, d.dy))).toFixed(1)
          : '0';
        $('#status').textContent =
          `aligned ${ok.length} of ${res.detail.length} tile(s), largest correction ${shift} px` +
          (bad.length ? ` — ${bad.length} left alone: ${bad[0].reason}` : '');
      }
    }
  } catch (e) { hideProgress($('#progress')); alert(e.message); }
}

async function pushPoses(withQC) {
  if (!S.sid) return null;
  const payload = S.poses.map((p, i) => ({ index: i, ...p }));
  return api(`/api/session/${S.sid}/poses`, { poses: payload, qc: !!withQC });
}

/* ------------------------------------------------------------ export etc */
$('#btn-export').onclick = async () => {
  try {
    await pushPoses();
    const { job } = await api(`/api/session/${S.sid}/export`, {
      scale: +$('#e-scale').value, island: $('#e-island').value,
      mode: $('#e-mode').value, out_dir: $('#e-dir').value.trim() || null,
    });
    const r = await runJob(job, $('#progress'), 'exporting');
    $('#export-out').innerHTML =
      `<b>mosaic</b> ${r.mosaic}\n${r.shape[1]} × ${r.shape[0]} px` +
      (r.px_um ? `, ${r.px_um.toFixed(3)} µm/px` : '') +
      `\n<b>layout</b> ${r.layout}\n<b>quality</b> ${r.quality}`;
  } catch (e) { hideProgress($('#progress')); alert(e.message); }
};

$('#btn-savelayout').onclick = async () => {
  const path = prompt('Save layout to:', '');
  if (path === null) return;
  await pushPoses();
  try {
    const r = await api(`/api/session/${S.sid}/save_layout`, { path: path || null });
    $('#export-out').textContent = `saved ${r.path}\n       ${r.csv}`;
  } catch (e) { alert(e.message); }
};

$('#btn-loadlayout').onclick = async () => {
  const path = prompt('Load layout from (.json):', '');
  if (!path) return;
  try {
    pushHistory('load layout');
    const r = await api(`/api/session/${S.sid}/load_layout`, { path });
    S.poses = r.poses; applyQC(r.qc); refreshIslandSelects(); fitView(); draw();
    $('#status').textContent = `restored ${r.n_restored} tile positions`;
  } catch (e) { alert(e.message); }
};

$('#btn-lock').onclick = () => { toggleLock(); };
$('#btn-fit').onclick = () => { fitView(); draw(); };

$('#btn-sess-save').onclick = () => saveSession();
$('#btn-sess-open').onclick = () => openSession($('#sess-list').value);
$('#btn-sess-del').onclick = async () => {
  const name = $('#sess-list').value;
  if (!name) return sessionNote('pick a session first', true);
  await api('/api/sessions/delete', { name });
  sessionNote(`deleted "${name}"`);
  await refreshSessions();
};
refreshSessions();

$('#p-ghost').oninput = () => {
  S.ghost = Math.max(0.05, Math.min(1, (+$('#p-ghost').value || 100) / 100));
  $('#ghost-note').textContent = S.ghost >= 1
    ? 'Off. Turn this down and whatever you are moving — a tile, or a whole island — is laid over '
      + 'what is under it so both sets of tissue show at once and you can line them up directly. '
      + 'Blank slide stops covering anything; only the stained tissue of each is drawn, so the two '
      + 'walls cross where they agree.'
    : `Both layers shown, the moving one at ${Math.round(S.ghost * 100)}%. Line the two walls up `
      + 'until they run as one; arrow keys nudge, and the frame and name stay solid so it can '
      + 'still be grabbed.';
  draw();
};
$('#tool-move').onclick = () => setTool('move');
$('#tool-rect').onclick = () => setTool('rect');
$('#tool-lasso').onclick = () => setTool('lasso');
$('#btn-selall').onclick = () => {
  applySelection(S.poses.map((p, i) => i).filter(i => S.poses[i].placed), 'replace');
  draw();
};
$('#btn-selnone').onclick = () => { S.sel.clear(); updateToolUI(); draw(); };
$('#btn-undo').onclick = () => undo();
$('#btn-redo').onclick = () => redo();

function toggleLock() {
  if (!S.sel.size) return;
  pushHistory('lock');
  const anyUnlocked = [...S.sel].some(i => !S.poses[i].locked);
  S.sel.forEach(i => S.poses[i].locked = anyUnlocked);
  pushPoses(); draw();
}

function refreshIslandSelects() {
  const ids = [...new Set(S.poses.map(p => p.island))].sort((a, b) => a - b);
  const counts = {};
  S.poses.forEach(p => counts[p.island] = (counts[p.island] || 0) + 1);
  const opts = ids.map(i => `<option value="${i}">${i} (${counts[i]} tiles)</option>`).join('');
  $('#e-island').innerHTML = '<option value="all">all</option>' + opts;
  // Every pane that can measure an island: the aorta wall, and each organ.
  document.querySelectorAll('#w-island, .tabpane[data-organ] select[id$="-island"]')
    .forEach(sel => { sel.innerHTML = opts; });
}

/* ==================================================================== canvas */
const cv = $('#canvas');
const ctx = cv.getContext('2d');

function resize() {
  const r = cv.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.max(1, Math.round(r.width * dpr));
  cv.height = Math.max(1, Math.round(r.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}
window.addEventListener('resize', resize);

const w2s = (x, y) => [(x - S.view.ox) * S.view.z, (y - S.view.oy) * S.view.z];
const s2w = (x, y) => [x / S.view.z + S.view.ox, y / S.view.z + S.view.oy];

function bounds() {
  if (!S.tiles.length) return [0, 0, 1, 1];
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  S.poses.forEach((p, i) => {
    const t = S.tiles[i];
    x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y);
    x1 = Math.max(x1, p.x + t.width); y1 = Math.max(y1, p.y + t.height);
  });
  return [x0, y0, x1, y1];
}

function fitView() {
  const r = cv.parentElement.getBoundingClientRect();
  if (r.width < 4 || r.height < 4) {
    // The pane has not been laid out yet -- fitting now would set zoom to zero
    // and the canvas would render nothing until the next window resize.
    requestAnimationFrame(() => { fitView(); draw(); });
    return;
  }
  const [x0, y0, x1, y1] = bounds();
  const z = Math.min(r.width / Math.max(x1 - x0, 1), r.height / Math.max(y1 - y0, 1)) * 0.94;
  if (!isFinite(z) || z <= 0) return;
  S.view.z = z;
  S.view.ox = x0 - (r.width / z - (x1 - x0)) / 2;
  S.view.oy = y0 - (r.height / z - (y1 - y0)) / 2;
}

/* The canvas paints with the stylesheet's colours, so the theme is defined in
   one place.  Read once and cached: qcColour runs per tile per frame, and
   getComputedStyle forces a style resolve every time it is called. */
const PALETTE = (() => {
  const css = getComputedStyle(document.documentElement);
  const get = (n, fallback) => (css.getPropertyValue(n).trim() || fallback);
  return {
    good: get('--good', '#7f8792'),
    warn: get('--warn', '#c8735f'),
    bad: get('--bad', '#ff5d5d'),
    unknown: get('--unknown', '#5a616b'),
    lock: get('--lock', '#b9c1cb'),
    sel: get('--sel', '#f2f5f8'),
    accent: get('--accent', '#e5484d'),
    canvas: get('--canvas', '#191b1f'),
    tileBlank: get('--tile-blank', '#2a2e34'),
    ink: get('--ink', '#dfe3e8'),
    dim: get('--dim', '#939aa4'),
  };
})();

function qcColour(i) {
  const q = S.qc && S.qc[i];
  if (!q) return PALETTE.unknown;
  if (q.n_conflicts > 0) return PALETTE.bad;
  if (q.score === null) return PALETTE.unknown;
  if (q.score >= 0.7) return PALETTE.good;
  if (q.score >= 0.45) return PALETTE.warn;
  return PALETTE.bad;
}

function draw() {
  const r = cv.parentElement.getBoundingClientRect();
  if (S.tiles.length && (!isFinite(S.view.z) || S.view.z <= 0)) fitView();
  ctx.clearRect(0, 0, r.width, r.height);
  ctx.fillStyle = PALETTE.canvas;
  ctx.fillRect(0, 0, r.width, r.height);
  if (!S.tiles.length) {
    ctx.fillStyle = PALETTE.dim;
    ctx.font = '13px -apple-system, sans-serif';
    ctx.fillText('Load a folder of tiles to begin.', 18, 28);
    return;
  }

  // What is being moved: the drag's own set while a drag is running, which is
  // the whole island when one is being dragged, and the selection otherwise.
  // It draws last, so it is the thing you see *through* rather than the thing
  // hidden underneath.
  const moving = S.drag && S.drag.kind === 'tiles' && S.drag.idx.length
    ? new Set(S.drag.idx) : S.sel;
  const order = S.tiles.map((_, i) => i)
    .sort((a, b) => (moving.has(a) ? 1 : 0) - (moving.has(b) ? 1 : 0));

  // Laying one tile over another and fading it does nothing here, and the
  // first version of this control did exactly that: brightfield tiles are
  // white with a little dark tissue on them, and white over white is white at
  // any opacity.  Measured over the neighbouring tile, dropping the moving
  // tile to 40 % moved the pixels underneath it by 14 of 255 -- the tile
  // dimmed against the dark stage and hid its neighbour just as completely.
  //
  // What matters in these images is only where the tissue is dark, so the
  // moving tile is multiplied into what is under it instead: white stops
  // covering anything, and both walls stay their own darkness and both stay
  // visible, crossing where they agree.  Multiplying over the dark stage would
  // sink the tile into it, so its footprint gets a white sheet first -- which
  // the tiles already placed then draw over, leaving the sheet only where
  // nothing else is.
  const lightTable = S.ghost < 1 && moving.size > 0;
  const place = (i) => {
    const t = S.tiles[i], p = S.poses[i];
    const [sx, sy] = w2s(p.x, p.y);
    const w = t.width * S.view.z, h = t.height * S.view.z;
    if (sx > r.width || sy > r.height || sx + w < 0 || sy + h < 0) return null;
    ctx.save();
    if (p.theta) {
      ctx.translate(sx + w / 2, sy + h / 2);
      ctx.rotate(p.theta * Math.PI / 180);
      ctx.translate(-w / 2, -h / 2);
    } else {
      ctx.translate(sx, sy);
    }
    return { w, h };
  };
  if (lightTable) {
    moving.forEach(i => {
      const box = place(i);
      if (!box) return;
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, box.w, box.h);
      ctx.restore();
    });
  }

  order.forEach(i => {
    const t = S.tiles[i], p = S.poses[i], im = S.thumbs[i];
    const [sx, sy] = w2s(p.x, p.y);
    const w = t.width * S.view.z, h = t.height * S.view.z;
    if (sx > r.width || sy > r.height || sx + w < 0 || sy + h < 0) return;
    ctx.save();
    if (p.theta) {
      ctx.translate(sx + w / 2, sy + h / 2);
      ctx.rotate(p.theta * Math.PI / 180);
      ctx.translate(-w / 2, -h / 2);
    } else {
      ctx.translate(sx, sy);
    }
    // Only the picture is blended.  The frame and the name stay solid, or a
    // faded tile becomes hard to find and impossible to keep hold of.
    const ghosted = lightTable && moving.has(i);
    if (ghosted) {
      ctx.globalCompositeOperation = 'multiply';
      ctx.globalAlpha = S.ghost;
    }
    if (im && im.complete && im.naturalWidth) {
      ctx.drawImage(im, 0, 0, w, h);
    } else {
      ctx.fillStyle = PALETTE.tileBlank;
      ctx.fillRect(0, 0, w, h);
    }
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
    const selected = S.sel.has(i);
    ctx.lineWidth = selected ? 3 : (S.hover === i ? 2 : 1);
    ctx.strokeStyle = selected ? PALETTE.sel : qcColour(i);
    if (p.locked) { ctx.setLineDash([6, 4]); ctx.strokeStyle = selected ? PALETTE.sel : PALETTE.lock; }
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
    ctx.setLineDash([]);
    if (selected || S.hover === i || S.view.z * t.width > 320) {
      ctx.fillStyle = 'rgba(22,24,28,.78)';
      const label = S.tiles[i].name + (p.locked ? ' 🔒' : '');
      ctx.font = '11px ui-monospace, Menlo, monospace';
      const tw = ctx.measureText(label).width + 8;
      ctx.fillRect(2, 2, tw, 15);
      ctx.fillStyle = PALETTE.ink;
      ctx.fillText(label, 6, 13);
    }
    ctx.restore();
  });

  // Conflicting pairs: a line between the two tiles that disagree.
  ctx.strokeStyle = PALETTE.bad;
  ctx.lineWidth = 1.5;
  S.conflictPairs.forEach(c => {
    const a = S.poses[c.i], b = S.poses[c.j];
    if (!a || !b) return;
    const [ax, ay] = w2s(a.x + S.tiles[c.i].width / 2, a.y + S.tiles[c.i].height / 2);
    const [bx, by] = w2s(b.x + S.tiles[c.j].width / 2, b.y + S.tiles[c.j].height / 2);
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
  });

  // The region being drawn, and what it would take.
  if (S.marquee) {
    const m = S.marquee;
    const prev = new Set(m.preview || []);
    ctx.save();
    ctx.lineWidth = 2;
    ctx.strokeStyle = PALETTE.sel;
    prev.forEach(i => {
      const q = tileCorners(i).map(([x, y]) => w2s(x, y));
      ctx.beginPath();
      q.forEach(([sx, sy], k) => (k ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy)));
      ctx.closePath();
      ctx.stroke();
    });
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = PALETTE.accent;
    ctx.fillStyle = 'rgba(229,72,77,.14)';
    ctx.beginPath();
    if (m.kind === 'rect') {
      const [x0, y0] = w2s(m.box[0], m.box[1]);
      const [x1, y1] = w2s(m.box[2], m.box[3]);
      ctx.rect(x0, y0, x1 - x0, y1 - y0);
    } else {
      m.points.forEach(([x, y], k) => {
        const [sx, sy] = w2s(x, y);
        k ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy);
      });
      ctx.closePath();
    }
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  const sel = [...S.sel];
  $('#overlay-info').textContent =
    `zoom ${(S.view.z * 100).toFixed(1)}%  ·  ` +
    `${S.tool}  ·  ` +
    (S.marquee ? `${(S.marquee.preview || []).length} tile(s) in the region` :
     sel.length === 1
      ? `${S.tiles[sel[0]].name}  x=${S.poses[sel[0]].x.toFixed(0)} y=${S.poses[sel[0]].y.toFixed(0)} θ=${S.poses[sel[0]].theta.toFixed(2)}° island ${S.poses[sel[0]].island}`
      : `${sel.length || 'no'} tile(s) selected`);
}

/* ================================================= selecting many at once
 * Both selection tools work in mosaic coordinates, so the region a user drew
 * keeps meaning the same thing if the view is zoomed afterwards.
 */

function tileCorners(i) {
  const t = S.tiles[i], p = S.poses[i];
  const cx = p.x + t.width / 2, cy = p.y + t.height / 2;
  const hw = t.width / 2, hh = t.height / 2;
  const pts = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]];
  if (!p.theta) return pts.map(([dx, dy]) => [cx + dx, cy + dy]);
  const a = p.theta * Math.PI / 180, c = Math.cos(a), sn = Math.sin(a);
  return pts.map(([dx, dy]) => [cx + dx * c - dy * sn, cy + dx * sn + dy * c]);
}

function tileBBox(i) {
  const q = tileCorners(i);
  const xs = q.map(v => v[0]), ys = q.map(v => v[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

function pointInPolygon(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-12) + xi) inside = !inside;
  }
  return inside;
}

/* Which tiles does the drawn region pick up?
 *
 * The rectangle takes anything it touches, which is what a marquee is expected
 * to do.  The lasso takes a tile when the tile's centre or any of its corners is
 * inside the loop -- a lasso is usually drawn round a cluster rather than
 * precisely along tile edges, and requiring full enclosure would mean nothing is
 * ever caught when the pieces are larger than the gesture.
 *
 * Holding alt while releasing requires *full* enclosure instead, for the times
 * you mean exactly the pieces inside the line and nothing else.
 */
function tilesInRegion(region, strict) {
  const out = [];
  for (let i = 0; i < S.tiles.length; i++) {
    if (!S.poses[i].placed) continue;
    if (region.kind === 'rect') {
      const [x0, y0, x1, y1] = region.box;
      const [bx0, by0, bx1, by1] = tileBBox(i);
      const hit = strict
        ? (bx0 >= x0 && by0 >= y0 && bx1 <= x1 && by1 <= y1)
        : !(bx1 < x0 || bx0 > x1 || by1 < y0 || by0 > y1);
      if (hit) out.push(i);
    } else {
      const poly = region.points;
      if (poly.length < 3) continue;
      const q = tileCorners(i);
      const cx = (q[0][0] + q[2][0]) / 2, cy = (q[0][1] + q[2][1]) / 2;
      const inside = q.map(([x, y]) => pointInPolygon(x, y, poly));
      const hit = strict
        ? inside.every(Boolean)
        : (pointInPolygon(cx, cy, poly) || inside.some(Boolean));
      if (hit) out.push(i);
    }
  }
  return out;
}

const strictSelection = () => !!($('#chk-enclosed') && $('#chk-enclosed').checked);

function applySelection(indices, mode) {
  if (mode === 'add') indices.forEach(i => S.sel.add(i));
  else if (mode === 'subtract') indices.forEach(i => S.sel.delete(i));
  else { S.sel.clear(); indices.forEach(i => S.sel.add(i)); }
  updateToolUI();
}

function setTool(name) {
  S.tool = name;
  S.marquee = null;
  updateToolUI();
  draw();
}

function updateToolUI() {
  ['move', 'rect', 'lasso'].forEach(t => {
    const b = $('#tool-' + t);
    if (b) b.classList.toggle('on', S.tool === t);
  });
  const el = $('#sel-count');
  if (!el) return;
  if (S.marquee) {
    // While a region is being drawn, show what it would take, not what is
    // currently selected -- the count you want is the one you are aiming at.
    const k = (S.marquee.preview || []).length;
    const verb = S.marquee.mode === 'subtract' ? 'to remove'
      : S.marquee.mode === 'add' ? 'to add' : 'under the cursor';
    el.textContent = `${k} tile${k === 1 ? '' : 's'} ${verb}`;
    return;
  }
  const n = S.sel.size;
  el.textContent = n ? `${n} tile${n === 1 ? '' : 's'} selected` : 'nothing selected';
}

const ts_name = (i) => (S.tiles[i] ? S.tiles[i].name : `tile ${i}`);

function tileAt(wx, wy) {
  for (let k = S.tiles.length - 1; k >= 0; k--) {
    const i = k, t = S.tiles[i], p = S.poses[i];
    let lx = wx - p.x - t.width / 2, ly = wy - p.y - t.height / 2;
    if (p.theta) {
      const a = -p.theta * Math.PI / 180;
      const rx = lx * Math.cos(a) - ly * Math.sin(a);
      const ry = lx * Math.sin(a) + ly * Math.cos(a);
      lx = rx; ly = ry;
    }
    if (Math.abs(lx) <= t.width / 2 && Math.abs(ly) <= t.height / 2) return i;
  }
  return -1;
}

cv.addEventListener('mousedown', ev => {
  const r = cv.getBoundingClientRect();
  const [wx, wy] = s2w(ev.clientX - r.left, ev.clientY - r.top);
  const hit = tileAt(wx, wy);

  // Space or the middle button always pans, whichever tool is active, so
  // switching back to the move tool is never needed just to look somewhere else.
  if (S.spaceHeld || ev.button === 1) {
    S.drag = { kind: 'pan', x: ev.clientX, y: ev.clientY, ox: S.view.ox, oy: S.view.oy };
    return;
  }

  if (S.tool !== 'move' && !(hit >= 0 && S.sel.has(hit))) {
    // In a selection tool, pressing anywhere starts a region -- except on a tile
    // that is already selected, which drags the selection instead, so a marquee
    // and the move that usually follows it do not need a mode switch between.
    const mode = ev.shiftKey ? 'add' : (ev.altKey ? 'subtract' : 'replace');
    S.marquee = {
      kind: S.tool === 'rect' ? 'rect' : 'lasso',
      mode, start: [wx, wy], box: [wx, wy, wx, wy], points: [[wx, wy]],
      base: new Set(S.sel), preview: [],
    };
    S.drag = { kind: 'marquee' };
    draw();
    return;
  }

  if (hit < 0) {
    S.drag = { kind: 'pan', x: ev.clientX, y: ev.clientY, ox: S.view.ox, oy: S.view.oy };
    if (!ev.shiftKey) { S.sel.clear(); updateToolUI(); draw(); }
    return;
  }
  if (ev.shiftKey && !$('#chk-island').checked) {
    const had = S.sel.has(hit);
    had ? S.sel.delete(hit) : S.sel.add(hit);
    $('#status').textContent =
      `${ts_name(hit)} ${had ? 'removed from' : 'added to'} the selection — ${S.sel.size} now selected`;
  } else if (!S.sel.has(hit)) {
    S.sel.clear(); S.sel.add(hit);
  }
  const island = (ev.shiftKey || $('#chk-island').checked) ? S.poses[hit].island : null;
  const moving = island !== null
    ? S.poses.map((p, i) => i).filter(i => S.poses[i].island === island)
    : [...S.sel];
  S.drag = {
    kind: 'tiles', x: ev.clientX, y: ev.clientY,
    idx: moving.filter(i => !S.poses[i].locked),
    start: moving.map(i => ({ i, x: S.poses[i].x, y: S.poses[i].y })),
    before: snapshotPoses(), moved: false,
  };
  updateToolUI();
  draw();
});

window.addEventListener('mousemove', ev => {
  const r = cv.getBoundingClientRect();
  if (!S.drag) {
    if (ev.target === cv) {
      const [wx, wy] = s2w(ev.clientX - r.left, ev.clientY - r.top);
      const h = tileAt(wx, wy);
      if (h !== S.hover) {
        S.hover = h;
        cv.style.cursor = S.spaceHeld ? 'grab'
          : S.tool !== 'move' && !(h >= 0 && S.sel.has(h)) ? 'crosshair'
          : h >= 0 ? 'move' : 'default';
        draw();
      }
    }
    return;
  }
  if (S.drag.kind === 'pan') {
    S.view.ox = S.drag.ox - (ev.clientX - S.drag.x) / S.view.z;
    S.view.oy = S.drag.oy - (ev.clientY - S.drag.y) / S.view.z;
    draw(); return;
  }
  if (S.drag.kind === 'marquee' && S.marquee) {
    const m = S.marquee;
    const [wx, wy] = s2w(ev.clientX - r.left, ev.clientY - r.top);
    if (m.kind === 'rect') {
      m.box = [Math.min(m.start[0], wx), Math.min(m.start[1], wy),
               Math.max(m.start[0], wx), Math.max(m.start[1], wy)];
    } else {
      const last = m.points[m.points.length - 1];
      // Thin the path: a point every couple of screen pixels is plenty and keeps
      // the point-in-polygon test cheap on a long gesture.
      if (Math.hypot(wx - last[0], wy - last[1]) * S.view.z > 2) m.points.push([wx, wy]);
    }
    m.preview = tilesInRegion(m, strictSelection());
    updateToolUI();
    draw();
    return;
  }
  const dx = (ev.clientX - S.drag.x) / S.view.z;
  const dy = (ev.clientY - S.drag.y) / S.view.z;
  if (!S.drag.moved && (Math.abs(dx) > 0 || Math.abs(dy) > 0)) {
    // Recorded on the first movement, not on mousedown, so a plain click to
    // select a tile does not leave an empty step in the history.
    S.drag.moved = true;
    S.undo.push({ poses: S.drag.before, label: 'drag' });
    if (S.undo.length > HISTORY_LIMIT) S.undo.shift();
    S.redo.length = 0;
    S.lastPush = Date.now();
    updateHistoryUI();
  }
  S.drag.start.forEach(s => {
    if (S.poses[s.i].locked) return;
    S.poses[s.i].x = s.x + dx;
    S.poses[s.i].y = s.y + dy;
  });
  draw();
});

window.addEventListener('mouseup', async (ev) => {
  if (S.drag && S.drag.kind === 'marquee' && S.marquee) {
    const m = S.marquee;
    const strict = strictSelection();
    let picked = tilesInRegion(m, strict);
    if (!picked.length && m.kind === 'lasso' && m.points.length < 3) {
      // A click rather than a gesture: behave like clicking the tile.
      const hit = tileAt(m.start[0], m.start[1]);
      if (hit >= 0) picked = [hit];
    }
    S.sel = new Set(m.base);
    applySelection(picked, m.mode);
    S.marquee = null; S.drag = null;
    const verb = m.mode === 'subtract' ? 'removed from the selection'
      : m.mode === 'add' ? 'added to the selection' : 'selected';
    $('#status').textContent =
      `${picked.length} tile(s) ${verb} — ${S.sel.size} now selected` +
      (strict ? ' (fully enclosed only)' : '');
    updateToolUI();
    draw();
    return;
  }
  if (S.drag && S.drag.kind === 'tiles' && S.drag.idx.length) {
    const r = await pushPoses(true);
    if (r && r.qc) applyQC(r.qc);
    draw();
  }
  S.drag = null;
});

cv.addEventListener('wheel', ev => {
  ev.preventDefault();
  const r = cv.getBoundingClientRect();
  const mx = ev.clientX - r.left, my = ev.clientY - r.top;
  const [wx, wy] = s2w(mx, my);
  const f = Math.exp(-ev.deltaY * 0.0016);
  S.view.z = Math.min(4, Math.max(0.002, S.view.z * f));
  S.view.ox = wx - mx / S.view.z;
  S.view.oy = wy - my / S.view.z;
  draw();
}, { passive: false });

window.addEventListener('keydown', ev => {
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) return;
  if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === 'z') {
    ev.preventDefault();
    ev.shiftKey ? redo() : undo();
    return;
  }
  if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === 'y') {
    ev.preventDefault(); redo(); return;
  }
  if (!$('#tab-assemble').classList.contains('active')) return;
  if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === 'a') {
    ev.preventDefault();
    applySelection(S.poses.map((p, i) => i).filter(i => S.poses[i].placed), 'replace');
    draw();
    return;
  }
  if (ev.code === 'Space' && !S.spaceHeld) {
    S.spaceHeld = true; cv.style.cursor = 'grab'; ev.preventDefault(); return;
  }
  const step = ev.shiftKey ? 20 : 1;
  let handled = true;
  switch (ev.key) {
    case 'a': alignSelection(); break;
    case 'A': snap(null); break;
    case 'l': case 'L': toggleLock(); break;
    case 'f': case 'F': fitView(); draw(); break;
    case 'i': case 'I': $('#chk-island').checked = !$('#chk-island').checked; break;
    case 'Escape': S.sel.clear(); S.marquee = null; setTool('move'); break;
    case 'v': case 'V': setTool('move'); break;
    case 'r': case 'R': setTool('rect'); break;
    case 'q': case 'Q': setTool('lasso'); break;
    case 'e': case 'E':
      // grow the selection to every tile in the islands already selected
      applySelection([...new Set([...S.sel].map(i => S.poses[i].island))]
        .flatMap(isl => S.poses.map((p, i) => i).filter(i => S.poses[i].island === isl)), 'add');
      draw();
      break;
    case '[': rotateSel(ev.shiftKey ? -5 : -0.5); break;
    case ']': rotateSel(ev.shiftKey ? 5 : 0.5); break;
    case 'ArrowLeft': nudge(-step, 0); break;
    case 'ArrowRight': nudge(step, 0); break;
    case 'ArrowUp': nudge(0, -step); break;
    case 'ArrowDown': nudge(0, step); break;
    default: handled = false;
  }
  if (handled) ev.preventDefault();
});

window.addEventListener('keyup', ev => {
  if (ev.code === 'Space') { S.spaceHeld = false; cv.style.cursor = 'default'; }
});

function nudge(dx, dy) {
  if (!S.sel.size) return;
  pushHistory('nudge');
  S.sel.forEach(i => { if (!S.poses[i].locked) { S.poses[i].x += dx; S.poses[i].y += dy; } });
  draw(); pushPoses();
}
function rotateSel(d) {
  if (!S.sel.size) return;
  pushHistory('rotate');
  S.sel.forEach(i => { if (!S.poses[i].locked) S.poses[i].theta += d; });
  draw(); pushPoses();
}

/* ============================================================ wall analysis */
/* One reading of the wall controls, for the single section and for the tray.
 * The batch used to send none of these and silently run on defaults, which is
 * survivable for a smoothing length and not for `media_edge_rule`: it decides
 * what "the media" means, and a tray answering that differently from the
 * section you set it on is a cohort measured two ways. */
function wallParams() {
  const params = {
    analysis_px_um: +$('#w-apx').value,
    media_quantile: +$('#w-mq').value,
    smooth_centerline_um: +$('#w-sm').value,
    recentre_passes: +$('#w-rec').value,
    recentre_smooth_um: +$('#w-recsm').value,
    arc_step_um: +$('#w-arc').value,
    max_depth_um: +$('#w-depth').value,
    exclude_non_wall: $('#chk-exclude').checked,
    media_edge_rule: $('#w-edge').value,
  };
  if ($('#w-blue').value !== '') params.blue_threshold = +$('#w-blue').value;
  return params;
}

/* `extra` carries a hand-drawn outline when the editor asks for a re-measure;
 * everything else about the run is read from the same controls either way, so
 * a redirected trace is not a different analysis with different settings. */
async function runAnalysis(extra) {
  const useSession = $('#wsrc-sess').checked;
  if (useSession && !S.sid) return alert('Load and stitch a folder first, or choose an image file.');
  const params = wallParams();
  const body = {
    params, name: $('#w-name').value.trim() || 'wall',
    out_dir: $('#w-out').value.trim() || null,
    points: OUT.points.filter(pt => isFinite(pt[0]) && isFinite(pt[1])),
    lesion_k_mad: +$('#w-lesion-k').value || 3,
    lesion_min_area_um2: +$('#w-lesion-min').value || 2000,
    exclude_lesions_at: OUT.rejected.filter(pt => isFinite(pt[0]) && isFinite(pt[1])),
  };
  if (useSession) {
    body.sid = S.sid;
    body.island = $('#w-island').value;
    body.render_scale = +$('#w-render').value;
  } else {
    body.image = $('#w-file').value.trim();
    if ($('#w-px').value !== '') body.px_um = +$('#w-px').value;
  }
  Object.assign(body, extra || {});
  delete body.show;                 // which vessel to look at, not a setting
  try {
    await pushPoses();
    const { job } = await api('/api/analyze', body);
    const r = await runJob(job, $('#w-progress'), 'analysis');
    showWallResult(r, !!(extra && extra.guides), extra && extra.show);
  } catch (e) { hideProgress($('#w-progress')); alert(e.message); }
}
$('#btn-analyze').onclick = () => runAnalysis(null);

/* ------------------------------------------------------------------ outline
 *
 * Redirecting the outline by hand.
 *
 * No rule gets every section right, and the ones it gets wrong are not defects
 * so much as judgements: which of two touching vessels to follow, whether to go
 * up a branch or past it, where a torn wall ought to be joined.  Those are
 * questions about the specimen, and the person looking at it can answer them in
 * a second.
 *
 * Drawing replaces a *stretch* rather than the whole curve, because the trace is
 * usually right nearly everywhere; the stroke's two ends are matched to the two
 * nearest points on the existing outline and the shorter way round between them
 * is what gets replaced.  Accuracy is not required of the drawing -- the
 * re-centring pass pulls it onto the middle of the media before anything is
 * measured, which it does for the automatic curve too.
 */
/* `um` is the outline being drawn or edited; `kept` holds the ones already
 * finished. A section with two vessels needs two outlines, and the analysis
 * measures one wall per outline, so they are a list rather than a special
 * case for the second. */
const OUT = { um: [], base: [], closed: false, img: null, scale: 1, undo: [], points: [],
             lesions: [], rejected: [],
             drawing: null, kept: [],
             // Which vessel is being edited, and which other vessels the
             // section has.  An outline belongs to a wall, and the walls it
             // does not belong to have to survive being ignored.
             rank: 0, others: [] };

/* ------------------------------------------------------------- sessions ---
 * Saving the working state, not just what it exported.  See `sessionfile.py`
 * for the format and for why the images and the tables are pointed at rather
 * than copied.
 *
 * Every control is collected by sweeping the DOM for inputs with an id, rather
 * than by listing them here.  A session then picks up a control added later
 * without anyone remembering to add it in a second place -- which is the way
 * this kind of code usually goes quietly out of date -- and an older session
 * simply restores the ids it knows and leaves the rest at their defaults. */
const ORGAN_PANES = {};

function controlsIn(root) {
  const out = {};
  if (!root) return out;
  root.querySelectorAll('input[id], select[id], textarea[id]').forEach((el) => {
    out[el.id] = (el.type === 'checkbox' || el.type === 'radio') ? el.checked : el.value;
  });
  return out;
}

function applyControls(vals) {
  Object.entries(vals || {}).forEach(([id, v]) => {
    const el = document.getElementById(id);
    if (!el) return;                       // a control this version no longer has
    if (el.type === 'checkbox' || el.type === 'radio') el.checked = !!v;
    else el.value = v;
    // The wiring hangs off these, so restoring a value silently would leave the
    // page showing one thing and behaving as another.
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  });
}

function collectSession() {
  const ui = {
    tab: (document.querySelector('.tab.active') || {}).dataset?.tab || null,
    controls: controlsIn(document),
    aorta: {
      outline: { um: OUT.um, closed: OUT.closed, points: OUT.points, rejected: OUT.rejected },
      figure: FIGSTYLE_PATH || null,
    },
    organs: {},
  };
  Object.entries(ORGAN_PANES).forEach(([organ, pane]) => {
    if (pane && pane.getState) ui.organs[organ] = pane.getState();
  });
  return ui;
}

/* Opening clears first, and clears everything, before a byte of the new
   session is applied.  Restoring only what loads and leaving the rest is how
   you get a page showing the previous section's outline under the new
   session's name, with any warning reading as if it were about something
   else. */
function clearSessionView() {
  OUT.um = []; OUT.base = []; OUT.points = []; OUT.rejected = [];
  OUT.lesions = []; OUT.undo = []; OUT.kept = []; OUT.img = null;
  OUT.rank = 0; OUT.others = [];
  const ed = $('#w-editor'); if (ed) ed.classList.add('hidden');
  const nums = $('#w-numbers'); if (nums) nums.classList.add('hidden');
  Object.values(ORGAN_PANES).forEach((pane) => pane && pane.setState && pane.setState(null));
  document.querySelectorAll('.stage img[data-result]').forEach((im) => im.remove());
  S.sel.clear();
}

function sessionNote(msg, bad) {
  const el = $('#sess-note');
  if (el) { el.textContent = msg || ''; el.classList.toggle('bad', !!bad); }
}

async function refreshSessions() {
  const sel = $('#sess-list');
  if (!sel) return;
  try {
    const r = await api('/api/sessions');
    const keep = sel.value;
    sel.innerHTML = (r.sessions || []).map((s) => {
      const when = s.saved ? new Date(s.saved * 1000).toLocaleString() : '';
      const gone = s.folder && !s.folder_exists ? ' — images moved' : '';
      return `<option value="${s.name}">${s.name} · ${when}${gone}</option>`;
    }).join('') || '<option value="">no saved sessions</option>';
    if (keep) sel.value = keep;
  } catch (e) { sessionNote(e.message, true); }
}

async function saveSession() {
  const raw = ($('#sess-name').value || '').trim();
  if (!raw) return sessionNote('give the session a name first', true);
  // Cleaned here as well as on the server: a name with a slash in it is
  // rejected by the router before any handler runs, so the server's own
  // tidying would never get the chance.
  const clean = raw.replace(/[^\w\-. ]+/g, '-').replace(/\s+/g, ' ').replace(/^[ .-]+|[ .-]+$/g, '').slice(0, 80);
  if (!clean) return sessionNote('that name has nothing usable in it', true);
  try {
    // The poses are read back off the server, and the canvas is what the
    // person has been dragging: send them up first, as the layout export
    // already does, or a session can be saved a drag behind what is on screen.
    if (S.sid) await pushPoses();
    const r = await api('/api/sessions/save', { name: clean, sid: S.sid || null, ui: collectSession() });
    $('#sess-name').value = r.name;
    sessionNote(`saved "${r.name}"${r.name !== raw ? ` (renamed from "${raw}")` : ''}`
                + ` · ${(r.bytes / 1024).toFixed(0)} kB`
                + (S.sid ? '' : ' · no tiles loaded, so only the settings'));
    await refreshSessions();
  } catch (e) { sessionNote(e.message, true); }
}

async function openSession(name) {
  if (!name) return sessionNote('pick a session first', true);
  let data;
  try {
    data = await api('/api/sessions/open', { name });
  } catch (e) {
    return sessionNote(e.message, true);   // nothing cleared: the tab still holds what it had
  }
  clearSessionView();
  const st = data.stitch;
  try {
    if (st && st.folder) {
      $('#folder').value = st.folder;
      sessionNote(`opening "${data.name}" — reading ${st.folder}`);
      const { job, sid } = await api('/api/session/create', { folder: st.folder, params: st.params || undefined });
      const res = await runJob(job, $('#progress'), 'loading tiles');
      S.sid = sid;
      await loadSession();
      tilesAreLoaded(res, st.folder);
      const r = await api(`/api/session/${sid}/load_layout`, { layout: st });
      S.poses = r.poses; applyQC(r.qc); refreshIslandSelects();
      S.undo = []; S.redo = []; updateHistoryUI();
      fitView(); draw();
      sessionNote(`opened "${data.name}" — ${r.n_restored} of ${st.poses.length} tiles back in place`);
    } else {
      sessionNote(`opened "${data.name}" — settings only, no tiles were saved with it`);
    }
    applyControls((data.ui || {}).controls);
    const ao = ((data.ui || {}).aorta || {}).outline;
    if (ao) {
      OUT.um = ao.um || []; OUT.base = (ao.um || []).slice();
      OUT.closed = !!ao.closed; OUT.points = ao.points || []; OUT.rejected = ao.rejected || [];
    }
    Object.entries((data.ui || {}).organs || {}).forEach(([organ, stt]) => {
      const pane = ORGAN_PANES[organ];
      if (pane && pane.setState) pane.setState(stt);
    });
    const tab = (data.ui || {}).tab;
    if (tab) { const b = document.querySelector(`.tab[data-tab="${tab}"]`); if (b) b.click(); }
  } catch (e) {
    sessionNote(`could not finish opening "${name}": ${e.message}`, true);
  }
}

function outlineFit() {
  const c = $('#w-canvas');
  if (!OUT.img) return { s: 1, ox: 0, oy: 0 };
  const s = Math.min(c.width / OUT.img.width, c.height / OUT.img.height);
  return { s, ox: (c.width - OUT.img.width * s) / 2, oy: (c.height - OUT.img.height * s) / 2 };
}
/* microns -> canvas, and back */
const umToCv = (p, f) => [f.ox + (p[0] / OUT.scale) * f.s, f.oy + (p[1] / OUT.scale) * f.s];
const cvToUm = (x, y, f) => [((x - f.ox) / f.s) * OUT.scale, ((y - f.oy) / f.s) * OUT.scale];

function drawOutline() {
  const c = $('#w-canvas'), g = c.getContext('2d');
  g.clearRect(0, 0, c.width, c.height);
  if (!OUT.img) return;
  const f = outlineFit();
  g.drawImage(OUT.img, f.ox, f.oy, OUT.img.width * f.s, OUT.img.height * f.s);
  const path = (pts, close) => {
    if (pts.length < 2) return;
    g.beginPath();
    pts.forEach((p, i) => { const q = umToCv(p, f); i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]); });
    if (close) g.closePath();
    g.stroke();
  };
  // Every outline white on a dark halo, as the figure draws it. The live one
  // is told apart by weight rather than by colour: a coloured line laid along
  // a wall is the thing you are trying to see through when you score by eye,
  // and it reads as stain.
  OUT.kept.forEach((o) => {
    g.lineWidth = 2.4; g.strokeStyle = 'rgba(0,0,0,.35)'; path(o.um, o.closed);
    g.lineWidth = 1.0; g.strokeStyle = 'rgba(255,255,255,.75)'; path(o.um, o.closed);
  });
  g.lineWidth = 3; g.strokeStyle = 'rgba(0,0,0,.45)'; path(OUT.um, OUT.closed);
  g.lineWidth = 1.6; g.strokeStyle = '#ffffff'; path(OUT.um, OUT.closed);
  // Detected lesions, outlined on the section itself.  This is where the
  // shapes belong: the figure panel gets an arrowhead, because there it would
  // be drawing over the tissue being scored, but here you are looking at the
  // stitched section on purpose and want to see what was counted.  One that
  // has been clicked away goes red and dashed rather than vanishing outright
  // -- it is still there until the next measurement actually drops it.
  OUT.lesions.forEach((poly) => {
    const rejected = OUT.rejected.some(pt => pathInside(pt, poly));
    g.lineWidth = 2.4; g.strokeStyle = 'rgba(0,0,0,.45)'; path(poly, true);
    if (rejected) {
      g.setLineDash([5, 4]);
      g.lineWidth = 1.4; g.strokeStyle = 'rgba(217,51,38,.95)'; path(poly, true);
      g.setLineDash([]);
    } else {
      g.lineWidth = 1.1; g.strokeStyle = 'rgba(255,255,255,.95)'; path(poly, true);
    }
  });
  // Marks last, so they sit on top of the outline they are placed against.
  OUT.points.forEach((pt, i) => {
    const [x, y] = umToCv(pt, f);
    g.beginPath(); g.arc(x, y, 5, 0, Math.PI * 2);
    g.fillStyle = 'rgba(255,176,0,.85)'; g.fill();
    g.lineWidth = 1.4; g.strokeStyle = '#1a1a1a'; g.stroke();
    g.font = '600 11px system-ui'; g.textAlign = 'left'; g.textBaseline = 'middle';
    g.lineWidth = 3; g.strokeStyle = 'rgba(0,0,0,.6)';
    g.strokeText(String(i + 1), x + 7, y);
    g.fillStyle = '#ffb000'; g.fillText(String(i + 1), x + 7, y);
  });
  if (OUT.drawing && OUT.drawing.length > 1) {
    g.lineWidth = 2.6; g.strokeStyle = '#ffffff'; path(OUT.drawing, false);
    g.lineWidth = 1.6; g.strokeStyle = '#2f7d4f'; path(OUT.drawing, false);
  }
  // Start and end told apart by shape, not by hue -- a circle and a square,
  // both white with a black edge, the same as the figure draws them. Red and
  // green markers on a red-and-blue section read as tissue.
  const ends = OUT.closed ? [] : [OUT.um[0], OUT.um[OUT.um.length - 1]];
  ends.forEach((p, i) => {
    if (!p) return;
    const q = umToCv(p, f);
    g.beginPath();
    if (i) g.rect(q[0] - 3.5, q[1] - 3.5, 7, 7); else g.arc(q[0], q[1], 4, 0, 7);
    g.fillStyle = '#ffffff'; g.fill();
    g.strokeStyle = '#000'; g.lineWidth = 1; g.stroke();
  });
}

function nearestIndex(pt) {
  let best = 0, bd = Infinity;
  OUT.um.forEach((p, i) => {
    const d = (p[0] - pt[0]) ** 2 + (p[1] - pt[1]) ** 2;
    if (d < bd) { bd = d; best = i; }
  });
  return best;
}

/* Splice a stroke into the outline between the two points it starts and ends
 * nearest.  On a closed ring either arc could be meant; the shorter one is,
 * because nobody redraws the long way round to fix a wobble. */
function spliceStroke(stroke) {
  if (stroke.length < 2) return;
  // Nothing to splice into: the first stroke on an empty canvas *is* the
  // outline. This is how a second vessel gets drawn from scratch, where
  // every other stroke is an edit to a curve that already exists.
  if (OUT.um.length < 4) {
    OUT.undo.push(OUT.um.slice());
    OUT.um = stroke.slice();
    outlineNote(`new outline, ${stroke.length} points` +
                (OUT.closed ? '' : ' — tick "closed ring" if it is a full ring'));
    drawOutline();
    return;
  }
  OUT.undo.push(OUT.um.slice());
  let a = nearestIndex(stroke[0]), b = nearestIndex(stroke[stroke.length - 1]);
  const n = OUT.um.length;
  if (OUT.closed) {
    const fwd = (b - a + n) % n, back = (a - b + n) % n;
    if (fwd <= back) {
      const keep = [];
      for (let i = b; i !== a; i = (i + 1) % n) keep.push(OUT.um[i]);
      keep.push(OUT.um[a]);
      OUT.um = keep.concat(stroke);
    } else {
      const keep = [];
      for (let i = a; i !== b; i = (i + 1) % n) keep.push(OUT.um[i]);
      keep.push(OUT.um[b]);
      OUT.um = keep.concat(stroke.slice().reverse());
    }
  } else {
    if (a > b) { a = [b, b = a][0]; stroke = stroke.slice().reverse(); }
    OUT.um = OUT.um.slice(0, a).concat(stroke, OUT.um.slice(b + 1));
  }
  outlineNote(`redrew ${stroke.length} points`);
  drawOutline();
}

/* Cut the outline back to a point on it.
 *
 * Redrawing a stretch cannot fix a trace that is right for its whole length
 * and then carries on somewhere it should not -- into the junction the two
 * limbs of an arch share, or off along the next vessel.  There is nothing to
 * redirect there; the wall simply stops, and where it stops is a judgement
 * about the specimen.  So the stretch between the click and the *nearer* end
 * goes, and two clicks set both ends.
 *
 * A closed ring has no ends to cut, so the first click opens it at the click
 * instead -- the ring becomes an arc starting and ending there -- and the
 * clicks after that trim it as any other arc.
 */
function trimOutlineAt(pt) {
  if (OUT.um.length < 6) return outlineNote('draw or measure an outline first');
  const i = nearestIndex(pt);
  const n = OUT.um.length;
  if (OUT.closed) {
    OUT.undo.push(OUT.um.slice());
    OUT.um = OUT.um.slice(i).concat(OUT.um.slice(0, i));
    OUT.closed = false;
    $('#w-outline-closed').checked = false;
    return void (outlineNote('ring opened here — click on either side to cut it back'),
                 drawOutline());
  }
  const cut = Math.min(i, n - 1 - i);
  if (n - cut < 4) return outlineNote('that would leave too little wall to measure');
  if (cut === 0) return outlineNote('that is already the end — click further along the wall');
  OUT.undo.push(OUT.um.slice());
  OUT.um = i <= n - 1 - i ? OUT.um.slice(i) : OUT.um.slice(0, i + 1);
  outlineNote(`${i <= n - 1 - i ? 'start' : 'end'} moved — ${cut} point(s) cut, ` +
              `${OUT.um.length} left · Undo puts them back`);
  drawOutline();
}

function outlineNote(msg) { $('#w-outline-note').textContent = msg || ''; }

function openOutlineEditor(o, rank, others) {
  if (!o || !o.um || !o.um.length || !o.image) { $('#w-editor').classList.add('hidden'); return; }
  OUT.rank = rank || 0;
  OUT.others = (others || []).map(Number);
  OUT.base = o.um.map((p) => [p[0], p[1]]);
  OUT.um = OUT.base.slice();
  OUT.closed = OUT.baseClosed = !!o.closed;
  OUT.scale = o.image_px_um || 1;
  OUT.undo = [];
  OUT.kept = [];        // a fresh result replaces whatever was being drawn
  $('#w-outline-closed').checked = OUT.closed;
  $('#w-editor').classList.remove('hidden');
  outlineNote('');
  const im = new Image();
  im.onload = () => {
    OUT.img = im;
    const c = $('#w-canvas');
    // A cached picture can load before the panel it was just un-hidden into
    // has any layout, and `min(920, 0 - 24)` is a canvas of no size at all --
    // the editor comes up blank and stays blank until a vessel is reselected.
    const w = Math.min(920, (c.parentElement.clientWidth || 920) - 24);
    c.width = w; c.height = Math.round(w * im.height / im.width);
    drawOutline();
  };
  im.src = '/api/file?path=' + encodeURIComponent(o.image) + '&t=' + Date.now();
}

(function wireOutlineEditor() {
  const c = $('#w-canvas');
  if (!c) return;
  const at = (ev) => {
    const r = c.getBoundingClientRect();
    return cvToUm((ev.clientX - r.left) * c.width / r.width,
                  (ev.clientY - r.top) * c.height / r.height, outlineFit());
  };
  const wPointing = () => $('#w-tool').value === 'point';
  const wRejecting = () => $('#w-tool').value === 'reject';
  const wTrimming = () => $('#w-tool').value === 'trim';

  // A click toggles the whole lesion under it, by geometry rather than by
  // index: the outlines are rebuilt from scratch on every measurement, so an
  // index recorded now could point at the wrong patch, or none, next time. A
  // raw point inside the patch survives that -- on its own it means nothing,
  // it simply erases whichever lesion the server finds at it.
  function toggleWallLesionAt(pt) {
    const hit = OUT.lesions.find(poly => pathInside(pt, poly));
    if (!hit) return outlineNote('no detected focus here — click inside an outlined one');
    const already = OUT.rejected.filter(p2 => pathInside(p2, hit));
    if (already.length) {
      OUT.rejected = OUT.rejected.filter(p2 => !already.includes(p2));
      outlineNote('restored — re-measure to bring it back');
    } else {
      OUT.rejected.push(pt);
      outlineNote(`${OUT.rejected.length} focus/foci marked for removal — re-measure to drop ` +
                  `${OUT.rejected.length === 1 ? 'it' : 'them'} from the count`);
    }
    drawOutline();
  }

  c.onpointerdown = (ev) => {
    const pt = at(ev);
    if (!isFinite(pt[0]) || !isFinite(pt[1])) {
      return outlineNote('the picture is still loading — try again in a moment');
    }
    if (wRejecting()) return toggleWallLesionAt(pt);
    if (wTrimming()) return trimOutlineAt(pt);
    if (wPointing()) {
      // Alt-click removes the nearest mark, as in the organ tabs.
      if (ev.altKey && OUT.points.length) {
        let best = 0, bd = Infinity;
        OUT.points.forEach((q, i) => {
          const d = Math.hypot(q[0] - pt[0], q[1] - pt[1]);
          if (d < bd) { bd = d; best = i; }
        });
        OUT.points.splice(best, 1);
      } else {
        OUT.points.push(pt);
      }
      outlineNote(`${OUT.points.length} mark(s) — re-measure to get a number for each ` +
                  `(alt-click removes one)`);
      return drawOutline();
    }
    c.setPointerCapture(ev.pointerId); OUT.drawing = [at(ev)];
  };
  c.onpointermove = (ev) => {
    if (!OUT.drawing || wPointing() || wRejecting() || wTrimming()) return;
    const p = at(ev), q = OUT.drawing[OUT.drawing.length - 1];
    if (Math.hypot(p[0] - q[0], p[1] - q[1]) > 2 * OUT.scale) { OUT.drawing.push(p); drawOutline(); }
  };
  c.onpointerup = () => { const st = OUT.drawing; OUT.drawing = null; if (st) spliceStroke(st); };
  $('#btn-outline-undo').onclick = () => {
    if (OUT.undo.length) { OUT.um = OUT.undo.pop(); outlineNote('undone'); drawOutline(); }
  };
  $('#btn-outline-reset').onclick = () => {
    // The ring/arc flag is part of the outline, so restoring one without the
    // other hands back the automatic curve labelled as something it is not.
    OUT.undo = []; OUT.kept = []; OUT.um = OUT.base.slice();
    OUT.closed = !!OUT.baseClosed;
    $('#w-outline-closed').checked = OUT.closed;
    outlineNote('back to the automatic outline'); drawOutline();
  };
  $('#btn-outline-add').onclick = () => {
    if (OUT.um.length < 4) return outlineNote('draw this outline first');
    OUT.kept.push({ um: OUT.um.slice(), closed: OUT.closed, rank: OUT.rank });
    // The next outline is a wall the detector did not find, so it needs a
    // rank of its own rather than the one just used.
    OUT.rank = 1 + Math.max(OUT.rank, ...OUT.others, ...OUT.kept.map((g) => g.rank));
    OUT.um = []; OUT.undo = []; OUT.closed = false;
    $('#w-outline-closed').checked = false;
    outlineNote(`${OUT.kept.length} outline(s) kept — draw the next vessel, ` +
                `then re-measure to get one result per outline`);
    drawOutline();
  };
  $('#btn-w-clearpts').onclick = () => { OUT.points = []; outlineNote('marks cleared'); drawOutline(); };
  $('#btn-w-restore').onclick = () => {
    OUT.rejected = []; outlineNote('every removed focus restored — re-measure to bring them back');
    drawOutline();
  };
  $('#w-tool').onchange = () => {
    const tool = $('#w-tool').value;
    c.style.cursor = tool === 'point' ? 'copy' : tool === 'reject' ? 'not-allowed'
                   : tool === 'trim' ? 'cell' : 'crosshair';
    outlineNote({
      point: 'click to mark a lesion, alt-click to remove one',
      reject: 'click a detected focus to remove it from the count · click again to restore it',
      trim: 'click on the wall where it should stop — everything between there and the '
            + 'nearer end goes · the circle and the square are the two ends',
    }[tool] || 'drag along the wall to redraw a stretch');
  };
  $('#w-outline-closed').onchange = (ev) => { OUT.closed = ev.target.checked; drawOutline(); };
  $('#btn-outline-run').onclick = () => {
    const all = OUT.kept.concat(
      OUT.um.length >= 4 ? [{ um: OUT.um, closed: OUT.closed, rank: OUT.rank }] : []);
    if (!all.length) return outlineNote('draw an outline first');
    // The editor edits one wall at a time.  The other walls in the section are
    // named so the server measures them too -- by rank, not by sending their
    // outlines back, because a curve fed through the re-centring a second time
    // comes back a few per cent longer.  Leaving them out is what used to make
    // the second vessel disappear the moment the first one was redrawn.
    const drawn = all.map((g) => g.rank);
    runAnalysis({ guides: all, show: OUT.rank,
                  keep_ranks: OUT.others.filter((k) => !drawn.includes(k)) });
  };
})();

/* ============================================================= figure style
 * The full analysis figure is matplotlib's, not a client reconstruction of
 * it: a hexbin, a fitted regression line, shared axes and legends would be a
 * second copy of plot_rectangular's logic, redrawn in a different language
 * and free to drift from it. Instead the server keeps the WallResult behind
 * each figure it draws (see FIGURE_CACHE in app.py), and a style change here
 * asks it to redraw that same figure -- rotation, size, type scale, boxes,
 * grid -- overwriting the PNG and SVG a plain analysis run would have saved.
 *
 * Style is one shared, sticky state rather than per-vessel: it is meant to
 * answer "how should this kind of figure look", and switching vessels or
 * re-running the analysis should not throw that away.
 */
const FIGSTYLE_DEFAULT = { rotation: 0, rowHeight: 1.7, leftWidth: 9, rightWidth: 2.5,
                          titlePt: 10.5, grid: true, spines: true,
                          // Display only -- see restain() in wall_analysis.py.
                          gainRed: 1, gainBlue: 1, gainNuc: 1, trace: true };
const FIGSTYLE = Object.assign({}, FIGSTYLE_DEFAULT);
let FIGSTYLE_PATH = null;

function figStyleIsDefault() {
  return Object.keys(FIGSTYLE_DEFAULT).every((k) => FIGSTYLE[k] === FIGSTYLE_DEFAULT[k]);
}

function figStyleNote(msg) { const s = $('#fs-status'); if (s) s.textContent = msg || ''; }

async function applyFigStyle(path) {
  if (!path) return;
  FIGSTYLE_PATH = path;
  figStyleNote('redrawing…');
  try {
    await api('/api/replot', {
      path,
      style: {
        rotation_deg: FIGSTYLE.rotation, row_height_in: FIGSTYLE.rowHeight,
        left_width_in: FIGSTYLE.leftWidth, right_width_in: FIGSTYLE.rightWidth,
        title_pt: FIGSTYLE.titlePt, show_grid: FIGSTYLE.grid, show_spines: FIGSTYLE.spines,
        show_trace: FIGSTYLE.trace, gain_muscle: FIGSTYLE.gainRed,
        gain_collagen: FIGSTYLE.gainBlue, gain_nuclei: FIGSTYLE.gainNuc,
      },
    });
    const img = document.querySelector('#w-summary img');
    if (img) img.src = `/api/file?path=${encodeURIComponent(path)}&t=${Date.now()}`;
    figStyleNote('');
  } catch (e) {
    figStyleNote('could not redraw: ' + e.message);
  }
}

(function wireFigStyle() {
  const box = $('#w-figstyle');
  if (!box) return;
  const onChange = (id, apply) => {
    $(id).onchange = () => { apply($(id).value); applyFigStyle(FIGSTYLE_PATH); };
  };
  onChange('#fs-rotation', (v) => { FIGSTYLE.rotation = Number(v) || 0; });
  onChange('#fs-rowh', (v) => { FIGSTYLE.rowHeight = Number(v) || FIGSTYLE_DEFAULT.rowHeight; });
  onChange('#fs-leftw', (v) => { FIGSTYLE.leftWidth = Number(v) || FIGSTYLE_DEFAULT.leftWidth; });
  onChange('#fs-rightw', (v) => { FIGSTYLE.rightWidth = Number(v) || FIGSTYLE_DEFAULT.rightWidth; });
  onChange('#fs-font', (v) => { FIGSTYLE.titlePt = Number(v) || FIGSTYLE_DEFAULT.titlePt; });
  // Typing a gain is several input events and each one is a matplotlib render,
  // so this one is debounced where the others are not.
  // Typing a level is several input events and each one is a matplotlib
  // render, so these are debounced where the plain selects are not.
  const gainReplot = LK.debounce(() => applyFigStyle(FIGSTYLE_PATH), 260);
  [['#fs-gain-red', 'gainRed'], ['#fs-gain-blue', 'gainBlue'],
   ['#fs-gain-nuc', 'gainNuc']].forEach(([id, key]) => {
    $(id).oninput = () => {
      const v = Number($(id).value);
      FIGSTYLE[key] = isFinite(v) ? Math.max(0, Math.min(2, v)) : 1;
      gainReplot();
    };
  });
  $('#fs-trace').onchange = () => { FIGSTYLE.trace = $('#fs-trace').checked; applyFigStyle(FIGSTYLE_PATH); };
  $('#fs-grid').onchange = () => { FIGSTYLE.grid = $('#fs-grid').checked; applyFigStyle(FIGSTYLE_PATH); };
  $('#fs-spines').onchange = () => { FIGSTYLE.spines = $('#fs-spines').checked; applyFigStyle(FIGSTYLE_PATH); };
  $('#btn-fs-reset').onclick = () => {
    Object.assign(FIGSTYLE, FIGSTYLE_DEFAULT);
    $('#fs-rotation').value = 0;
    $('#fs-rowh').value = FIGSTYLE_DEFAULT.rowHeight;
    $('#fs-leftw').value = FIGSTYLE_DEFAULT.leftWidth;
    $('#fs-rightw').value = FIGSTYLE_DEFAULT.rightWidth;
    $('#fs-font').value = FIGSTYLE_DEFAULT.titlePt; $('#fs-grid').checked = true; $('#fs-spines').checked = true;
    $('#fs-gain-red').value = 1; $('#fs-gain-blue').value = 1; $('#fs-gain-nuc').value = 1;
    $('#fs-trace').checked = FIGSTYLE_DEFAULT.trace;
    applyFigStyle(FIGSTYLE_PATH);
  };
})();

/* Which vessel is on screen.  A section that catches the aorta twice produces
 * two of everything, and the tab used to show the first and say nothing about
 * the rest. */
function showVesselPicker(r, chosen) {
  const box = $('#w-vessels');
  const vs = r.vessels || [];
  if (vs.length < 2) { box.classList.add('hidden'); box.innerHTML = ''; return; }
  box.classList.remove('hidden');
  box.innerHTML = `<span class="small" style="align-self:center">${vs.length} vessels in this
    section:</span>` + vs.map((v) => {
    const mm = (v.totals.wall_length_mm || 0).toFixed(2);
    return `<button class="ghost small ${v.rank === chosen ? 'on' : ''}"
      data-rank="${v.rank}" title="${mm} mm of wall">${v.label} · ${mm} mm</button>`;
  }).join('');
  box.querySelectorAll('button').forEach((b) => {
    b.onclick = () => showWallResult(r, false, +b.dataset.rank);
  });
}

/* Built while the side-panel summary is assembled, appended under the figure.
   A module-level handoff rather than a return value because showWallResult
   already writes into three places and threading a fourth through it would be
   worse than one clearly named variable. */
let WALL_TABLES = '';

function showWallResult(r, guided, rank) {
  /* The vessel on screen is one of possibly several; everything below reads
   * from `r` for the shared parts and from the chosen vessel for the rest. */
  const vs = r.vessels || [];
  let pick = (rank === undefined || rank === null) ? 0 : rank;
  let v = vs.find((x) => x.rank === pick) || null;
  // Asking for a vessel this run does not have -- the wall that was on screen
  // was merged away, or the settings changed under it -- shows the first one
  // rather than an empty panel labelled with a rank nothing answers to.
  if (!v && vs.length) { v = vs[0]; pick = v.rank; }
  if (v) {
    r = Object.assign({}, r, {
      totals: v.totals, paths: v.paths,
      rectangular_channels_png: v.rectangular_channels_png,
      rectangular_channels_svg: v.rectangular_channels_svg,
      straight_png: v.straight_png,
      rectangular_png: v.paths.rectangular || r.rectangular_png,
      outline: v.outline || r.outline,
    });
  }
  showVesselPicker(arguments[0], pick);
  openOutlineEditor(r.outline, pick, vs.map((x) => x.rank).filter((k) => k !== pick));
  if (guided) outlineNote('measured from your outline');
  const t = r.totals;
  // A vessel is either a wall closed around a lumen or a separate piece of the
  // band, so the count is not the lumen count -- on a section caught at a
  // branch neither wall closes and both are arcs.  Older saved runs have no
  // n_vessels, and for those the lumen count was the answer.
  const nv = t.n_vessels === undefined ? t.n_lumina : t.n_vessels;
  const vesselsWhy = t.guided
    ? 'the trace came from the outline you drew, so there was nothing to enumerate'
    : nv === 0
      ? 'no wall closes around a lumen here, and no piece of the band is long enough to be one — the trace came from the band itself'
      : nv === t.n_lumina
        ? 'each closes around its own lumen; use the buttons above the figure to switch between them'
        : `${t.n_lumina} close around a lumen and ${nv - t.n_lumina} are open arcs — pieces of band that enclose nothing, each traced and measured on its own; use the buttons above the figure to switch between them`;
  const rows = [
    ['wall length', t.wall_length_mm.toFixed(3) + ' mm', 'traced circumference of the media'],
    ['closed ring', t.closed_ring ? 'yes' : 'no (open arc)', ''],
    ['media thickness', `${t.mean_media_thickness_um.toFixed(1)} ± ${t.sd_media_thickness_um.toFixed(1)} µm`, 'mean ± SD along the wall'],
    ['media area', Math.round(t.media_area_um2).toLocaleString() + ' µm²', 'cross-sectional area of the muscle layer'],
    ['collagen per mm of wall', Math.round(t.collagen_od_um2_per_mm_length).toLocaleString() + ' OD·µm²/mm', 'the per-length figure'],
    ['collagen, total', Math.round(t.collagen_od_um2_total).toLocaleString() + ' OD·µm²', 'integrated over the whole wall'],
    ['blue area in media', (100 * t.collagen_area_fraction_media).toFixed(2) + ' %',
     t.collagen_rule === 'dominance'
       ? 'fraction of the muscle layer where collagen outweighs muscle — no threshold'
       : 'fraction of the muscle layer above the absolute threshold'],
    ['blue area per mm', Math.round(t.collagen_area_um2_per_mm_length).toLocaleString() + ' µm²/mm', ''],
    ['collagen ÷ muscle', t.collagen_to_muscle_ratio.toFixed(3),
     'the thickness-independent readout — both stains are integrated through the same wall, so thickness divides out'],
    ['collagen / (collagen+muscle)', (100 * t.collagen_fraction_of_stain).toFixed(2) + ' %', 'the same thing, bounded 0–1'],
    ['thickness confound', `r = ${t.r_thickness_vs_collagen_to_muscle.toFixed(2)} (ratio) vs ${t.r_thickness_vs_collagen_per_length.toFixed(2)} (per length)`,
     'how strongly each readout tracks media thickness; the per-length figure is largely a thickness measurement'],
    ['mean collagen concentration', t.collagen_mean_conc_media.toFixed(3), 'in the media'],
    ['collagen in adventitia', Math.round(t.collagen_adventitia_od_um2_total).toLocaleString() + ' OD·µm²', 'for comparison'],
    ['blue rule', t.collagen_rule === 'dominance' ? 'collagen > muscle, per pixel'
       : `absolute, threshold ${t.collagen_threshold === null ? '?' : Number(t.collagen_threshold).toFixed(3)}`,
     'an absolute threshold derived per section makes sections incomparable'],
    ['unrolling stretch', `${t.stretch_p1.toFixed(2)} – ${t.stretch_p99.toFixed(2)}`,
     'how much laying the curved wall flat had to squash or spread it (1 = no distortion)'],
    ['folded samples', (100 * t.folded_fraction).toFixed(3) + ' %',
     'sampling lines that crossed and were discarded; should be 0'],
    ['step along the wall', t.arc_step_um + ' µm', ''],
    ['branches found', String(t.n_branches),
     'places where the wall forks — found by what the traced strip leaves behind, not by thickness'],
    ['branch / thickened', `${(100 * t.branch_length_fraction).toFixed(1)} % / ${(100 * t.thickened_length_fraction).toFixed(1)} %`,
     'of the traced length; "thickened" is wall that is thick without forking, i.e. possible pathology'],
    ['length measured', `${t.wall_length_mm.toFixed(3)} of ${t.traced_length_mm.toFixed(3)} mm`,
     t.excluded_non_wall ? 'junctions left out of the totals' : 'everything included'],
    ['vessels in the section', t.guided ? 'outline drawn by hand' : String(nv), vesselsWhy],
  ];
  if (t.muscle_variation !== undefined) {
    rows.push(['muscle variation across the wall',
      t.muscle_variation.toFixed(3),
      'how unevenly the muscle is laid down across the wall, lumen to outer edge, measured at a scale coarser than a lamella and skipping the boundary ramps. Higher means the media has holes in it. Unlike the rows below it describes the media itself, so it does not depend on the outer edge being in the right place']);
    rows.push(['lamellar contrast',
      t.lamellar_contrast.toFixed(3),
      'the banding the measure above deliberately excludes: elastic lamellae on a 4-5 µm pitch. Not a finding on its own — a crisply cut section bands strongly and is not a broken one. Read it against the edge width below']);
    rows.push(['lamellar spacing',
      t.lamellar_spacing_um.toFixed(2) + ' µm',
      'distance between successive layers, from the same banding. Blur merges neighbouring lamellae and inflates this, so check the edge width before comparing sections']);
    rows.push(['lamellar disorder',
      t.lamellar_disorder.toFixed(3) + '  (on ' + (100 * t.lamellar_coverage).toFixed(0) + ' % of the wall)',
      'spread of the gaps between successive layers, over their mean. Low means evenly stacked lamellae; high means layers that have merged, split or gone missing. Only measurable where three layers could be found, hence the coverage — and blur inflates it, so check the edge width']);
    rows.push(['section edge width',
      t.section_edge_width_um.toFixed(2) + ' µm',
      'how sharply this section resolves anything, from the rise of muscle across the luminal boundary — measured without reference to any lamella. Blur attenuates both layering figures above, so two sections are only comparable on them at a similar edge width']);
  }
  if (t.median_muscle_beyond_edge_fill !== undefined) {
    const solid = t.median_muscle_beyond_edge_fill >= 0.8;
    rows.push(['muscle past the media edge',
      `${t.median_muscle_beyond_edge_um.toFixed(1)} µm, ${(100 * t.median_muscle_beyond_edge_fill).toFixed(0)} % solid`,
      solid
        ? 'muscle continues past the edge as a solid layer, so the edge stopped short of it'
        : 'the muscle out there is in pieces — an outer media that has broken up, which no boundary rule can follow']);
    rows.push(['wall with a fragmented outer media',
      (100 * t.fragmented_length_fraction).toFixed(1) + ' %',
      'of the counted length, where muscle reaches more than 12 µm past the edge and less than 80 % of that reach is muscle']);
  }
  rows.push(['focal lesions', `${t.n_lesions} · ${(100 * t.lesion_area_fraction_media).toFixed(2)} % of media`,
    'patches of collagen standing above this wall\u2019s own level — see the table below']);
  if (t.lesion_cut_share !== undefined) {
    rows.push(['lesion cut / wall max', `${t.lesion_cut_share.toFixed(3)} / ${t.lesion_share_max.toFixed(3)}`,
      t.lesion_share_max < t.lesion_cut_share
        ? 'nothing in this wall reached the cut — the collagen here is diffuse, not focal'
        : 'the cut the lesion search used, and the highest the wall actually reached']);
  }
  if (t.n_lumina_rejected) {
    rows.push(['holes not counted as vessels', String(t.n_lumina_rejected),
      'enclosed by the band but too small, or ringed by too little muscle to be a wall']);
  }
  $('#w-totals').innerHTML = '<table class="grid"><tbody>' + rows.map(
    ([a, b, c]) => `<tr title="${c}"><th>${a}</th><td class="num">${b}</td></tr>`).join('') +
    '</tbody></table>';
  // Lesions and marks, when there are any.  Appended to the summary below the
  // figure rather than squeezed into the side panel: they are tables, and a
  // 310 px column is not where a table belongs.
  WALL_TABLES = '';
  if (r.lesions && r.lesions.length) {
    WALL_TABLES +=
      `<h3 class="sec">Focal lesions — ${r.lesions.length}</h3>` +
      `<div class="hint">Collagen standing more than ${t.lesion_k_mad}× the robust deviation above
        <i>this wall's own</i> level, smoothed to lesion scale. <b>arc_mm</b> is where along the
        traced wall each one sits, so it lines up with the profile panels. A healthy media
        returns none, and that is the right answer: medial collagen is lamellar and runs the
        whole thickness, so there is nothing focal to find. Lower the cut in the side panel to
        look for relative hot spots — but a lesion set that appears only below 2× is telling
        you the wall is diffuse, not that it has lesions.</div>` +
      gridTable(r.lesions, ['lesion', 'arc_mm', 'area_um2', 'equivalent_diameter_um',
                            'collagen_share_mean', 'collagen_share_peak',
                            'collagen_to_counterstain']);
  }
  if (r.points && r.points.length) {
    WALL_TABLES +=
      `<h3 class="sec">Marks — ${r.points.length}</h3>` +
      `<div class="hint">Each mark measured over the same ${r.points[0].radius_um} µm disc of
        media. <code>in_media</code> is false for a mark that landed off the wall — in the
        adventitia, or on another vessel — and those carry no numbers.</div>` +
      gridTable(r.points, ['point', 'arc_mm', 'x_um', 'y_um', 'in_media', 'collagen_share',
                           'collagen_to_counterstain', 'collagen_mean_od', 'in_lesion']);
  }
  $('#w-files').innerHTML = Object.entries(r.paths).map(
    ([k, v]) => `<a href="/api/file?path=${encodeURIComponent(v)}" target="_blank">${k}</a>`).join(' · ') +
    `\n${r.paths.totals.replace(/[^/]+$/, '')}`;
  const bust = '&t=' + Date.now();
  const img = (p) => `<img src="/api/file?path=${encodeURIComponent(p)}${bust}">`;
  const svg = `/api/file?path=${encodeURIComponent(r.rectangular_channels_svg)}`;
  $('#w-summary').innerHTML =
    `<div class="row" style="justify-content:space-between;align-items:center;margin-bottom:6px">
       <b>Full analysis figure</b>
       <a class="btn ghost" href="${svg}" download>Download SVG</a></div>` +
    `<div class="hint" style="margin-bottom:8px">The map, every profile plot -- thickness,
      collagen per length, collagen ÷ muscle running along the wall -- and the regression
      against thickness, all from the same trace the numbers above were computed from. Use
      the figure style controls above to rotate, resize or restyle it.</div>` +
    img(r.rectangular_channels_png) + WALL_TABLES;
  // Outlines are in microns of the analysis frame, the same units the editor
  // draws in, so they need no conversion -- which is the reason they are sent
  // that way rather than as pixels of whichever image produced them.
  OUT.lesions = r.lesion_outlines || [];
  drawOutline();
  $('#w-numbers').classList.remove('hidden');
  $('#w-figstyle').classList.toggle('hidden', !r.rectangular_channels_png);
  if (r.rectangular_channels_png) {
    if (figStyleIsDefault()) FIGSTYLE_PATH = r.rectangular_channels_png;
    else applyFigStyle(r.rectangular_channels_png);
  }
}

/* ==================================================================== batch */

/* One table renderer for every cohort table: they differ only in which
   columns they have, and hand-writing four of these would guarantee they
   drift apart. */
function gridTable(rows, keys, fmt) {
  if (!rows || !rows.length) return '<div class="hint">nothing to show</div>';
  const present = keys.filter(k => rows.some(r => r[k] !== undefined && r[k] !== null));
  const cell = (v, k) => {
    if (v === undefined || v === null || v === '') return '';
    // A NaN survives JSON as null; showing the word "null" in a results table
    // reads as a bug rather than as "there was nothing here to measure".
    if (v === null || v === undefined) return '—';
    if (typeof v === 'boolean') return v ? 'yes' : 'no';
    if (typeof v !== 'number') return String(v);
    if (!isFinite(v)) return '—';
    if (fmt && fmt[k]) return fmt[k](v);
    if (Number.isInteger(v)) return String(v);   // counts are counts, not 3.00
    return Math.abs(v) < 1 ? v.toFixed(4) : v.toFixed(2);
  };
  return '<table class="grid"><thead><tr>' + present.map(k => `<th>${k}</th>`).join('') +
    '</tr></thead><tbody>' + rows.map(r => '<tr>' + present.map(k =>
      `<td class="${typeof r[k] === 'number' ? 'num' : ''}">${cell(r[k], k)}</td>`
    ).join('') + '</tr>').join('') + '</tbody></table>';
}

/* --- organs ---------------------------------------------------------------
 * An aorta is measured along a wall; a heart or a kidney is measured over an
 * area.  Everything else about the two is the same -- the same tray, the same
 * pooling, the same mouse-as-unit statistics -- so the difference lives here,
 * in a table, and the panes themselves are built once from templates.  Two
 * near-identical copies of a tab is how heart and kidney end up disagreeing
 * about something neither of them meant to change.
 */
const ORGANS = {
  aorta: {
    label: 'Aorta', kind: 'wall', batchApi: '/api/batch',
    analyseLabel: 'also run the wall analysis',
    readouts: [
      ['collagen_to_muscle_ratio', 'collagen ÷ muscle', true],
      ['collagen_fraction_of_stain', 'collagen fraction of stain', true],
      ['mean_media_thickness_um', 'media thickness', true],
      ['collagen_od_um2_per_mm_length', 'collagen per mm of wall', true],
      ['muscle_od_um2_per_mm_length', 'muscle per mm of wall', false],
      ['collagen_area_fraction_media', 'collagen > muscle, % of media', true],
      // Averaged over a mouse's sections weighted by wall length, not summed
      // like the rest: these are medians, and medians do not add up.
      ['muscle_variation', 'muscle unevenness across the wall', false],
      ['lamellar_contrast', 'lamellar contrast', false],
      ['section_edge_width_um', 'section edge width', false],
      ['fragmented_length_fraction', 'wall with a fragmented outer media', false],
    ],
    sectionCols: ['name', 'mouse', 'genotype', 'segment', 'slide', 'section',
      'n_tiles', 'largest_island', 'n_islands', 'conflicts', 'median_agreement', 'unreadable',
      'wall_length_mm', 'mean_media_thickness_um', 'collagen_od_um2_per_mm_length',
      'collagen_to_muscle_ratio', 'collagen_area_fraction_media', 'error'],
    pooledCols: ['genotype', 'mouse', 'segment', 'n_sections', 'wall_length_mm',
      'mean_media_thickness_um', 'collagen_od_um2_per_mm_length', 'collagen_to_muscle_ratio',
      'collagen_fraction_of_stain', 'collagen_area_fraction_media',
      // Not summed like the rest: 'medians_from' says whether the block was
      // rebuilt from every position of the mouse or averaged over sections.
      'muscle_variation', 'medians_from'],
    regionWord: 'segment', pooledWord: 'whole-aorta',
  },
  heart: {
    label: 'Heart', kind: 'area', batchApi: '/api/fibrosis/batch',
    analyseLabel: 'also measure the fibrosis',
    readouts: [
      ['collagen_to_muscle_ratio', 'collagen ÷ muscle', true],
      ['collagen_area_fraction_tissue', 'collagen > muscle, % of tissue', true],
      ['collagen_fraction_of_stain', 'collagen fraction of stain', false],
      ['collagen_od_um2_per_mm2_tissue', 'collagen per mm² of tissue', true],
      ['muscle_od_um2_per_mm2_tissue', 'muscle per mm² of tissue', false],
      ['tissue_fraction_of_image', 'tissue, % of what was imaged', false],
    ],
    sectionCols: ['name', 'mouse', 'genotype', 'segment', 'section', 'slide',
      'n_tiles', 'largest_island', 'n_islands', 'median_agreement', 'unreadable',
      'tissue_area_mm2', 'tissue_fraction_of_image', 'collagen_to_muscle',
      'collagen_area_fraction', 'collagen_od_um2_per_mm2_tissue', 'error'],
    pooledCols: ['genotype', 'mouse', 'segment', 'n_sections', 'n_fields', 'tissue_area_mm2',
      'collagen_to_muscle_ratio', 'collagen_area_fraction_tissue',
      'collagen_od_um2_per_mm2_tissue', 'tissue_fraction_of_image'],
    regionWord: 'region', pooledWord: 'whole-heart',
    // What the red channel of a trichrome is in this organ.  It stains
    // cytoplasm; in a ventricle that is muscle, in a kidney it is tubular
    // epithelium and there is barely any muscle in the field at all.
    counterstain: 'muscle',
    parts: ['left ventricle', 'right ventricle', 'septum'],
  },
};
ORGANS.kidney = {
  ...ORGANS.heart, label: 'Kidney', pooledWord: 'whole-kidney',
  counterstain: 'parenchyma',
  // Inner and outer medulla listed separately, and both kept alongside the
  // general 'medulla' rather than replacing it: a section cut where the two
  // cannot be told apart should be labelled 'medulla' and not forced into a
  // distinction the tissue does not support. cohort.py already reads all four
  // out of folder names and orders them anatomically, so a region named here
  // groups with the same region read from a name.
  parts: ['cortex', 'outer medulla', 'inner medulla', 'medulla', 'papilla'],
  // Same readouts, said correctly: trichrome's red is cytoplasm, and a
  // kidney's cytoplasm is tubular epithelium rather than muscle.
  readouts: ORGANS.heart.readouts.map(([k, label, on]) =>
    [k, label.replace('muscle', 'parenchyma'), on]),
};

/* The columns a metadata.csv may carry, mirroring cohort.OVERRIDE_FIELDS.
   Region first because it is the field folder names miss most often, and the
   only one with a vocabulary behind it. */
const META_FIELDS = ['segment', 'mouse', 'genotype', 'age', 'slide', 'section'];

/* --- the tray: one parent folder, one sub-folder per section -------------
   The row navigates; the tick on its left chooses.  Keeping those apart
   matters because "holds images" and "is a section" are not the same thing:
   Documents has three stray images and thirty sub-folders, and any rule that
   makes holding images mean "this is a leaf" traps you there.

   Built per organ rather than once, because each tab keeps its own tray, its
   own chosen sections and its own output folder.  Sharing them would mean
   picking kidney sections in the heart tab and never being told. */
function makeBatch(organ, panelHost, stageHost) {
  const cfg = ORGANS[organ];
  const p = `b-${organ}`;
  const B = { scan: null, chosen: new Set(), plotDrawn: false, panel: null };
  // 'segment' is the column; what it is called depends on the organ.
  const fieldWord = (f) => (f === 'segment' ? cfg.regionWord : f);

  panelHost.innerHTML = `
    <div class="group">
      <h3 class="sec-title">Batch — a tray of sections</h3>
      <div class="row">
        <input id="${p}-parent" type="text" placeholder="folder of section folders" spellcheck="false">
        <button id="${p}-browse" class="ghost small" title="Pick the tray, then list the sections in it">Browse…</button>
      </div>
      <div id="${p}-browser" class="browser hidden"></div>
      <div class="row">
        <button id="${p}-all" class="wide">Choose all here</button>
        <button id="${p}-none" class="wide">Clear</button>
      </div>
      <div class="row" title="Say what the folder names did not. What you type is written
into a metadata.csv beside the sections — the same file a hand-made one would be, read back by
the scan, the batch and the pooling alike. Leave it blank to clear that field and let the
folder name have its say back.">
        <select id="${p}-segfield" title="which field this sets">
          ${META_FIELDS.map(f => `<option value="${f}">${fieldWord(f)}</option>`).join('')}
        </select>
        <input id="${p}-seg" type="text" list="${p}-seglist" spellcheck="false"
               placeholder="${cfg.regionWord} for the ticked sections">
        <datalist id="${p}-seglist"></datalist>
        <button id="${p}-setseg" class="ghost small">Set</button>
      </div>
      <div class="hint">Click a <b>name</b> to open the folder; click the <b>○</b> beside it
        to tick that section into the run. Names are read for mouse, genotype, ${cfg.regionWord},
        slide and section; a <b>metadata.csv</b> beside them (columns
        <code>folder,mouse,genotype,age,segment</code>) overrides anything misread — and the
        box above writes that file for you, for whatever the names never said.</div>
      <div id="${p}-scan" class="readout small">no tray scanned</div>
    </div>

    <div class="group">
      <h3>Folders to run</h3>
      <textarea id="${p}-folders" rows="8" placeholder="one folder per line"></textarea>
      <input id="${p}-out" type="text" placeholder="output folder">
      <label class="row"><span><input id="${p}-analyze" type="checkbox" checked> ${cfg.analyseLabel}</span></label>
      <label class="row"><span>Render scale</span>
        <input id="${p}-render" type="number" value="0.35" min="0.05" max="1" step="0.05"></label>
      ${cfg.kind === 'area' ? `
      <label class="row"><span><input id="${p}-shared" type="checkbox"> one stain matrix for the whole run</span></label>
      <div class="hint">Off by default. Reading the stain vectors off each image separately
        called healthy myocardium 23% fibrotic on this data — nearly pure muscle gives
        the estimator no collagen to find, so it fits the second direction to noise.
        Estimating one matrix across the run is safer than per image but still drifts
        that way; the fixed trichrome vectors do not.</div>` : ''}
      <button id="${p}-run" class="primary wide">Run batch</button>
    </div>

    <div class="group">
      <h3>Plots</h3>
      <label class="row"><span>A dot is</span>
        <select id="${p}-unit">
          <option value="mouse_segment" selected>one mouse, per ${cfg.regionWord}</option>
          <option value="mouse">one mouse, pooled</option>
          <option value="section">one section</option>
        </select></label>
      <label class="row" title="How many panels go across. This figure grows with the run
— one panel per readout per ${cfg.regionWord} — so set the panel size itself under Axes in
the plot controls below, and the figure becomes whatever that makes it."><span>Columns</span>
        <input id="${p}-ncol" type="number" min="1" max="8" step="1" placeholder="auto"></label>
      ${cfg.kind === 'area' ? '' : `
      <label class="row" title="Through the wall: one panel per condition with the
counterstain and the collagen in it, instead of one panel of each with the genotypes
overlaid. The first compares the two stains within a wall, the second compares
genotypes."><span><input id="${p}-stains" type="checkbox"> both stains in one panel</span></label>`}
      <div id="${p}-controls"></div>
      <div id="${p}-readouts" class="checks"></div>
      <button id="${p}-plot" class="primary wide">Plot</button>
      <a id="${p}-svg" class="btn wide hidden" download>Download SVG</a>
      <div class="hint">Drawn from the <code>cohort_*.csv</code> already on disk, so the
        controls cost nothing to turn. Each dot is one animal; the brackets carry
        the Welch p from <code>cohort_comparison.csv</code>, corrected within each panel.</div>
    </div>`;

  stageHost.innerHTML = `
    <div id="${p}-progress" class="progress hidden"><div class="bar"></div><span></span></div>
    <div id="${p}-plotimg"></div>
    <div id="${p}-table"></div>
    <div id="${p}-cohort"></div>`;

  const q = (k) => panelHost.querySelector(`#${p}-${k}`) || stageHost.querySelector(`#${p}-${k}`);

  async function browseTo(path) {
    const j = await api('/api/cohort/scan', { path: path || null, organ });
    B.scan = j;
    q('parent').value = j.path;
    // The results belong beside the data, not inside the application folder:
    // an app you can move or reinstall is a bad place to keep results.
    if (!q('out').value.trim()) q('out').value = j.path.replace(/\/+$/, '') + '/analysis';
    drawBrowser();
  }

  function drawBrowser() {
    const j = B.scan;
    if (!j) return;
    const box = q('browser');
    box.classList.remove('hidden');
    // Ticking redraws the list; without this, choosing a section near the bottom
    // throws you back to the top and you lose your place.
    const scroll = box.scrollTop;
    const rows = j.dirs.map((d, i) => {
      const on = B.chosen.has(d.folder);
      const tick = d.is_section
        ? `<span class="tick" data-act="pick" title="use this section">${on ? '✓' : '○'}</span>`
        : `<span class="tick dim">·</span>`;
      const note = d.is_section
        ? `${d.images} img · ${d.mouse || '?'} · ${d.genotype || '?'} · ${d.segment || '?'}`
        : '';
      return `<div data-i="${i}" class="${on ? 'on' : ''}">${tick}` +
        `<span class="grow" data-act="nav" title="open">${d.name}</span>` +
        `<span>${note}</span></div>`;
    });
    box.innerHTML =
      `<div data-up="1"><span class="tick dim">‥</span>` +
      `<span class="grow" data-act="nav">up one level</span><span></span></div>` +
      (rows.length ? rows.join('')
                   : '<div><span class="grow">this folder has no sub-folders</span><span></span></div>');
    box.scrollTop = scroll;

    const secs = j.dirs.filter(d => d.is_section);
    fillMeta();
    const missing = secs.filter(d => d.missing.length);
    // A grade written into a folder name is somebody's eye, not a measurement.
    // It is counted here so it is visible, and never fed into anything.
    const graded = secs.filter(d => d.grade);
    q('scan').innerHTML =
      `<b>${secs.length}</b> section(s) here · <b>${B.chosen.size}</b> chosen · ` +
      `<b>${new Set(secs.map(d => d.mouse || '?')).size}</b> mouse/mice · ` +
      `<b>${new Set(secs.map(d => d.genotype || '?')).size}</b> genotype(s) · ` +
      `<b>${new Set(secs.map(d => d.segment || '?')).size}</b> ${cfg.regionWord}(s)` +
      (missing.length
        ? `\n${missing.length} folder(s) missing ${[...new Set(missing.flatMap(m => m.missing))].join('/')}` +
          ` — they would be grouped as "unknown"`
        : '') +
      (graded.length
        ? `\n${graded.length} folder name(s) carry a fibrosis grade (${[...new Set(graded.map(d => d.grade))].join(', ')})` +
          ` — recorded, never measured on`
        : '') +
      ((j.skipped && j.skipped.length) ? `\ncould not read: ${j.skipped.join(', ')} — skipped` : '') +
      (j.has_metadata_csv ? '\nmetadata.csv found and applied' : '');
    q('folders').value = [...B.chosen].join('\n');
  }

  // What the box is asking for, and what it suggests: the parser's own
  // vocabulary when the field is the region, and in every case whatever this
  // tray already says.  A value typed by hand has to land on a spelling that
  // groups with the folders that named their own -- "KO" and "ko" in one
  // column are two genotypes where the mice are one.
  function fillMeta() {
    const field = q('segfield').value;
    const secs = B.scan ? B.scan.dirs.filter(d => d.is_section) : [];
    q('seg').placeholder = `${fieldWord(field)} for the ticked sections`;
    q('seglist').innerHTML = [...new Set([
      ...(field === 'segment' && B.scan ? B.scan.segments || [] : []),
      ...secs.map(d => d[field]).filter(Boolean)])]
      .map(x => `<option value="${x}">`).join('');
  }
  q('segfield').onchange = fillMeta;
  fillMeta();

  q('browse').onclick = async () => {
    const picked = await LK.browse({
      mode: 'dir', exts: IMAGE_EXTS, path: q('parent').value.trim(),
      title: `Tray of ${cfg.label.toLowerCase()} sections`,
      hint: 'The parent folder, one sub-folder per section.',
    });
    if (picked) browseTo(picked).catch(e => alert(e.message));
  };

  q('browser').onclick = (ev) => {
    const div = ev.target.closest('div[data-up], div[data-i]');
    if (!div || !B.scan) return;
    if (div.dataset.up) return void browseTo(B.scan.parent).catch(e => alert(e.message));
    const d = B.scan.dirs[+div.dataset.i];
    const act = (ev.target.closest('[data-act]') || {}).dataset;
    if (act && act.act === 'pick' && d.is_section) {
      B.chosen.has(d.folder) ? B.chosen.delete(d.folder) : B.chosen.add(d.folder);
      return drawBrowser();
    }
    browseTo(d.folder).catch(e => alert(e.message));
  };

  q('all').onclick = () => {
    if (!B.scan) return;
    B.scan.dirs.filter(d => d.is_section).forEach(d => B.chosen.add(d.folder));
    drawBrowser();
  };
  q('none').onclick = () => { B.chosen.clear(); drawBrowser(); };

  // Saying by hand what a folder name did not.  It writes the metadata.csv
  // beside those sections rather than holding the answer in the page, so it
  // reaches the batch and the pooling by the path overrides always took, and
  // it is still there tomorrow.  Re-scanning afterwards is the proof: the list
  // redraws from the file, so what you see is what was written.
  q('setseg').onclick = async () => {
    if (!B.chosen.size) return alert(`Tick the sections to label first.`);
    const field = q('segfield').value;
    const value = q('seg').value.trim();
    if (!value && !confirm(`Clear the ${fieldWord(field)} on ${B.chosen.size} ticked section(s), `
      + `back to what their folder names say?`)) return;
    try {
      await api('/api/cohort/meta', { folders: [...B.chosen], fields: { [field]: value } });
      await browseTo(q('parent').value.trim());
    } catch (err) { alert(err.message); }
  };

  q('run').onclick = async () => {
    const folders = q('folders').value.split('\n').map(s => s.trim()).filter(Boolean);
    if (!folders.length) return alert('Scan a tray, or list at least one folder.');
    const body = {
      organ, folders, parent: q('parent').value.trim() || null,
      out_dir: q('out').value.trim() || null,
      analyze: q('analyze').checked, render_scale: +q('render').value,
      stitch_params: stitchParams(),
    };
    if (cfg.kind === 'area') {
      body.shared_stains = q('shared').checked;
      body.params = organParams(organ);
    }
    if (cfg.kind === 'wall') body.wall_params = wallParams();
    try {
      const r = await runJob((await api(cfg.batchApi, body)).job, q('progress'), 'batch');
      q('table').innerHTML =
        `<h3 class="sec">Per section</h3>` +
        `<div class="hint">Written to ${r.csv}` +
        (r.depth_relative_csv
          ? `<br>Panel E, every section's depth profile through the media pooled into one
             long table: ${r.depth_relative_csv}` : '') +
        (r.depth_relative_avg_csv
          ? `<br>Panel E averaged per genotype and ${cfg.regionWord}, sections into mice then
             mice into groups, so n is animals: ${r.depth_relative_avg_csv}` : '') +
        `</div>` + gridTable(forCounterstain(r.rows, cfg.counterstain || 'muscle'),
                             cfg.sectionCols.map(c => (cfg.counterstain && cfg.counterstain !== 'muscle')
                               ? c.replace(/muscle/g, cfg.counterstain) : c));
      renderCohort(r.cohort, r.cohort_csv);
      if (r.cohort && !r.cohort.error) drawPlot(true);
    } catch (e) { hideProgress(q('progress')); alert(e.message); }
  };

  /* --- cohort plots ------------------------------------------------------
     Server-side matplotlib, same as the analysis figure, so the two share a
     type scale and both come out as PNG to look at and SVG to edit. */
  q('readouts').innerHTML = cfg.readouts.map(([k, label, on]) =>
    `<label><input type="checkbox" value="${k}"${on ? ' checked' : ''}>${label}</label>`).join('');

  /* Debounced: typing "180" into the width box is three input events, and each
     redraw is a matplotlib render on the server. */
  const replot = LK.debounce(() => drawPlot(true), 260);
  B.panel = LK.PlotControls(q('controls'), {
    spec: { width_mm: 180, font_pt: 8, error: 'sem', kind: 'bar', p_test: 'welch' },
    onchange: replot,
  });
  /* The test is not a choice here.  cohort.py ran Welch's t when it built
     cohort_comparison.csv -- with three or four mice a side you cannot check the
     equal-variance assumption, so not making it is free -- and the brackets read
     that file rather than recomputing, so offering a test picker would be
     offering a setting that changes nothing. */
  B.panel.node.querySelectorAll('#lk-ptest, #lk-ppairs').forEach((field) => {
    const row = field.closest('label');
    if (row) row.hidden = true;
  });

  async function drawPlot(quiet) {
    if (quiet && !B.plotDrawn) return;
    const readouts = [...q('readouts').querySelectorAll('input:checked')].map(i => i.value);
    if (!readouts.length) { if (!quiet) alert('Tick at least one readout.'); return; }
    try {
      const num = (k) => (q(k).value.trim() === '' ? null : +q(k).value);
      const r = await api('/api/cohort/plot', {
        organ, readouts, out_dir: q('out').value.trim() || null,
        unit: q('unit').value, spec: B.panel.spec(), ncol: num('ncol'),
        stains_together: !!(q('stains') && q('stains').checked),
        title: q('parent').value.trim().split('/').filter(Boolean).pop() || '',
      });
      const file = (p) => `/api/file?path=${encodeURIComponent(p)}`;
      q('plotimg').innerHTML = `<h3 class="sec">Quantifications</h3>` +
        `<img src="${file(r.png)}&t=${Date.now()}">` +
        // The curve through the wall: not a readout, so not a panel in the
        // grid above.  Absent for an area organ, or a batch run without the
        // wall analysis.
        (r.depth_png ? `<h3 class="sec">Through the wall</h3>` +
          `<img src="${file(r.depth_png)}&t=${Date.now()}">` +
          `<div class="hint"><a href="${file(r.depth_svg)}" download>this one as SVG</a>` +
          (r.depth_source_csv ? ` · <a href="${file(r.depth_source_csv)}" download>source data
             (CSV)</a> — one row per depth per condition, mean, SD between animals, and n` : '') +
          `</div>` : '');
      const a = q('svg');
      a.href = `/api/file?path=${encodeURIComponent(r.svg)}`;
      a.classList.remove('hidden');
      // A colour box per genotype, named by the figure rather than guessed at
      // here -- the server says which genotypes it just drew.
      B.panel.setColourKeys(r.groups || []);
      B.plotDrawn = true;
    } catch (e) { if (!quiet) alert(e.message); }
  }

  q('plot').onclick = () => drawPlot(false);
  // Turning a control is a request to see the result of turning it.  The size
  // boxes are debounced like labkit's own: typing "55" is two input events and
  // each redraw is a matplotlib render on the server.
  q('unit').onchange = () => drawPlot(true);
  q('readouts').onchange = () => drawPlot(true);
  q('ncol').oninput = replot;
  if (q('stains')) q('stains').onchange = replot;

  function renderCohort(c, csv) {
    const box = q('cohort');
    if (!c || c.error) {
      box.innerHTML = c && c.error ? `<div class="hint">cohort summary failed: ${c.error}</div>` : '';
      return;
    }
    const fmt = { p: v => (v < 0.001 ? v.toExponential(1) : v.toFixed(3)) };
    box.innerHTML =
      `<h3 class="sec">Per mouse, per ${cfg.regionWord}</h3>` +
      `<div class="hint">Slices of one ${cfg.regionWord} in one mouse, pooled by <b>adding the
        integrals</b> — total collagen over total muscle, not the mean of the section ratios,
        so a short off-cut cannot weigh as much as a whole ring.</div>` +
      gridTable(c.by_mouse_segment, cfg.pooledCols) +

      `<h3 class="sec">Per mouse</h3>` +
      `<div class="hint">Every ${cfg.regionWord} pooled: the ${cfg.pooledWord} number for that
        animal.${cfg.kind === 'area' ? ' Worth reading beside the per-' + cfg.regionWord +
        ' table rather than instead of it: baseline collagen differs between regions by more' +
        ' than genotype moves it within one, so a pooled number partly reports which regions' +
        ' happened to be sampled.' : ''}</div>` +
      gridTable(c.by_mouse, cfg.pooledCols) +

      `<h3 class="sec">Per genotype</h3>` +
      `<div class="hint">Mean ± SD across <b>mice</b>. <b>n_mice</b> is the n that counts;
        <b>n_sections</b> is shown beside it so the difference is visible — twenty sections
        from three animals is an n of 3, not 20.</div>` +
      gridTable(c.by_genotype, ['segment', 'genotype', 'readout', 'n_mice', 'n_sections',
        'mean', 'sd', 'sem', 'min', 'max']) +

      `<h3 class="sec">Genotype comparison</h3>` +
      (c.comparison && c.comparison.length
        ? `<div class="hint">Welch's t-test on the per-mouse values, two-sided. Blank p means
            fewer than two mice in a group, which has a direction but no p-value.</div>` +
          gridTable(c.comparison, ['segment', 'readout', 'genotype_a', 'genotype_b', 'n_a', 'n_b',
            'mean_a', 'mean_b', 'difference', 'ratio', 't', 'p'], fmt)
        : `<div class="hint">Only one genotype in this batch — nothing to compare.</div>`) +

      (csv && Object.keys(csv).length
        ? `<div class="hint">Tables written: ${Object.values(csv).join(' · ')}</div>` : '');
  }

  return B;
}

/* --- editable paths ------------------------------------------------------
 * A region drawn freehand arrives as several hundred points two microns
 * apart, which is a fine mask and a hopeless thing to edit: there is no
 * "the point" to grab. So a stroke is reduced to a few dozen anchors on the
 * way in, and it is the anchors you move afterwards.
 *
 * What is drawn, and what is sent to be measured, is a smooth curve resampled
 * through those anchors -- not the anchor polygon itself. The two must be the
 * same curve or the mask would not be the shape on screen, which is the one
 * thing a masking tool cannot get wrong.
 */

/** Ramer-Douglas-Peucker: keep the points that carry the shape. */
function simplifyPath(points, tolUm) {
  if (points.length < 3) return points.slice();
  const keep = new Array(points.length).fill(false);
  keep[0] = keep[points.length - 1] = true;
  const stack = [[0, points.length - 1]];
  while (stack.length) {
    const [a, b] = stack.pop();
    if (b <= a + 1) continue;
    const [ax, ay] = points[a], [bx, by] = points[b];
    const dx = bx - ax, dy = by - ay;
    const len = Math.hypot(dx, dy) || 1e-9;
    let far = -1, fd = tolUm;
    for (let i = a + 1; i < b; i++) {
      const d = Math.abs((points[i][0] - ax) * dy - (points[i][1] - ay) * dx) / len;
      if (d > fd) { fd = d; far = i; }
    }
    if (far > 0) { keep[far] = true; stack.push([a, far], [far, b]); }
  }
  return points.filter((_, i) => keep[i]);
}

/** A closed Catmull-Rom through the anchors, sampled every *stepUm*. */
function smoothClosed(anchors, stepUm) {
  const n = anchors.length;
  if (n < 3) return anchors.slice();
  const at = (i) => anchors[((i % n) + n) % n];
  const out = [];
  for (let i = 0; i < n; i++) {
    const p0 = at(i - 1), p1 = at(i), p2 = at(i + 1), p3 = at(i + 2);
    // Enough samples that the curve is smooth at this segment's length, and
    // no more: the polygon is rasterised on the server and extra vertices
    // buy nothing but bytes.
    const steps = Math.max(2, Math.min(40, Math.round(Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / Math.max(stepUm, 1e-6))));
    for (let s = 0; s < steps; s++) {
      const t = s / steps, t2 = t * t, t3 = t2 * t;
      out.push([
        0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t +
               (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
               (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3),
        0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t +
               (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
               (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3),
      ]);
    }
  }
  return out;
}

/** Distance from a point to a segment, and where along it the foot falls. */
function segDistance(pt, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const l2 = dx * dx + dy * dy;
  const t = l2 ? Math.max(0, Math.min(1, ((pt[0] - a[0]) * dx + (pt[1] - a[1]) * dy) / l2)) : 0;
  return { d: Math.hypot(pt[0] - (a[0] + t * dx), pt[1] - (a[1] + t * dy)), t };
}

function pathInside(pt, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > pt[1]) !== (yj > pt[1]) &&
        pt[0] < ((xj - xi) * (pt[1] - yi)) / ((yj - yi) || 1e-9) + xi) inside = !inside;
  }
  return inside;
}

/* --- one organ section, measured by area --------------------------------
 * No wall is straightened here because there is none to follow: interstitial
 * fibrosis is diffuse, scattered between myocytes or around tubules, and the
 * question is simply how much of the tissue is collagen.
 */
function organParams(organ) {
  const g = (k) => document.querySelector(`#f-${organ}-${k}`);
  const num = (k) => (g(k) && g(k).value.trim() !== '' ? +g(k).value : null);
  return {
    analysis_px_um: num('apx'),
    tissue_od: num('tod'),
    min_tissue_area_um2: num('minarea'),
    blue_threshold: num('blue'),
    blue_rule: (g('blue') && g('blue').value.trim() !== '') ? 'absolute' : 'dominance',
    stain_estimation: g('stains') ? g('stains').value : 'fixed',
  };
}

function makeOrganPane(organ, panelHost, stageHost) {
  const cfg = ORGANS[organ];
  const p = `f-${organ}`;
  panelHost.innerHTML = `
    <div class="group">
      <h3>What to measure</h3>
      <label class="row"><span><input type="radio" name="${p}src" id="${p}-src-sess" checked> stitched session</span></label>
      <label class="row"><span>Island</span><select id="${p}-island"><option value="0">0</option></select></label>
      <label class="row"><span>Render scale</span>
        <input id="${p}-render" type="number" value="0.35" min="0.05" max="1" step="0.05"></label>
      <label class="row"><span><input type="radio" name="${p}src" id="${p}-src-file"> image file</span></label>
      <div class="row">
        <input id="${p}-file" type="text" placeholder="/path/to/mosaic.png">
        <button id="${p}-pick" class="ghost small">Browse…</button>
      </div>
      <label class="row"><span>µm per pixel</span>
        <input id="${p}-px" type="number" value="" step="0.01" placeholder="from file"></label>
    </div>

    <div class="group">
      <h3>Measurement</h3>
      <label class="row"><span>Analysis µm/px</span>
        <input id="${p}-apx" type="number" value="0.5" min="0.1" max="4" step="0.1"></label>
      <label class="row"><span>Tissue OD threshold</span>
        <input id="${p}-tod" type="number" value="" step="0.02" placeholder="automatic"></label>
      <label class="row"><span>Smallest speck µm²</span>
        <input id="${p}-minarea" type="number" value="2000" min="0" step="500"></label>
      <label class="row"><span>Blue threshold</span>
        <input id="${p}-blue" type="number" value="" step="0.05" placeholder="collagen &gt; muscle"></label>
      <div class="hint">Left empty, a pixel counts as fibrotic when its collagen outweighs its
        muscle — no threshold, and comparable between slides. Enter a number for the classical
        absolute-threshold index.</div>
      <label class="row"><span>Stain vectors</span>
        <select id="${p}-stains">
          <option value="fixed" selected>fixed trichrome</option>
          <option value="per_image">read off this image</option>
        </select></label>
      <div class="hint"><b>Fixed</b> unless you have a reason. Reading the vectors off a single
        field needs both stains to be in it: nearly pure myocardium has almost no collagen, the
        estimator fits its second direction to noise, and that noise comes back as collagen —
        on this data it called healthy heart 23% fibrotic where fixed vectors read 0.4%.</div>
      <input id="${p}-name" type="text" placeholder="output name" value="${organ}">
      <input id="${p}-out" type="text" placeholder="output folder">
      <button id="${p}-run" class="primary wide">Measure</button>
    </div>

    <div class="group">
      <h3>Lesions</h3>
      <label class="row"><span>Lesion threshold</span>
        <input id="${p}-lesion-k" type="number" value="3" min="0.5" max="8" step="0.5"></label>
      <label class="row"><span>Smallest lesion µm²</span>
        <input id="${p}-lesion-min" type="number" value="2000" min="0" step="500"></label>
      <div class="hint">Collagen standing this many robust deviations above <i>this section's
        own</i> level. The readout says what the cut was and what the section reached, so
        <b>none found</b> is an answer you can check.</div>
    </div>`;

  stageHost.innerHTML = `
    <div id="${p}-progress" class="progress hidden"><div class="bar"></div><span></span></div>
    <div id="${p}-numbers" class="editor hidden">
      <div class="row" style="gap:8px;align-items:center;margin-bottom:6px"><b>Numbers</b></div>
      <div id="${p}-totals" class="readout">—</div>
      <div id="${p}-files" class="readout small"></div>
    </div>
    <div id="${p}-figstyle" class="editor hidden">
      <div class="row" style="gap:8px;align-items:center;margin-bottom:6px">
        <b>Figure style</b>
        <span class="hint" style="margin:0">Restyles the figure below in place — the same PNG
          and SVG a run saves, redrawn from the same numbers. Rotating turns the picture
          panels only: area, optical density and the comparison between two stains do not
          care which way the section was laid on the stage, so this is the same analysis at
          a different angle.</span>
      </div>
      <div class="row" style="gap:14px;flex-wrap:wrap;align-items:center">
        <label style="flex:0 0 auto;display:flex;gap:5px;align-items:center">Rotate section
          <input id="${p}-fs-rot" type="number" value="0" step="1"></label>
        <label style="flex:0 0 auto;display:flex;gap:5px;align-items:center">Panel size (in)
          <input id="${p}-fs-panel" type="number" value="3" min="1" step="0.25"></label>
        <label style="flex:0 0 auto;display:flex;gap:5px;align-items:center">Font size (pt)
          <input id="${p}-fs-font" type="number" value="10.5" min="6" max="24" step="0.5"></label>
        <label style="flex:0 0 auto;display:flex;gap:5px;align-items:center"
               title="Display levels, one per stain, for the picture panels only. Turning the red down leaves the blue exactly as it was — which is what makes it usable for scoring by eye. No number in any table is affected.">Levels
          <span class="pair">
            <label><span class="dim">red</span>
              <input id="${p}-fs-gain" type="number" value="1" min="0" max="2" step="0.05"></label>
            <label><span class="dim">blue</span>
              <input id="${p}-fs-gain-blue" type="number" value="1" min="0" max="2" step="0.05"></label>
            <label><span class="dim">nuc</span>
              <input id="${p}-fs-gain-nuc" type="number" value="1" min="0" max="2" step="0.05"></label>
          </span></label>
        <label class="inline small"><input id="${p}-fs-grid" type="checkbox" checked> gridlines</label>
        <label class="inline small"><input id="${p}-fs-spines" type="checkbox" checked> plot boxes</label>
        <span class="grow"></span>
        <span id="${p}-fs-note" class="hint" style="margin:0"></span>
        <button id="${p}-fs-reset" class="ghost small">Reset</button>
      </div>
    </div>
    <div id="${p}-editor" class="editor hidden">
      <div class="row" style="gap:8px;align-items:center;margin-bottom:6px">
        <b>Regions</b>
        <span class="hint" style="margin:0">Draw a loop on the section. <b>Mask off</b>
          takes what you enclose out of the tissue entirely — a fold, a tear, mounting
          medium — so it counts towards neither the collagen nor the area it is measured
          against. Any other label measures that part on its own <i>as well as</i> within
          the whole, which is what a ${cfg.label.toLowerCase()} spanning several regions
          needs: baseline collagen differs between them by more than genotype moves it
          within one.</span>
      </div>
      <canvas id="${p}-canvas"></canvas>
      <div class="row" style="gap:6px;margin-top:6px;align-items:center;flex-wrap:wrap">
        <label class="inline small">Tool
          <select id="${p}-rtool">
            <option value="region">draw a region</option>
            <option value="edit">edit points</option>
            <option value="point">mark lesions</option>
            <option value="reject">remove a focus</option>
          </select></label>
        <label class="inline small">Draw as
          <select id="${p}-rlabel">
            <option value="__exclude__">mask off (remove from tissue)</option>
            ${cfg.parts.map(x => `<option value="${x}">${x}</option>`).join('')}
            <option value="__other__">other…</option>
          </select></label>
        <input id="${p}-rother" type="text" placeholder="region name"
               style="width:130px;display:none">
        <button id="${p}-rundo" class="ghost small">Undo</button>
        <button id="${p}-rclear" class="ghost small">Clear</button>
        <button id="${p}-rclearpts" class="ghost small">Clear marks</button>
        <button id="${p}-rrestore" class="ghost small">Restore removed foci</button>
        ${organ === 'kidney' ? `
        <button id="${p}-rguide" class="ghost small"
                title="Texture gradient across the section: red where it looks papillary (parallel tubules, more matrix), blue where it looks cortical. It asserts no boundary — you draw it.">Texture guide</button>
        <button id="${p}-rsuggest" class="ghost small"
                title="A first guess, and not a good one: checked against the region lists in your folder names it got the number of zones right in 5 of 17 sections, and found three zones in crops cut entirely from cortex. Correct it, or clear it and use the guide.">Propose zones</button>` : ''}
        <span id="${p}-rnote" class="hint" style="margin:0"></span>
        <span class="hint" style="margin:0">Draw roughly, then <b>edit points</b>: drag an
          anchor to move it, click the path to add one, alt-click an anchor to remove it.</span>
        <span class="grow"></span>
        <button id="${p}-rrun" class="primary small">Re-measure</button>
      </div>
    </div>
    <div id="${p}-summary"></div>`;

  const q = (k) => panelHost.querySelector(`#${p}-${k}`) || stageHost.querySelector(`#${p}-${k}`);

  q('pick').onclick = async () => {
    const picked = await LK.browse({
      mode: 'file', exts: IMAGE_EXTS, path: q('file').value.trim(),
      title: 'Image to measure', hint: 'A stitched mosaic, or any single field.',
    });
    if (picked) { q('file').value = picked; q('src-file').checked = true; }
  };

  /* --- the region editor ------------------------------------------------
     One tool for two jobs, because they are the same drawing: a loop you
     enclose is either taken out of the tissue or measured as a part of it,
     and which of those it is is a label rather than a different gesture. */
  const R = { img: null, section: null, guideImg: null, scale: 1,
              regions: [], points: [], lesions: [], rejected: [], drawing: null,
              sel: -1, grab: null, hover: null };

  // A stroke's worth of freehand becomes a handful of anchors, and the curve
  // through them is what gets drawn and what gets measured.
  const rebuild = (reg) => {
    reg.um = smoothClosed(reg.anchors, 6 * R.scale);
    return reg;
  };
  const makeRegion = (name, stroke) => rebuild({
    name,
    anchors: simplifyPath(stroke, 8 * R.scale),
  });

  const rlabel = () => (q('rlabel').value === '__other__'
    ? (q('rother').value.trim() || 'region')
    : q('rlabel').value);
  const rnote = (m) => { q('rnote').textContent = m || ''; };
  q('rlabel').onchange = () => {
    q('rother').style.display = q('rlabel').value === '__other__' ? '' : 'none';
  };

  function rfit() {
    const c = q('canvas');
    if (!R.img) return { s: 1, ox: 0, oy: 0 };
    const s = Math.min(c.width / R.img.width, c.height / R.img.height);
    return { s, ox: (c.width - R.img.width * s) / 2, oy: (c.height - R.img.height * s) / 2 };
  }
  const rToCv = (pt, f) => [f.ox + (pt[0] / R.scale) * f.s, f.oy + (pt[1] / R.scale) * f.s];
  const rToUm = (x, y, f) => [((x - f.ox) / f.s) * R.scale, ((y - f.oy) / f.s) * R.scale];

  function drawRegions() {
    const c = q('canvas'), g = c.getContext('2d');
    g.clearRect(0, 0, c.width, c.height);
    if (!R.img) return;
    const f = rfit();
    g.drawImage(R.img, f.ox, f.oy, R.img.width * f.s, R.img.height * f.s);
    const trace = (pts, close) => {
      g.beginPath();
      pts.forEach((pt, i) => { const [x, y] = rToCv(pt, f); i ? g.lineTo(x, y) : g.moveTo(x, y); });
      if (close) g.closePath();
    };
    R.regions.forEach((reg, ri) => {
      const cut = reg.name === '__exclude__';
      trace(reg.um, true);
      g.fillStyle = cut ? 'rgba(217,51,38,.28)' : 'rgba(11,89,242,.18)';
      g.fill();
      g.lineWidth = ri === R.sel ? 2.6 : 1.8;
      g.strokeStyle = cut ? '#d93326' : '#0b59f2'; g.stroke();
      const cx = reg.um.reduce((a, b) => a + b[0], 0) / reg.um.length;
      const cy = reg.um.reduce((a, b) => a + b[1], 0) / reg.um.length;
      const [x, y] = rToCv([cx, cy], f);
      g.font = '600 12px system-ui'; g.textAlign = 'center'; g.textBaseline = 'middle';
      g.lineWidth = 3; g.strokeStyle = 'rgba(255,255,255,.85)';
      const text = cut ? 'masked off' : reg.name;
      g.strokeText(text, x, y); g.fillStyle = cut ? '#d93326' : '#0b59f2';
      g.fillText(text, x, y);

      // Anchors only on the region being edited.  Showing every anchor of
      // every region at once turns the picture into a field of squares and
      // makes it impossible to see the tissue underneath, which is the thing
      // you are drawing against.
      if (ri === R.sel && q('rtool').value === 'edit') {
        reg.anchors.forEach((a, ai) => {
          const [ax, ay] = rToCv(a, f);
          const on = R.hover && R.hover.region === ri && R.hover.anchor === ai;
          g.beginPath(); g.rect(ax - 4, ay - 4, 8, 8);
          g.fillStyle = on ? '#ffffff' : (cut ? '#d93326' : '#0b59f2');
          g.fill();
          g.lineWidth = 1.2; g.strokeStyle = '#1a1a1a'; g.stroke();
        });
      }
    });
    // Detected lesions outlined on the section.  The figure panel gets an
    // arrowhead instead, because there the outline would sit on the tissue
    // being scored; here you are looking at the section to check them.  One
    // a person has clicked away goes red and dashed rather than disappearing
    // outright -- it is still there until the next measurement actually
    // drops it, and the dashed red is the same "marked for removal" language
    // the mask-off region uses.
    R.lesions.forEach((poly) => {
      const rejected = R.rejected.some(pt => pathInside(pt, poly));
      trace(poly, true);
      g.lineWidth = 2.4; g.strokeStyle = 'rgba(0,0,0,.45)'; g.stroke();
      trace(poly, true);
      if (rejected) {
        g.setLineDash([5, 4]);
        g.lineWidth = 1.4; g.strokeStyle = 'rgba(217,51,38,.95)'; g.stroke();
        g.setLineDash([]);
      } else {
        g.lineWidth = 1.1; g.strokeStyle = 'rgba(255,255,255,.95)'; g.stroke();
      }
    });
    // Marks last, so they sit on top of any region they fall inside.
    R.points.forEach((pt, i) => {
      const [x, y] = rToCv(pt, f);
      g.beginPath(); g.arc(x, y, 5, 0, Math.PI * 2);
      g.fillStyle = 'rgba(255,176,0,.85)'; g.fill();
      g.lineWidth = 1.4; g.strokeStyle = '#1a1a1a'; g.stroke();
      g.font = '600 11px system-ui'; g.textAlign = 'left'; g.textBaseline = 'middle';
      g.lineWidth = 3; g.strokeStyle = 'rgba(0,0,0,.6)';
      g.strokeText(String(i + 1), x + 7, y);
      g.fillStyle = '#ffb000'; g.fillText(String(i + 1), x + 7, y);
    });
    if (R.drawing && R.drawing.length > 1) {
      trace(R.drawing, false);
      g.lineWidth = 2.4; g.strokeStyle = '#ffffff'; g.stroke();
      g.lineWidth = 1.5; g.strokeStyle = '#2f7d4f'; g.stroke();
    }
  }

  function showEditor(over, pxum) {
    const im = new Image();
    im.onload = () => {
      R.img = im; R.section = im; R.guideImg = null; R.scale = pxum;
      // Unhide first: a hidden element measures zero wide, and a canvas sized
      // from that keeps its 300x150 default and draws the section into a
      // postage stamp you cannot draw a region on.
      q('editor').classList.remove('hidden');
      // Fit inside a box rather than to a width.  A kidney section can be five
      // times taller than it is wide, and sizing by width alone turns it into
      // a strip several screens long that you cannot see a region on.
      const c = q('canvas');
      const boxW = Math.max(320, Math.min(1100, c.parentElement.clientWidth - 24));
      const boxH = 1100;
      const k = Math.min(boxW / im.width, boxH / im.height);
      c.width = Math.round(im.width * k);
      c.height = Math.round(im.height * k);
      drawRegions();
    };
    im.src = '/api/file?path=' + encodeURIComponent(over) + '&t=' + Date.now();
  }

  (function wireRegions() {
    const c = q('canvas');
    // A canvas that has not been laid out yet measures zero wide, and dividing
    // by that gives Infinity -- which travels all the way to the server as a
    // null coordinate and comes back as a mark with no tissue at it.  Refuse
    // the click instead, and say why.
    const at = (ev) => {
      const r = c.getBoundingClientRect();
      if (!(r.width > 0 && r.height > 0)) return null;
      const pt = rToUm((ev.clientX - r.left) * c.width / r.width,
                       (ev.clientY - r.top) * c.height / r.height, rfit());
      return (isFinite(pt[0]) && isFinite(pt[1])) ? pt : null;
    };
    const isPointTool = () => q('rtool').value === 'point';
    const isEditTool = () => q('rtool').value === 'edit';
    const isRejectTool = () => q('rtool').value === 'reject';

    // A click toggles the whole lesion under it, by geometry rather than by
    // index: the outlines get rebuilt from scratch on every measurement, so
    // an index recorded now would point at the wrong patch, or none, next
    // time. A raw point inside the patch survives that -- it is meaningless
    // on its own and simply erases whichever lesion the server finds at it.
    function toggleLesionAt(pt) {
      const hit = R.lesions.find(poly => pathInside(pt, poly));
      if (!hit) return rnote('no detected focus here — click inside an outlined one');
      const already = R.rejected.filter(p2 => pathInside(p2, hit));
      if (already.length) {
        R.rejected = R.rejected.filter(p2 => !already.includes(p2));
        rnote('restored — re-measure to bring it back');
      } else {
        R.rejected.push(pt);
        rnote(`${R.rejected.length} focus/foci marked for removal — re-measure to drop ` +
              `${R.rejected.length === 1 ? 'it' : 'them'} from the count`);
      }
      drawRegions();
    }

    // Hit tests in microns, using a tolerance that is a fixed number of screen
    // pixels: a grab radius fixed in microns would be unusable when zoomed out
    // and enormous when zoomed in.
    const grabUm = () => {
      const rect = c.getBoundingClientRect();
      const pxPerUm = (rect.width / c.width) * rfit().s / R.scale;
      return 9 / Math.max(pxPerUm, 1e-9);
    };

    function nearestAnchor(pt, tol) {
      let best = null, bd = tol;
      R.regions.forEach((reg, ri) => {
        if (R.sel >= 0 && ri !== R.sel) return;
        reg.anchors.forEach((a, ai) => {
          const d = Math.hypot(a[0] - pt[0], a[1] - pt[1]);
          if (d < bd) { bd = d; best = { region: ri, anchor: ai }; }
        });
      });
      return best;
    }

    function nearestSegment(pt, tol) {
      let best = null, bd = tol;
      R.regions.forEach((reg, ri) => {
        if (R.sel >= 0 && ri !== R.sel) return;
        const n = reg.anchors.length;
        for (let i = 0; i < n; i++) {
          const { d } = segDistance(pt, reg.anchors[i], reg.anchors[(i + 1) % n]);
          if (d < bd) { bd = d; best = { region: ri, after: i }; }
        }
      });
      return best;
    }

    c.onpointerdown = (ev) => {
      if (isRejectTool()) {
        const pt = at(ev);
        if (!pt) return rnote('the picture is still loading — try again in a moment');
        return toggleLesionAt(pt);
      }
      if (isEditTool()) {
        const pt = at(ev);
        if (!pt) return rnote('the picture is still loading — try again in a moment');
        const tol = grabUm();
        const hit = nearestAnchor(pt, tol);
        if (hit) {
          if (ev.altKey) {
            // Three anchors is the least that encloses anything; below that
            // there is no region left to edit.
            const reg = R.regions[hit.region];
            if (reg.anchors.length <= 3) return rnote('a region needs at least three points');
            reg.anchors.splice(hit.anchor, 1); rebuild(reg);
            rnote('point removed'); return drawRegions();
          }
          R.sel = hit.region;
          R.grab = hit;
          c.setPointerCapture(ev.pointerId);
          return drawRegions();
        }
        const seg = nearestSegment(pt, tol);
        if (seg) {
          const reg = R.regions[seg.region];
          reg.anchors.splice(seg.after + 1, 0, pt); rebuild(reg);
          R.sel = seg.region;
          R.grab = { region: seg.region, anchor: seg.after + 1 };
          c.setPointerCapture(ev.pointerId);
          rnote('point added — drag it into place');
          return drawRegions();
        }
        // Nothing near the path: pick whichever region encloses the click, so
        // switching between overlapping regions is a click rather than a hunt.
        const inside = R.regions.findIndex(reg => pathInside(pt, reg.um));
        R.sel = inside;
        rnote(inside < 0 ? 'no region here — click inside one to edit it'
                         : `editing "${R.regions[inside].name === '__exclude__' ? 'masked off' : R.regions[inside].name}"`);
        return drawRegions();
      }
      if (isPointTool()) {
        // Alt-click removes the nearest mark: a counting tool you cannot
        // correct is a counting tool you have to start over.
        const pt = at(ev);
        if (!pt) return rnote('the picture is still loading — click again in a moment');
        if (ev.altKey && R.points.length) {
          let best = 0, bd = Infinity;
          R.points.forEach((q2, i) => {
            const d = Math.hypot(q2[0] - pt[0], q2[1] - pt[1]);
            if (d < bd) { bd = d; best = i; }
          });
          R.points.splice(best, 1);
        } else {
          R.points.push(pt);
        }
        rnote(`${R.points.length} mark(s) — re-measure to get a number for each ` +
              `(alt-click removes one)`);
        return drawRegions();
      }
      const start = at(ev);
      if (!start) return rnote('the picture is still loading — try again in a moment');
      c.setPointerCapture(ev.pointerId); R.drawing = [start];
    };
    c.onpointermove = (ev) => {
      if (isEditTool()) {
        const pt = at(ev);
        if (!pt) return;
        if (R.grab) {
          const reg = R.regions[R.grab.region];
          reg.anchors[R.grab.anchor] = pt; rebuild(reg);
          return drawRegions();
        }
        const was = R.hover;
        R.hover = nearestAnchor(pt, grabUm());
        if (JSON.stringify(was) !== JSON.stringify(R.hover)) drawRegions();
        return;
      }
      if (!R.drawing || isPointTool() || isRejectTool()) return;
      const pt = at(ev), last = R.drawing[R.drawing.length - 1];
      if (pt && Math.hypot(pt[0] - last[0], pt[1] - last[1]) > 2 * R.scale) {
        R.drawing.push(pt); drawRegions();
      }
    };
    c.onpointerup = () => {
      if (R.grab) { R.grab = null; rnote('moved — re-measure to apply'); return drawRegions(); }
      const stroke = R.drawing; R.drawing = null;
      // A loop, not a line: three points is the least that encloses anything,
      // and anything shorter is a misclick rather than a region.
      if (stroke && stroke.length >= 3) {
        R.regions.push(makeRegion(rlabel(), stroke));
        R.sel = R.regions.length - 1;
        const cut = R.regions.filter(x => x.name === '__exclude__').length;
        rnote(`${R.regions.length - cut} region(s), ${cut} masked off — ` +
              `switch to "edit points" to adjust, then re-measure`);
      }
      drawRegions();
    };
    q('rundo').onclick = () => {
      R.regions.pop(); R.sel = Math.min(R.sel, R.regions.length - 1); R.hover = null;
      rnote('undone'); drawRegions();
    };
    if (q('rguide')) {
      // The guide is a background, not a boundary.  Toggling between it and
      // the section is the whole interaction: you look at the gradient, then
      // draw on the tissue.
      q('rguide').onclick = async () => {
        if (R.section && R.img === R.guideImg) { R.img = R.section; return drawRegions(); }
        if (R.guideImg) { R.img = R.guideImg; return drawRegions(); }
        rnote('measuring texture…');
        try {
          const r = await runJob((await api('/api/zones', zoneBody())).job,
                                 q('progress'), 'texture');
          await loadGuide(r.guide);
          rnote('red looks papillary, blue cortical — no boundary is claimed here');
        } catch (e) { rnote(e.message); }
      };
      q('rsuggest').onclick = async () => {
        rnote('proposing…');
        try {
          const r = await runJob((await api('/api/zones', zoneBody())).job,
                                 q('progress'), 'zones');
          if (!r.polygons.length) return rnote(r.note || 'nothing proposed');
          // Proposed boundaries go in as anchors, not as fixed outlines: the
          // whole point of a proposal you are told to distrust is that you can
          // take hold of it and move it.
          r.polygons.forEach(z => R.regions.push(makeRegion(z.name, z.polygon)));
          R.sel = R.regions.length - 1;
          q('rtool').value = 'edit'; q('rtool').dispatchEvent(new Event('change'));
          rnote(`${r.n_zones} zone(s) proposed — now in "edit points": drag the anchors ` +
                `onto the real boundaries` + (r.note ? ` (${r.note})` : ''));
        } catch (e) { rnote(e.message); }
      };
    }
    function zoneBody() {
      const useSession = q('src-sess').checked;
      const b = { organ, out_dir: q('out').value.trim() || null,
                  name: q('name').value.trim() || organ };
      if (useSession) { b.sid = S.sid; b.island = q('island').value; b.render_scale = +q('render').value; }
      else { b.image = q('file').value.trim(); if (q('px').value.trim() !== '') b.px_um = +q('px').value; }
      return b;
    }
    async function loadGuide(path) {
      await new Promise((done, fail) => {
        const im = new Image();
        im.onload = () => { R.guideImg = im; R.img = im; drawRegions(); done(); };
        im.onerror = fail;
        im.src = '/api/file?path=' + encodeURIComponent(path) + '&t=' + Date.now();
      });
    }
    q('rclear').onclick = () => {
      R.regions = []; R.sel = -1; R.hover = null; rnote('regions cleared'); drawRegions();
    };
    q('rclearpts').onclick = () => { R.points = []; rnote('marks cleared'); drawRegions(); };
    q('rrestore').onclick = () => {
      R.rejected = []; rnote('every removed focus restored — re-measure to bring them back');
      drawRegions();
    };
    q('rtool').onchange = () => {
      const tool = q('rtool').value;
      q('rlabel').disabled = (tool !== 'region');
      c.style.cursor = tool === 'point' ? 'copy'
                     : tool === 'edit' ? 'default'
                     : tool === 'reject' ? 'not-allowed' : 'crosshair';
      if (tool === 'edit' && R.sel < 0 && R.regions.length) R.sel = R.regions.length - 1;
      R.hover = null;
      rnote({
        point: 'click to mark a lesion, alt-click to remove one',
        edit: 'drag an anchor to move it · click the path to add one · alt-click to remove one',
        region: 'drag to enclose a region',
        reject: 'click a detected focus to remove it from the count · click again to restore it',
      }[tool]);
      drawRegions();
    };
    q('rrun').onclick = () => {
      // No precondition: this button now applies whatever the editor holds --
      // regions, marks, or a focus clicked away -- and a region was the only
      // one of the three that used to be required.  Guarding on it left marks
      // and removals with a button that silently did nothing.
      q('run').click();
    };
  })();

  /* --- figure style --------------------------------------------------
     The figure is matplotlib's, not a client redraw of it: the server keeps
     the result behind each figure (FIGURE_CACHE in app.py) and a style change
     asks it to redraw that same figure, overwriting the PNG and the SVG a
     plain run would have saved.  Sticky across runs, because it answers "how
     should this kind of figure look" and re-measuring should not undo it. */
  const FS_DEFAULT = { rot: 0, panel: 3, font: 10.5, grid: true, spines: true,
                       gain: 1, gainBlue: 1, gainNuc: 1 };
  const FS = { ...FS_DEFAULT };
  let fsPath = null;

  function readFigStyle() {
    FS.rot = +q('fs-rot').value || 0;
    FS.panel = +q('fs-panel').value || FS_DEFAULT.panel;
    FS.font = +q('fs-font').value || FS_DEFAULT.font;
    FS.grid = q('fs-grid').checked;
    FS.spines = q('fs-spines').checked;
    const lvl = (k) => { const v = Number(q(k).value); return isFinite(v) ? Math.max(0, Math.min(2, v)) : 1; };
    FS.gain = lvl('fs-gain'); FS.gainBlue = lvl('fs-gain-blue'); FS.gainNuc = lvl('fs-gain-nuc');
  }
  function writeFigStyle() {
    q('fs-rot').value = FS.rot; q('fs-panel').value = FS.panel;
    q('fs-font').value = FS.font;
    q('fs-grid').checked = FS.grid; q('fs-spines').checked = FS.spines;
    q('fs-gain').value = FS.gain;
    q('fs-gain-blue').value = FS.gainBlue; q('fs-gain-nuc').value = FS.gainNuc;
  }
  async function applyFigStyle() {
    // Nothing to restyle until something has been measured.  Said rather than
    // ignored: a control that does nothing and reports nothing reads as broken.
    if (!fsPath) { q('fs-note').textContent = 'measure something first'; return; }
    readFigStyle();
    q('fs-note').textContent = 'redrawing…';
    try {
      await api('/api/replot', {
        path: fsPath,
        style: { rotation_deg: FS.rot, panel_in: FS.panel, title_pt: FS.font,
                 show_grid: FS.grid, show_spines: FS.spines,
                 gain_muscle: FS.gain, gain_collagen: FS.gainBlue, gain_nuclei: FS.gainNuc },
      });
      const img = q('summary').querySelector('img');
      if (img) img.src = `/api/file?path=${encodeURIComponent(fsPath)}&t=${Date.now()}`;
      q('fs-note').textContent = '';
    } catch (e) { q('fs-note').textContent = 'could not redraw: ' + e.message; }
  }
  const figReplot = LK.debounce(applyFigStyle, 260);
  ['fs-rot', 'fs-panel', 'fs-font', 'fs-gain', 'fs-gain-blue', 'fs-gain-nuc']
    .forEach(k => { q(k).oninput = figReplot; });
  ['fs-grid', 'fs-spines'].forEach(k => { q(k).onchange = applyFigStyle; });
  q('fs-reset').onclick = () => { Object.assign(FS, FS_DEFAULT); writeFigStyle(); applyFigStyle(); };

  q('run').onclick = async () => {
    const useSession = q('src-sess').checked;
    if (useSession && !S.sid) return alert('Load and stitch a folder first, or choose an image file.');
    const src = useSession ? $('#folder').value.trim() : q('file').value.trim();
    if (!useSession && !src) return alert('Choose an image file.');
    if (!q('out').value.trim() && src) {
      q('out').value = src.replace(/\/+$/, '').replace(/\/[^/]*$/, '') + '/analysis';
    }
    const body = {
      organ, params: organParams(organ),
      name: q('name').value.trim() || organ,
      out_dir: q('out').value.trim() || null,
      points: R.points.filter(pt => isFinite(pt[0]) && isFinite(pt[1])),
      lesion_k_mad: +q('lesion-k').value || 3,
      lesion_min_area_um2: +q('lesion-min').value || 2000,
      exclude_lesions_at: R.rejected.filter(pt => isFinite(pt[0]) && isFinite(pt[1])),
      exclude: R.regions.filter(x => x.name === '__exclude__').map(x => x.um),
      regions: R.regions.filter(x => x.name !== '__exclude__')
                        .map(x => ({ name: x.name, polygon: x.um })),
    };
    if (useSession) {
      body.sid = S.sid;
      body.island = q('island').value;
      body.render_scale = +q('render').value;
    } else {
      body.image = src;
      if (q('px').value.trim() !== '') body.px_um = +q('px').value;
    }
    try {
      const r = await runJob((await api('/api/fibrosis', body)).job, q('progress'), 'measuring');
      showOrganResult(organ, q, r);
      q('numbers').classList.remove('hidden');
      // The picture to draw on is the one that was just measured, so a second
      // pass refines the regions on the same frame rather than on a stale one.
      R.lesions = r.lesion_outlines || [];
      showEditor(r.overview, r.overview_px_um);
      fsPath = r.figure_png;
      q('figstyle').classList.remove('hidden');
      // A new run draws at the default angle; put the chosen style back on it
      // rather than making you set it again after every measurement.
      if (JSON.stringify(FS) !== JSON.stringify(FS_DEFAULT)) applyFigStyle();
    } catch (e) { hideProgress(q('progress')); alert(e.message); }
  };

  /* What a saved session needs back from this pane.  Only the anchors are
     kept: the drawn curve is `rebuild`'s pure function of them, so replaying
     it is exact where storing the smoothed points would freeze one rendering
     of a shape the editor is still free to change how it draws. */
  const getState = () => ({
    regions: R.regions.map(x => ({ name: x.name, anchors: x.anchors })),
    points: R.points, rejected: R.rejected,
  });
  const setState = (st) => {
    R.regions = (st && st.regions || []).map(x => rebuild({ name: x.name, anchors: x.anchors }));
    R.points = (st && st.points) || [];
    R.rejected = (st && st.rejected) || [];
    R.sel = -1;
    drawRegions();
  };

  return { q, getState, setState };
}

/* The columns are named for the aorta, where the red channel really is muscle,
   and they stay that way because the pooling and the CSVs are keyed on them.
   What a reader sees does not have to be. */
function forCounterstain(rows, cs) {
  if (cs === 'muscle') return rows;
  const rename = (k) => k.replace(/muscle/g, cs);
  return rows.map(row => Object.fromEntries(
    Object.entries(row).map(([k, v]) => [rename(k), v])));
}

function showOrganResult(organ, q, r) {
  const t = r.totals;
  // Whatever the red channel is in this organ: it stains cytoplasm, which is
  // muscle in a ventricle and tubular epithelium in a kidney.
  const cs = r.counterstain || ORGANS[organ].counterstain || 'muscle';
  const pct = (v) => (100 * v).toFixed(2) + '%';
  q('totals').innerHTML =
    `<b>collagen ÷ ${cs}</b>  ${t.collagen_to_muscle.toFixed(4)}\n` +
    `<b>collagen > ${cs}</b>  ${pct(t.collagen_area_fraction)} of tissue\n` +
    `collagen fraction of stain  ${pct(t.collagen_fraction_of_stain)}\n` +
    `tissue area  ${t.tissue_area_mm2.toFixed(3)} mm²  (${pct(t.tissue_fraction_of_image)} of what was imaged)` +
    (t.excluded_area_mm2 ? `  · ${t.excluded_area_mm2.toFixed(3)} mm² masked off` : '') + `\n` +
    `collagen area  ${t.collagen_area_mm2.toFixed(4)} mm²\n` +
    `collagen per mm² tissue  ${t.collagen_od_um2_per_mm2_tissue.toFixed(0)} OD·µm²\n` +
    `${cs} per mm² tissue  ${t.muscle_od_um2_per_mm2_tissue.toFixed(0)} OD·µm²\n` +
    `<b>lesions</b>  ${t.n_lesions} · ${t.lesion_area_mm2.toFixed(4)} mm² ` +
    `(${(100 * t.lesion_area_fraction).toFixed(2)}% of tissue, ${t.lesions_per_mm2_tissue.toFixed(1)}/mm²)\n` +
    `resolution  ${t.analysis_px_um.toFixed(2)} µm/px · tissue cut at OD ${t.tissue_od_threshold.toFixed(3)}\n` +
    `background removed  collagen ${t.collagen_background_od.toFixed(4)} · ${cs} ${t.counterstain_background_od.toFixed(4)} OD`;
  q('files').innerHTML = Object.entries(r.paths)
    .filter(([k]) => k !== 'overview_px_um')
    .map(([k, v]) => `${k}: ${v}`).join('\n');
  const lesionTable = (r.lesions && r.lesions.length)
    ? `<h3 class="sec">Lesions — ${t.n_lesions} found, ${(100 * t.lesion_area_fraction).toFixed(2)}% of tissue</h3>` +
      `<div class="hint">Patches where collagen stands more than three robust deviations above
        <i>this section's own</i> level, smoothed to lesion scale. Being relative is the point
        and the limitation: it finds <b>focal</b> disease against its surroundings, and is blind
        to <b>diffuse</b> fibrosis — uniformly fibrotic tissue has no focus to stand out from and
        returns no lesions while being thoroughly diseased. The ratio and the area fraction answer
        that; this answers where and how many. Outlined on the section above; pointed at with a
        small white arrowhead on the figure, which is meant to be looked through rather than
        drawn on.</div>` +
      gridTable(r.lesions, ['lesion', 'area_um2', 'equivalent_diameter_um', 'x_um', 'y_um',
                            'collagen_share_mean', 'collagen_share_peak',
                            'collagen_to_counterstain'])
    : '';
  const pointTable = (r.points && r.points.length)
    ? `<h3 class="sec">Marks — ${r.points.length}</h3>` +
      `<div class="hint">Each mark measured over the same ${r.points[0].radius_um} µm disc, so the
        numbers compare between marks. <code>in_lesion</code> is which detected lesion it landed
        in, 0 for none.</div>` +
      gridTable(r.points, ['point', 'x_um', 'y_um', 'n_tissue_px', 'collagen_share',
                           'collagen_to_counterstain', 'collagen_mean_od', 'in_lesion'])
    : '';
  const regionTable = (r.regions && r.regions.length)
    ? `<h3 class="sec">By region</h3>` +
      `<div class="hint">Each region measured on its own, and the whole section beside it.
        Regions are the comparison to make in a ${ORGANS[organ].label.toLowerCase()}: a
        section that spans several of them has one number that mostly reports which ones
        the knife caught.</div>` +
      gridTable(
        forCounterstain([{ name: 'whole section', ...t }].concat(r.regions), cs),
        ['name', 'tissue_area_mm2', `collagen_to_${cs}`, 'collagen_area_fraction',
         'collagen_area_mm2', 'collagen_od_um2_per_mm2_tissue',
         `${cs}_od_um2_per_mm2_tissue`])
    : '';
  q('summary').innerHTML =
    `<h3 class="sec">${r.title || organ}</h3>` + lesionTable + pointTable + regionTable +
    `<img src="/api/file?path=${encodeURIComponent(r.figure_png)}&t=${Date.now()}">` +
    `<div class="hint">Panel F is the one to check: every tissue pixel plotted as its
      ${cs} against its collagen once the background has been taken off both, with the
      diagonal drawn on it. Everything above the diagonal is counted. The dashed line is
      the background level that was subtracted — it is measured on blank slide, so how
      much of the answer rests on that correction stays visible.
      <a href="/api/file?path=${encodeURIComponent(r.figure_svg)}" download>Download SVG</a></div>`;
}

/* Build the panes.  The aorta tab keeps its hand-written wall controls and
   only takes the batch block; heart and kidney are built entirely here. */
document.querySelectorAll('.tabpane[data-organ]').forEach((pane) => {
  const organ = pane.dataset.organ;
  const os = pane.querySelector('.organ-slot');
  if (os) ORGAN_PANES[organ] = makeOrganPane(organ, os, pane.querySelector('.organ-stage'));
  makeBatch(organ, pane.querySelector('.batch-slot'), pane.querySelector('.batch-stage'));
});


resize();
updateHistoryUI();
updateToolUI();


/* One toggle, the same place in every tool, and the choice is remembered for
   the machine rather than for the app -- switching tools should not switch
   look.  Dark is the default: a microscope room is not a bright place. */
LK.mountThemeToggle('header');
