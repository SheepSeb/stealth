// A team of model-based RL agents in a gridworld (Sutton & Barto, ch. 8).
// They share one world model built from what each of them sees (a radio map),
// and each plans on it with prioritized sweeping, so news of food or a new wall
// reaches every agent's values within a tick or two. They split the food: if a
// teammate is closer to the food you're heading for, you leave it to them.
// The world keeps changing (food regrows, rocks shift, visitors drop food and
// build walls), so learning never stops. Places the team hasn't seen in a
// while are worth a little, so idle agents go and check on them.
const DIRS = [[1, 0], [0, 1], [-1, 0], [0, -1]], ARROWS = '>v<^';
const GAMMA = .98, KAPPA = .002, SWEEPS = 200, THETA = 1e-4, VIEW = 6, TICK = 1 / 10;
const IDLE = 20, GIVE_UP = 150; // no food target for IDLE ticks (2 s) -> go and look; abandon a search after GIVE_UP
const BYTES_PER_TRANSITION = 6; // int8 Q, uint16 next state, int8 reward, uint16 last-tried

const FONT = '"JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace';
const canvas = document.getElementById('world');
const ctx = canvas.getContext('2d');
const hud = document.getElementById('hud');
const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
const canHover = matchMedia('(hover: hover)').matches;
let cell, cols, rows, S, wall, blocked, ripe, rocks, food, agents, tick, focus, builtW, warm;
let next, rew, known, seen, seenFlag, lastSeen, bush; // the team's shared world model
let dist, first, queue; // scratch for breadth-first search
let pings = [], hover = -1, drag = null, hudAt = 0, acc = 0, last = performance.now();

const rnd = n => Math.random() * n | 0;
const xy = s => [s % cols, s / cols | 0];
const occupied = s => wall[s] || food.some(f => f.s === s) || agents.some(a => a.s === s);
const freeCell = () => { for (;;) { const s = rnd(S); if (!blocked[s] && !occupied(s)) return s; } };
const maxQ = (a, s) => Math.max(a.Q[s * 4], a.Q[s * 4 + 1], a.Q[s * 4 + 2], a.Q[s * 4 + 3]);
function greedy(a, s) {
  let b = 0;
  for (let u = 1; u < 4; u++) if (a.Q[s * 4 + u] > a.Q[s * 4 + b]) b = u;
  return b;
}
// greedy on Q plus a bonus for actions not tried in a while
function choose(a, s) {
  let b = 0, bv = -Infinity;
  for (let u = 0; u < 4; u++) {
    const v = a.Q[s * 4 + u] + KAPPA * Math.sqrt(Math.min(tick - a.tried[s * 4 + u], 2000));
    if (v > bv) { bv = v; b = u; }
  }
  return b;
}
function move(s, u) {
  const [x0, y0] = xy(s), x = x0 + DIRS[u][0], y = y0 + DIRS[u][1];
  if (x < 0 || y < 0 || x >= cols || y >= rows) return s;
  return wall[y * cols + x] ? s : y * cols + x;
}

// max-heap of transition indices keyed by priority
class Heap {
  p = []; v = [];
  get size() { return this.p.length; }
  push(pr, val) {
    let i = this.p.length;
    this.p.push(pr); this.v.push(val);
    while (i) {
      const q = (i - 1) >> 1;
      if (this.p[q] >= pr) break;
      this.p[i] = this.p[q]; this.v[i] = this.v[q]; i = q;
    }
    this.p[i] = pr; this.v[i] = val;
  }
  pop() {
    const top = this.v[0], pr = this.p.pop(), val = this.v.pop(), n = this.p.length;
    if (n) {
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= n) break;
        if (c + 1 < n && this.p[c + 1] > this.p[c]) c++;
        if (this.p[c] <= pr) break;
        this.p[i] = this.p[c]; this.v[i] = this.v[c]; i = c;
      }
      this.p[i] = pr; this.v[i] = val;
    }
    return top;
  }
}

// reward as this agent sees it: food it has left to a teammate is worth nothing to it
const R = (a, j) => a.skip.has(next[j]) ? 0 : rew[j];
const backup = (a, j) => R(a, j) + GAMMA * maxQ(a, next[j]);
function push(a, j) {
  const p = Math.abs(backup(a, j) - a.Q[j]);
  if (p > THETA) a.pq.push(p, j);
}
// queue every known transition that leads into s (from a neighbour, or s bumping a wall)
function pushInto(a, s) {
  const [x, y] = xy(s);
  for (let u = 0; u < 4; u++) {
    const cx = x - DIRS[u][0], cy = y - DIRS[u][1];
    if (cx >= 0 && cy >= 0 && cx < cols && cy < rows && next[(cy * cols + cx) * 4 + u] === s) push(a, (cy * cols + cx) * 4 + u);
    if (next[s * 4 + u] === s) push(a, s * 4 + u);
  }
}
// prioritized sweeping: back up the biggest surprises first, then their predecessors
function sweep(a) {
  for (let k = 0; k < SWEEPS && a.pq.size; k++) {
    const j = a.pq.pop();
    a.Q[j] = backup(a, j);
    pushInto(a, j >> 2);
  }
}

function setModel(j, n, r, a) {
  if (next[j] === n && rew[j] === r) return;
  if (next[j] < 0) known.push(j);
  if (r > rew[j] && warm && tick - a.pingAt > 10) { // spotted food: tell the team
    a.pingAt = tick;
    pings.push({ s: a.s, k: agents.indexOf(a), t: performance.now() });
  }
  next[j] = n; rew[j] = r;
  for (const b of agents) push(b, j);
}
// what an agent sees around it goes straight into the shared model
function look(a) {
  const [ax, ay] = xy(a.s);
  for (let y = Math.max(0, ay - VIEW); y <= Math.min(rows - 1, ay + VIEW); y++)
    for (let x = Math.max(0, ax - VIEW); x <= Math.min(cols - 1, ax + VIEW); x++) {
      const c = y * cols + x;
      if ((x - ax) ** 2 + (y - ay) ** 2 > VIEW * VIEW) continue;
      lastSeen[c] = tick;
      if (ripe[c]) bush[c] = 1; // remember where food grows
      else if (bush[c] && !food.some(f => f.s === c)) bush[c] = 0; // it was a one-off
      if (wall[c]) continue;
      if (!seenFlag[c]) { seenFlag[c] = 1; seen.push(c); }
      for (let u = 0; u < 4; u++) { const n = move(c, u); setModel(c * 4 + u, n, ripe[n], a); }
    }
}

// breadth-first search over the team's map: dist to every reachable cell, and the first move to get there
function bfs(s0) {
  dist.fill(-1); dist[s0] = 0; first[s0] = -1;
  let h = 0, t = 0;
  queue[t++] = s0;
  while (h < t) {
    const s = queue[h++];
    for (let u = 0; u < 4; u++) {
      const n = next[s * 4 + u];
      if (n < 0 || dist[n] >= 0) continue;
      dist[n] = dist[s] + 1; first[n] = s === s0 ? u : first[s]; queue[t++] = n;
    }
  }
}
// no food in sight for a while: pick somewhere worth checking, favouring spots where food
// has grown before and places nobody has looked at lately, near rather than far, away from teammates' goals
function pickGoal(a) {
  let best = -1, bs = 0;
  for (let c = 0; c < S; c++) {
    const age = Math.min(tick - lastSeen[c], 600);
    if (dist[c] < 1 || age < IDLE || agents.some(b => b !== a && b.goal >= 0 && Math.abs(b.goal % cols - c % cols) + Math.abs((b.goal / cols | 0) - (c / cols | 0)) < 8)) continue;
    const score = (bush[c] ? 3 : 1) * age / (dist[c] + 8);
    if (score > bs) { bs = score; best = c; }
  }
  return best;
}

// where an agent's greedy plan ends: food it believes is ripe, or -1
function target(a) {
  for (let s = a.s, k = 0; k < 120; k++) {
    const j = s * 4 + greedy(a, s), n = next[j];
    if (n < 0 || n === s) return -1;
    if (R(a, j) > .5) return n;
    s = n;
  }
  return -1;
}
// split the food: the agent closest (highest value) to a target keeps it
function coordinate() {
  for (const a of agents) for (const [n, until] of a.skip) if (tick > until) { a.skip.delete(n); pushInto(a, n); }
  const best = new Map();
  for (const a of agents) {
    a.target = target(a);
    if (a.target < 0) continue;
    const v = maxQ(a, a.s), b = best.get(a.target);
    if (!b || v > b.v) best.set(a.target, { a, v });
  }
  for (const a of agents) if (a.target >= 0 && best.get(a.target).a !== a) {
    a.skip.set(a.target, tick + 150);
    pushInto(a, a.target);
    a.target = -1;
  }
}

function addRock() {
  const cx = rnd(cols), cy = rnd(rows), r = 1 + Math.random() * 2.2, cells = [];
  for (let y = Math.floor(cy - r); y <= cy + r; y++) for (let x = Math.floor(cx - r); x <= cx + r; x++) {
    const s = y * cols + x;
    if (x < 0 || y < 0 || x >= cols || y >= rows || blocked[s] || occupied(s)) continue;
    if (Math.hypot(x - cx, y - cy) < r * (.7 + Math.random() * .5)) { wall[s] = 1; cells.push(s); }
  }
  rocks.push(cells);
}
function removeRock(s) {
  wall[s] = 0;
  for (const r of rocks) { const i = r.indexOf(s); if (i >= 0) r.splice(i, 1); }
}

function makeAgent(k) {
  return {
    id: k + 1, s: freeCell(), trail: [], eaten: 0, target: -1, idle: 0, goal: -1, goalAt: 0, pingAt: -99, skip: new Map(), pq: new Heap(),
    Q: Float32Array.from({ length: S * 4 }, () => Math.random() * 1e-4), // tiny noise breaks ties
    tried: new Float32Array(S * 4),
  };
}

function step() {
  tick++;
  ripe.fill(0);
  for (const f of food) { if (!f.ripe && --f.timer <= 0) f.ripe = true; if (f.ripe) ripe[f.s] = 1; }
  for (const a of agents) look(a);
  coordinate();
  for (const a of agents) {
    a.idle = a.target >= 0 ? 0 : a.idle + 1;
    if (a.target >= 0 || lastSeen[a.goal] === tick || tick - a.goalAt > GIVE_UP) a.goal = -1; // found food, looked, or gave up
    let u = -1;
    if (a.goal >= 0 || a.idle > IDLE) {
      bfs(a.s);
      if (a.goal < 0) { a.goal = pickGoal(a); a.goalAt = tick; }
      if (a.goal >= 0) u = first[a.goal];
      if (u < 0) a.goal = -1; // unreachable
    }
    if (u < 0) u = Math.random() < .02 ? rnd(4) : choose(a, a.s);
    const s2 = move(a.s, u);
    a.tried[a.s * 4 + u] = tick;
    const k = food.findIndex(f => f.s === s2 && f.ripe);
    if (k >= 0) {
      const f = food[k];
      a.eaten++; ripe[s2] = 0;
      if (f.once) food.splice(k, 1); // dropped by a visitor
      else { f.ripe = false; f.timer = 200 + rnd(250); }
    }
    a.trail.push(a.s);
    if (a.trail.length > 10) a.trail.shift();
    a.s = s2;
    sweep(a);
  }
  if (tick % 400 === 0) { // rocks shift
    for (const s of rocks.splice(rnd(rocks.length), 1)[0] ?? []) wall[s] = 0;
    addRock();
  }
}

function sizeCanvas() {
  const dpr = devicePixelRatio || 1;
  canvas.width = innerWidth * dpr;
  canvas.height = innerHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.font = `${cell * .72}px ${FONT}`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
}

function build() {
  builtW = innerWidth;
  cell = innerWidth < 640 ? 16 : 22;
  cols = Math.ceil(innerWidth / cell);
  rows = Math.ceil(innerHeight / cell);
  S = cols * rows;
  wall = new Uint8Array(S); ripe = new Uint8Array(S); rocks = []; food = []; agents = []; pings = [];
  next = new Int32Array(S * 4).fill(-1); rew = new Float32Array(S * 4); known = []; seen = []; seenFlag = new Uint8Array(S);
  lastSeen = new Float32Array(S).fill(-1e9); bush = new Uint8Array(S);
  dist = new Int16Array(S); first = new Int8Array(S); queue = new Int32Array(S);
  tick = 0; focus = 0; warm = false;
  blocked = new Uint8Array(S); // keep rocks and food out from under the text
  if (hud && !hud.textContent) hud.textContent = (' '.repeat(42) + '\n').repeat(6); // reserve its size
  for (const el of document.querySelectorAll('nav, #hud, main')) {
    const r = el.getBoundingClientRect();
    for (let y = Math.floor(r.top / cell) - 1; y <= r.bottom / cell; y++)
      for (let x = Math.floor(r.left / cell) - 1; x <= r.right / cell; x++)
        if (x >= 0 && y >= 0 && x < cols && y < rows) blocked[y * cols + x] = 1;
  }
  for (let k = 0; k < S / 300; k++) addRock();
  for (let k = 0; k < Math.max(4, S / 500); k++) { // berry bushes: a few food cells around a center
    const c = freeCell();
    for (let j = 0; j < 6; j++) {
      const s = Math.max(0, Math.min(S - 1, c + rnd(5) - 2 + (rnd(5) - 2) * cols));
      if (!blocked[s] && !occupied(s)) food.push({ s, ripe: true, timer: 0 });
    }
  }
  for (let k = 0; k < Math.max(4, Math.min(6, S / 600 | 0)); k++) agents.push(makeAgent(k));
  for (let k = 0; k < 400; k++) step(); // open on a team that already knows a little
  warm = true;
  sizeCanvas();
}

function palette() {
  const css = getComputedStyle(document.documentElement), v = k => css.getPropertyValue('--' + k).trim();
  return { agents: [v('y'), v('r'), v('b'), v('g')], bg: v('bg'), ink: v('ink'), fg: v('fg'), rock: v('d'), food: v('s'), faint: v('trail') };
}

function draw(now) {
  const P = palette(), a = agents[focus], col = P.agents[focus % 4];
  const px = s => (s % cols) * cell, py = s => (s / cols | 0) * cell;
  const box = (s, color, alpha = 1) => { ctx.globalAlpha = alpha; ctx.fillStyle = color; ctx.fillRect(px(s) + 1, py(s) + 1, cell - 2, cell - 2); };
  const glyph = (s, ch, color, alpha = 1) => { ctx.globalAlpha = alpha; ctx.fillStyle = color; ctx.fillText(ch, px(s) + cell / 2, py(s) + cell / 2 + 1); };
  const outline = (s, color, alpha = 1) => { ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.strokeRect(px(s) + .5, py(s) + .5, cell - 1, cell - 1); };
  ctx.clearRect(0, 0, innerWidth, innerHeight);

  // what the followed agent knows: its value map and greedy policy over the team's map
  let vmax = .05;
  for (const s of seen) vmax = Math.max(vmax, maxQ(a, s));
  for (const s of seen) {
    const val = maxQ(a, s) / vmax;
    if (val > .2) { box(s, col, .4 * val * val); glyph(s, ARROWS[greedy(a, s)], col, .2 + .6 * val); }
    else glyph(s, '·', val > .1 ? col : P.faint, val > .1 ? .5 : 1);
  }

  for (let s = 0; s < S; s++) if (wall[s]) { box(s, P.rock, .35); glyph(s, '#', P.rock); }
  for (const f of food) {
    if (f.ripe) { box(f.s, P.food); glyph(f.s, '*', P.ink); }
    else glyph(f.s, ',', P.food, .6); // regrowing
  }

  // every agent's plan through the shared model, the followed one drawn last and brightest
  for (const k of [...agents.keys()].filter(k => k !== focus).concat(focus)) {
    const b = agents[k], c = P.agents[k % 4], mine = k === focus, path = new Set([b.s]);
    for (let s = b.s, i = 0; i < 60; i++) {
      const n = next[s * 4 + greedy(b, s)];
      if (n < 0 || path.has(n) || n === b.target) break;
      path.add(n);
      glyph(n, '+', c, mine ? Math.max(.35, .9 - i * .02) : .3);
      s = n;
    }
    if (b.goal >= 0) glyph(b.goal, '?', c, mine ? 1 : .6); // somewhere it is going to check
    ctx.lineWidth = 2;
    if (b.target >= 0) outline(b.target, c, mine ? 1 : .7);
    ctx.lineWidth = 1;
  }

  // radio pings: an agent telling the team it spotted food
  pings = pings.filter(p => now - p.t < 900);
  for (const p of pings) {
    const age = (now - p.t) / 900, r = 1 + age * 8, [x0, y0] = xy(p.s), n = Math.ceil(r * 7);
    ctx.globalAlpha = (1 - age) * .8;
    ctx.fillStyle = P.agents[p.k % 4];
    for (let i = 0; i < n; i++) {
      const x = Math.round(x0 + r * Math.cos(2 * Math.PI * i / n)), y = Math.round(y0 + r * Math.sin(2 * Math.PI * i / n));
      ctx.fillText('o', x * cell + cell / 2, y * cell + cell / 2 + 1);
    }
  }

  agents.forEach((b, k) => {
    b.trail.forEach((s, i) => glyph(s, '.', P.agents[k % 4], (i + 1) / b.trail.length * .5));
    box(b.s, P.agents[k % 4]);
    glyph(b.s, '@', P.ink);
  });
  ctx.globalAlpha = 1;
  ctx.fillStyle = P.fg;
  ctx.fillText('[', px(a.s) - cell / 2, py(a.s) + cell / 2 + 1);
  ctx.fillText(']', px(a.s) + cell * 1.5, py(a.s) + cell / 2 + 1);

  if (canHover && hover >= 0) outline(hover, P.fg, .5);
  ctx.globalAlpha = 1;
}

function updateHud() {
  if (!hud) return;
  const P = palette(), a = agents[focus], kb = known.length * BYTES_PER_TRANSITION / 1024, mine = P.agents[focus % 4];
  const c = (color, ch) => `<span style="color:${color}">${ch}</span>`, me = c(mine, '@');
  hud.innerHTML = [
    `${me} agent ${String(a.id).padStart(2, '0')}  ate ${a.eaten} · team ${agents.reduce((n, b) => n + b.eaten, 0)}`,
    `  map ${seen.length} cells · ${kb.toFixed(1)} / 256 kB`,
    '',
    `${me} agent ${c(P.food, '*')} food ${c(P.food, ',')} regrowing ${c(P.rock, '#')} rock`,
    `${c(mine, '+')} plan ${c(mine, '□')} claimed ${c(mine, '>')} policy ${c(mine, 'o')} ping ${c(mine, '?')} search`,
    `click food · drag rocks · click ${me} follow`,
  ].join('\n');
}

function render() { draw(performance.now()); updateHud(); }

function frame(now) {
  acc += Math.min(.25, (now - last) / 1000);
  last = now;
  while (acc >= TICK) { acc -= TICK; step(); }
  draw(now);
  if (now - hudAt > 250) { updateHud(); hudAt = now; }
  requestAnimationFrame(frame);
}

// pointer play: click drops food (or removes a rock, or follows an agent), drag builds or erases rocks
const cellAt = e => { const x = e.clientX / cell | 0, y = e.clientY / cell | 0; return x < cols && y < rows ? y * cols + x : -1; };
function tap(s) {
  const [x, y] = xy(s);
  const hit = agents.findIndex(a => { const [ax, ay] = xy(a.s); return Math.abs(ax - x) <= 1 && Math.abs(ay - y) <= 1; });
  if (hit >= 0) focus = hit;
  else if (wall[s]) removeRock(s);
  else if (!occupied(s)) {
    food.push({ s, ripe: true, timer: 0, once: true });
    if (food.length > 80) food.splice(food.findIndex(f => f.once), 1);
  }
}
function paint(from, to, build) {
  const [x0, y0] = xy(from), [x1, y1] = xy(to), n = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1);
  for (let i = 0; i <= n; i++) {
    const s = Math.round(y0 + (y1 - y0) * i / n) * cols + Math.round(x0 + (x1 - x0) * i / n);
    if (build && !occupied(s)) wall[s] = 1;
    else if (!build && wall[s]) removeRock(s);
  }
}
addEventListener('pointerdown', e => {
  if (e.button || e.target.closest('a, button')) return;
  const s = cellAt(e);
  if (s >= 0) drag = { start: s, last: s, build: !wall[s], moved: false };
});
addEventListener('pointermove', e => {
  hover = cellAt(e);
  if (!drag || hover < 0 || hover === drag.last) return;
  drag.moved = true;
  paint(drag.last, hover, drag.build);
  drag.last = hover;
  if (still) render();
});
addEventListener('pointerup', () => {
  if (drag && !drag.moved) tap(drag.start);
  drag = null;
  updateHud();
  if (still) render();
});
document.documentElement.addEventListener('mouseleave', () => { hover = -1; });
addEventListener('resize', () => {
  if (innerWidth !== builtW) build(); else sizeCanvas(); // ignore mobile URL-bar height changes
  if (still) render();
});

build();
render();
if (!still) requestAnimationFrame(frame);
document.fonts?.ready.then(() => { sizeCanvas(); if (still) render(); }); // redraw once the web font is in

// the bulb switches light and dark; the new colors spread outward from the bulb
const root = document.documentElement, bulb = document.getElementById('lamp');
function setTheme(dark) {
  const apply = () => {
    root.dataset.theme = dark ? 'dark' : 'light';
    bulb.setAttribute('aria-pressed', !dark);
    render();
  };
  try { localStorage.setItem('theme', dark ? 'dark' : 'light'); } catch {}
  if (!document.startViewTransition || still) return apply();
  const r = bulb.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top + r.height / 2;
  const reach = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y));
  document.startViewTransition(apply).ready.then(() => root.animate(
    { clipPath: [`circle(0 at ${x}px ${y}px)`, `circle(${reach}px at ${x}px ${y}px)`] },
    { duration: 800, easing: 'ease-in-out', pseudoElement: '::view-transition-new(root)' }));
}
bulb.setAttribute('aria-pressed', root.dataset.theme !== 'dark');
bulb.onclick = () => setTheme(root.dataset.theme !== 'dark');
