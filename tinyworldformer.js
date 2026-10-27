// tinyworldformer in the browser: one WebAssembly module per game (wasm/<env>.wasm), no imports.
// The module holds the int8 transformer, the game state and the agent; this file only calls
// tick() once per frame, forwards keys, and paints the RGBA frame it exposes.
(() => {
const KEYS = {
  breakout: [['a', '◀ left'], ['s', 'stop'], ['d', 'right ▶']],
  doomlite: [['a', '↺ turn'], ['w', '▲ forward'], ['d', 'turn ↻'], [' ', 'fire'], ['s', 'wait']],
  flappy:   [[' ', 'flap']],
  pacman:   [['a', '◀'], ['w', '▲'], ['s', '▼'], ['d', '▶'], [' ', 'keep going']],
  pong:     [['w', '▲ up'], [' ', 'stop'], ['s', '▼ down']],
};
const WON = { breakout: 'bricks', doomlite: 'kills', flappy: 'pipes', pacman: 'pellets', pong: 'points' };
const ARROWS = { ArrowUp: 'w', ArrowLeft: 'a', ArrowDown: 's', ArrowRight: 'd' };

const $ = id => document.getElementById(id);
const screen = $('screen'), pix = screen.getContext('2d');
let placed = false, wasm = null, game = '', envs = [], fast = false, lastStep = 0, modelMs = 0, generation = 0;

// the sim behind the page listens on window: keep clicks on the demo to the demo
$('screen').closest('main').addEventListener('pointerdown', e => e.stopPropagation());

async function load(env) {
  const gen = ++generation;
  const url = `../wasm/${env}.wasm`;
  const { instance } = await WebAssembly.instantiateStreaming(fetch(url), {})
    .catch(async () => WebAssembly.instantiate(await (await fetch(url)).arrayBuffer(), {})); // host without application/wasm
  if (gen !== generation) return; // a newer pick won
  const e = instance.exports;
  e.init((Math.random() * 2 ** 32) >>> 0);
  const str = i => new TextDecoder().decode(new Uint8Array(e.memory.buffer, e.name_ptr(i), e.name_len(i)));
  wasm = e;
  game = str(0);
  screen.width = e.width();
  screen.height = e.height();
  // as wide as the column allows, but short enough to leave room for the controls below
  screen.style.width = `min(100%, ${(42 * e.width() / e.height()).toFixed(1)}vh)`;
  if (!placed) { placed = true; requestAnimationFrame(() => build()); } // the sim keeps clear of the stage, now it has its size
  for (const b of $('games').children) b.toggleAttribute('aria-current', b.dataset.env === env);
  $('pad').innerHTML = '';
  for (const [k, label] of KEYS[game] ?? []) {
    const b = document.createElement('button');
    b.textContent = label;
    b.addEventListener('pointerdown', ev => { ev.preventDefault(); press(k); });
    $('pad').append(b);
  }
  $('help').textContent = `${game}: env ${str(1)}, agent ${str(2)}. Keys: `
    + (KEYS[game] ?? []).map(([k]) => (k === ' ' ? 'space' : k)).join(' ') + `; g agent, l learning, f fast, 1-${e.max_loops()} loops.`;
  draw();
}

function press(k) {
  if (!wasm) return;
  if (k === 'f') { fast = !fast; return; }
  if (k === 'n') { press(String(wasm.loops() % wasm.max_loops() + 1)); return; }
  wasm.key(k.charCodeAt(0));
}

function draw() {
  const px = new Uint8ClampedArray(wasm.memory.buffer, wasm.rgba(), wasm.width() * wasm.height() * 4);
  pix.putImageData(new ImageData(px, wasm.width(), wasm.height()), 0, 0);
  const who = wasm.human() ? 'you' : wasm.learning() ? 'stream-x, learning' : 'stream-x, pretrained';
  $('stats').textContent =
    `${who}: ${WON[game] ?? 'wins'} ${wasm.wins()}, lives ${wasm.lives()}, episode ${wasm.episode()}
` +
    `${wasm.loops()} loops, ${modelMs.toFixed(1)} ms per dreamed frame` +
    (wasm.learning() ? `, learned ${wasm.learn_steps()} steps, mean of the last 10 episodes ${wasm.avg10()}` : '');
  $('learn').setAttribute('aria-pressed', String(!!wasm.learning()));
  $('agent').hidden = !wasm.human();
}

// Human speed: one dreamed frame every human_frame_ms. Fast (not while you play): as many as fit in ~12 ms.
function frame(now) {
  requestAnimationFrame(frame);
  if (!wasm) return;
  const turbo = fast && !wasm.human();
  if (!turbo && now - lastStep < wasm.human_frame_ms()) return;
  lastStep = now;
  const t0 = performance.now();
  let n = 0;
  do { wasm.tick(); n++; } while (turbo && performance.now() - t0 < 12);
  modelMs = (performance.now() - t0) / n;
  draw();
}

addEventListener('keydown', ev => {
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  const k = ARROWS[ev.key] ?? ev.key.toLowerCase();
  if (k.length !== 1) return;
  ev.preventDefault();
  press(k);
});
$('agent').onclick = () => press('g');
$('learn').onclick = () => press('l');

fetch('../wasm/envs.json').then(r => r.json()).then(list => {
  envs = list.includes('doomlite') ? ['doomlite', ...list.filter(e => e !== 'doomlite')] : list; // doom-lite first: the most striking dream
  for (const env of envs) {
    const b = document.createElement('button');
    b.textContent = env;
    b.dataset.env = env;
    b.onclick = () => load(env);
    $('games').append(b);
  }
  const pick = new URLSearchParams(location.search).get('env');
  return load(envs.includes(pick) ? pick : envs[0]);
}).catch(err => { $('stats').textContent = `could not load the models (${err})`; });
requestAnimationFrame(frame);

// the board video plays while it is on screen, in case the browser skipped autoplay
const video = document.querySelector('.video video');
if (video) new IntersectionObserver(([e]) => e.isIntersecting ? video.play().catch(() => {}) : video.pause(), { threshold: .25 }).observe(video);
})();
