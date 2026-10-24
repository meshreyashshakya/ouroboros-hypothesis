import * as THREE from 'three';
import { STAGES } from './shapes.js';

const isMobile = matchMedia('(max-width: 768px), (pointer: coarse)').matches;
const N = isMobile ? 6000 : 20000;
const ZOOM = 12;    // how much bigger each stage is than the last, visually
const LAST = STAGES.length - 1;

// Scroll timeline, in units of 0.75 screen heights. Each stage holds, mirror
// stages morph into their twin and back, then the camera zooms out.
const HOLD = 0.45, MOVE = 0.8, MORPH = 0.6, TWIN = 0.7, UNIT_VH = 75;

const timeline = [];
let TOTAL = 0;
const seg = (type, k, len) => { timeline.push({ type, k, from: TOTAL, len }); TOTAL += len; };
STAGES.forEach((s, k) => {
  seg('hold', k, HOLD);
  if (s.mirror) { seg('in', k, MORPH); seg('twin', k, TWIN); seg('out', k, MORPH); seg('hold', k, HOLD); }
  if (k < LAST) seg('move', k, MOVE);
});

const ease = (t) => t * t * t * (t * (t * 6 - 15) + 10);
const clamp01 = (t) => Math.min(Math.max(t, 0), 1);

function evaluate(u) {
  let s = timeline[timeline.length - 1];
  for (const x of timeline) if (u < x.from + x.len) { s = x; break; }
  const t = ease(clamp01((u - s.from) / s.len));
  if (s.type === 'move') return { a: s.k, e: t, m: 0 };
  const m = s.type === 'in' ? t : s.type === 'twin' ? 1 : s.type === 'out' ? 1 - t : 0;
  return { a: s.k, e: 0, m };
}

// ?stage=7 or ?twin=7 (1-based) or ?u=12.3 pins the journey for testing.
const params = new URLSearchParams(location.search);
let pinned = null;
if (params.has('u')) pinned = +params.get('u');
if (params.has('stage')) pinned = timeline.find((x) => x.k === +params.get('stage') - 1).from + 0.1;
if (params.has('twin')) pinned = timeline.find((x) => x.type === 'twin' && x.k === +params.get('twin') - 1).from + 0.3;

// ---------- shapes (generated lazily, warmed in the background) ----------

const cache = [], twinCache = [];
const shape = (k) => (cache[k] ||= STAGES[k].gen(N, 1000 + k));
const twinShape = (k) => (twinCache[k] ||= STAGES[k].mirror.gen(N, 5000 + k));

const warmQueue = STAGES.flatMap((s, k) => (s.mirror ? [() => shape(k), () => twinShape(k)] : [() => shape(k)]));
function warm() {
  const job = warmQueue.shift();
  if (!job) return;
  job();
  setTimeout(warm, 16);
}

// ---------- renderer ----------

const canvas = document.getElementById('cosmos');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false });
renderer.setClearColor(0x05060a, 1);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
camera.position.z = 4;
const group = new THREE.Group();
scene.add(group);

const shared = {
  uTime: { value: 0 },
  uSize: { value: isMobile ? 2.6 : 2.1 },
  uPR: { value: 1 },
  uCam: { value: 4 },
};

const vertexShader = /* glsl */ `
  attribute vec3 aColor;
  attribute vec3 aTwin;
  attribute vec3 aTwinColor;
  attribute float aRand;
  uniform float uScale, uAlpha, uMorph, uTime, uSize, uPR, uCam;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    // staggered per particle, so the morph ripples instead of snapping
    float m = clamp(uMorph * 1.5 - aRand * 0.5, 0.0, 1.0);
    m = m * m * (3.0 - 2.0 * m);
    vec3 p = mix(position, aTwin, m);
    p += normalize(p + 1e-4) * sin(m * 3.14159) * 0.18 * (aRand - 0.3);
    p *= uScale;
    float t = uTime * 0.6 + aRand * 6.2831;
    p += vec3(sin(t), cos(t * 1.3), sin(t * 0.7)) * 0.006 * uScale;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    float depth = -mv.z;
    gl_PointSize = min(uSize * uPR * (0.55 + aRand * 0.9) * (4.0 / max(depth, 0.05)), 40.0 * uPR);
    gl_Position = projectionMatrix * mv;
    float twinkle = 0.78 + 0.22 * sin(uTime * 1.7 + aRand * 40.0);
    vColor = mix(aColor, aTwinColor, m);
    // depth cue: the far side of a shape dims, so spheres read as solid
    float side = (mv.z + uCam) / (1.25 * uScale);
    vAlpha = uAlpha * twinkle * smoothstep(0.15, 1.6, depth) * (0.35 + 0.65 * smoothstep(-1.0, 1.0, side));
  }
`;

const fragmentShader = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    if (d > 0.5) discard;
    gl_FragColor = vec4(vColor, pow(1.0 - d * 2.0, 1.6) * vAlpha);
  }
`;

function makeLayer() {
  const geo = new THREE.BufferGeometry();
  const rand = new Float32Array(N);
  for (let i = 0; i < N; i++) rand[i] = Math.random();
  for (const name of ['position', 'aColor', 'aTwin', 'aTwinColor']) {
    geo.setAttribute(name, new THREE.BufferAttribute(new Float32Array(N * 3), 3));
  }
  geo.setAttribute('aRand', new THREE.BufferAttribute(rand, 1));
  const mat = new THREE.ShaderMaterial({
    uniforms: { ...shared, uScale: { value: 1 }, uAlpha: { value: 1 }, uMorph: { value: 0 } },
    vertexShader, fragmentShader,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  });
  const points = new THREE.Points(geo, mat);
  points.frustumCulled = false;
  group.add(points);
  return { geo, mat, stage: -1 };
}

function upload(geo, name, data) {
  geo.attributes[name].array.set(data);
  geo.attributes[name].needsUpdate = true;
}

function showStage(layer, k) {
  if (layer.stage === k) return;
  const s = shape(k);
  upload(layer.geo, 'position', s.pos);
  upload(layer.geo, 'aColor', s.col);
  const t = STAGES[k].mirror ? twinShape(k) : s;
  upload(layer.geo, 'aTwin', t.pos);
  upload(layer.geo, 'aTwinColor', t.col);
  layer.stage = k;
}

// Layer A is the stage we're leaving (shrinks away), B the one arriving
// (descends from a larger scale) — a log-space zoom out.
const layerA = makeLayer();
const layerB = makeLayer();

// ---------- scroll ----------

const journey = document.getElementById('journey');
journey.style.height = `calc(${TOTAL * UNIT_VH}vh + 100vh)`;
let target = pinned ?? 0;
let current = target;

function readScroll() {
  if (pinned !== null) return;
  const span = journey.offsetHeight - innerHeight;
  target = clamp01((scrollY - journey.offsetTop) / span) * TOTAL;
}

function resize() {
  const w = innerWidth, h = innerHeight;
  const pr = Math.min(devicePixelRatio, 2);
  renderer.setPixelRatio(pr);
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  // keep shapes fully in frame on portrait screens
  camera.position.z = w < h ? 4 * Math.min(1.7, (h / w) * 0.85) : 4;
  camera.updateProjectionMatrix();
  shared.uPR.value = pr;
  shared.uCam.value = camera.position.z;
  readScroll();
}

// ---------- HUD ----------

const $ = (id) => document.getElementById(id);
const hero = $('hero'), hud = $('hud'), hudText = $('hud-text'), mirror = $('mirror');
const ring = $('ring-progress');
const RING = 2 * Math.PI * 28;
ring.style.strokeDasharray = RING;
$('hud-total').textContent = String(STAGES.length).padStart(2, '0');

const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
const sup = (n) => String(n).split('').map((c) => SUP[c]).join('');

function formatScale(exp) {
  if (exp === null) return 'beyond measurement';
  const whole = Math.floor(exp + 1e-6);
  return `${Math.pow(10, exp - whole).toFixed(2)} × 10${sup(whole)} m`;
}

function scaleBetween(a, b, e) {
  if (a === null || b === null) return formatScale(e < 0.5 ? a : b);
  return formatScale(a + (b - a) * e);
}

function ratio(a, b) {
  if (a === null || b === null) return 'scale-free: detail at every magnification';
  const d = Math.round(b - a);
  return d >= 0 ? `10${sup(d)} times larger` : `10${sup(-d)} times smaller`;
}

let shown = -1, swapTimer;
function setHudStage(k) {
  if (k === shown) return;
  shown = k;
  const s = STAGES[k];
  hudText.classList.add('swap');
  clearTimeout(swapTimer);
  swapTimer = setTimeout(() => {
    $('hud-index').textContent = String(k + 1).padStart(2, '0');
    $('hud-name').textContent = s.name;
    $('hud-line').textContent = s.line;
    $('hud-zone').textContent = s.zone === 'observed' ? 'Zone I — Observed' : 'Zone II — Hypothesis';
    document.body.classList.toggle('hypothesis', s.zone === 'hypothesis');
    if (s.mirror) {
      $('m-a').textContent = s.name;
      $('m-b').textContent = s.mirror.name;
      $('m-a-s').textContent = formatScale(s.exp);
      $('m-b-s').textContent = s.mirror.exp === null ? 'scale-free' : formatScale(s.mirror.exp);
      $('m-ratio').textContent = ratio(s.exp, s.mirror.exp);
      $('m-note').textContent = s.mirror.note;
      $('m-src').textContent = s.mirror.source;
    }
    hudText.classList.remove('swap');
  }, 220);
}

// ---------- pointer parallax ----------

const mouse = { x: 0, y: 0, sx: 0, sy: 0 };
addEventListener('pointermove', (e) => {
  mouse.x = e.clientX / innerWidth - 0.5;
  mouse.y = e.clientY / innerHeight - 0.5;
});

// ---------- loop ----------

const clock = new THREE.Clock();
function frame() {
  const dt = Math.min(clock.getDelta(), 0.05);
  shared.uTime.value += dt;
  current += (target - current) * Math.min(1, dt * 5);

  const { a, e, m } = evaluate(current);
  const b = Math.min(a + 1, LAST);
  showStage(layerA, a);
  showStage(layerB, b);
  layerA.mat.uniforms.uScale.value = Math.pow(ZOOM, -e);
  layerA.mat.uniforms.uAlpha.value = Math.pow(1 - e, 1.5);
  layerA.mat.uniforms.uMorph.value = m;
  layerB.mat.uniforms.uScale.value = Math.pow(ZOOM, 1 - e);
  layerB.mat.uniforms.uAlpha.value = a === LAST ? 0 : Math.pow(e, 0.8);

  // gentle sway rather than a full spin, so flat shapes never turn edge-on
  mouse.sx += (mouse.x - mouse.sx) * dt * 2;
  mouse.sy += (mouse.y - mouse.sy) * dt * 2;
  group.rotation.y = Math.sin(shared.uTime.value * 0.1) * 0.3 + mouse.sx * 0.35;
  group.rotation.x = mouse.sy * 0.25;

  const k = e < 0.5 ? a : b;
  setHudStage(k);
  $('hud-scale').textContent = scaleBetween(STAGES[a].exp, STAGES[b].exp, e);
  ring.style.strokeDashoffset = RING * (1 - current / TOTAL);
  mirror.style.opacity = clamp01((m - 0.4) / 0.4);

  hero.style.opacity = 1 - clamp01(current / 0.2);
  hero.style.transform = `translateY(${-clamp01(current / 0.2) * 30}px)`;
  hud.style.opacity = clamp01((current - 0.1) / 0.15);

  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}

addEventListener('resize', resize);
addEventListener('scroll', readScroll, { passive: true });
resize();
current = target;
requestAnimationFrame(frame);
setTimeout(warm, 300);
