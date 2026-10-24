// Particle shapes for every stage of the journey.
// Each generator returns { pos, col } — Float32Arrays of N * 3 — built from a
// seeded RNG so every visit renders the exact same universe.

// ---------- random ----------

export function rng(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(r) {
  return Math.sqrt(-2 * Math.log(r() + 1e-9)) * Math.cos(2 * Math.PI * r());
}

function onSphere(r, rad = 1) {
  const z = r() * 2 - 1, a = r() * Math.PI * 2, s = Math.sqrt(1 - z * z);
  return [Math.cos(a) * s * rad, Math.sin(a) * s * rad, z * rad];
}

function inSphere(r, rad = 1) {
  return onSphere(r, rad * Math.cbrt(r()));
}

function fibSphere(n) {
  const pts = [], ga = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = 1 - ((i + 0.5) / n) * 2, s = Math.sqrt(1 - y * y), a = i * ga;
    pts.push([Math.cos(a) * s, y, Math.sin(a) * s]);
  }
  return pts;
}

// ---------- vectors ----------

const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const unit = (a) => mul(a, 1 / (len(a) || 1));

function lerp3(a, b, t) {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

function jitter(r, p, amt) {
  return [p[0] + gauss(r) * amt, p[1] + gauss(r) * amt, p[2] + gauss(r) * amt];
}

function rotX(p, a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [p[0], p[1] * c - p[2] * s, p[1] * s + p[2] * c];
}

function rotY(p, a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c];
}

function rotZ(p, a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [p[0] * c - p[1] * s, p[0] * s + p[1] * c, p[2]];
}

// ---------- noise ----------

function hash3(x, y, z, s) {
  let h = Math.imul(x, 374761393) ^ Math.imul(y, 668265263) ^ Math.imul(z, 1440662683) ^ Math.imul(s, 2246822519);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function noise3(x, y, z, s = 0) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  const f = (t) => t * t * (3 - 2 * t);
  const u = f(x - xi), v = f(y - yi), w = f(z - zi);
  let out = 0;
  for (let dx = 0; dx < 2; dx++) for (let dy = 0; dy < 2; dy++) for (let dz = 0; dz < 2; dz++) {
    out += (dx ? u : 1 - u) * (dy ? v : 1 - v) * (dz ? w : 1 - w) * hash3(xi + dx, yi + dy, zi + dz, s);
  }
  return out;
}

function fbm(p, s = 0, oct = 4) {
  let amp = 0.5, freq = 1, total = 0, norm = 0;
  for (let i = 0; i < oct; i++) {
    total += amp * noise3(p[0] * freq, p[1] * freq, p[2] * freq, s + i);
    norm += amp; amp *= 0.5; freq *= 2.03;
  }
  return total / norm;
}

// ---------- samplers ----------

function picker(items, weight) {
  const cum = [];
  let sum = 0;
  for (const it of items) { sum += weight(it); cum.push(sum); }
  return (r) => {
    const x = r() * sum;
    let lo = 0, hi = cum.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (cum[mid] < x) lo = mid + 1; else hi = mid; }
    return items[lo];
  };
}

// Samples points along line segments {a, b, w, c?}; w is the thickness.
function segSampler(segs, weight = (s) => len(sub(s.b, s.a)) * (s.w + 0.004)) {
  const pick = picker(segs, weight);
  return (r) => {
    const s = pick(r);
    const p = jitter(r, lerp3(s.a, s.b, r()), s.w);
    return s.c ? [...p, ...s.c] : p;
  };
}

// Recursive branching growth: trees, neurons, roots, river deltas.
function grow(r, o) {
  const {
    origin, dir, length, width, depth, spread = 0.5, decay = 0.72,
    kids = [2, 3], bend = 0.2, pull = [0, 0, 0], steps = 4, flat = false,
  } = o;
  const segs = [], tips = [];
  (function step(p, d, l, w, k) {
    let q = p;
    for (let s = 0; s < steps; s++) {
      d = unit([d[0] + gauss(r) * bend + pull[0], d[1] + gauss(r) * bend + pull[1], flat ? 0 : d[2] + gauss(r) * bend + pull[2]]);
      const n = add(q, mul(d, l / steps));
      segs.push({ a: q, b: n, w });
      q = n;
    }
    if (k === 0) { tips.push(q); return; }
    const count = kids[0] + Math.floor(r() * (kids[1] - kids[0] + 1));
    for (let i = 0; i < count; i++) {
      const nd = unit([d[0] + gauss(r) * spread, d[1] + gauss(r) * spread, flat ? 0 : d[2] + gauss(r) * spread]);
      step(q, nd, l * decay * (0.8 + r() * 0.4), w * 0.68, k - 1);
    }
  })(origin, unit(dir), length, width, depth);
  return { segs, tips };
}

// Applies a transform to every point a list of parts produces.
function xform(parts, fn, wScale = 1) {
  return parts.map((p) => ({
    w: p.w * wScale,
    c: p.c,
    f: (r) => {
      const v = p.f(r);
      const q = fn([v[0], v[1], v[2]]);
      return v.length > 3 ? [...q, v[3], v[4], v[5]] : q;
    },
  }));
}

// Fills N particles from weighted parts. A part's f(r) returns [x,y,z] or
// [x,y,z,r,g,b]; without its own colour the part's `c` is used.
function build(N, seed, parts, radius = 1.25) {
  const r = rng(seed);
  const pos = new Float32Array(N * 3);
  const col = new Float32Array(N * 3);
  const total = parts.reduce((s, p) => s + p.w, 0);
  let i = 0;
  parts.forEach((part, k) => {
    const count = k === parts.length - 1 ? N - i : Math.floor((part.w / total) * N);
    for (let n = 0; n < count; n++, i++) {
      const v = part.f(r);
      const c = v.length > 3 ? [v[3], v[4], v[5]] : part.c;
      const b = 0.7 + r() * 0.45;
      pos.set([v[0], v[1], v[2]], i * 3);
      col.set([c[0] * b, c[1] * b, c[2] * b], i * 3);
    }
  });
  normalize(pos, radius);
  return { pos, col };
}

function normalize(pos, radius) {
  let max = 0;
  for (let i = 0; i < pos.length; i += 3) {
    max = Math.max(max, Math.hypot(pos[i], pos[i + 1], pos[i + 2]));
  }
  const k = radius / max;
  for (let i = 0; i < pos.length; i++) pos[i] *= k;
}

// ---------- palette ----------

const C = {
  white: [1.0, 1.0, 1.0],
  ice: [0.72, 0.84, 1.0],
  blue: [0.45, 0.62, 1.0],
  deep: [0.32, 0.45, 0.95],
  dim: [0.22, 0.3, 0.62],
  warm: [1.0, 0.86, 0.74],
  gold: [1.0, 0.76, 0.36],
  amber: [1.0, 0.6, 0.24],
  honey: [1.0, 0.88, 0.6],
};

const starfield = (w, rad = 1.25) => ({ w, c: C.dim, f: (r) => inSphere(r, rad) });

// ================= Zone I — the microscopic world =================

// Atom: a nucleus inside a quantum probability cloud (s + p orbitals),
// not the outdated planetary Bohr picture.
function atomShape(pal) {
  return (N, seed) => {
    const r0 = rng(seed + 7);
    const nucleons = Array.from({ length: 14 }, () => inSphere(r0, 0.09));
    const pLobe = (r) => {
      const axis = Math.floor(r() * 3);
      for (;;) {
        const p = inSphere(r, 1.3);
        const d = len(p) || 1e-6;
        const cos = p[axis] / d;
        const radial = (d / 0.55) ** 2 * Math.exp(2 - (2 * d) / 0.55);
        if (r() < cos * cos * radial) return p;
      }
    };
    return build(N, seed, [
      { w: 0.14, c: pal.a, f: (r) => { const k = Math.floor(r() * nucleons.length); return [...jitter(r, nucleons[k], 0.028), ...(k % 2 ? pal.a : pal.b)]; } },
      { w: 0.36, c: pal.s, f: (r) => onSphere(r, Math.min(-Math.log(r() + 1e-9) * 0.3, 1.3)) },
      { w: 0.5, c: pal.p, f: pLobe },
    ]);
  };
}
const atom = atomShape({ a: C.warm, b: C.ice, s: C.blue, p: C.ice });
const atomReturn = atomShape({ a: C.white, b: C.honey, s: C.amber, p: C.gold });

// Molecule: a glucose ring — five carbons and an oxygen, with its
// hydroxyl groups and hydrogens, drawn ball-and-stick.
function molecule(N, seed) {
  const atoms = [], bonds = [];
  const addAtom = (p, rad, c) => atoms.push({ p, rad, c }) - 1;
  const ring = [];
  for (let k = 0; k < 6; k++) {
    const a = (k / 6) * Math.PI * 2;
    const z = k % 2 ? 0.12 : -0.12; // chair conformation pucker
    const isO = k === 0;
    ring.push(addAtom([Math.cos(a) * 0.55, Math.sin(a) * 0.55, z], isO ? 0.12 : 0.13, isO ? C.blue : C.ice));
  }
  for (let k = 0; k < 6; k++) bonds.push([ring[k], ring[(k + 1) % 6]]);
  for (let k = 1; k < 6; k++) {
    const a = (k / 6) * Math.PI * 2;
    const up = k % 2 ? 1 : -1;
    const o = addAtom([Math.cos(a) * 1.0, Math.sin(a) * 1.0, 0.3 * up], 0.12, C.blue);
    const h = addAtom([Math.cos(a + 0.25) * 1.25, Math.sin(a + 0.25) * 1.25, 0.42 * up], 0.07, C.white);
    const hc = addAtom([Math.cos(a) * 0.72, Math.sin(a) * 0.72, -0.48 * up], 0.07, C.white);
    bonds.push([ring[k], o], [o, h], [ring[k], hc]);
  }
  const pickAtom = picker(atoms, (a) => a.rad * a.rad);
  const tilt = (p) => rotX(rotZ(p, 0.3), -0.9);
  return build(N, seed, xform([
    { w: 0.62, c: C.ice, f: (r) => { const a = pickAtom(r); return [...jitter(r, a.p, a.rad * 0.45), ...a.c]; } },
    { w: 0.38, c: C.deep, f: (r) => { const [i, j] = bonds[Math.floor(r() * bonds.length)]; return jitter(r, lerp3(atoms[i].p, atoms[j].p, r()), 0.014); } },
  ], tilt));
}

// DNA: a right-handed double helix, ~10.5 base pairs per turn, with the
// strands offset to form the major and minor grooves.
function dna(N, seed) {
  const H = 2.6, R = 0.36, turns = 2.5;
  const bp = Math.round(turns * 10.5);
  const groove = Math.PI * 0.78;
  const strand = (y, off) => {
    const a = (y / H) * turns * Math.PI * 2 + off;
    return [Math.cos(a) * R, y - H / 2, Math.sin(a) * R];
  };
  return build(N, seed, xform([
    { w: 0.56, c: C.ice, f: (r) => jitter(r, strand(r() * H, r() < 0.5 ? 0 : groove), 0.03) },
    {
      w: 0.44, c: C.blue,
      f: (r) => {
        const y = ((Math.floor(r() * bp) + 0.5) / bp) * H;
        const t = r();
        // each rung is two paired bases, so colour the halves apart
        return [...jitter(r, lerp3(strand(y, 0), strand(y, groove), t), 0.014), ...(t < 0.5 ? C.blue : C.white)];
      },
    },
  ], (p) => rotX(rotZ(p, -0.55), 0.25)));
}

// Ribosome: a large and a small subunit clamped around a strand of mRNA,
// with a new protein chain leaving through the exit tunnel.
function ribosome(N, seed) {
  const r0 = rng(seed + 3);
  const lumps = (n, c, rx, ry, rz) => Array.from({ length: n }, () => {
    const p = inSphere(r0, 1);
    return { p: [c[0] + p[0] * rx, c[1] + p[1] * ry, c[2] + p[2] * rz], rad: 0.1 + r0() * 0.12 };
  });
  const blob = (list) => {
    const pick = picker(list, (l) => l.rad * l.rad);
    return (r) => { const l = pick(r); return add(l.p, onSphere(r, l.rad * (0.8 + r() * 0.2))); };
  };
  const chain = [];
  let q = [0.1, 0.7, 0];
  for (let i = 0; i < 40; i++) {
    const n = add(q, mul(unit([gauss(r0) * 0.7, 1, gauss(r0) * 0.7]), 0.025));
    chain.push({ a: q, b: n, w: 0.012 });
    q = n;
  }
  return build(N, seed, xform([
    { w: 0.5, c: C.ice, f: blob(lumps(46, [0, 0.28, 0], 0.75, 0.45, 0.6)) },
    { w: 0.3, c: C.blue, f: blob(lumps(30, [0.05, -0.45, 0], 0.62, 0.22, 0.45)) },
    { w: 0.1, c: C.white, f: (r) => { const x = r() * 2.8 - 1.4; return jitter(r, [x, -0.2 + Math.sin(x * 5) * 0.05, Math.cos(x * 3) * 0.1], 0.012); } },
    { w: 0.1, c: C.warm, f: segSampler(chain) },
  ], (p) => rotX(rotY(p, 0.5), 0.2)));
}

// Virus: a protein capsid studded with spike proteins, genome coiled inside.
function virus(N, seed) {
  const R = 0.72;
  const spikes = fibSphere(64);
  const r0 = rng(seed + 5);
  const genome = [];
  let q = [0, 0, 0];
  for (let i = 0; i < 400; i++) {
    let n = add(q, mul(onSphere(r0, 1), 0.06));
    if (len(n) > 0.55) n = mul(n, 0.5 / len(n));
    genome.push({ a: q, b: n, w: 0.01 });
    q = n;
  }
  const spike = (r) => spikes[Math.floor(r() * spikes.length)];
  return build(N, seed, [
    { w: 0.4, c: C.ice, f: (r) => jitter(r, onSphere(r, R), 0.018) },
    { w: 0.22, c: C.blue, f: (r) => jitter(r, mul(spike(r), R + r() * 0.24), 0.012) },
    { w: 0.2, c: C.white, f: (r) => add(mul(spike(r), R + 0.28), inSphere(r, 0.06)) },
    { w: 0.18, c: C.deep, f: segSampler(genome) },
  ]);
}

// Bacterium: a rod-shaped cell with whip-like helical flagella and pili.
function bacterium(N, seed) {
  const h = 0.6, R = 0.3;
  const body = (r) => {
    const capA = 4 * Math.PI * R * R, cylA = 2 * Math.PI * R * 2 * h;
    if (r() < capA / (capA + cylA)) { const p = onSphere(r, R); p[0] += p[0] > 0 ? h : -h; return p; }
    const a = r() * Math.PI * 2;
    return [(r() * 2 - 1) * h, Math.cos(a) * R, Math.sin(a) * R];
  };
  const r0 = rng(seed + 9);
  const flagella = [];
  for (let k = 0; k < 5; k++) {
    const phase = r0() * 6.28, sy = gauss(r0) * 0.35, sz = gauss(r0) * 0.35;
    let prev = null;
    for (let i = 0; i <= 120; i++) {
      const t = i / 120;
      const p = [
        -(h + R * 0.8) - t * 1.6,
        sy * t * 1.6 + Math.cos(t * 30 + phase) * 0.05 + Math.sin(t * 4 + k) * 0.08,
        sz * t * 1.6 + Math.sin(t * 30 + phase) * 0.05,
      ];
      if (prev) flagella.push({ a: prev, b: p, w: 0.008 });
      prev = p;
    }
  }
  const pili = Array.from({ length: 50 }, () => {
    const b = body(r0);
    return { a: b, b: add(b, mul(unit([b[0] * 0.3, b[1], b[2]]), 0.12 + r0() * 0.08)), w: 0.005 };
  });
  return build(N, seed, xform([
    { w: 0.55, c: C.ice, f: (r) => jitter(r, body(r), 0.01) },
    { w: 0.12, c: C.deep, f: (r) => { const p = inSphere(r, 1); return [p[0] * h * 1.1, p[1] * R * 0.6, p[2] * R * 0.6]; } },
    { w: 0.25, c: C.blue, f: segSampler(flagella) },
    { w: 0.08, c: C.white, f: segSampler(pili) },
  ], (p) => rotZ(rotY([p[0] + 0.75, p[1], p[2]], 0.5), 0.35)));
}

// Human cell: membrane, nucleus with nucleolus, endoplasmic reticulum,
// mitochondria, Golgi apparatus and vesicles.
function cell(N, seed) {
  const r0 = rng(seed + 11);
  const nuc = [0.12, 0.05, 0], nr = 0.36;
  const outside = (min, max) => { let p; do { p = inSphere(r0, max); } while (len(sub(p, nuc)) < min); return p; };
  const mito = Array.from({ length: 18 }, () => ({ p: outside(0.55, 0.85), d: onSphere(r0, 1) }));
  const vesicles = Array.from({ length: 40 }, () => ({ p: outside(0.5, 0.9), rad: 0.02 + r0() * 0.03 }));
  return build(N, seed, [
    { w: 0.22, c: [0.35, 0.48, 0.9], f: (r) => { const d = onSphere(r, 1); return mul(d, 1 + (fbm(mul(d, 2), 3) - 0.5) * 0.15); } },
    { w: 0.16, c: C.ice, f: (r) => add(nuc, onSphere(r, nr)) },
    { w: 0.1, c: C.blue, f: (r) => add(nuc, inSphere(r, nr * 0.9)) },
    { w: 0.05, c: C.white, f: (r) => add([nuc[0] + 0.1, nuc[1] + 0.08, 0.05], inSphere(r, 0.09)) },
    {
      w: 0.14, c: C.blue,
      f: (r) => {
        for (;;) {
          const d = onSphere(r, 1), k = Math.floor(r() * 4);
          if (noise3(d[0] * 3, d[1] * 3, d[2] * 3, k) > 0.55) return add(nuc, mul(d, 0.46 + k * 0.06));
        }
      },
    },
    { w: 0.17, c: C.warm, f: (r) => { const m = mito[Math.floor(r() * mito.length)]; return add(add(m.p, mul(m.d, (r() * 2 - 1) * 0.09)), onSphere(r, 0.045)); } },
    { w: 0.06, c: C.white, f: (r) => { const v = vesicles[Math.floor(r() * vesicles.length)]; return add(v.p, onSphere(r, v.rad)); } },
    {
      w: 0.1, c: C.ice,
      f: (r) => {
        const k = Math.floor(r() * 5), a = Math.PI - 0.9 + r() * 1.3, rad = 0.55 + k * 0.05;
        return jitter(r, [Math.cos(a) * rad + nuc[0], Math.sin(a) * rad + nuc[1], (r() - 0.5) * 0.15], 0.008);
      },
    },
  ]);
}

// Neuron: a soma with branching dendrites and a myelinated axon ending in
// terminal branches.
function neuron(N, seed) {
  const r = rng(seed + 13);
  const dend = [];
  for (let k = 0; k < 7; k++) {
    const d = unit(add(onSphere(r, 1), [-0.9, 0, 0]));
    dend.push(...grow(r, { origin: mul(d, 0.1), dir: d, length: 0.42, width: 0.016, depth: 4, spread: 0.7, decay: 0.7, kids: [2, 2], bend: 0.18 }).segs);
  }
  const axon = [];
  let q = [0.1, 0, 0], d = [1, 0, 0];
  for (let i = 0; i < 40; i++) {
    d = unit([1, d[1] + gauss(r) * 0.08, d[2] + gauss(r) * 0.08]);
    const n = add(q, mul(d, 0.045));
    axon.push({ a: q, b: n, w: i % 6 === 0 ? 0.006 : 0.02 }); // gaps are the nodes of Ranvier
    q = n;
  }
  const term = grow(r, { origin: q, dir: [1, 0, 0], length: 0.3, width: 0.008, depth: 4, spread: 0.8, kids: [2, 3], bend: 0.2 });
  return build(N, seed, xform([
    { w: 0.12, c: C.white, f: (r) => onSphere(r, 0.1 * (0.7 + 0.3 * r())) },
    { w: 0.46, c: C.ice, f: segSampler(dend) },
    { w: 0.17, c: C.blue, f: segSampler(axon) },
    { w: 0.17, c: C.ice, f: segSampler(term.segs) },
    { w: 0.08, c: C.white, f: (r) => add(term.tips[Math.floor(r() * term.tips.length)], inSphere(r, 0.02)) },
  ], (p) => rotZ([p[0] - 0.75, p[1], p[2]], 0.25)));
}

// Grain of sand: a worn, faceted quartz grain with smaller grains nearby.
function sandGrain(N, seed) {
  const r0 = rng(seed + 19);
  const planes = Array.from({ length: 16 }, () => ({ n: onSphere(r0, 1), d: 0.8 + r0() * 0.3 }));
  const surf = (dir) => {
    let m = 1.15;
    for (const p of planes) { const k = dot(dir, p.n); if (k > 1e-3) m = Math.min(m, p.d / k); }
    return m;
  };
  const shape = (p) => [p[0] * 1.1, p[1] * 0.85, p[2] * 0.95];
  const grains = Array.from({ length: 14 }, () => ({ p: mul(onSphere(r0, 1), 1.2 + r0() * 0.3), s: 0.05 + r0() * 0.08 }));
  return build(N, seed, [
    {
      w: 0.72, c: C.ice,
      f: (r) => {
        const d = onSphere(r, 1);
        return [...shape(mul(d, surf(d) * (0.97 + fbm(mul(d, 4), 2) * 0.06))), ...(r() < 0.25 ? C.warm : C.ice)];
      },
    },
    { w: 0.16, c: C.deep, f: (r) => { const d = onSphere(r, 1); return shape(mul(d, surf(d) * Math.cbrt(r()))); } },
    { w: 0.12, c: C.blue, f: (r) => { const g = grains[Math.floor(r() * grains.length)]; return add(g.p, onSphere(r, g.s)); } },
  ]);
}

// ================= Zone I — the human scale =================

function irisParts(R) {
  const ring = (rad, spread) => (r) => {
    const a = r() * Math.PI * 2, rr = rad + gauss(r) * spread;
    return [Math.cos(a) * rr, Math.sin(a) * rr, gauss(r) * 0.01];
  };
  return [
    {
      w: 0.5, c: C.ice,
      f: (r) => {
        const k = Math.floor(r() * 180), t = 0.3 + r() * 0.62;
        const a = (k / 180) * Math.PI * 2 + 0.05 * Math.sin(t * 16 + k);
        return [...jitter(r, [Math.cos(a) * t * R, Math.sin(a) * t * R, 0], 0.018 * R), ...(r() < 0.5 ? C.ice : C.blue)];
      },
    },
    { w: 0.1, c: C.white, f: (r) => { const a = r() * Math.PI * 2, rr = R * (0.48 + 0.03 * Math.sin(a * 9)); return jitter(r, [Math.cos(a) * rr, Math.sin(a) * rr, 0], 0.012 * R); } },
    { w: 0.06, c: C.white, f: ring(R * 0.3, 0.008 * R) },
    { w: 0.1, c: C.deep, f: ring(R * 0.95, 0.03 * R) },
  ];
}

// Human eye: iris fibres, collarette and pupil inside an almond eyelid.
function eye(N, seed) {
  const W = 1.6, H = 0.78;
  const lid = (x) => H * Math.pow(Math.max(0, 1 - (x / W) ** 2), 1.1);
  return build(N, seed, xform([
    ...irisParts(0.62),
    { w: 0.12, c: C.ice, f: (r) => { const x = (r() * 2 - 1) * W; return jitter(r, [x, lid(x) * (r() < 0.5 ? 1 : -1), 0], 0.012); } },
    {
      w: 0.1, c: C.dim,
      f: (r) => {
        for (;;) {
          const x = (r() * 2 - 1) * W, y = (r() * 2 - 1) * H;
          if (Math.abs(y) < lid(x) && Math.hypot(x, y) > 0.62) return [x, y, gauss(r) * 0.02];
        }
      },
    },
    { w: 0.04, c: C.ice, f: (r) => { const x = (r() * 2 - 1) * W * 0.85, t = r() * 0.14; return jitter(r, [x * (1 + t * 0.3), lid(x) + t, 0], 0.004); } },
  ], (p) => rotX(p, 0.15)));
}

// Human: the Vitruvian figure — arm span equals height, inside a circle
// and a square, with Leonardo's second pose ghosted behind.
function human(N, seed) {
  // tapered limb from a (radius ra) to b (radius rb)
  const limb = (a, b, ra, rb) => ({ a, b, ra, rb });
  const side = (sx, arm, leg) => [
    limb([sx * 0.19, 0.56, 0], [sx * 0.5, 0.56 + arm * 0.31, 0], 0.045, 0.036),
    limb([sx * 0.5, 0.56 + arm * 0.31, 0], [sx * 0.8, 0.56 + arm * 0.62, 0], 0.036, 0.027),
    limb([sx * 0.8, 0.56 + arm * 0.62, 0], [sx * 0.87, 0.56 + arm * 0.69, 0], 0.03, 0.018), // hand
    limb([sx * 0.085, 0.04, 0], [sx * (0.12 + leg * 0.2), -0.44, 0], 0.07, 0.045),
    limb([sx * (0.12 + leg * 0.2), -0.44, 0], [sx * (0.16 + leg * 0.4), -0.88, 0], 0.045, 0.03),
    limb([sx * (0.16 + leg * 0.4), -0.9, 0], [sx * (0.18 + leg * 0.4), -0.92, 0.09], 0.028, 0.022), // foot
  ];
  const pose = [...side(-1, 0.06, 0), ...side(1, 0.06, 0), limb([0, 0.56, 0], [0, 0.7, 0], 0.048, 0.036), limb([-0.05, 0.6, 0], [-0.19, 0.56, 0], 0.035, 0.045), limb([0.05, 0.6, 0], [0.19, 0.56, 0], 0.035, 0.045)];
  const ghost = [...side(-1, 0.55, 0.35), ...side(1, 0.55, 0.35)];
  const limbSampler = (limbs) => {
    const pick = picker(limbs, (l) => len(sub(l.b, l.a)) * (l.ra + l.rb));
    return (r) => { const l = pick(r), t = r(); return add(lerp3(l.a, l.b, t), onSphere(r, l.ra + (l.rb - l.ra) * t)); };
  };
  // torso: hips, waist, chest, shoulders
  const profile = [[0.0, 0.15], [0.12, 0.155], [0.28, 0.118], [0.45, 0.165], [0.55, 0.195], [0.6, 0.11]];
  // head, bottom to top: chin, jaw, cheekbones, temples, crown
  const head = [[0.0, 0.025], [0.12, 0.055], [0.35, 0.074], [0.6, 0.08], [0.82, 0.068], [0.95, 0.04], [1.0, 0.0]];
  const halfAt = (prof, y) => {
    for (let i = 1; i < prof.length; i++) {
      if (y <= prof[i][0]) { const [y0, w0] = prof[i - 1], [y1, w1] = prof[i]; return w0 + ((w1 - w0) * (y - y0)) / (y1 - y0); }
    }
    return prof[prof.length - 1][1];
  };
  const torso = (r) => { const y = r() * 0.6, w = halfAt(profile, y), a = r() * Math.PI * 2; return [Math.cos(a) * w, y, Math.sin(a) * w * 0.55]; };
  const C0 = [0, 0.1, 0], CR = 1.03;
  const sq = (r) => {
    const t = r() * 2 - 1, h = 0.9;
    const p = [[t * h, h], [t * h, -h], [h, t * h], [-h, t * h]][Math.floor(r() * 4)];
    return jitter(r, [p[0], p[1] - 0.05, 0], 0.004);
  };
  return build(N, seed, xform([
    {
      w: 0.045, c: C.ice,
      f: (r) => {
        const y = r(), w = halfAt(head, y), a = r() * Math.PI * 2;
        return [Math.cos(a) * w, 0.67 + y * 0.23, Math.sin(a) * w * 1.15];
      },
    },
    { w: 0.24, c: C.ice, f: torso },
    { w: 0.34, c: C.ice, f: limbSampler(pose) },
    { w: 0.08, c: C.dim, f: limbSampler(ghost) },
    { w: 0.12, c: C.deep, f: (r) => { const a = r() * Math.PI * 2; return jitter(r, add(C0, [Math.cos(a) * CR, Math.sin(a) * CR, 0]), 0.004); } },
    { w: 0.1, c: C.deep, f: sq },
    starfield(0.04, 1.2),
  ], (p) => [p[0], p[1] - 0.1, p[2]]));
}

// Brain: folded cortex over two hemispheres, the striated cerebellum and
// brainstem, with a neural network inside — the structure Vazza & Feletti
// compared with the cosmic web.
function brain(N, seed) {
  const r0 = rng(seed + 83);
  const S = [1.0, 0.72, 0.82];
  const shell = (d) => {
    const p = [d[0] * S[0], d[1] * S[1], d[2] * S[2]];
    if (p[1] < -0.25) p[1] = -0.25 + (p[1] + 0.25) * 0.5; // flatter base
    return p;
  };
  const ridge = (d) => 1 - Math.abs(noise3(d[0] * 4.5, d[1] * 4.5, d[2] * 4.5, 5) * 2 - 1);
  const net = webParts(r0, { count: 120, R: 0.8, width: 0.01, knot: 0.02, w: 0.35, fil: C.blue });
  return build(N, seed, xform([
    {
      w: 0.5, c: C.ice,
      f: (r) => {
        for (;;) {
          const d = onSphere(r, 1), g = ridge(d);
          if (Math.abs(d[2]) < 0.04 && d[1] > -0.2) continue; // longitudinal fissure
          if (r() < g ** 6) return [...mul(shell(d), 1 + g * 0.06), ...(g > 0.9 ? C.white : C.ice)];
        }
      },
    },
    ...xform(net, (p) => [p[0] * S[0], p[1] * S[1], p[2] * S[2]]),
    {
      w: 0.1, c: C.blue,
      f: (r) => {
        for (;;) {
          const d = onSphere(r, 1), p = [-0.62 + d[0] * 0.3, -0.42 + d[1] * 0.18, d[2] * 0.4];
          if (Math.sin(p[1] * 90) > -0.2) return p;
        }
      },
    },
    { w: 0.05, c: C.deep, f: (r) => { const t = r(); return add(lerp3([-0.3, -0.4, 0], [-0.24, -0.95, 0], t), onSphere(r, 0.08 - t * 0.02)); } },
  ], (p) => rotY([p[0], p[1] + 0.05, p[2]], -0.5)));
}

// Tree: recursive branching above, roots mirroring it below.
function tree(N, seed) {
  const r = rng(seed + 23);
  const t = grow(r, { origin: [0, -1.0, 0], dir: [0, 1, 0], length: 0.55, width: 0.035, depth: 6, spread: 0.7, decay: 0.74, kids: [2, 3], bend: 0.05, pull: [0, 0.05, 0], steps: 3 });
  const roots = grow(r, { origin: [0, -1.0, 0], dir: [0, -1, 0], length: 0.25, width: 0.025, depth: 3, spread: 0.9, decay: 0.7, kids: [2, 3], bend: 0.25, pull: [0, 0.03, 0] });
  return build(N, seed, [
    { w: 0.55, c: C.ice, f: segSampler(t.segs) },
    { w: 0.3, c: C.blue, f: (r) => [...add(t.tips[Math.floor(r() * t.tips.length)], inSphere(r, 0.045)), ...(r() < 0.3 ? C.white : C.blue)] },
    { w: 0.1, c: C.deep, f: segSampler(roots.segs) },
    { w: 0.05, c: C.dim, f: (r) => { const a = r() * Math.PI * 2, rr = Math.sqrt(r()) * 0.6; return [Math.cos(a) * rr, -1.0, Math.sin(a) * rr]; } },
  ]);
}

// River delta seen from orbit: one channel splitting into distributaries.
function riverDelta(N, seed) {
  const r = rng(seed + 29);
  const d = grow(r, { origin: [0, 1.1, 0], dir: [0, -1, 0], length: 0.5, width: 0.03, depth: 6, spread: 0.5, decay: 0.78, kids: [2, 2], bend: 0.15, flat: true });
  return build(N, seed, xform([
    { w: 0.8, c: C.ice, f: segSampler(d.segs) },
    { w: 0.2, c: C.deep, f: (r) => add(d.tips[Math.floor(r() * d.tips.length)], [gauss(r) * 0.08, gauss(r) * 0.08, 0]) },
  ], (p) => rotX(p, -0.5)));
}

// City at night from above: a warped street grid, a river, radial
// highways, a ring road and downtown towers.
function city(N, seed) {
  const r0 = rng(seed + 31);
  const river = (x) => 0.25 * Math.sin(x * 2.2) + 0.1;
  const G = 0.07;
  const towers = Array.from({ length: 60 }, () => ({ x: gauss(r0) * 0.15, z: gauss(r0) * 0.15, h: 0.05 + r0() * 0.25 }));
  return build(N, seed, xform([
    {
      w: 0.55, c: C.ice,
      f: (r) => {
        for (;;) {
          const horiz = r() < 0.5, line = Math.round(((r() * 2 - 1) * 1.2) / G) * G, t = (r() * 2 - 1) * 1.2;
          let x = horiz ? t : line, z = horiz ? line : t;
          x += (noise3(x * 1.5, z * 1.5, 0, 1) - 0.5) * 0.15;
          z += (noise3(x * 1.5, z * 1.5, 0, 2) - 0.5) * 0.15;
          const d2 = x * x + z * z;
          if (d2 > 1.4 || r() > Math.exp(-d2 / 0.45) || Math.abs(z - river(x)) < 0.06) continue;
          return [x, gauss(r) * 0.004, z, ...(r() < 0.15 ? C.warm : C.ice)];
        }
      },
    },
    {
      w: 0.12, c: C.white,
      f: (r) => {
        if (r() < 0.4) { const a = r() * Math.PI * 2; return jitter(r, [Math.cos(a) * 0.75, 0, Math.sin(a) * 0.75], 0.004); }
        const a = (Math.floor(r() * 6) / 6) * Math.PI * 2 + 0.3, t = r() * 1.2;
        return jitter(r, [Math.cos(a) * t, 0, Math.sin(a) * t], 0.004);
      },
    },
    { w: 0.06, c: C.deep, f: (r) => { const x = (r() * 2 - 1) * 1.15; return jitter(r, [x, 0, river(x) + (r() < 0.5 ? -0.065 : 0.065)], 0.004); } },
    { w: 0.15, c: C.white, f: (r) => { const t = towers[Math.floor(r() * towers.length)]; return jitter(r, [t.x, r() * t.h, t.z], 0.004); } },
    starfield(0.12, 1.25),
  ], (p) => rotX(p, 0.8)));
}

// Hurricane: an eyewall around a calm eye, four logarithmic rain bands,
// cirrus outflow at the edge.
function spiralParts(r0, kind) {
  const bands = 4, b = 0.2, r1 = 0.14;
  const armPoint = (r, k, th) => {
    const rr = r1 * Math.exp(b * th), a = -th + (k * Math.PI * 2) / bands;
    return jitter(r, [Math.cos(a) * rr, 0, Math.sin(a) * rr], 0.015 + rr * 0.03);
  };
  const flatten = (p, h) => [p[0], p[1] * h, p[2]];
  const storm = kind === 'storm';
  const knots = Array.from({ length: 90 }, () => armPoint(r0, Math.floor(r0() * bands), r0() * 10.7));
  return [
    storm
      ? { w: 0.14, c: C.white, f: (r) => { const a = r() * Math.PI * 2, rr = 0.11 + Math.abs(gauss(r)) * 0.04; return [Math.cos(a) * rr, gauss(r) * 0.04, Math.sin(a) * rr]; } }
      : { w: 0.14, c: C.warm, f: (r) => [...flatten(jitter(r, [0, 0, 0], 0.07), 0.6), ...(r() < 0.3 ? C.white : C.warm)] },
    {
      w: storm ? 0.6 : 0.5, c: C.ice,
      f: (r) => {
        const p = armPoint(r, Math.floor(r() * bands), r() * 10.7);
        p[1] = gauss(r) * 0.02;
        return [...p, ...(storm ? (r() < 0.3 ? C.white : C.ice) : r() < 0.5 ? C.ice : C.blue)];
      },
    },
    ...(storm ? [] : [{ w: 0.1, c: C.white, f: (r) => add(knots[Math.floor(r() * knots.length)], inSphere(r, 0.014)) }]),
    {
      w: 0.2, c: storm ? C.blue : C.deep,
      f: (r) => {
        for (;;) {
          const rr = Math.min(0.12 + -Math.log(r() + 1e-9) * 0.35, 1.2), a = r() * Math.PI * 2;
          const th = Math.log(rr / r1) / b;
          if (r() < Math.pow(0.5 + 0.5 * Math.sin((a + th) * bands), 3)) return [Math.cos(a) * rr, gauss(r) * 0.02, Math.sin(a) * rr];
        }
      },
    },
    { w: 0.06, c: storm ? C.deep : C.dim, f: (r) => { const a = r() * Math.PI * 2, rr = 0.9 + r() * 0.35; return [Math.cos(a) * rr, gauss(r) * 0.02, Math.sin(a) * rr]; } },
  ];
}

function hurricane(N, seed) {
  return build(N, seed, xform(spiralParts(rng(seed + 5), 'storm'), (p) => rotX(p, 1.05)));
}

// ================= Zone I — the cosmos =================

function earthParts(R = 1, at = [0, 0, 0], w = 1) {
  const tilt = (p) => add(at, rotZ(p, 0.41)); // 23.4° axial tilt
  return [
    {
      w: 0.62 * w, c: C.ice,
      f: (r) => {
        for (;;) {
          const d = onSphere(r, 1);
          if (fbm(mul(d, 1.8), 41) > 0.53) return [...tilt(mul(d, R)), ...(r() < 0.5 ? C.white : C.ice)];
          if (r() < 0.2) return [...tilt(mul(d, R)), ...C.dim];
        }
      },
    },
    { w: 0.12 * w, c: C.blue, f: (r) => { for (;;) { const d = onSphere(r, 1); if (fbm(add(mul(d, 3), [5, 0, 0]), 43) > 0.58) return tilt(mul(d, R * 1.02)); } } },
    { w: 0.16 * w, c: C.blue, f: (r) => tilt(mul(onSphere(r, 1), R * (1.03 + Math.abs(gauss(r)) * 0.03))) },
  ];
}

function earth(N, seed) {
  return build(N, seed, earthParts(1), 1.15);
}

function earthMoon(N, seed) {
  const orbit = (p) => rotX(p, 1.1);
  const moon = orbit([Math.cos(-0.6) * 1.05, 0, Math.sin(-0.6) * 1.05]);
  return build(N, seed, [
    ...earthParts(0.15, [0, 0, 0], 0.14),
    { w: 0.04, c: C.ice, f: (r) => add(moon, onSphere(r, 0.045)) },
    { w: 0.3, c: C.deep, f: (r) => { const a = r() * Math.PI * 2; return jitter(r, orbit([Math.cos(a) * 1.05, 0, Math.sin(a) * 1.05]), 0.004); } },
    starfield(0.52),
  ]);
}

// The Sun: granulated photosphere, coronal streamers, prominence loops.
function sun(N, seed) {
  const r0 = rng(seed + 47);
  const R = 0.62;
  const streamers = Array.from({ length: 60 }, () => onSphere(r0, 1));
  const loops = Array.from({ length: 6 }, () => {
    const a = onSphere(r0, 1);
    return { a, b: unit(add(a, mul(onSphere(r0, 1), 0.25))), h: 0.12 + r0() * 0.15 };
  });
  return build(N, seed, [
    { w: 0.45, c: C.warm, f: (r) => { const d = onSphere(r, 1); return [...mul(d, R), ...lerp3(C.warm, C.white, fbm(mul(d, 9), 51))]; } },
    { w: 0.12, c: C.warm, f: (r) => inSphere(r, R) },
    {
      w: 0.3, c: C.ice,
      f: (r) => {
        const d = unit(add(streamers[Math.floor(r() * streamers.length)], mul(onSphere(r, 1), 0.18)));
        return mul(d, R * Math.min(1 + -Math.log(r() + 1e-9) * 0.35, 2));
      },
    },
    { w: 0.13, c: C.warm, f: (r) => { const l = loops[Math.floor(r() * loops.length)], t = r(); return jitter(r, mul(unit(lerp3(l.a, l.b, t)), R * (1 + l.h * Math.sin(t * Math.PI))), 0.01); } },
  ]);
}

// A star with planets, orbits and belts, laid in the xz plane.
function systemParts({ sun: sunR, planets, belts, ring }) {
  const pickPlanet = (r) => planets[Math.floor(r() * planets.length)];
  return [
    { w: 0.08, c: C.warm, f: (r) => (r() < 0.6 ? inSphere(r, sunR) : jitter(r, [0, 0, 0], sunR * 0.8)) },
    { w: 0.2, c: C.deep, f: (r) => { const p = pickPlanet(r), a = r() * Math.PI * 2; return jitter(r, [Math.cos(a) * p.r, 0, Math.sin(a) * p.r], 0.003); } },
    { w: 0.08, c: C.white, f: (r) => { const p = pickPlanet(r); return add([Math.cos(p.a) * p.r, 0, Math.sin(p.a) * p.r], inSphere(r, p.size)); } },
    ...belts.map((b) => ({
      w: b.w, c: C.ice,
      f: (r) => {
        for (;;) {
          const rr = b.r0 + r() * (b.r1 - b.r0);
          if (b.gap && Math.abs(rr - b.gap) < 0.012) continue; // Kirkwood gap
          const a = r() * Math.PI * 2;
          return [Math.cos(a) * rr, gauss(r) * b.thick, Math.sin(a) * rr];
        }
      },
    })),
    ...(ring ? [{
      w: 0.03, c: C.honey,
      f: (r) => { const a = r() * Math.PI * 2, rr = ring.size * (1.6 + r() * 0.8); return add([Math.cos(ring.a) * ring.r, 0, Math.sin(ring.a) * ring.r], rotX([Math.cos(a) * rr, 0, Math.sin(a) * rr], 0.45)); },
    }] : []),
  ];
}

function innerSolar(N, seed) {
  return build(N, seed, xform([
    ...systemParts({
      sun: 0.06,
      planets: [{ r: 0.24, a: 0.4, size: 0.012 }, { r: 0.42, a: 2.1, size: 0.022 }, { r: 0.6, a: 3.9, size: 0.024 }, { r: 0.88, a: 5.2, size: 0.016 }],
      belts: [{ r0: 1.0, r1: 1.25, w: 0.45, thick: 0.03, gap: 1.12 }],
    }),
    starfield(0.18),
  ], (p) => rotX(p, 1.1)));
}

function solarSystem(N, seed) {
  const saturn = { r: 0.62, a: 1.2, size: 0.03 };
  return build(N, seed, xform([
    ...systemParts({
      sun: 0.04,
      planets: [
        { r: 0.1, a: 0.4, size: 0.006 }, { r: 0.14, a: 2.1, size: 0.01 }, { r: 0.18, a: 3.9, size: 0.01 }, { r: 0.23, a: 5.2, size: 0.008 },
        { r: 0.45, a: 4.4, size: 0.04 }, saturn, { r: 0.8, a: 2.8, size: 0.022 }, { r: 0.96, a: 0.2, size: 0.022 },
      ],
      belts: [{ r0: 0.27, r1: 0.33, w: 0.14, thick: 0.01 }, { r0: 1.02, r1: 1.25, w: 0.34, thick: 0.04 }],
      ring: saturn,
    }),
    starfield(0.14),
  ], (p) => rotX(p, 1.15)));
}

// The Bohr atom: electrons on fixed orbits like planets. Superseded by
// quantum mechanics, but the classic picture of the resemblance.
function bohrAtom(N, seed) {
  const r0 = rng(seed + 7);
  const nucleons = Array.from({ length: 14 }, () => inSphere(r0, 0.09));
  const orbit = (k, t) => rotX(rotZ([Math.cos(t) * 1.1, Math.sin(t) * 0.38, 0], (k * Math.PI) / 3), 0.4);
  return build(N, seed, [
    { w: 0.2, c: C.warm, f: (r) => { const k = Math.floor(r() * nucleons.length); return [...jitter(r, nucleons[k], 0.03), ...(k % 2 ? C.warm : C.ice)]; } },
    { w: 0.6, c: C.blue, f: (r) => jitter(r, orbit(Math.floor(r() * 3), r() * Math.PI * 2), 0.006) },
    { w: 0.2, c: C.white, f: (r) => { const k = Math.floor(r() * 3); return add(orbit(k, k * 2.1 + 0.6), inSphere(r, 0.05)); } },
  ]);
}

// Oort cloud: the planetary disc at the centre, the flattened inner Hills
// cloud, and the vast spherical shell of comets.
function oort(N, seed) {
  return build(N, seed, xform([
    { w: 0.04, c: C.white, f: (r) => { const p = inSphere(r, 0.05); return [p[0], p[1] * 0.2, p[2]]; } },
    { w: 0.04, c: C.ice, f: (r) => { const a = r() * Math.PI * 2, rr = 0.06 + r() * 0.03; return [Math.cos(a) * rr, gauss(r) * 0.004, Math.sin(a) * rr]; } },
    { w: 0.27, c: C.blue, f: (r) => { const d = onSphere(r, 0.15 + r() * 0.3); return [d[0], d[1] * 0.45, d[2]]; } },
    { w: 0.65, c: C.ice, f: (r) => onSphere(r, 0.6 + Math.pow(r(), 0.7) * 0.65) },
  ], (p) => rotX(p, 1.1)));
}

// Stellar neighbourhood: the Sun at centre, the nearest stars around it,
// with map lines to the closest ones.
function neighbourhood(N, seed) {
  const r0 = rng(seed + 53);
  const stars = [
    { p: [0, 0, 0], m: 0.8, c: C.honey },              // the Sun
    { p: [0.42, -0.2, 0.25], m: 0.9, c: C.white },    // Alpha Centauri A/B
    { p: [0.45, -0.18, 0.27], m: 0.5, c: C.warm },
    { p: [0.4, -0.24, 0.22], m: 0.25, c: C.warm },    // Proxima
    { p: [-0.5, 0.45, 0.3], m: 0.3, c: C.warm },      // Barnard's Star
    { p: [-0.8, -0.3, -0.55], m: 1, c: C.white },     // Sirius
  ];
  for (let i = 0; i < 44; i++) {
    stars.push({ p: inSphere(r0, 1.2), m: 0.15 + r0() ** 2 * 0.8, c: [C.white, C.ice, C.warm][Math.floor(r0() * 3)] });
  }
  const near = stars.slice(1).sort((a, b) => len(a.p) - len(b.p)).slice(0, 10);
  const lines = near.map((s) => ({ a: [0, 0, 0], b: s.p, w: 0.002 }));
  const pick = picker(stars, (s) => s.m);
  return build(N, seed, [
    { w: 0.55, c: C.white, f: (r) => { const s = pick(r); return [...jitter(r, s.p, 0.006 * (0.6 + s.m)), ...s.c]; } },
    { w: 0.15, c: C.deep, f: (r) => { const s = pick(r); return jitter(r, s.p, 0.03 * s.m); } },
    { w: 0.1, c: C.deep, f: segSampler(lines) },
    starfield(0.2),
  ]);
}

// Globular cluster: ancient stars in a Plummer-profile sphere.
function starCluster(N, seed) {
  const r0 = rng(seed + 57);
  const plummer = (r) => {
    for (;;) {
      const rr = 0.16 / Math.sqrt(Math.pow(r() + 1e-9, -2 / 3) - 1);
      if (rr < 1.25) return onSphere(r, rr);
    }
  };
  const giants = Array.from({ length: 40 }, () => plummer(r0));
  return build(N, seed, [
    { w: 0.92, c: C.ice, f: (r) => [...plummer(r), ...[C.white, C.ice, C.warm, C.honey][Math.floor(r() * 4)]] },
    { w: 0.08, c: C.warm, f: (r) => add(giants[Math.floor(r() * giants.length)], inSphere(r, 0.012)) },
  ]);
}

// A barred spiral galaxy in the xz plane: bulge, bar, two major and two
// minor logarithmic arms, a diffuse disc and bright star-forming knots.
function galaxyParts({ r0, s = 1, at = [0, 0, 0], rot = (p) => p, w = 1, bar = true, arms = 2 }) {
  const place = (p) => add(at, rot(mul(p, s)));
  const wind = 0.209;
  const armPoint = (r, k, th) => {
    const rr = 0.2 * Math.exp(wind * th), major = k < arms;
    const a = th + ((k % arms) * Math.PI * 2) / arms + (major ? 0 : Math.PI / arms);
    const sp = (major ? 0.028 : 0.02) + rr * 0.03;
    return [Math.cos(a) * rr + gauss(r) * sp, gauss(r) * 0.02, Math.sin(a) * rr + gauss(r) * sp];
  };
  const knots = Array.from({ length: 70 }, () => armPoint(r0, Math.floor(r0() * arms), r0() * 2.6 * Math.PI));
  return [
    { w: 0.1 * w, c: C.warm, f: (r) => place([gauss(r) * 0.1, gauss(r) * 0.05, gauss(r) * 0.1]) },
    ...(bar ? [{ w: 0.06 * w, c: C.warm, f: (r) => place([gauss(r) * 0.25, gauss(r) * 0.03, gauss(r) * 0.06]) }] : []),
    { w: 0.6 * w, c: C.ice, f: (r) => { const k = Math.floor(r() * arms * 2); return [...place(armPoint(r, k, r() * 2.6 * Math.PI)), ...(k < arms ? C.ice : C.blue)]; } },
    { w: 0.14 * w, c: C.deep, f: (r) => { const rr = Math.min(-Math.log(r() + 1e-9) * 0.35, 1.2), a = r() * Math.PI * 2; return place([Math.cos(a) * rr, gauss(r) * 0.02, Math.sin(a) * rr]); } },
    { w: 0.1 * w, c: C.white, f: (r) => place(add(knots[Math.floor(r() * knots.length)], inSphere(r, 0.015))) },
  ];
}

function milkyWay(N, seed) {
  return build(N, seed, galaxyParts({ r0: rng(seed + 3), rot: (p) => rotX(p, 1.05) }));
}

// Local Group: the Milky Way, Andromeda, Triangulum and their dwarfs.
function localGroup(N, seed) {
  const r0 = rng(seed + 59);
  const mw = [-0.55, -0.05, 0], m31 = [0.5, 0.15, -0.1];
  const dwarfs = Array.from({ length: 40 }, () => ({
    p: add(r0() < 0.5 ? mw : m31, mul(onSphere(r0, 1), 0.15 + r0() * 0.35)),
    s: 0.01 + r0() * 0.025,
  }));
  return build(N, seed, [
    ...galaxyParts({ r0, s: 0.32, at: mw, rot: (p) => rotX(rotZ(p, 0.3), 1.1), w: 0.3 }),
    ...galaxyParts({ r0, s: 0.45, at: m31, rot: (p) => rotX(rotZ(p, -0.6), 0.6), w: 0.38 }),
    ...galaxyParts({ r0, s: 0.15, at: [0.85, -0.45, 0.1], rot: (p) => rotX(p, 1.3), w: 0.08, bar: false }),
    { w: 0.14, c: C.ice, f: (r) => { const d = dwarfs[Math.floor(r() * dwarfs.length)]; return jitter(r, d.p, d.s); } },
    starfield(0.1),
  ]);
}

// Virgo Supercluster: galaxy groups strung along filaments around the
// dense Virgo Cluster; the Local Group sits near the centre of the view.
function virgo(N, seed) {
  const r0 = rng(seed + 61);
  const flat = (p) => [p[0], p[1] * 0.35, p[2]];
  const vc = [0.45, 0.05, 0.1];
  const gal = [];
  const g3 = () => [gauss(r0), gauss(r0), gauss(r0)];
  for (let i = 0; i < 350; i++) gal.push(add(vc, mul(g3(), 0.1)));
  for (let f = 0; f < 7; f++) {
    const end = flat(mul(onSphere(r0, 1), 1.0 + r0() * 0.2));
    for (let g = 0; g < 6; g++) {
      const c = add(lerp3(vc, end, (g + 0.5) / 6 + gauss(r0) * 0.03), mul(onSphere(r0, 1), 0.06));
      const n = 8 + Math.floor(r0() * 30);
      for (let i = 0; i < n; i++) gal.push(add(c, mul(g3(), 0.035)));
    }
  }
  for (let i = 0; i < 30; i++) gal.push(mul(g3(), 0.03)); // the Local Group
  for (let i = 0; i < 300; i++) gal.push(flat(inSphere(r0, 1.2)));
  return build(N, seed, xform([
    { w: 0.8, c: C.ice, f: (r) => [...jitter(r, gal[Math.floor(r() * gal.length)], 0.005), ...[C.ice, C.white, C.blue][Math.floor(r() * 3)]] },
    { w: 0.2, c: C.dim, f: (r) => jitter(r, vc, 0.2) },
  ], (p) => rotX(p, 0.5)));
}

// Laniakea: streamlines of galaxies flowing toward the Great Attractor,
// after Tully et al. (2014). The bright dot marks the Milky Way.
function laniakea(N, seed) {
  const r0 = rng(seed + 67);
  const lines = [];
  let home = null;
  for (let k = 0; k < 240; k++) {
    const d = onSphere(r0, 1);
    let p = mul([d[0] * 1.15, d[1] * 0.6, d[2] * 0.95], 0.85 + fbm(mul(d, 2), 71) * 0.4);
    for (let step = 0; step < 90; step++) {
      const dist = len(p);
      if (dist < 0.04) break;
      const n = [noise3(p[0] * 2.5, p[1] * 2.5, p[2] * 2.5, 1) - 0.5, noise3(p[0] * 2.5, p[1] * 2.5, p[2] * 2.5, 2) - 0.5, noise3(p[0] * 2.5, p[1] * 2.5, p[2] * 2.5, 3) - 0.5];
      const q = add(p, mul(unit(add(unit(mul(p, -1)), mul(n, 1.6))), 0.035));
      lines.push({ a: p, b: q, w: 0.008, d: dist, c: dist < 0.35 ? C.white : dist < 0.7 ? C.ice : C.blue });
      if (k === 7 && step === 4) home = q;
      p = q;
    }
  }
  return build(N, seed, xform([
    { w: 0.8, c: C.ice, f: segSampler(lines, (s) => 1 / (s.d + 0.25)) },
    { w: 0.06, c: C.white, f: (r) => jitter(r, [0, 0, 0], 0.05) },
    { w: 0.01, c: C.white, f: (r) => add(home, inSphere(r, 0.014)) },
    { w: 0.13, c: C.dim, f: (r) => { const d = onSphere(r, 1); return mul([d[0] * 1.15, d[1] * 0.6, d[2] * 0.95], 0.85 + fbm(mul(d, 2), 71) * 0.4); } },
  ], (p) => rotX(p, 0.4)));
}

// Cosmic web: galaxy clusters (nodes) joined by filaments around voids.
function webParts(r0, { count = 90, R = 1.15, k = 3, width = 0.02, knot = 0.035, w = 1, fil = C.blue }) {
  const nodes = [];
  for (let tries = 0; nodes.length < count && tries < 20000; tries++) {
    const p = inSphere(r0, R);
    if (nodes.every((q) => len(sub(p, q)) > R * 0.18)) nodes.push(p);
  }
  const keys = new Set(), segs = [], degree = nodes.map(() => 0);
  nodes.forEach((p, i) => {
    nodes.map((q, j) => [len(sub(p, q)), j]).sort((a, b) => a[0] - b[0]).slice(1, k + 1).forEach(([, j]) => {
      const key = i < j ? `${i}-${j}` : `${j}-${i}`;
      if (keys.has(key)) return;
      keys.add(key);
      segs.push({ a: p, b: nodes[j], w: width });
      degree[i]++; degree[j]++;
    });
  });
  const idx = nodes.map((_, i) => i);
  const pickNode = picker(idx, (i) => degree[i] + 1);
  return [
    { w: 0.55 * w, c: fil, f: (r) => [...segSampler(segs)(r), ...(r() < 0.35 ? C.ice : fil)] },
    { w: 0.35 * w, c: C.white, f: (r) => jitter(r, nodes[pickNode(r)], knot) },
    { w: 0.1 * w, c: C.dim, f: (r) => inSphere(r, R) },
  ];
}

function cosmicWeb(N, seed) {
  return build(N, seed, webParts(rng(seed + 3), {}));
}

// Observable universe: the cosmic web inside, the cosmic microwave
// background as its boundary, and us at the centre.
function observableUniverse(N, seed) {
  return build(N, seed, [
    ...webParts(rng(seed + 3), { count: 260, R: 0.92, width: 0.012, knot: 0.018, w: 0.5, fil: C.deep }),
    {
      w: 0.45, c: C.ice,
      f: (r) => {
        const d = onSphere(r, 1), t = fbm(mul(d, 5), 81);
        return [...mul(d, 1 + gauss(r) * 0.01), ...(t > 0.52 ? C.white : t > 0.47 ? C.ice : C.blue)];
      },
    },
    { w: 0.005, c: C.white, f: (r) => inSphere(r, 0.012) },
  ]);
}

// Helix Nebula: a dying star's shed layers — central white dwarf, inner
// disc, main ring lined with radial cometary knots, and a tilted outer ring.
function helixNebula(N, seed) {
  return build(N, seed, [
    { w: 0.03, c: C.white, f: (r) => inSphere(r, 0.03) },
    { w: 0.12, c: C.dim, f: (r) => { const p = inSphere(r, 0.42); return [p[0], p[1], p[2] * 0.3]; } },
    { w: 0.4, c: C.blue, f: (r) => { const a = r() * Math.PI * 2, rr = 0.62 + gauss(r) * 0.09 + 0.04 * Math.sin(a * 5); return [Math.cos(a) * rr, Math.sin(a) * rr * 0.92, gauss(r) * 0.06]; } },
    {
      w: 0.25, c: C.ice,
      f: (r) => {
        const k = Math.floor(r() * 260), a = (k / 260) * Math.PI * 2, t = r(), rr = 0.42 + t * 0.14;
        return [...jitter(r, [Math.cos(a) * rr, Math.sin(a) * rr * 0.92, 0], 0.006), ...(t < 0.25 ? C.white : C.ice)];
      },
    },
    { w: 0.2, c: C.deep, f: (r) => { const a = r() * Math.PI * 2, rr = 1.0 + gauss(r) * 0.08; return rotX(rotZ([Math.cos(a) * rr, Math.sin(a) * rr * 0.85, gauss(r) * 0.05], 0.6), 0.3); } },
  ]);
}

// ================= Zone II — the hypothesis =================

// Multiverse: bubble universes in a froth; ours (white) at the centre.
function multiverse(N, seed) {
  const r0 = rng(seed + 73);
  const bubbles = [{ p: [0, 0, 0], rad: 0.14, ours: true }];
  for (let tries = 0; bubbles.length < 28 && tries < 4000; tries++) {
    const rad = 0.1 + r0() ** 2 * 0.32, p = inSphere(r0, 1.2 - rad);
    if (bubbles.every((b) => len(sub(p, b.p)) > (b.rad + rad) * 0.85)) bubbles.push({ p, rad });
  }
  const shell = picker(bubbles, (b) => b.rad * b.rad);
  const fill = picker(bubbles, (b) => b.rad ** 3);
  return build(N, seed, [
    { w: 0.6, c: C.gold, f: (r) => { const b = shell(r); return [...add(b.p, onSphere(r, b.rad)), ...(b.ours ? C.white : r() < 0.5 ? C.gold : C.honey)]; } },
    { w: 0.3, c: C.amber, f: (r) => { const b = fill(r); return add(b.p, inSphere(r, b.rad * 0.9)); } },
    { w: 0.1, c: [0.5, 0.36, 0.16], f: (r) => inSphere(r, 1.25) },
  ]);
}

// Ultraverse: the serpent itself — a ring of countless small universes,
// tapering from tail to head, the head closing on its own tail.
function ultraverse(N, seed) {
  const r0 = rng(seed + 79);
  const RING = 0.9, SPAN = Math.PI * 2 * 0.96;
  const tube = (u) => 0.04 + 0.16 * u;
  const center = (u) => { const a = u * SPAN; return [Math.cos(a) * RING, Math.sin(a) * RING, Math.sin(a * 3) * 0.06]; };
  const cells = Array.from({ length: 520 }, () => {
    const u = r0(), t = tube(u);
    return { p: add(center(u), mul(onSphere(r0, 1), t * Math.sqrt(r0()) * 0.8)), rad: t * 0.35 * (0.5 + r0() * 0.5) };
  });
  const pickCell = picker(cells, (c) => c.rad * c.rad);
  const ha = SPAN;
  const T = [-Math.sin(ha), Math.cos(ha), 0], Rd = [Math.cos(ha), Math.sin(ha), 0], Z = [0, 0, 1];
  const headC = add(center(1), mul(T, 0.05));
  const head = (q) => add(headC, add(add(mul(T, q[0] * 0.26), mul(Rd, q[1] * 0.18)), mul(Z, q[2] * 0.14)));
  return build(N, seed, xform([
    { w: 0.6, c: C.gold, f: (r) => { const c = pickCell(r); return [...add(c.p, onSphere(r, c.rad)), ...(r() < 0.5 ? C.gold : C.honey)]; } },
    { w: 0.2, c: C.amber, f: (r) => { const u = r(); return jitter(r, center(u), tube(u) * 0.5); } },
    { w: 0.14, c: C.honey, f: (r) => head(r() < 0.7 ? onSphere(r, 1) : inSphere(r, 1)) },
    { w: 0.02, c: C.white, f: (r) => add(add(headC, add(mul(T, 0.12), add(mul(Rd, 0.1), mul(Z, 0.08)))), inSphere(r, 0.02)) },
    { w: 0.04, c: [0.5, 0.36, 0.16], f: (r) => inSphere(r, 1.25) },
  ], (p) => rotX(p, 0.35)));
}

// Convergence: scale folds inward — a vortex spiralling into one point.
function convergence(N, seed) {
  return build(N, seed, xform([
    {
      w: 0.85, c: C.gold,
      f: (r) => {
        const k = Math.floor(r() * 9), rr = Math.pow(r(), 1.8) * 1.2 + 0.01;
        const a = (k / 9) * Math.PI * 2 + Math.log(rr) * 2.2 + gauss(r) * 0.08;
        const y = Math.max(-0.9, -0.25 / (rr + 0.15) + 0.2);
        return [Math.cos(a) * rr, y, Math.sin(a) * rr, ...(rr < 0.15 ? C.white : rr < 0.5 ? C.honey : r() < 0.5 ? C.gold : C.amber)];
      },
    },
    { w: 0.15, c: C.white, f: (r) => jitter(r, [0, -0.9, 0], 0.04) },
  ], (p) => rotX([p[0], p[1] + 0.35, p[2]], 0.7)));
}

// ---------- stages ----------
// exp: log10 of the subject's size in metres; null where the hypothesis
// goes beyond anything measurable.

export const STAGES = [
  { name: 'Atom', exp: -10, zone: 'observed', gen: atom, line: 'A nucleus held inside a cloud of probability. Almost entirely empty space.' },
  { name: 'Molecule', exp: -9, zone: 'observed', gen: molecule, line: 'Atoms bind into structure. A ring of glucose, the fuel of living things.' },
  { name: 'DNA', exp: -8, zone: 'observed', gen: dna, line: 'Two strands, coiled. The instruction set of every organism on Earth.' },
  { name: 'Ribosome', exp: -7.6, zone: 'observed', gen: ribosome, line: 'A molecular machine that reads genetic code and builds proteins, one amino acid at a time.' },
  { name: 'Virus', exp: -7, zone: 'observed', gen: virus, line: 'A shell of protein around a strand of code, at the border of what we call alive.' },
  { name: 'Bacterium', exp: -5.7, zone: 'observed', gen: bacterium, line: 'A single cell, complete in itself. For two billion years, this was all of life on Earth.' },
  {
    name: 'Human Cell', exp: -4.7, zone: 'observed', gen: cell, line: 'A nucleus, a membrane, and thousands of machines at work inside one sphere.',
    mirror: { name: 'Observable Universe', exp: 26.9, gen: observableUniverse, note: 'A bounded sphere dense with inner structure: a membrane around a cell, the cosmic microwave background around everything we can see.', source: 'Visual comparison' },
  },
  { name: 'Neuron', exp: -3.3, zone: 'observed', gen: neuron, line: 'A single cell that branches to connect, carrying signals as electrical pulses.' },
  { name: 'Grain of Sand', exp: -2.7, zone: 'observed', gen: sandGrain, line: 'Rock worn down by time. By most estimates, the universe holds more stars than Earth holds grains like this.' },
  {
    name: 'Human Eye', exp: -1.6, zone: 'observed', gen: eye, line: 'The instrument through which the universe looks back at itself.',
    mirror: { name: 'Helix Nebula', exp: 16.4, gen: helixNebula, note: 'A dark centre, a ring of radial fibres, a fading outer edge: an iris twelve millimetres wide and a nebula three light-years across.', source: 'Visual comparison · NASA / ESA Hubble imagery' },
  },
  {
    name: 'Brain', exp: -0.8, zone: 'observed', gen: brain, line: 'Eighty-six billion neurons and a hundred trillion connections. The most complex object we know.',
    mirror: { name: 'Cosmic Web', exp: 25.5, gen: cosmicWeb, note: 'Measured, not only seen: the brain’s neuronal network and the cosmic web share statistically similar structure.', source: 'Vazza & Feletti, Frontiers in Physics (2020)' },
  },
  { name: 'Human', exp: 0.2, zone: 'observed', gen: human, line: 'The observer. Built from atoms, standing on a planet, asking what contains it all.' },
  {
    name: 'Tree', exp: 1.3, zone: 'observed', gen: tree, line: 'Branches above, roots below. A pattern of distribution that nature repeats everywhere.',
    mirror: { name: 'River Delta', exp: 5, gen: riverDelta, note: 'Networks that distribute flow at the least cost branch the same way: trees, rivers, lungs, lightning.', source: 'Bejan, Constructal Law (1996)' },
  },
  { name: 'City', exp: 4, zone: 'observed', gen: city, line: 'Ten million lights, connected by streets that branch like veins.' },
  { name: 'Hurricane', exp: 5.7, zone: 'observed', gen: hurricane, line: 'A storm five hundred kilometres wide, spiralling around a calm eye.' },
  { name: 'Earth', exp: 7.1, zone: 'observed', gen: earth, line: 'Every human who has ever lived, on a sphere twelve thousand kilometres across.' },
  { name: 'Earth & Moon', exp: 8.9, zone: 'observed', gen: earthMoon, line: 'Bound by gravity, three hundred and eighty-four thousand kilometres apart.' },
  { name: 'The Sun', exp: 9.1, zone: 'observed', gen: sun, line: 'A sphere of fusing hydrogen, holding 99.8% of the solar system’s mass.' },
  { name: 'Inner Solar System', exp: 11.5, zone: 'observed', gen: innerSolar, line: 'Mercury, Venus, Earth and Mars. The rocky worlds, fenced by the asteroid belt.' },
  {
    name: 'Solar System', exp: 13, zone: 'observed', gen: solarSystem, line: 'Eight planets orbiting one star. A nucleus, and what circles it.',
    mirror: { name: 'Bohr Atom', exp: -10, gen: bohrAtom, note: 'The famous resemblance. Physics has since replaced Bohr’s orbits with probability clouds: a similar picture, a different reality.', source: 'Bohr (1913), superseded by quantum mechanics' },
  },
  { name: 'Oort Cloud', exp: 16.2, zone: 'observed', gen: oort, line: 'A sphere of icy bodies a light-year or more from the Sun. The edge of its reach.' },
  { name: 'Stellar Neighbourhood', exp: 17.3, zone: 'observed', gen: neighbourhood, line: 'The nearest stars. Proxima Centauri, the closest, is 4.2 light-years away.' },
  { name: 'Star Cluster', exp: 18, zone: 'observed', gen: starCluster, line: 'Hundreds of thousands of ancient stars, bound into one sphere by gravity.' },
  { name: 'Milky Way', exp: 21, zone: 'observed', gen: milkyWay, line: 'A barred spiral of one to four hundred billion stars. The Sun is one of them.' },
  { name: 'Local Group', exp: 23, zone: 'observed', gen: localGroup, line: 'The Milky Way, Andromeda, and dozens of smaller galaxies, falling toward one another.' },
  { name: 'Virgo Supercluster', exp: 24, zone: 'observed', gen: virgo, line: 'Thousands of galaxies in groups and clusters, dominated by the Virgo Cluster.' },
  { name: 'Laniakea', exp: 24.7, zone: 'observed', gen: laniakea, line: '“Immeasurable heaven” in Hawaiian. A hundred thousand galaxies flowing toward one point.' },
  { name: 'Cosmic Web', exp: 25.5, zone: 'observed', gen: cosmicWeb, line: 'Filaments of galaxies around vast voids. The largest structure known.' },
  { name: 'Observable Universe', exp: 26.9, zone: 'observed', gen: observableUniverse, line: 'Everything whose light has had time to reach us. Its edge is the cosmic microwave background.' },
  { name: 'Multiverse', exp: null, zone: 'hypothesis', gen: multiverse, line: 'Here observation ends and the hypothesis begins: our universe as one bubble among many.' },
  { name: 'Ultraverse', exp: null, zone: 'hypothesis', gen: ultraverse, line: 'The structure that holds the multiverse. The serpent itself.' },
  { name: 'Convergence', exp: null, zone: 'hypothesis', gen: convergence, line: 'Go beyond the largest, and scale folds inward.' },
  { name: 'Atom', exp: -10, zone: 'hypothesis', gen: atomReturn, line: 'The smallest thing in existence is also the largest. The loop closes.' },
];
