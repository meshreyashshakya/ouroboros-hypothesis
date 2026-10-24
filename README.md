# The Ouroboros Hypothesis

A hypothesis by Shreyash Shakya: the universe is a closed loop of scale, where the smallest thing in existence is also the largest.

An interactive scroll journey across 33 scales — from an atom, out through cells, planets, galaxies and the cosmic web, into the multiverse, and back to the atom. Every shape is generated in code from ~20,000 particles; nothing is an image.

## Run locally

No build step. Serve the folder and open it in a browser:

```
python3 -m http.server 8765
```

Then visit http://localhost:8765. Add `?stage=12` or `?twin=11` to jump to a stage or a mirror moment.

## Files

- `shapes.js` — particle generators for every stage, and the stage list
- `main.js` — renderer, scroll timeline, zoom and morph engine, HUD
- `index.html`, `style.css` — page and layout

Built with [Three.js](https://threejs.org).
