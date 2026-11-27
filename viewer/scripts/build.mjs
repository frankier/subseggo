// Build orchestration for the viewer.
//
//   node scripts/build.mjs                 dev build once into dist/ (stable asset names)
//   node scripts/build.mjs --watch         rebuild on change (implies serving dist/)
//   node scripts/build.mjs --bundle        production bundle into dist/ (clean, minified, hashed)
//
// public/index.html references the emitted assets via __MAIN_JS__ / __MAIN_CSS__
// placeholders; this script replaces them with the real file names and copies the
// rest of public/ verbatim. If DATA_DIR is set, transcript.json / segments.json /
// config.json are copied from there instead, so a real workdir can be bundled in.
import * as esbuild from 'esbuild';
import { createServer } from 'node:http';
import { createReadStream, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync, copyFileSync } from 'node:fs';
import { extname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(fileURLToPath(new URL('..', import.meta.url)));
const publicDir = join(root, 'public');
const outdir = join(root, 'dist');

const args = new Set(process.argv.slice(2));
const watch = args.has('--watch');
const bundle = args.has('--bundle');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

function serveDist(port) {
  const server = createServer((req, res) => {
    const url = new URL(req.url ?? '/', 'http://localhost');
    let path = decodeURIComponent(url.pathname);
    if (path.endsWith('/')) path += 'index.html';
    const file = join(outdir, path);
    if (!file.startsWith(outdir) || !existsSync(file) || !statSync(file).isFile()) {
      res.writeHead(404).end('not found');
      return;
    }
    res.writeHead(200, { 'content-type': MIME[extname(file)] ?? 'application/octet-stream', 'cache-control': 'no-store' });
    createReadStream(file).pipe(res);
  });
  server.listen(port, () => console.log(`serving dist/ at http://localhost:${port}/`));
  return server;
}

// Copies public/ into dist/ (without touching dist/assets, which esbuild owns) and
// writes index.html with the emitted asset names filled in.
function copyPublic(assetJs, assetCss) {
  mkdirSync(outdir, { recursive: true });
  const dataDir = process.env.DATA_DIR ? resolve(process.env.DATA_DIR) : null;
  const dataOverrides = dataDir ? ['transcript.json', 'segments.json', 'config.json'] : [];
  const walk = (dir) => {
    for (const entry of readdirSync(dir)) {
      const src = join(dir, entry);
      if (statSync(src).isDirectory()) { walk(src); continue; }
      const rel = relative(publicDir, src);
      // index.html is templated with the emitted asset names; styles.css is
      // bundled by esbuild into assets/ instead of being copied verbatim
      if (rel === 'index.html' || rel === 'styles.css') continue;
      if (dataOverrides.includes(rel) && existsSync(join(dataDir, rel))) continue; // real data wins
      const dest = join(outdir, rel);
      mkdirSync(join(dest, '..'), { recursive: true });
      copyFileSync(src, dest);
    }
  };
  walk(publicDir);
  for (const rel of dataOverrides) {
    const src = join(dataDir, rel);
    if (existsSync(src)) copyFileSync(src, join(outdir, rel));
  }
  const html = readFileSync(join(publicDir, 'index.html'), 'utf8')
    .replaceAll('__MAIN_JS__', `./assets/${assetJs}`)
    .replaceAll('__MAIN_CSS__', `./assets/${assetCss}`);
  writeFileSync(join(outdir, 'index.html'), html);
}

function esbuildOptions() {
  return {
    entryPoints: [join(root, 'src/main.ts')],
    outdir: join(outdir, 'assets'),
    bundle: true,
    format: 'esm',
    target: 'es2020',
    sourcemap: true,
    minify: bundle,
    // [hash] goes in entryNames; outExtension does not expand placeholders
    entryNames: bundle ? '[name]-[hash]' : '[name]',
    metafile: true,
    logLevel: 'info',
  };
}

function assetNames(outputs) {
  const js = Object.keys(outputs).find((f) => f.endsWith('.js'));
  const css = Object.keys(outputs).find((f) => f.endsWith('.css'));
  return { js: js.replace(/^.*assets\//, ''), css: css ? css.replace(/^.*assets\//, '') : '' };
}

async function buildOnce() {
  rmSync(outdir, { recursive: true, force: true }); // fresh dist per one-shot run
  const result = await esbuild.build(esbuildOptions());
  const { js, css } = assetNames(result.metafile.outputs);
  copyPublic(js, css);
  return { js, css };
}

if (watch) {
  const ctx = await esbuild.context(esbuildOptions());
  // dist assets + index.html are rewritten on every rebuild so the served page is current
  const refresh = async () => {
    const result = await ctx.rebuild();
    const { js, css } = assetNames(result.metafile.outputs);
    copyPublic(js, css);
  };
  await refresh();
  await ctx.watch({ delay: 100 });
  console.log('watching src/ for changes...');
  serveDist(Number(process.env.PORT ?? 8080));
} else {
  const { js, css } = await buildOnce();
  console.log(`built assets/${js}${css ? ` + assets/${css}` : ''} -> dist/`);
  if (bundle) {
    const files = readdirSync(outdir, { recursive: true })
      .map((f) => join(outdir, f))
      .filter((f) => statSync(f).isFile())
      .sort();
    console.log(`bundle ready in dist/ (${files.length} files):`);
    for (const f of files) console.log(`  ${relative(outdir, f)}  ${statSync(f).size} B`);
    console.log('deploy dist/ as a static site (any static host); serve over http, fetch() needs it');
  }
}
