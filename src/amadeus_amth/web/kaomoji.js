/* Pixel kaomoji for the chat sidebar. Sprites and emotions come from
   pixel_itsuki.html; Kaomoji.set(name) swaps the emotion being animated. */
(() => {
const W = 35, H = 24, S = 6;
const COLORS = {
  k: "#5a2323", u: "#2f4fc4", w: "#ffffff", r: "#ff4f7b", b: "#ff8fa3", c: "#8fd0ff",
  y: "#ffd23a", o: "#ff9f1a", v: "#e0332f",
  h: "#d93a32", m: "#ec5a3a", l: "#f89a5e", s: "#ffe3cf", p: "#f6f4f1", g: "#cfc9c2",
};

const mirror = rows => rows.map(r => [...r].reverse().join(""));

// eye sprites sit in a 4-wide slot; blue "u" with a white "w" glint, dark "k" lash
const EYES = {
  up:    [".kk.", "k..k"],
  down:  ["k..k", ".kk."],
  line:  ["kkkk"],
  dot:   ["kkkk", "uwuu", ".uu."],
  half:  ["kkkk", "uwuu"],
  gt:    ["k.", ".k", "k."],
  angry: ["kk..", ".kkk", "uwuu"],
  cry:   ["kkkk", "wuuu", "uuuu", "uwuu"],
  ring:  [".kk.", "kwuk", "kuuk", ".kk."],
  x:     ["k.k", ".k.", "k.k"],
  spark: ["kkkk", "wwuu", "uwuu", "uuuu"],
  heart: ["rr.rr", "rrrrr", ".rrr.", "..r.."],
};
const MOUTHS = {
  grin:  ["kkkkk", "krrrk", ".kkk."],
  smile: ["k...k", ".kkk."],
  w:     ["k.k.k", ".kkk."],
  pout:  [".k.", "k.k"],
  flat:  ["kkk"],
  frown: [".kkk.", "k...k"],
  wail:  ["kkkkk", "k...k", ".kkk."],
  grit:  ["kkkkk", "k.k.k"],
  o:     [".k.", "k.k", ".k."],
  wavy:  ["k.k.k", ".k.k."],
  smirk: ["...kk", "kkk.."],
};
const EXTRAS = {
  spark: [".#.", "###", ".#."],
  heart: ["#.#", "###", ".#."],
  tear:  ["c", "c"],
  drop:  [".c.", "ccc", ".c."],
  vein:  ["##.##", "#...#", ".....", "#...#", "##.##"],
  bang:  ["#", "#", "#", ".", "#"],
  q:     ["##.", "..#", ".#.", "...", ".#."],
  z:     ["####", "..#.", ".#..", "####"],
};

const F = (e, m, p, x = [], b = 1) => ({ e, m, p, x, b });
const EMOTIONS = [
  { name:"Happy",       jp:"嬉しい",   text:"＼(＾▽＾)／", frames:[
    F("up","grin",0,[["spark",1,4,"o"]]),                   F("up","grin",1,[["spark",31,5,"o"]]) ]},
  { name:"Overjoyed",   jp:"大喜び",   text:"＼(≧▽≦)／", frames:[
    F("gt","grin",-1,[["spark",1,3,"o"],["spark",31,6,"o"]]), F("gt","grin",0,[["spark",2,7,"o"],["spark",30,2,"o"]]) ]},
  { name:"Smiling",     jp:"にこにこ", text:"(◕‿◕)✿",     frames:[
    F("dot","smile",0),                                     F("line","smile",1) ]},
  { name:"Pouting",     jp:"ぷくー",   text:"(｀ε´)",      frames:[
    F("angry","pout",0),                                    F("angry","flat",1,[["vein",30,1,"v"]]) ]},
  { name:"Sad",         jp:"悲しい",   text:"(´；ω；`)",  frames:[
    F("dot","frown",1),                                     F("dot","frown",2,[["tear",13,14]]) ]},
  { name:"Crying",      jp:"泣いてる", text:"(╥_╥)",       frames:[
    F("cry","wail",0,[["tear",13,14],["tear",21,14]]),      F("cry","frown",1,[["tear",13,16],["tear",21,16]]) ]},
  { name:"Angry",       jp:"怒り",     text:"(╬ಠ益ಠ)",     frames:[
    F("angry","grit",0,[["vein",30,1,"v"]],0),              F("angry","grit",1,[["vein",30,3,"v"]],0) ]},
  { name:"Surprised",   jp:"びっくり", text:"Σ(°□°)",      frames:[
    F("ring","o",0,[["bang",31,2]]),                        F("ring","o",1,[["bang",30,2],["bang",32,2]]) ]},
  { name:"Embarrassed", jp:"照れる",   text:"(〃▽〃)",     frames:[
    F("line","w",0,[],2),                                   F("down","w",1,[["drop",2,4]],2) ]},
  { name:"Sleepy",      jp:"眠い",     text:"(－ω－) zzZ", frames:[
    F("line","flat",0,[["z",30,5,"u"]]),                    F("line","o",1,[["z",30,5,"u"],["z",31,0,"u"]]) ]},
  { name:"In Love",     jp:"大好き",   text:"(♡´▽`♡)",    frames:[
    F("heart","smile",0,[["heart",1,4,"r"]]),               F("heart","smile",1,[["heart",31,3,"r"]]) ]},
  { name:"Winking",     jp:"ウインク", text:"(＾_－)☆",    frames:[
    F("dot","w",0),                                         F(["dot","line"],"w",1,[["spark",31,5,"o"]]) ]},
  { name:"Confused",    jp:"困惑",     text:"(・_・?)",     frames:[
    F("dot","flat",0,[["q",30,2,"u"]]),                     F("dot","wavy",1,[["q",30,3,"u"]]) ]},
  { name:"Cool",        jp:"クール",   text:"(⌐■_■)",      frames:[
    F("half","smirk",0,[],0),                               F("dot","smirk",1,[["spark",31,5,"o"]],0) ]},
  { name:"Dizzy",       jp:"目が回る", text:"(@_@)",       frames:[
    F("x","wavy",0,[["spark",1,4,"o"]]),                    F("spark","wavy",1,[["spark",31,4,"o"]]) ]},
  { name:"Hug",         jp:"ぎゅー",   text:"(づ｡◕‿‿◕｡)づ", frames:[
    F("up","smile",-1,[["heart",30,3,"r"]]),                F("up","smile",0,[["heart",30,2,"r"]]) ]},
];

function build(fr) {
  const g = Array.from({ length: H }, () => Array(W).fill("."));
  const put = (x, y, ch) => { if (x >= 0 && x < W && y >= 0 && y < H) g[y][x] = ch; };
  const stamp = (rows, x, y, c = "k") => rows.forEach((row, j) => [...row].forEach((ch, i) => {
    if (ch !== ".") put(x + i, y + j, ch === "#" ? c : ch);
  }));
  const d = fr.p;                                // the whole head bobs

  // star clips
  stamp([".y.", "yyy", ".y."], 8, 9 + d, "y");
  stamp([".y.", "yyy", ".y."], 25, 6 + d, "y");

  if (fr.b) {                                    // blush
    const big = fr.b === 2;
    stamp(big ? ["bbb", "bbb", "bbb"] : ["bb", "bb"], big ? 11 : 11, (big ? 14 : 15) + d);
    stamp(big ? ["bbb", "bbb", "bbb"] : ["bb", "bb"], big ? 21 : 22, (big ? 14 : 15) + d);
  }

  const lName = Array.isArray(fr.e) ? fr.e[0] : fr.e;
  const rName = Array.isArray(fr.e) ? fr.e[1] : fr.e;
  const l = EYES[lName];
  const r = Array.isArray(fr.e) ? EYES[rName] : mirror(EYES[lName]);
  stamp(l, 12 + Math.floor((4 - l[0].length) / 2), 14 - l.length + d);
  stamp(r, 19 + Math.ceil((4 - r[0].length) / 2), 14 - r.length + d);

  const m = MOUTHS[fr.m];
  stamp(m, 17 - Math.floor(m[0].length / 2), 16 + d);

  for (const [name, x, y, c] of fr.x) stamp(EXTRAS[name], x, y, c || "k");
  return g;
}

function paint(ctx, grid) {
  ctx.clearRect(0, 0, W * S, H * S);
  grid.forEach((row, y) => row.forEach((ch, x) => {
    if (ch === ".") return;
    ctx.fillStyle = COLORS[ch];
    ctx.fillRect(x * S, y * S, S, S);
  }));
}

const canvas = document.getElementById("kaomoji");
if (!canvas) return;
const ctx = canvas.getContext("2d");
canvas.width = W * S;
canvas.height = H * S;

const byName = Object.fromEntries(
  EMOTIONS.map(e => [e.name, { ...e, grids: e.frames.map(build) }])
);
let current = byName.Smiling;
let frame = 0;
const FRAME_MS = 450;

const draw = () => paint(ctx, current.grids[frame]);
draw();

// Two-frame loop. It only runs while the face is enabled and the tab is
// visible, and not at all for people who ask the OS for less motion (they get
// the first frame of each emotion).
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
let timer = null;

function tick() {
  frame = 1 - frame;
  draw();
}

function run() {
  stop();
  if (!reduced && enabled && !document.hidden) timer = setInterval(tick, FRAME_MS);
}

function stop() {
  clearInterval(timer);
  timer = null;
}

let enabled = true;
document.addEventListener("visibilitychange", run);
run();

window.Kaomoji = {
  names: EMOTIONS.map(e => e.name),
  set(name) {
    if (!byName[name]) return;
    current = byName[name];
    frame = 0;
    draw();
  },
  enable(on) {
    enabled = on;
    run();
  },
};
})();
