const API = "/api";
const TIMEOUT_MS = 8000;
// The first load may wake a stopped machine, which takes a while to load its
// vectors and answers 503 meanwhile, so keep retrying for up to a minute.
const START_TIMEOUT_MS = 20000;
const START_DEADLINE_MS = 60000;
const START_RETRY_MS = 2000;
const STORAGE_PREFIX = "guesses:";
const APP_NAME = document.querySelector('meta[name="application-name"]').content;

// Keep in sync with app/text.py.
const WORD = /^[a-záčďéěíňóřšťúůýž]{1,40}$/u;

// How the reference ranks from the API are named in the scale sentence.
const RANK_NAMES = { 1: "nejbližší slovo ve slovníku má", 10: "desáté", 1000: "tisící" };

const percent = new Intl.NumberFormat("cs-CZ", {
  style: "percent",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});
const integer = new Intl.NumberFormat("cs-CZ");
const list = new Intl.ListFormat("cs-CZ", { type: "conjunction" });
const attempts = new Intl.PluralRules("cs-CZ");
const ATTEMPT_FORMS = { one: "pokus", few: "pokusy", many: "pokusu", other: "pokusů" };
const longDate = new Intl.DateTimeFormat("cs-CZ", {
  weekday: "long",
  day: "numeric",
  month: "long",
  year: "numeric",
});

const el = {
  title: document.getElementById("title"),
  date: document.getElementById("date"),
  vocabulary: document.getElementById("vocabulary"),
  scale: document.getElementById("scale"),
  form: document.getElementById("guess-form"),
  input: document.getElementById("guess"),
  button: document.querySelector("#guess-form button"),
  status: document.getElementById("status"),
  table: document.getElementById("guesses"),
  rows: document.querySelector("#guesses tbody"),
};

const state = {
  puzzle: null, // { id, number, date, preview, vocabulary_size, neighbours: [{ rank, similarity }] }
  guesses: [], // { n, word, lemma, similarity, solved }
  busy: false,
};

/* ---------- API ---------- */

class ApiError extends Error {
  constructor(status) {
    super(`API error ${status}`);
    this.status = status; // 0 = network failure or timeout
  }
}

async function api(path, { body, timeout = TIMEOUT_MS } = {}) {
  let response;
  try {
    response = await fetch(API + path, {
      method: body ? "POST" : "GET",
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body && JSON.stringify(body),
      signal: AbortSignal.timeout(timeout),
    });
  } catch {
    throw new ApiError(0);
  }
  if (!response.ok) throw new ApiError(response.status);
  return response.json();
}

/* ---------- Persistence: progress survives a reload, one puzzle at a time ---------- */

// Keyed by the puzzle's id, not its number: if the schedule is ever reset,
// puzzle 1 may be a different word, and old guesses must not carry over.
const storageKey = (id) => STORAGE_PREFIX + id;

function loadGuesses(id) {
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey(id)));
    return Array.isArray(saved) ? saved : [];
  } catch {
    return [];
  }
}

function saveGuesses(id, guesses) {
  try {
    localStorage.setItem(storageKey(id), JSON.stringify(guesses));
    for (const key of Object.keys(localStorage)) {
      if (key.startsWith(STORAGE_PREFIX) && key !== storageKey(id)) {
        localStorage.removeItem(key);
      }
    }
  } catch {
    // Private mode or storage full: the game still works, it just won't survive a reload.
  }
}

/* ---------- Game logic ---------- */

const isSolved = () => state.guesses.some((g) => g.solved);

function normalize(raw) {
  return raw.normalize("NFC").trim().toLocaleLowerCase("cs");
}

async function submitGuess(raw) {
  const word = normalize(raw);
  if (!word || state.busy) return;

  if (!WORD.test(word)) {
    say("Napiš jedno slovo jen z písmen, bez mezer a číslic.");
    return;
  }
  if (state.guesses.some((g) => g.word === word)) {
    say("Slovo ", quoted(word), " už v seznamu máš.");
    el.input.select();
    return;
  }

  setBusy(true);
  try {
    const result = await api(`/puzzles/${state.puzzle.number}/guesses`, { body: { word } });
    if (!result.known) {
      say("Slovo ", quoted(word), " ve slovníku není. Zkus jiný tvar nebo jiné slovo.");
      el.input.select();
      return;
    }
    const guess = {
      n: state.guesses.length + 1,
      word: result.word,
      lemma: result.lemma,
      similarity: result.similarity,
      solved: result.solved,
    };
    state.guesses.push(guess);
    saveGuesses(state.puzzle.id, state.guesses);
    el.input.value = "";
    render(guess);
  } catch (error) {
    say(errorMessage(error));
  } finally {
    setBusy(false);
  }
}

function errorMessage(error) {
  switch (error.status) {
    case 0:
      return "Server neodpovídá. Zkontroluj připojení a zkus to znovu.";
    case 404:
      return "Tohle kolo už skončilo. Obnov stránku.";
    case 422:
      return "Napiš jedno slovo jen z písmen, bez mezer a číslic.";
    case 429:
      return "Moc pokusů za minutu. Chvíli počkej a zkus to znovu.";
    case 503:
      return "Server se ještě probouzí. Zkus to za pár vteřin.";
    default:
      return "Na serveru se něco pokazilo. Zkus to za chvíli znovu.";
  }
}

/* ---------- Rendering ---------- */

function node(tag, props = {}, ...children) {
  const element = Object.assign(document.createElement(tag), props);
  element.append(...children);
  return element;
}

const quoted = (word) => node("span", { className: "word" }, `„${word}“`);

function say(...parts) {
  el.status.className = "status";
  el.status.replaceChildren(...parts);
}

function setBusy(busy) {
  state.busy = busy;
  el.button.disabled = busy || isSolved();
}

function renderHeader() {
  const { number, date, preview, vocabulary_size, neighbours } = state.puzzle;
  const [y, m, d] = date.split("-").map(Number);
  const day = longDate.format(new Date(y, m - 1, d));

  document.title = `${APP_NAME} #${number}`;
  el.title.textContent = `${APP_NAME} #${number}`;
  el.date.textContent = preview ? `Zkušební provoz. Hra začne ${day}.` : day;
  el.vocabulary.textContent = integer.format(vocabulary_size);

  if (neighbours.length) {
    const parts = neighbours.map(
      ({ rank, similarity }) => `${RANK_NAMES[rank] ?? `${rank}.`} ${percent.format(similarity)}`,
    );
    el.scale.textContent = `Pro srovnání: ${list.format(parts)}.`;
    el.scale.hidden = false;
  }
}

function renderRows(latest) {
  const sorted = [...state.guesses].sort((a, b) => b.similarity - a.similarity);
  const lastN = state.guesses.length;

  el.rows.replaceChildren(
    ...sorted.map((guess) => {
      const word = node("td", { className: "word", title: guess.word }, guess.word);
      if (guess.lemma) word.append(node("span", { className: "lemma" }, guess.lemma));

      const row = node(
        "tr",
        {},
        node("td", { className: "n" }, String(guess.n)),
        word,
        node("td", { className: "pct" }, percent.format(guess.similarity)),
      );
      row.classList.toggle("is-latest", guess.n === lastN);
      row.classList.toggle("is-solved", guess.solved);
      row.classList.toggle("is-new", guess === latest);
      return row;
    }),
  );
  el.table.hidden = state.guesses.length === 0;
}

function renderStatus(latest) {
  const solved = state.guesses.find((g) => g.solved);
  if (solved) {
    say(`Uhodnuto na ${solved.n}. pokus. Další slovo bude zítra.`);
    el.status.classList.add("is-solved");
  } else if (latest) {
    say(quoted(latest.word), `: ${percent.format(latest.similarity)}`);
  } else if (state.guesses.length === 0) {
    say("Napiš první slovo, které tě napadne.");
  } else {
    const count = state.guesses.length;
    say(`Máš za sebou ${count} ${ATTEMPT_FORMS[attempts.select(count)]}. Pokračuj.`);
  }
}

function render(latest = null) {
  renderRows(latest);
  renderStatus(latest);

  const solved = isSolved();
  el.input.disabled = solved;
  el.button.disabled = solved || state.busy;
  if (!solved) el.input.focus();
}

/* ---------- Start ---------- */

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function loadPuzzle() {
  const deadline = Date.now() + START_DEADLINE_MS;
  for (let attempt = 0; ; attempt++) {
    try {
      return await api("/puzzles/today", { timeout: START_TIMEOUT_MS });
    } catch (error) {
      const clientError = error.status >= 400 && error.status < 500;
      if (clientError || Date.now() > deadline) throw error;
      if (attempt === 0) say("Server se probouzí, chvilku strpení…");
      await sleep(START_RETRY_MS);
    }
  }
}

async function start() {
  try {
    state.puzzle = await loadPuzzle();
  } catch (error) {
    console.error("Loading today's puzzle failed:", error);
    say("Dnešní slovo se nepodařilo načíst. Zkontroluj připojení a obnov stránku.");
    return;
  }
  state.guesses = loadGuesses(state.puzzle.id);
  renderHeader();
  render();
}

el.form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitGuess(el.input.value);
});

start();
