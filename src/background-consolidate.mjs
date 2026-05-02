#!/usr/bin/env node
/**
 * Detached shutdown consolidator for pi-memory.
 *
 * Runs independently from the Pi TUI so quitting is never blocked by the
 * memory LLM call. Input is a JSON payload path containing { prompt, dbPath,
 * cwd, source }. This script invokes `pi -p ... --print --no-extensions`,
 * parses the JSON extraction, and applies it directly to the SQLite store.
 */
import { DatabaseSync } from "node:sqlite";
import { appendFileSync, mkdirSync, readFileSync, unlinkSync } from "node:fs";
import { dirname, join } from "node:path";
import { homedir } from "node:os";
import { spawn } from "node:child_process";

const logPath = join(homedir(), ".pi", "memory", "consolidate.log");
function log(message) {
  try {
    mkdirSync(dirname(logPath), { recursive: true });
    appendFileSync(logPath, `${new Date().toISOString()} ${message}\n`);
  } catch {}
}

const payloadPath = process.argv[2];
if (!payloadPath) {
  log("error missing payload path");
  console.error("pi-memory background: missing payload path");
  process.exit(1);
}

let payload;
try {
  payload = JSON.parse(readFileSync(payloadPath, "utf8"));
} catch (err) {
  log(`error failed to read payload: ${err.message}`);
  console.error(`pi-memory background: failed to read payload: ${err.message}`);
  process.exit(1);
}

const prompt = String(payload.prompt || "");
const dbPath = String(payload.dbPath || "");
const cwd = String(payload.cwd || process.cwd());
const source = String(payload.source || "consolidation");

if (!prompt || !dbPath) {
  log("error payload missing prompt or dbPath");
  console.error("pi-memory background: payload missing prompt or dbPath");
  process.exit(1);
}

log(`start pid=${process.pid} cwd=${cwd} db=${dbPath} payload=${payloadPath}`);

try {
  const result = await runPi(prompt, cwd);
  if (result.code !== 0 || !result.stdout) {
    log(`error pi exited code=${result.code} stdout=${result.stdout.length} stderr=${JSON.stringify(result.stderr.slice(0, 1000))}`);
    console.error(`pi-memory background: pi exited ${result.code}`);
    if (result.stderr) console.error(result.stderr.slice(0, 1000));
    process.exit(result.code || 1);
  }

  const extracted = parseConsolidationResponse(result.stdout);
  log(`pi complete stdout=${result.stdout.length} stderr=${result.stderr.length} extracted=${extracted.semantic.length} facts/${extracted.lessons.length} lessons`);
  const applied = applyExtracted(dbPath, extracted, source);
  log(`success applied=${applied.semantic} facts/${applied.lessons} lessons`);
  if (applied.semantic + applied.lessons > 0) {
    console.error(`pi-memory: consolidated ${applied.semantic} facts, ${applied.lessons} lessons`);
  }
} catch (err) {
  log(`error ${err.stack || err.message}`);
  console.error(`pi-memory background: ${err.message}`);
  process.exit(1);
} finally {
  try {
    unlinkSync(payloadPath);
    log(`cleanup payload=${payloadPath}`);
  } catch (err) {
    log(`cleanup failed payload=${payloadPath}: ${err.message}`);
  }
}

function runPi(prompt, cwd) {
  return new Promise((resolve, reject) => {
    const child = spawn("pi", [
      "-p", prompt,
      "--print",
      "--no-extensions",
      "--model", "claude-sonnet-4-20250514",
    ], {
      cwd,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    const timeout = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error("pi consolidation timed out"));
    }, 45_000);

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", chunk => { stdout += chunk; });
    child.stderr.on("data", chunk => { stderr += chunk; });
    child.on("error", err => {
      clearTimeout(timeout);
      reject(err);
    });
    child.on("close", code => {
      clearTimeout(timeout);
      resolve({ code: code ?? 1, stdout, stderr });
    });
  });
}

function parseConsolidationResponse(text) {
  const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/) || text.match(/(\{[\s\S]*\})/);
  if (!jsonMatch) return { semantic: [], lessons: [] };

  try {
    const parsed = JSON.parse(jsonMatch[1].trim());
    const result = { semantic: [], lessons: [] };

    if (Array.isArray(parsed.semantic)) {
      for (const s of parsed.semantic) {
        if (typeof s.key === "string" && typeof s.value === "string" && typeof s.confidence === "number") {
          if (s.confidence >= 0.8 && isValidKey(s.key) && s.value.length <= 500) {
            result.semantic.push({ key: s.key, value: s.value, confidence: s.confidence });
          }
        }
      }
    }

    if (Array.isArray(parsed.lessons)) {
      for (const l of parsed.lessons) {
        if (typeof l.rule === "string" && l.rule.trim().length > 0) {
          result.lessons.push({
            rule: l.rule.trim(),
            category: typeof l.category === "string" ? l.category : "general",
            negative: !!l.negative,
          });
        }
      }
    }

    return result;
  } catch {
    return { semantic: [], lessons: [] };
  }
}

function applyExtracted(dbPath, extracted, source) {
  mkdirSync(dirname(dbPath), { recursive: true });
  const db = new DatabaseSync(dbPath);
  db.exec("PRAGMA journal_mode = WAL");
  db.exec("PRAGMA busy_timeout = 5000");
  db.exec("PRAGMA foreign_keys = ON");
  migrate(db);

  let semantic = 0;
  let lessons = 0;

  try {
    for (const s of extracted.semantic) {
      if (isDerivableOrEphemeral(s.key, s.value)) continue;
      setSemantic(db, s.key, s.value, s.confidence);
      semantic++;
    }

    for (const l of extracted.lessons) {
      if (isDerivableLesson(l.rule)) continue;
      if (addLesson(db, l.rule, l.category, source, l.negative)) lessons++;
    }
  } finally {
    db.close();
  }

  return { semantic, lessons };
}

function migrate(db) {
  db.exec(`
    CREATE TABLE IF NOT EXISTS semantic (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL,
      confidence REAL NOT NULL DEFAULT 0.8,
      source TEXT NOT NULL DEFAULT 'consolidation',
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS lessons (
      id TEXT PRIMARY KEY,
      rule TEXT NOT NULL,
      category TEXT NOT NULL DEFAULT 'general',
      source TEXT NOT NULL DEFAULT 'consolidation',
      negative INTEGER NOT NULL DEFAULT 0,
      is_deleted INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS events (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      event_type TEXT NOT NULL,
      memory_type TEXT NOT NULL,
      memory_key TEXT NOT NULL,
      details TEXT NOT NULL DEFAULT '',
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
  `);

  try { db.exec(`ALTER TABLE semantic ADD COLUMN last_accessed TEXT`); } catch {}

  try {
    db.exec(`
      CREATE VIRTUAL TABLE IF NOT EXISTS semantic_fts USING fts5(key, value, content='semantic', content_rowid='rowid');
      CREATE TRIGGER IF NOT EXISTS semantic_ai AFTER INSERT ON semantic BEGIN
        INSERT INTO semantic_fts(rowid, key, value) VALUES (new.rowid, new.key, new.value);
      END;
      CREATE TRIGGER IF NOT EXISTS semantic_ad AFTER DELETE ON semantic BEGIN
        INSERT INTO semantic_fts(semantic_fts, rowid, key, value) VALUES('delete', old.rowid, old.key, old.value);
      END;
      CREATE TRIGGER IF NOT EXISTS semantic_au AFTER UPDATE ON semantic BEGIN
        INSERT INTO semantic_fts(semantic_fts, rowid, key, value) VALUES('delete', old.rowid, old.key, old.value);
        INSERT INTO semantic_fts(rowid, key, value) VALUES (new.rowid, new.key, new.value);
      END;
    `);
    db.exec(`
      CREATE VIRTUAL TABLE IF NOT EXISTS lessons_fts USING fts5(rule, category, content='lessons', content_rowid='rowid');
      CREATE TRIGGER IF NOT EXISTS lessons_fts_ai AFTER INSERT ON lessons BEGIN
        INSERT INTO lessons_fts(rowid, rule, category) VALUES (new.rowid, new.rule, new.category);
      END;
      CREATE TRIGGER IF NOT EXISTS lessons_fts_ad AFTER DELETE ON lessons BEGIN
        INSERT INTO lessons_fts(lessons_fts, rowid, rule, category) VALUES('delete', old.rowid, old.rule, old.category);
      END;
      CREATE TRIGGER IF NOT EXISTS lessons_fts_au AFTER UPDATE ON lessons BEGIN
        INSERT INTO lessons_fts(lessons_fts, rowid, rule, category) VALUES('delete', old.rowid, old.rule, old.category);
        INSERT INTO lessons_fts(rowid, rule, category) VALUES (new.rowid, new.rule, new.category);
      END;
    `);
  } catch {}
}

function setSemantic(db, key, value, confidence) {
  const normalized = key.toLowerCase();
  transaction(db, () => {
    const existing = db.prepare("SELECT * FROM semantic WHERE key = ?").get(normalized);
    if (existing && existing.confidence > confidence) return;

    db.prepare(`
      INSERT INTO semantic (key, value, confidence, source, updated_at)
      VALUES (?, ?, ?, 'consolidation', datetime('now'))
      ON CONFLICT(key) DO UPDATE SET
        value = excluded.value,
        confidence = excluded.confidence,
        source = excluded.source,
        updated_at = datetime('now')
    `).run(normalized, value, confidence);

    logEvent(db, existing ? "update" : "create", "semantic", normalized);
  });
}

function addLesson(db, rule, category = "general", source = "consolidation", negative = false) {
  const trimmed = rule.trim();
  if (!trimmed) return false;
  const normalizedCategory = category.trim().toLowerCase() || "general";

  return transaction(db, () => {
    const existing = db.prepare(
      "SELECT id FROM lessons WHERE LOWER(TRIM(rule)) = LOWER(?) AND is_deleted = 0"
    ).get(trimmed.toLowerCase());
    if (existing) return false;

    const allRules = db.prepare("SELECT id, rule FROM lessons WHERE is_deleted = 0").all();
    for (const r of allRules) {
      if (jaccard(trimmed, r.rule) >= 0.7) return false;
    }

    const id = crypto.randomUUID();
    db.prepare("INSERT INTO lessons (id, rule, category, source, negative) VALUES (?, ?, ?, ?, ?)")
      .run(id, trimmed, normalizedCategory, source, negative ? 1 : 0);

    logEvent(db, "create", "lesson", id, trimmed.slice(0, 100));
    return true;
  });
}

function transaction(db, fn) {
  db.exec("BEGIN IMMEDIATE");
  try {
    const result = fn();
    db.exec("COMMIT");
    return result;
  } catch (err) {
    db.exec("ROLLBACK");
    throw err;
  }
}

function logEvent(db, eventType, memoryType, key, details = "") {
  db.prepare("INSERT INTO events (event_type, memory_type, memory_key, details) VALUES (?, ?, ?, ?)")
    .run(eventType, memoryType, key, details);
}

const VALID_KEY_RE = /^[a-z][a-z0-9._-]*$/;
function isValidKey(key) {
  return VALID_KEY_RE.test(key) && key.length <= 100 && key.length >= 2;
}

function isDerivableOrEphemeral(key, value) {
  const kl = key.toLowerCase();
  const vl = value.toLowerCase();
  if (kl.includes("filepath") || kl.includes("file_path") || kl.includes("directory")) return true;
  if (/^project\.\w+\.(path|dir|location|structure|layout|architecture)$/.test(kl)) return true;
  if (kl.includes("commit") || kl.includes("git.history") || kl.includes("git.recent")) return true;
  if (vl.startsWith("today ") || vl.startsWith("we worked on") || vl.startsWith("this session")) return true;
  if (vl.includes("```") && vl.length > 300) return true;
  if (kl.includes("current_task") || kl.includes("in_progress") || kl.includes("investigating")) return true;
  return false;
}

function isDerivableLesson(rule) {
  const rl = rule.toLowerCase();
  if (/file .+ is (at|in|located) /.test(rl)) return true;
  if (/^the (project|codebase|repo) (uses|is written in) /.test(rl)) return true;
  if (/^(we|i|the agent) (fixed|deployed|updated|changed|modified|ran|executed) /.test(rl)) return true;
  if (/^when (encountering|bash fails|edit fails|.*error)/.test(rl) && /\b(run:|fix with:)/.test(rl)) return true;
  if (/^run: /.test(rl)) return true;
  if (rl.includes("command exited with code") && rl.length < 100) return true;
  return false;
}

function jaccard(a, b) {
  const setA = new Set(a.toLowerCase().split(/\s+/).filter(Boolean));
  const setB = new Set(b.toLowerCase().split(/\s+/).filter(Boolean));
  if (setA.size === 0 && setB.size === 0) return 1;
  const intersection = new Set([...setA].filter(x => setB.has(x)));
  const union = new Set([...setA, ...setB]);
  return intersection.size / union.size;
}
