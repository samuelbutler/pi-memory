/**
 * pi-memory — Persistent memory extension for pi.
 *
 * Learns corrections, preferences, and patterns from sessions.
 * Injects relevant memory into future conversations.
 *
 * Lifecycle:
 * - session_start: open store, inject memory into status
 * - before_agent_start: inject memory context into system prompt
 * - agent_end: queue messages for consolidation
 * - session_shutdown: consolidate and close store
 *
 * Tools:
 * - memory_search: search semantic memory
 * - memory_remember: manually add a memory
 * - memory_forget: delete a memory
 * - memory_lessons: list learned corrections
 * - memory_stats: show memory statistics
 */

import { spawn } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type {
	AgentToolResult,
	ExtensionAPI,
} from "@mariozechner/pi-coding-agent";
import { Type } from "@sinclair/typebox";
import { buildContextBlock, type InjectorConfig } from "./injector.js";
import { MemoryStore } from "./store.js";

type ToolResult = AgentToolResult<unknown>;
function ok(text: string): ToolResult {
	return { content: [{ type: "text", text }], details: {} };
}

/**
 * Strip one layer of surrounding quotes from a string value.
 * Some local models (e.g. Qwen on certain runners) double-JSON-encode tool
 * arguments, emitting `"\"fact\""` instead of `"fact"`. We defensively
 * unwrap so these calls don't fail schema validation / equality checks.
 */
function stripQuotes<T>(v: T): T {
	if (typeof v !== "string") return v;
	const s = v.trim();
	if (s.length >= 2) {
		const first = s[0];
		const last = s[s.length - 1];
		if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
			try {
				// Prefer JSON.parse for double-quoted (handles escapes)
				if (first === '"') return JSON.parse(s) as unknown as T;
			} catch {
				/* fall through */
			}
			return s.slice(1, -1) as unknown as T;
		}
	}
	return v;
}

import {
	applyExtracted,
	buildConsolidationPrompt,
	type ConsolidationInput,
	parseConsolidationResponse,
} from "./consolidator.js";

const DEFAULT_MEMORY_DIR = join(homedir(), ".pi", "memory");
const DEFAULT_DB_PATH = join(DEFAULT_MEMORY_DIR, "memory.db");
const GLOBAL_SETTINGS_PATH = join(homedir(), ".pi", "agent", "settings.json");

/**
 * Resolve the memory DB path for a given working directory.
 * Priority:
 *   1. "pi-memory".localPath from {cwd}/.pi/settings.json → join(localPath, "memory.db")
 *   2. Global default: ~/.pi/memory/memory.db  (preserves existing behavior)
 */
function resolveDbPath(cwd: string): string {
	// Try reading the local project settings for an explicit localPath override
	try {
		const localSettingsPath = join(cwd, ".pi", "settings.json");
		const raw = readFileSync(localSettingsPath, "utf-8");
		const settings = JSON.parse(raw);
		const piMemory = settings?.["pi-memory"];
		if (
			piMemory &&
			typeof piMemory === "object" &&
			typeof piMemory.localPath === "string" &&
			piMemory.localPath
		) {
			return join(piMemory.localPath, "memory.db");
		}
	} catch {
		// No local settings or parse error — use global default
	}
	// Default: global shared memory (preserves existing behavior)
	return DEFAULT_DB_PATH;
}

/**
 * Read pi-memory config from settings.json.
 * Looks for a "memory" key with extension-specific settings.
 *
 * Example settings.json:
 * {
 *   "memory": {
 *     "lessonInjection": "selective"
 *   }
 * }
 */
function readSettingsConfig(cwd?: string): InjectorConfig {
	const config: InjectorConfig = {};

	// Read global settings
	try {
		const raw = readFileSync(GLOBAL_SETTINGS_PATH, "utf-8");
		const settings = JSON.parse(raw);
		const memorySettings = settings?.memory;
		if (memorySettings && typeof memorySettings === "object") {
			if (
				memorySettings.lessonInjection === "all" ||
				memorySettings.lessonInjection === "selective"
			) {
				config.lessonInjection = memorySettings.lessonInjection;
			}
		}
	} catch {
		// no global settings
	}

	// Override with local project settings if available
	if (cwd) {
		try {
			const raw = readFileSync(join(cwd, ".pi", "settings.json"), "utf-8");
			const settings = JSON.parse(raw);
			const memorySettings = settings?.memory ?? settings?.["pi-memory"];
			if (memorySettings && typeof memorySettings === "object") {
				if (
					memorySettings.lessonInjection === "all" ||
					memorySettings.lessonInjection === "selective"
				) {
					config.lessonInjection = memorySettings.lessonInjection;
				}
			}
		} catch {
			// no local settings
		}
	}

	return config;
}

export default function (pi: ExtensionAPI) {
	let store: MemoryStore | null = null;
	let pendingUserMessages: string[] = [];
	let pendingAssistantMessages: string[] = [];
	let sessionCwd: string = "";
	let sessionId: string | undefined;
	let statusClearTimer: ReturnType<typeof setTimeout> | undefined;
	let resolvedDbPath: string = DEFAULT_DB_PATH;
	let injectorConfig: InjectorConfig = readSettingsConfig();

	function getContextCwd(ctx: unknown): string | undefined {
		return typeof (ctx as { cwd?: unknown })?.cwd === "string"
			? (ctx as { cwd: string }).cwd
			: undefined;
	}

	function initializeStore(
		cwd?: string,
		force: boolean = false,
	): { initialized: boolean; dbPath: string; message?: string } {
		if (store && !force) {
			return {
				initialized: true,
				dbPath: resolvedDbPath,
				message: "Memory store already initialized",
			};
		}

		if (store && force) {
			store.close();
			store = null;
		}

		sessionCwd = cwd || sessionCwd || process.cwd();
		resolvedDbPath = resolveDbPath(sessionCwd);
		injectorConfig = readSettingsConfig(sessionCwd);
		store = new MemoryStore(resolvedDbPath);
		return { initialized: true, dbPath: resolvedDbPath };
	}

	// ─── Lifecycle ───────────────────────────────────────────────────

	pi.on("session_start", async (event, ctx) => {
		if (event.reason === "reload") {
			store = null;
			pendingUserMessages = [];
			pendingAssistantMessages = [];
			return;
		}

		try {
			sessionCwd = ctx.cwd;
			sessionId = (ctx as any).sessionId ?? (ctx as any).session?.id;

			initializeStore(sessionCwd, true);
			const memoryStore = store!;

			// Seed pending messages from existing session history so that
			// /memory-consolidate works even when resuming a session (the
			// historical messages never fire agent_end).  See #5.
			pendingUserMessages = [];
			pendingAssistantMessages = [];
			try {
				const branch = ctx.sessionManager.getBranch();
				for (const entry of branch) {
					if (entry.type !== "message") continue;
					const msg = (entry as any).message;
					if (!msg) continue;
					if (msg.role === "user") {
						const text = extractText(msg.content);
						if (text) pendingUserMessages.push(text);
					} else if (msg.role === "assistant") {
						const text = extractText(msg.content);
						if (text) pendingAssistantMessages.push(text);
					}
				}
			} catch {
				// Session may not have entries yet (brand-new session)
			}

			const stats = memoryStore.stats();
			if (stats.semantic + stats.lessons > 0) {
				ctx.ui.setStatus(
					"pi-memory",
					`Memory: ${stats.semantic} facts, ${stats.lessons} lessons`,
				);
				statusClearTimer = setTimeout(() => {
					try {
						ctx.ui.setStatus("pi-memory", "");
					} catch {
						// The context may be stale after /reload; ignore status cleanup.
					}
				}, 5000);
			}
		} catch (err: any) {
			ctx.ui.notify(
				`pi-memory: failed to open store: ${err.message}`,
				"warning",
			);
		}
	});

	pi.on("before_agent_start", async (event, ctx) => {
		if (!store) return;

		const { text } = buildContextBlock(
			store,
			ctx.cwd,
			event.prompt,
			injectorConfig,
		);
		if (!text) return;

		return {
			systemPrompt: `${event.systemPrompt}\n\n${text}`,
		};
	});

	pi.on("agent_end", async (event, _ctx) => {
		// Collect messages for consolidation at shutdown
		for (const msg of event.messages) {
			if (msg.role === "user" && "content" in msg) {
				const text = extractText(msg.content);
				if (text) {
					pendingUserMessages.push(text);
					if (pendingUserMessages.length > 60) pendingUserMessages.shift();
				}
			} else if (msg.role === "assistant" && "content" in msg) {
				const text = extractText(msg.content);
				if (text) {
					pendingAssistantMessages.push(text);
					if (pendingAssistantMessages.length > 60)
						pendingAssistantMessages.shift();
				}
			}
		}
	});

	// Consolidate memory when switching sessions (/new, /resume)
	pi.on("session_before_switch", async (_event, ctx) => {
		if (!store) return;

		if (pendingUserMessages.length >= 3) {
			ctx.ui.setStatus("pi-memory", "🧠 Consolidating memory...");
			try {
				await consolidateSession();
			} catch {
				// Best-effort
			}
			ctx.ui.setStatus("pi-memory", "");
		}

		// Reset for the next session
		pendingUserMessages = [];
		pendingAssistantMessages = [];
	});

	pi.on("session_shutdown", async (event) => {
		if (!store) return;

		if ((event as any).reason === "reload") {
			if (statusClearTimer) {
				clearTimeout(statusClearTimer);
				statusClearTimer = undefined;
			}
			store.close();
			store = null;
			pendingUserMessages = [];
			pendingAssistantMessages = [];
			return;
		}

		// Start consolidation out-of-process so quitting Pi is not blocked by the
		// LLM call. The detached child owns its own `pi -p` run and DB connection.
		if (pendingUserMessages.length >= 3) {
			try {
				startBackgroundConsolidation();
			} catch {
				// Best-effort — don't crash or block shutdown
			}
		}

		store.close();
		store = null;
	});

	// ─── Consolidation ──────────────────────────────────────────────

	function buildCurrentConsolidationPrompt(): string | undefined {
		if (!store) return undefined;

		const input: ConsolidationInput = {
			userMessages: pendingUserMessages,
			assistantMessages: pendingAssistantMessages,
			cwd: sessionCwd,
			sessionId,
		};

		const currentFacts = store
			.listSemantic(undefined, 200)
			.map((f) => ({ key: f.key, value: f.value }));
		const currentLessons = store
			.listLessons(undefined, 100)
			.map((l) => ({ rule: l.rule, category: l.category }));
		return buildConsolidationPrompt(input, currentFacts, currentLessons);
	}

	async function consolidateSession(): Promise<void> {
		if (!store) return;

		const prompt = buildCurrentConsolidationPrompt();
		if (!prompt) return;

		// Use pi's exec to call the LLM via a lightweight pi session.
		// Use a fast model to avoid blocking shutdown for too long.
		try {
			const result = await pi.exec(
				"pi",
				[
					"-p",
					prompt,
					"--print",
					"--no-extensions",
					"--model",
					"claude-sonnet-4-20250514",
				],
				{
					timeout: 45_000,
					cwd: sessionCwd,
				},
			);

			if (result.code === 0 && result.stdout) {
				const extracted = parseConsolidationResponse(result.stdout);
				const applied = applyExtracted(
					store!,
					extracted,
					`session:${sessionId ?? "unknown"}`,
				);
				if (applied.semantic + applied.lessons > 0) {
					// Log but don't notify — we're shutting down
					console.error(
						`pi-memory: consolidated ${applied.semantic} facts, ${applied.lessons} lessons`,
					);
				}
			}
		} catch {
			// Timeout or exec failure — skip consolidation this session
		}
	}

	function startBackgroundConsolidation(): void {
		const prompt = buildCurrentConsolidationPrompt();
		if (!prompt) return;

		const payloadPath = join(
			tmpdir(),
			`pi-memory-${process.pid}-${Date.now()}.json`,
		);
		writeFileSync(
			payloadPath,
			JSON.stringify({
				prompt,
				dbPath: resolvedDbPath,
				cwd: sessionCwd,
				source: `session:${sessionId ?? "unknown"}`,
			}),
		);

		const scriptPath = join(
			dirname(fileURLToPath(import.meta.url)),
			"background-consolidate.mjs",
		);
		const child = spawn(process.execPath, [scriptPath, payloadPath], {
			cwd: sessionCwd || homedir(),
			detached: true,
			stdio: "ignore",
		});
		child.unref();
	}

	// ─── Tools ──────────────────────────────────────────────────────

	pi.registerTool({
		name: "memory_init",
		label: "Memory Init",
		description:
			"Initialize the persistent memory store for the current session. Use this if other memory tools report 'Memory store not initialized'.",
		parameters: Type.Object({
			force: Type.Optional(
				Type.Boolean({
					description: "Reopen the store even if already initialized",
				}),
			),
		}) as any,
		async execute(
			_id: unknown,
			params: any,
			_signal: unknown,
			_update: unknown,
			ctx: unknown,
		) {
			try {
				const result = initializeStore(
					getContextCwd(ctx),
					params?.force ?? false,
				);
				const stats = store!.stats();
				return ok(
					`${result.message ?? "Memory store initialized"}\nMemory: ${stats.semantic} semantic facts, ${stats.lessons} active lessons, ${stats.events} events logged\nDB: ${result.dbPath}`,
				);
			} catch (err: any) {
				return ok(`Memory init failed: ${err.message}`);
			}
		},
	});

	pi.registerTool({
		name: "memory_search",
		label: "Memory Search",
		description:
			"Search persistent memory for facts, preferences, and project patterns the user has established across sessions.",
		parameters: Type.Object({
			query: Type.String({ description: "Search query" }),
			limit: Type.Optional(
				Type.Number({ description: "Max results (default 10)" }),
			),
		}) as any,
		async execute(
			_id: unknown,
			params: any,
			_signal: unknown,
			_update: unknown,
			_ctx: unknown,
		) {
			if (!store) return ok("Memory store not initialized");

			const results = store.searchSemantic(params.query, params.limit ?? 10);
			if (results.length === 0) {
				return ok("No matching memories found.");
			}

			const text = results
				.map(
					(r) =>
						`${r.key}: ${r.value} (confidence: ${r.confidence}, source: ${r.source})`,
				)
				.join("\n");

			return ok(text);
		},
	});

	pi.registerTool({
		name: "memory_remember",
		label: "Memory Remember",
		description:
			"Store a fact, preference, or lesson in persistent memory. Use dotted keys like pref.editor, project.rosie.lang, tool.sed.usage. For corrections, use type='lesson'.",
		parameters: Type.Object({
			type: Type.String({
				description: "'fact' for key-value, 'lesson' for a correction",
			}),
			key: Type.Optional(
				Type.String({
					description: "Dotted key for facts (e.g. pref.commit_style)",
				}),
			),
			value: Type.Optional(Type.String({ description: "Value for facts" })),
			rule: Type.Optional(
				Type.String({ description: "Rule text for lessons" }),
			),
			category: Type.Optional(
				Type.String({ description: "Category for lessons (default: general)" }),
			),
			negative: Type.Optional(
				Type.Boolean({ description: "True if this is something to AVOID" }),
			),
		}) as any,
		async execute(
			_id: unknown,
			params: any,
			_signal: unknown,
			_update: unknown,
			_ctx: unknown,
		) {
			if (!store) return ok("Memory store not initialized");

			// Defensively unwrap double-quoted string args from misbehaving model runners.
			params = {
				...params,
				type: stripQuotes(params.type),
				key: stripQuotes(params.key),
				value: stripQuotes(params.value),
				rule: stripQuotes(params.rule),
				category: stripQuotes(params.category),
			};

			if (params.type !== "fact" && params.type !== "lesson") {
				return ok(`Invalid type: ${params.type}. Must be 'fact' or 'lesson'.`);
			}

			if (params.type === "fact") {
				if (!params.key || !params.value) {
					return ok("Both key and value required for facts");
				}
				store.setSemantic(params.key, params.value, 0.95, "user");
				return ok(`Remembered: ${params.key} = ${params.value}`);
			}

			if (params.type === "lesson") {
				if (!params.rule) {
					return ok("Rule text required for lessons");
				}
				const result = store.addLesson(
					params.rule,
					params.category ?? "general",
					"user",
					params.negative ?? false,
				);
				if (result.success) {
					return ok(`Lesson learned: ${params.rule}`);
				}
				return ok(`Already known (${result.reason}): ${params.rule}`);
			}

			return ok("Unknown type");
		},
	});

	pi.registerTool({
		name: "memory_forget",
		label: "Memory Forget",
		description: "Remove a fact or lesson from persistent memory.",
		parameters: Type.Object({
			type: Type.String(),
			key: Type.Optional(Type.String({ description: "Key for facts" })),
			id: Type.Optional(Type.String({ description: "ID for lessons" })),
		}) as any,
		async execute(
			_id: unknown,
			params: any,
			_signal: unknown,
			_update: unknown,
			_ctx: unknown,
		) {
			if (!store) return ok("Memory store not initialized");

			params = {
				...params,
				type: stripQuotes(params.type),
				key: stripQuotes(params.key),
				id: stripQuotes(params.id),
			};

			if (params.type !== "fact" && params.type !== "lesson") {
				return ok(`Invalid type: ${params.type}. Must be 'fact' or 'lesson'.`);
			}

			if (params.type === "fact" && params.key) {
				const deleted = store.deleteSemantic(params.key);
				return ok(
					deleted ? `Forgot: ${params.key}` : `Not found: ${params.key}`,
				);
			}

			if (params.type === "lesson" && params.id) {
				const deleted = store.deleteLesson(params.id);
				return ok(
					deleted ? `Forgot lesson ${params.id}` : `Not found: ${params.id}`,
				);
			}

			return ok("Provide key (for facts) or id (for lessons)");
		},
	});

	pi.registerTool({
		name: "memory_lessons",
		label: "Memory Lessons",
		description: "List learned corrections and lessons from past sessions.",
		parameters: Type.Object({
			category: Type.Optional(
				Type.String({ description: "Filter by category" }),
			),
			limit: Type.Optional(
				Type.Number({ description: "Max results (default 50)" }),
			),
		}) as any,
		async execute(
			_id: unknown,
			params: any,
			_signal: unknown,
			_update: unknown,
			_ctx: unknown,
		) {
			if (!store) return ok("Memory store not initialized");

			const lessons = store.listLessons(params.category, params.limit ?? 50);
			if (lessons.length === 0) {
				return ok("No lessons learned yet.");
			}

			const text = lessons
				.map(
					(l) =>
						`${l.negative ? "❌" : "✅"} [${l.category}] ${l.rule} (id: ${l.id.slice(0, 8)})`,
				)
				.join("\n");

			return ok(text);
		},
	});

	pi.registerTool({
		name: "memory_stats",
		label: "Memory Stats",
		description:
			"Show memory statistics — how many facts, lessons, and events are stored.",
		parameters: Type.Object({}) as any,
		async execute(
			_id: unknown,
			_params: unknown,
			_signal: unknown,
			_update: unknown,
			_ctx: unknown,
		) {
			if (!store) return ok("Memory store not initialized");

			const stats = store.stats();
			const text = `Memory: ${stats.semantic} semantic facts, ${stats.lessons} active lessons, ${stats.events} events logged\nDB: ${resolvedDbPath}`;
			return ok(text);
		},
	});

	// ─── Commands ──────────────────────────────────────────────────

	pi.registerCommand("memory-consolidate", {
		description:
			"Manually trigger memory consolidation for the current session",
		async handler(_args, ctx) {
			if (!store) {
				ctx.ui.notify("Memory store not initialized", "warning");
				return;
			}

			if (pendingUserMessages.length < 2) {
				ctx.ui.notify(
					"Not enough conversation to consolidate (need at least 2 user messages)",
					"warning",
				);
				return;
			}

			ctx.ui.notify("Consolidating session memory...", "info");
			try {
				await consolidateSession();
				const stats = store.stats();
				ctx.ui.notify(
					`Memory updated: ${stats.semantic} facts, ${stats.lessons} lessons`,
					"info",
				);
			} catch (err: any) {
				ctx.ui.notify(`Consolidation failed: ${err.message}`, "error");
			}
		},
	});
}

// ─── Helpers ─────────────────────────────────────────────────────────

function extractText(content: unknown): string {
	if (typeof content === "string") return content;
	if (Array.isArray(content)) {
		return content
			.filter((c: any) => c.type === "text" && typeof c.text === "string")
			.map((c: any) => c.text)
			.join("\n");
	}
	return "";
}
