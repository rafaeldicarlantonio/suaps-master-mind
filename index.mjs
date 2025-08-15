// index.mjs — Supabase Pooler + IPv4 + SSL + hardened inputs
console.log("Booting Startup Brain API...");

import express from "express";
import cors from "cors";
import helmet from "helmet";
import dotenv from "dotenv";
import pkg from "pg";
import OpenAI from "openai";
import dns from "dns";
import { URL } from "url";

dotenv.config();
const { Client } = pkg;

/** Prefer IPv4 globally (extra guard) */
if (typeof dns.setDefaultResultOrder === "function") {
  dns.setDefaultResultOrder("ipv4first");
}

// ---- Env ----
const PORT = process.env.PORT || 8787;
const API_TOKEN = process.env.API_TOKEN || "";
const DATABASE_URL = process.env.DATABASE_URL; // use *pooler* URI here
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

// Embeddings defaults (free tier)
const EMBEDDINGS_MODEL = process.env.EMBEDDINGS_MODEL || "text-embedding-3-small";
const EMBEDDING_DIM = parseInt(process.env.EMBEDDING_DIM || "1536", 10);

if (!DATABASE_URL) throw new Error("Missing DATABASE_URL");
if (!OPENAI_API_KEY) throw new Error("Missing OPENAI_API_KEY");

// ---- Express ----
const app = express();
app.use(helmet());
app.use(cors());
app.use(express.json({ limit: "2mb" }));

/** Build a pg Client pinned to the IPv4 A-record of the pooler host */
async function pgClientFromUrl(pgUrlString) {
  const u = new URL(pgUrlString); // postgresql://user:pass@host:port/db
  const host = u.hostname;
  const port = parseInt(u.port || "5432", 10);
  const database = decodeURIComponent(u.pathname.replace(/^\//, ""));
  const user = decodeURIComponent(u.username);
  const password = decodeURIComponent(u.password);

  // Resolve IPv4 explicitly to avoid IPv6 ENETUNREACH
  const { address: ipv4 } = await new Promise((resolve, reject) =>
    dns.lookup(host, { family: 4 }, (err, addr, fam) =>
      err ? reject(err) : resolve({ address: addr, family: fam })
    )
  );

  return new Client({
    host: ipv4,            // connect to IPv4 directly
    port,
    database,
    user,
    password,
    ssl: { rejectUnauthorized: false }, // required for Supabase from serverless hosts
    statement_timeout: 60000,
    query_timeout: 60000,
    connectionTimeoutMillis: 10000
  });
}

const db = await pgClientFromUrl(DATABASE_URL);
await db.connect();

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

// ---- Auth ----
function auth(req, res, next) {
  if (!API_TOKEN) return next();
  const token = req.headers.authorization?.replace("Bearer ", "");
  if (token !== API_TOKEN) return res.status(401).json({ error: "Unauthorized" });
  next();
}

// ---- Embedding Helper (returns pgvector text literal) ----
async function embed(text) {
  const resp = await openai.embeddings.create({ model: EMBEDDINGS_MODEL, input: text });
  const vec = resp.data[0].embedding;
  if (vec.length !== EMBEDDING_DIM) {
    throw new Error(`Embedding dim mismatch: got ${vec.length}, expected ${EMBEDDING_DIM}`);
  }
  // Return as pgvector string literal: "[0.1,0.2,...]"
  return `[${vec.join(",")}]`;
}


/* ========= ROUTES ========= */

// Root & health
app.get("/", (req, res) => res.type("text").send("Startup Brain API is running. Try /health"));
app.get("/health", (_, res) => res.json({ ok: true, uptime: process.uptime() }));

// ---- Upsert single ----
app.post("/remember", auth, async (req, res) => {
  try {
    let {
      user_id, key, value,
      type = "preference", scope = "global", category = "",
      tags = [], source = "user", confidence = null, pii = false,
      sensitivity = "low", version = "1.0", meta = {}, expires_at = null
    } = req.body || {};

    if (!user_id || !key || typeof value === "undefined" || value === null) {
      return res.status(400).json({ error: "Missing user_id, key, or value" });
    }

    // Normalize inputs
    if (typeof tags === "string") {
      // accept "formatting" or "a,b,c"
      tags = tags.includes(",")
        ? tags.split(",").map(s => s.trim()).filter(Boolean)
        : [tags.trim()].filter(Boolean);
    }
    if (!Array.isArray(tags)) tags = [];
    if (!meta || typeof meta !== "object") meta = {};

    const vec = await embed(`${key}\n${String(value)}`);

    const sql = `
      INSERT INTO memories (
        user_id, key, value, type, scope, category, tags, source, confidence,
        pii, sensitivity, version, created_at, updated_at, expires_at, meta, embedding
      ) VALUES (
        $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,NOW(),NOW(),$13,$14,$15::vector
      )
      ON CONFLICT (user_id, key)
      DO UPDATE SET
        value = EXCLUDED.value,
        type = EXCLUDED.type,
        scope = EXCLUDED.scope,
        category = EXCLUDED.category,
        tags = EXCLUDED.tags,
        source = EXCLUDED.source,
        confidence = EXCLUDED.confidence,
        pii = EXCLUDED.pii,
        sensitivity = EXCLUDED.sensitivity,
        version = EXCLUDED.version,
        expires_at = EXCLUDED.expires_at,
        meta = EXCLUDED.meta,
        embedding = EXCLUDED.embedding,
        updated_at = NOW();
    `;

    await db.query(sql, [
      user_id, key, String(value), type, scope, category, tags, source, confidence,
      pii, sensitivity, version, expires_at, meta, vec
    ]);

    res.json({ ok: true });
  } catch (e) {
    console.error("remember error:", e.stack || e);
    res.status(500).json({ error: e.message || String(e) });
  }
});

// ---- Upsert batch (OpenAPI-friendly alias) ----
// Body: { "items": [MemoryItem, ...] }
app.post("/remember-batch", auth, async (req, res) => {
  try {
    const items = Array.isArray(req.body?.items) ? req.body.items : [];
    let count = 0;
    for (const it of items) {
      const r = await fetch(`http://localhost:${PORT}/remember`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(API_TOKEN ? { Authorization: `Bearer ${API_TOKEN}` } : {})
        },
        body: JSON.stringify(it)
      });
      const js = await r.json().catch(() => ({}));
      if (js && js.ok) count++;
    }
    res.json({ ok: true, count });
  } catch (e) {
    console.error("remember-batch error:", e.stack || e);
    res.status(500).json({ error: e.message || String(e) });
  }
});

// ---- Recall (hybrid) ----
app.get("/recall", auth, async (req, res) => {
  try {
    const { user_id, query = "", type = "", category = "", tags = "", key = "", limit = 10 } = req.query;
    if (!user_id) return res.status(400).json({ error: "Missing user_id" });

    const lim = Math.max(1, Math.min(50, parseInt(limit, 10) || 10));
    const conds = ["user_id = $1"]; const params = [user_id];

    if (type) { params.push(type); conds.push(`type = $${params.length}`); }
    if (category) { params.push(category); conds.push(`category = $${params.length}`); }
    if (key) { params.push(key); conds.push(`key = $${params.length}`); }

    let tagList = [];
    if (tags) {
      tagList = String(tags).split(",").map(s => s.trim()).filter(Boolean);
      if (tagList.length) { params.push(tagList); conds.push(`tags && $${params.length}::text[]`); }
    }

    let vectorResults = [];
    if (query) {
      const qvec = await embed(String(query));
      const vsql = `SELECT user_id, key, value, type, scope, category, tags, source, confidence, pii, sensitivity, version, created_at, updated_at, expires_at, meta
                    FROM memories WHERE ${conds.join(" AND ")}
                    ORDER BY embedding <=> $${params.length + 1}::vector
                    LIMIT $${params.length + 2}`;
      const vres = await db.query(vsql, [...params, qvec, lim]);
      vectorResults = vres.rows;
    }

    let textResults = [];
    if (query) {
      const tsql = `SELECT user_id, key, value, type, scope, category, tags, source, confidence, pii, sensitivity, version, created_at, updated_at, expires_at, meta
                    FROM memories WHERE ${conds.join(" AND ")}
                      AND (value ILIKE $${params.length + 1} OR key ILIKE $${params.length + 1})
                    ORDER BY updated_at DESC
                    LIMIT $${params.length + 2}`;
      const tres = await db.query(tsql, [...params, `%${query}%`, lim]);
      textResults = tres.rows;
    }

    if (!query) {
      const lsql = `SELECT user_id, key, value, type, scope, category, tags, source, confidence, pii, sensitivity, version, created_at, updated_at, expires_at, meta
                    FROM memories WHERE ${conds.join(" AND ")}
                    ORDER BY updated_at DESC
                    LIMIT $${params.length + 1}`;
      const lres = await db.query(lsql, [...params, lim]);
      return res.json(lres.rows);
    }

    // Merge & dedupe (vector wins first)
    const map = new Map();
    for (const r of [...vectorResults, ...textResults]) {
      const id = `${r.user_id}::${r.key}`;
      if (!map.has(id)) map.set(id, r);
    }
    res.json(Array.from(map.values()).slice(0, lim));
  } catch (e) {
    console.error("recall error:", e.stack || e);
    res.status(500).json({ error: e.message || String(e) });
  }
});

// ---- List ----
app.get("/list", auth, async (req, res) => {
  try {
    const { user_id, type = "", category = "", tags = "", limit = 100 } = req.query;
    if (!user_id) return res.status(400).json({ error: "Missing user_id" });

    const lim = Math.max(1, Math.min(500, parseInt(limit, 10) || 100));
    const conds = ["user_id = $1"]; const params = [user_id];

    if (type) { params.push(type); conds.push(`type = $${params.length}`); }
    if (category) { params.push(category); conds.push(`category = $${params.length}`); }

    let tagList = [];
    if (tags) {
      tagList = String(tags).split(",").map(s => s.trim()).filter(Boolean);
      if (tagList.length) { params.push(tagList); conds.push(`tags && $${params.length}::text[]`); }
    }

    const sql = `SELECT user_id, key, value, type, scope, category, tags, source, confidence, pii, sensitivity, version, created_at, updated_at, expires_at, meta
                 FROM memories WHERE ${conds.join(" AND ")}
                 ORDER BY updated_at DESC
                 LIMIT $${params.length + 1}`;
    const q = await db.query(sql, [...params, lim]);
    res.json(q.rows);
  } catch (e) {
    console.error("list error:", e.stack || e);
    res.status(500).json({ error: e.message || String(e) });
  }
});

// ---- Forget ----
app.delete("/forget", auth, async (req, res) => {
  try {
    const { user_id, key, type, category, tags = [] } = req.body || {};
    if (!user_id) return res.status(400).json({ error: "Missing user_id" });

    if (key === "__ALL__") {
      await db.query("DELETE FROM memories WHERE user_id = $1", [user_id]);
      return res.json({ ok: true, cleared: true });
    }

    if (key) {
      await db.query("DELETE FROM memories WHERE user_id = $1 AND key = $2", [user_id, key]);
      return res.json({ ok: true, deleted: 1 });
    }

    const conds = ["user_id = $1"]; const params = [user_id];
    if (type) { params.push(type); conds.push(`type = $${params.length}`); }
    if (category) { params.push(category); conds.push(`category = $${params.length}`); }
    if (tags && tags.length) { params.push(tags); conds.push(`tags && $${params.length}::text[]`); }

    const sql = `DELETE FROM memories WHERE ${conds.join(" AND ")}`;
    await db.query(sql, params);
    res.json({ ok: true });
  } catch (e) {
    console.error("forget error:", e.stack || e);
    res.status(500).json({ error: e.message || String(e) });
  }
});

// ---- Export ----
app.get("/export", auth, async (req, res) => {
  try {
    const { user_id } = req.query || {};
    if (!user_id) return res.status(400).json({ error: "Missing user_id" });

    const q = await db.query("SELECT * FROM memories WHERE user_id = $1 ORDER BY key", [user_id]);
    res.json(q.rows);
  } catch (e) {
    console.error("export error:", e.stack || e);
    res.status(500).json({ error: e.message || String(e) });
  }
});

// ---- Start ----
app.listen(PORT, () => console.log(`Memory API (vector) running on port ${PORT}`));
