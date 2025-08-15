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
// ===== Documents module (ingest + search + get + delete) =====
import multer from "multer";
import pdfParse from "pdf-parse";
import mammoth from "mammoth";
import { parse as parseHTML } from "node-html-parser";
import { encoding_for_model } from "tiktoken";
import { createClient as createSupabaseClient } from "@supabase/supabase-js";

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 15 * 1024 * 1024 } }); // 15MB
const supabase = createSupabaseClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE);

function enc() { return encoding_for_model("gpt-4o-mini"); } // ok for token counting
function countTokens(text) { const e = enc(); const n = e.encode(text || "").length; e.free(); return n; }

function chunkText(text, maxTokens = parseInt(process.env.MAX_CHUNK_TOKENS || "800",10), overlap = parseInt(process.env.CHUNK_OVERLAP_TOKENS || "100",10)) {
  const e = enc(); const toks = e.encode(text || ""); const chunks = [];
  for (let i = 0; i < toks.length; i += Math.max(1, maxTokens - overlap)) {
    const slice = toks.slice(i, Math.min(i + maxTokens, toks.length));
    chunks.push(e.decode(slice));
    if (i + maxTokens >= toks.length) break;
  }
  e.free(); return chunks.length ? chunks : [text || ""];
}

async function extractTextFromBuffer(filename, buffer) {
  const lower = (filename || "").toLowerCase();
  if (lower.endsWith(".pdf")) {
    const out = await pdfParse(buffer); return out.text || "";
  } else if (lower.endsWith(".docx")) {
    const out = await mammoth.extractRawText({ buffer }); return out.value || "";
  } else if (lower.endsWith(".html") || lower.endsWith(".htm")) {
    const root = parseHTML(buffer.toString("utf8")); return root.textContent || "";
  } else if (lower.endsWith(".txt") || lower.endsWith(".md")) {
    return buffer.toString("utf8");
  }
  throw new Error("Unsupported file type (pdf, docx, html, txt, md)");
}

// Create doc row
async function upsertDocument({ title, author = null, published_at = null, tags = [], metadata = {}, source_uri = null }) {
  const q = await db.query(
    `INSERT INTO documents (title, author, published_at, tags, metadata, source_uri)
     VALUES ($1,$2,$3,$4,$5,$6) RETURNING id`,
    [title, author, published_at, tags, metadata, source_uri]
  );
  return q.rows[0].id;
}

// Insert chunk row
async function insertChunk(doc_id, order_index, content, embeddingLiteral) {
  const token_count = countTokens(content);
  await db.query(
    `INSERT INTO doc_chunks (doc_id, order_index, content, token_count, embedding)
     VALUES ($1,$2,$3,$4,$5::vector)`,
    [doc_id, order_index, content, token_count, embeddingLiteral]
  );
}

// POST /docs/ingest  (multipart OR JSON)
app.post("/docs/ingest", auth, upload.single("file"), async (req, res) => {
  try {
    const body = req.body || {};
    const title = body.title || (req.file ? req.file.originalname : null);
    if (!title) return res.status(400).json({ error: "Missing title" });

    let text = body.content || "";
    let uploadedPath = null;

    // If a file was uploaded, store it in Supabase Storage and extract text
    if (req.file) {
      const path = `${Date.now()}_${req.file.originalname}`;
      const { data, error } = await supabase
        .storage.from(process.env.SUPABASE_BUCKET)
        .upload(path, req.file.buffer, { cacheControl: "3600", contentType: req.file.mimetype, upsert: true });
      if (error) throw error;
      uploadedPath = data.path;
      text = await extractTextFromBuffer(req.file.originalname, req.file.buffer);
    }

    if (!text || !text.trim()) return res.status(400).json({ error: "No content found" });

    // Parse tags if they came as JSON string in multipart
    let tags = [];
    if (typeof body.tags === "string") {
      try { const arr = JSON.parse(body.tags); if (Array.isArray(arr)) tags = arr; } catch {}
    } else if (Array.isArray(body.tags)) {
      tags = body.tags;
    }

    const doc_id = await upsertDocument({
      title,
      author: body.author || null,
      published_at: body.published_at || null,
      tags,
      metadata: { uploadedPath },
      source_uri: body.source_uri || null
    });

    const pieces = chunkText(text);
    let order = 0;
    for (const p of pieces) {
      const vecLit = await embed(p);               // returns "[...]" string
      await insertChunk(doc_id, order++, p, vecLit);
    }

    res.json({ ok: true, doc_id, chunks: pieces.length });
  } catch (e) {
    console.error("docs/ingest error:", e);
    res.status(500).json({ error: e.message || String(e) });
  }
});

// GET /docs/search?query=...&top_k=8&tags=a,b&from=YYYY-MM-DD&to=YYYY-MM-DD&doc_id=uuid
app.get("/docs/search", auth, async (req, res) => {
  try {
    const { query = "", top_k = 8, from = null, to = null, tags = "", doc_id = null } = req.query;
    if (!query) return res.status(400).json({ error: "Missing query" });
    const limit = Math.max(1, Math.min(20, parseInt(top_k, 10) || 8));
    const qvec = await embed(String(query));       // "[...]" string

    const conds = ["1=1"]; const params = [];
    if (from)   { params.push(from);   conds.push(`d.published_at >= $${params.length}`); }
    if (to)     { params.push(to);     conds.push(`d.published_at <= $${params.length}`); }
    if (doc_id) { params.push(doc_id); conds.push(`d.id = $${params.length}`); }
    const tagList = String(tags || "").split(",").map(s => s.trim()).filter(Boolean);
    if (tagList.length) { params.push(tagList); conds.push(`d.tags && $${params.length}::text[]`); }

    // Vector search + join metadata
    const sql = `
      SELECT c.doc_id, c.order_index, c.content, d.title, d.source_uri, d.tags, d.metadata
      FROM doc_chunks c
      JOIN documents d ON d.id = c.doc_id
      WHERE ${conds.join(" AND ")}
      ORDER BY c.embedding <=> $${params.length + 1}::vector
      LIMIT $${params.length + 2}
    `;
    const v = await db.query(sql, [...params, qvec, limit]);

    // Hybrid: simple keyword fall-back and merge
    const ksql = `
      SELECT c.doc_id, c.order_index, c.content, d.title, d.source_uri, d.tags, d.metadata
      FROM doc_chunks c
      JOIN documents d ON d.id = c.doc_id
      WHERE ${conds.join(" AND ")} AND (c.content ILIKE $${params.length + 1})
      ORDER BY c.created_at DESC
      LIMIT $${params.length + 2}
    `;
    const k = await db.query(ksql, [...params, `%${query}%`, Math.min(10, limit)]);

    const seen = new Set(); const merged = [];
    for (const r of [...v.rows, ...k.rows]) {
      const id = `${r.doc_id}:${r.order_index}`;
      if (!seen.has(id)) { seen.add(id); merged.push(r); }
    }
    res.json(merged.slice(0, limit));
  } catch (e) {
    console.error("docs/search error:", e);
    res.status(500).json({ error: e.message || String(e) });
  }
});

// GET /docs/get?doc_id=...
app.get("/docs/get", auth, async (req, res) => {
  try {
    const { doc_id } = req.query || {};
    if (!doc_id) return res.status(400).json({ error: "Missing doc_id" });
    const d = await db.query("SELECT * FROM documents WHERE id = $1", [doc_id]);
    if (!d.rows.length) return res.status(404).json({ error: "Not found" });

    let signedUrl = null;
    const uploadedPath = d.rows[0].metadata?.uploadedPath;
    if (uploadedPath) {
      const { data, error } = await supabase
        .storage.from(process.env.SUPABASE_BUCKET)
        .createSignedUrl(uploadedPath, 3600);
      if (!error) signedUrl = data.signedUrl;
    }

    res.json({ ...d.rows[0], signedUrl });
  } catch (e) {
    console.error("docs/get error:", e);
    res.status(500).json({ error: e.message || String(e) });
  }
});

// DELETE /docs/delete  { doc_id }
app.delete("/docs/delete", auth, async (req, res) => {
  try {
    const { doc_id } = req.body || {};
    if (!doc_id) return res.status(400).json({ error: "Missing doc_id" });
    await db.query("DELETE FROM documents WHERE id = $1", [doc_id]);
    res.json({ ok: true });
  } catch (e) {
    console.error("docs/delete error:", e);
    res.status(500).json({ error: e.message || String(e) });
  }
});
