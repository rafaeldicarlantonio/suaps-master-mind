import express from "express";
import cors from "cors";
import helmet from "helmet";
import dotenv from "dotenv";
import pkg from "pg"; // { Client }
import OpenAI from "openai";

dotenv.config();
const { Client } = pkg;

// ---- Env ----
const PORT = process.env.PORT || 8787;
const API_TOKEN = process.env.API_TOKEN || "";
const DATABASE_URL = process.env.DATABASE_URL;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const EMBEDDINGS_MODEL = process.env.EMBEDDINGS_MODEL || "text-embedding-3-large"; // 3072 dims
const EMBEDDING_DIM = parseInt(process.env.EMBEDDING_DIM || "3072", 10);

if (!DATABASE_URL) throw new Error("Missing DATABASE_URL");
if (!OPENAI_API_KEY) throw new Error("Missing OPENAI_API_KEY");

// ---- Setup ----
const app = express();
app.use(helmet());
app.use(cors());
app.use(express.json({ limit: "2mb" }));

const db = new Client({ connectionString: DATABASE_URL });
await db.connect();

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

function auth(req, res, next) {
  if (!API_TOKEN) return next();
  const token = req.headers.authorization?.replace("Bearer ", "");
  if (token !== API_TOKEN) return res.status(401).json({ error: "Unauthorized" });
  next();
}

async function embed(text) {
  const resp = await openai.embeddings.create({ model: EMBEDDINGS_MODEL, input: text });
  const vec = resp.data[0].embedding;
  if (vec.length !== EMBEDDING_DIM) {
    throw new Error(`Embedding dim mismatch: got ${vec.length}, expected ${EMBEDDING_DIM}`);
  }
  return vec;
}

function toPgArray(arr) { return `{${(arr || []).map(s => (s || "").replace(/"/g, '\\"')).join(',')}}`; }

// ---- Upsert single ----
app.post("/remember", auth, async (req, res) => {
  try {
    const {
      user_id, key, value,
      type = "preference", scope = "global", category = "",
      tags = [], source = "user", confidence = null, pii = false,
      sensitivity = "low", version = "1.0", meta = {}, expires_at = null
    } = req.body || {};

    if (!user_id || !key || !value) return res.status(400).json({ error: "Missing user_id, key, or value" });

    const vec = await embed(`${key}\n${value}`);

    const sql = `
      INSERT INTO memories (
        user_id, key, value, type, scope, category, tags, source, confidence,
        pii, sensitivity, version, created_at, updated_at, expires_at, meta, embedding
      ) VALUES (
        $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,NOW(),NOW(),$13,$14,$15
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
      user_id, key, value, type, scope, category, tags, source, confidence,
      pii, sensitivity, version, expires_at, meta, vec
    ]);

    res.json({ ok: true });
  } catch (e) {
    console.error(e); res.status(500).json({ error: String(e.message || e) });
  }
});

// ---- Upsert batch ----
app.post("/remember:batch", auth, async (req, res) => {
  try {
    const items = Array.isArray(req.body) ? req.body : [];
    let count = 0;
    for (const it of items) {
      const r = await fetch(`http://localhost:${PORT}/remember`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...(API_TOKEN ? { Authorization: `Bearer ${API_TOKEN}` } : {}) },
        body: JSON.stringify(it)
      });
      const js = await r.json();
      if (js.ok) count++;
    }
    res.json({ ok: true, count });
  } catch (e) {
    console.error(e); res.status(500).json({ error: String(e.message || e) });
  }
});

// ---- Recall (hybrid) ----
app.get("/recall", auth, async (req, res) => {
  try {
    const { user_id, query = "", type = "", category = "", tags = "", key = "", limit = 10 } = req.query;
    if (!user_id) return res.status(400).json({ error: "Missing user_id" });
    const lim = Math.max(1, Math.min(50, parseInt(limit, 10) || 10));

    // Build filters
    const conds = ["user_id = $1"]; const params = [user_id];
    if (type) { params.push(type); conds.push(`type = $${params.length}`); }
    if (category) { params.push(category); conds.push(`category = $${params.length}`); }
    if (key) { params.push(key); conds.push(`key = $${params.length}`); }
    let tagList = [];
    if (tags) { tagList = String(tags).split(',').map(s=>s.trim()).filter(Boolean); }
    if (tagList.length) { params.push(tagList); conds.push(`tags && $${params.length}::text[]`); }

    // Vector search
    let vectorResults = [];
    if (query) {
      const qvec = await embed(String(query));
      const vsql = `SELECT user_id, key, value, type, scope, category, tags, source, confidence, pii, sensitivity, version, created_at, updated_at, expires_at, meta
                    FROM memories WHERE ${conds.join(" AND ")}
                    ORDER BY embedding <=> $${params.length + 1} LIMIT $${params.length + 2}`;
      const vres = await db.query(vsql, [...params, qvec, lim]);
      vectorResults = vres.rows;
    }

    // Keyword prefilter (ILIKE)
    let textResults = [];
    if (query) {
      const tsql = `SELECT user_id, key, value, type, scope, category, tags, source, confidence, pii, sensitivity, version, created_at, updated_at, expires_at, meta
                    FROM memories WHERE ${conds.join(" AND ")} AND (value ILIKE $${params.length + 1} OR key ILIKE $${params.length + 1})
                    ORDER BY updated_at DESC LIMIT $${params.length + 2}`;
      const tres = await db.query(tsql, [...params, `%${query}%`, lim]);
      textResults = tres.rows;
    }

    // No query → just list by filters
    if (!query) {
      const lsql = `SELECT user_id, key, value, type, scope, category, tags, source, confidence, pii, sensitivity, version, created_at, updated_at, expires_at, meta
                    FROM memories WHERE ${conds.join(" AND ")}
                    ORDER BY updated_at DESC LIMIT $${params.length + 1}`;
      const lres = await db.query(lsql, [...params, lim]);
      return res.json(lres.rows);
    }

    // Merge & dedupe (vector wins)
    const map = new Map();
    for (const r of [...vectorResults, ...textResults]) {
      const id = `${r.user_id}::${r.key}`;
      if (!map.has(id)) map.set(id, r);
    }
    res.json(Array.from(map.values()).slice(0, lim));
  } catch (e) {
    console.error(e); res.status(500).json({ error: String(e.message || e) });
  }
});

// ---- List (filters only) ----
app.get("/list", auth, async (req, res) => {
  try {
    const { user_id, type = "", category = "", tags = "", limit = 100 } = req.query;
    if (!user_id) return res.status(400).json({ error: "Missing user_id" });
    const lim = Math.max(1, Math.min(500, parseInt(limit, 10) || 100));

    const conds = ["user_id = $1"]; const params = [user_id];
    if (type) { params.push(type); conds.push(`type = $${params.length}`); }
    if (category) { params.push(category); conds.push(`category = $${params.length}`); }
    let tagList = [];
    if (tags) { tagList = String(tags).split(',').map(s=>s.trim()).filter(Boolean); }
    if (tagList.length) { params.push(tagList); conds.push(`tags && $${params.length}::text[]`); }

    const sql = `SELECT user_id, key, value, type, scope, category, tags, source, confidence, pii, sensitivity, version, created_at, updated_at, expires_at, meta
                 FROM memories WHERE ${conds.join(" AND ")}
                 ORDER BY updated_at DESC LIMIT $${params.length + 1}`;
    const q = await db.query(sql, [...params, lim]);
    res.json(q.rows);
  } catch (e) {
    console.error(e); res.status(500).json({ error: String(e.message || e) });
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

    // delete by filters
    const conds = ["user_id = $1"]; const params = [user_id];
    if (type) { params.push(type); conds.push(`type = $${params.length}`); }
    if (category) { params.push(category); conds.push(`category = $${params.length}`); }
    if (tags && tags.length) { params.push(tags); conds.push(`tags && $${params.length}::text[]`); }
    const sql = `DELETE FROM memories WHERE ${conds.join(" AND ")}`;
    await db.query(sql, params);
    res.json({ ok: true });
  } catch (e) {
    console.error(e); res.status(500).json({ error: String(e.message || e) });
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
    console.error(e); res.status(500).json({ error: String(e.message || e) });
  }
});

app.listen(PORT, () => console.log(`Memory API (vector) on :${PORT}`));
// ---- Documents module (ingest + search) ----
import multer from "multer";
import pdfParse from "pdf-parse";
import mammoth from "mammoth";
import { parse as parseHTML } from "node-html-parser";
import { encoding_for_model } from "tiktoken";
import { createClient as createSupabaseClient } from "@supabase/supabase-js";

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 15 * 1024 * 1024 } }); // 15MB
const supabase = createSupabaseClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE);

function enc() { return encoding_for_model("gpt-4o-mini"); } // ok for counting tokens; model agnostic
function countTokens(str) { const e = enc(); const n = e.encode(str).length; e.free(); return n; }

function chunkText(text, maxTokens = 800, overlap = 100) {
  const e = enc();
  const tokens = e.encode(text);
  const chunks = [];
  for (let i = 0; i < tokens.length; i += (maxTokens - overlap)) {
    const slice = tokens.slice(i, Math.min(i + maxTokens, tokens.length));
    chunks.push(e.decode(slice));
    if (i + maxTokens >= tokens.length) break;
  }
  e.free();
  return chunks;
}

async function embedStr(s) {
  const resp = await openai.embeddings.create({ model: EMBEDDINGS_MODEL, input: s });
  const vec = resp.data[0].embedding;
  if (vec.length !== EMBEDDING_DIM) throw new Error("Embedding dim mismatch");
  return vec;
}

async function upsertDocument({ title, author = null, published_at = null, tags = [], metadata = {}, source_uri = null }) {
  const q = await db.query(
    `INSERT INTO documents (title, author, published_at, tags, metadata, source_uri)
     VALUES ($1,$2,$3,$4,$5,$6) RETURNING id`,
    [title, author, published_at, tags, metadata, source_uri]
  );
  return q.rows[0].id;
}

async function insertChunk(doc_id, order_index, content, embedding) {
  const token_count = countTokens(content);
  await db.query(
    `INSERT INTO doc_chunks (doc_id, order_index, content, token_count, embedding)
     VALUES ($1,$2,$3,$4,$5)`,
    [doc_id, order_index, content, token_count, embedding]
  );
}

async function extractTextFromBuffer(filename, buffer) {
  const lower = filename.toLowerCase();
  if (lower.endsWith(".pdf")) {
    const out = await pdfParse(buffer);
    return out.text || "";
  } else if (lower.endsWith(".docx")) {
    const out = await mammoth.extractRawText({ buffer });
    return out.value || "";
  } else if (lower.endsWith(".html") || lower.endsWith(".htm")) {
    const root = parseHTML(buffer.toString("utf8"));
    return root.textContent || "";
  } else if (lower.endsWith(".txt") || lower.endsWith(".md")) {
    return buffer.toString("utf8");
  }
  throw new Error("Unsupported file type (pdf, docx, html, txt, md)");
}

// POST /docs/ingest  (multipart or JSON)
app.post("/docs/ingest", auth, upload.single("file"), async (req, res) => {
  try {
    const { title, author, published_at, tags, source_uri, content } = req.body || {};

    let text = content || "";
    let uploadedPath = null;

    if (req.file) {
      // store original file in Supabase Storage (optional but handy)
      const filePath = `${Date.now()}_${req.file.originalname}`;
      const { data, error } = await supabase.storage
        .from(process.env.SUPABASE_BUCKET)
        .upload(filePath, req.file.buffer, { contentType: req.file.mimetype, upsert: true });
      if (error) throw error;
      uploadedPath = data.path;

      text = await extractTextFromBuffer(req.file.originalname, req.file.buffer);
    }

    if (!text || !title) return res.status(400).json({ error: "Missing text content or title" });

    const doc_id = await upsertDocument({ title, author, published_at, tags: tags ? JSON.parse(tags) : [], metadata: { uploadedPath }, source_uri });

    const pieces = chunkText(text, parseInt(process.env.MAX_CHUNK_TOKENS || "800",10), parseInt(process.env.CHUNK_OVERLAP_TOKENS || "100",10));

    // embed & insert sequentially (simple); for speed, batch with Promise.allSettled and SQL COPY
    let order = 0;
    for (const p of pieces) {
      const vec = await embedStr(p);
      await insertChunk(doc_id, order++, p, vec);
    }

    res.json({ ok: true, doc_id, chunks: pieces.length });
  } catch (e) {
    console.error(e); res.status(500).json({ error: String(e.message || e) });
  }
});

// GET /docs/search?query=...&top_k=5&from=2024-01-01&to=2025-12-31&tags=a,b
app.get("/docs/search", auth, async (req, res) => {
  try {
    const { query = "", top_k = 8, from = null, to = null, tags = "", doc_id = null } = req.query;
    if (!query) return res.status(400).json({ error: "Missing query" });

    const qvec = await embedStr(String(query));

    // Filter by doc metadata
    const conds = ["1=1"]; const params = [];
    if (from) { params.push(from); conds.push(`d.published_at >= $${params.length}`); }
    if (to)   { params.push(to);   conds.push(`d.published_at <= $${params.length}`); }
    if (doc_id) { params.push(doc_id); conds.push(`d.id = $${params.length}`); }
    const tagList = String(tags).split(',').map(s=>s.trim()).filter(Boolean);
    if (tagList.length) { params.push(tagList); conds.push(`d.tags && $${params.length}::text[]`); }

    // Vector search + join back metadata
    const sql = `
      SELECT c.doc_id, c.order_index, c.content, d.title, d.source_uri, d.tags, d.metadata
      FROM doc_chunks c
      JOIN documents d ON d.id = c.doc_id
      WHERE ${conds.join(' AND ')}
      ORDER BY c.embedding <=> $${params.length + 1}
      LIMIT $${params.length + 2}
    `;
    const q = await db.query(sql, [...params, qvec, Math.max(1, Math.min(20, parseInt(top_k,10) || 8))]);

    // Also do lightweight keyword match and merge (hybrid)
    const ksql = `
      SELECT c.doc_id, c.order_index, c.content, d.title, d.source_uri, d.tags, d.metadata
      FROM doc_chunks c
      JOIN documents d ON d.id = c.doc_id
      WHERE ${conds.join(' AND ')} AND (c.content ILIKE $${params.length + 1})
      ORDER BY c.created_at DESC LIMIT $${params.length + 2}
    `;
    const kq = await db.query(ksql, [...params, `%${query}%`, 10]);

    const seen = new Set();
    const merged = [];
    for (const r of [...q.rows, ...kq.rows]) {
      const id = `${r.doc_id}:${r.order_index}`;
      if (!seen.has(id)) { seen.add(id); merged.push(r); }
    }

    res.json(merged.slice(0, Math.max(1, Math.min(20, parseInt(top_k,10) || 8))));
  } catch (e) {
    console.error(e); res.status(500).json({ error: String(e.message || e) });
  }
});

// GET /docs/get?doc_id=...
app.get("/docs/get", auth, async (req, res) => {
  try {
    const { doc_id } = req.query;
    if (!doc_id) return res.status(400).json({ error: "Missing doc_id" });
    const d = await db.query("SELECT * FROM documents WHERE id = $1", [doc_id]);
    if (!d.rows.length) return res.status(404).json({ error: "Not found" });

    // Optional: signed URL for original file
    let signedUrl = null;
    const uploadedPath = d.rows[0].metadata?.uploadedPath;
    if (uploadedPath) {
      const { data, error } = await supabase.storage
        .from(process.env.SUPABASE_BUCKET)
        .createSignedUrl(uploadedPath, 60 * 60); // 1h
      if (!error) signedUrl = data.signedUrl;
    }

    res.json({ ...d.rows[0], signedUrl });
  } catch (e) {
    console.error(e); res.status(500).json({ error: String(e.message || e) });
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
    console.error(e); res.status(500).json({ error: String(e.message || e) });
  }
});
