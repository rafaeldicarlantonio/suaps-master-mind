// ingest.mjs — local folder → Supabase (documents + chunks)
import fs from "fs";
import path from "path";
import dns from "dns";
import dotenv from "dotenv";
import OpenAI from "openai";
import { URL } from "url";
import { parse as parseHTML } from "node-html-parser";
import mammoth from "mammoth";
import pkg from "pg";
import { encoding_for_model } from "tiktoken";

dotenv.config();
const { Client } = pkg;

/* ---------- Config ---------- */
const FOLDER = process.argv[2] || "./docs";     // folder of PDFs/DOCX/TXT/HTML/MD
const DEFAULT_TAGS = ["kb"];                    // optional default tags
const MAX_CHUNK_TOKENS = parseInt(process.env.MAX_CHUNK_TOKENS || "800", 10);
const CHUNK_OVERLAP_TOKENS = parseInt(process.env.CHUNK_OVERLAP_TOKENS || "100", 10);

// DB & OpenAI envs
const DATABASE_URL = process.env.DATABASE_URL;  // use Supabase *pooler* URL
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const EMBEDDINGS_MODEL = process.env.EMBEDDINGS_MODEL || "text-embedding-3-small";
const EMBEDDING_DIM = parseInt(process.env.EMBEDDING_DIM || "1536", 10);

if (!DATABASE_URL) throw new Error("Missing DATABASE_URL");
if (!OPENAI_API_KEY) throw new Error("Missing OPENAI_API_KEY");

// Prefer IPv4 (avoids ENETUNREACH in some hosts)
if (typeof dns.setDefaultResultOrder === "function") dns.setDefaultResultOrder("ipv4first");

/* ---------- Helpers ---------- */
function enc() { return encoding_for_model("gpt-4o-mini"); } // ok for counting
function countTokens(text) { const e = enc(); const n = e.encode(text || "").length; e.free(); return n; }

function chunkText(text, maxTokens = MAX_CHUNK_TOKENS, overlap = CHUNK_OVERLAP_TOKENS) {
  const e = enc(); const toks = e.encode(text || ""); const chunks = [];
  for (let i = 0; i < toks.length; i += Math.max(1, maxTokens - overlap)) {
    const slice = toks.slice(i, Math.min(i + maxTokens, toks.length));
    chunks.push(e.decode(slice));
    if (i + maxTokens >= toks.length) break;
  }
  e.free(); 
  return chunks.length ? chunks : [text || ""];
}

// Lazy-load pdf-parse safely (avoid package test harness path issues)
let _pdfParseFn = null;
async function pdfParse(buffer) {
  if (_pdfParseFn) return _pdfParseFn(buffer);
  try {
    const mod = await import("pdf-parse/lib/pdf-parse.js");
    _pdfParseFn = mod.default || mod;
  } catch {
    const mod = await import("pdf-parse");
    _pdfParseFn = mod.default || mod;
  }
  return _pdfParseFn(buffer);
}

async function extractTextFromFile(filePath) {
  const lower = filePath.toLowerCase();
  const buf = fs.readFileSync(filePath);

  if (lower.endsWith(".pdf")) {
    const out = await pdfParse(buf);        // { text }
    return out.text || "";
  }
  if (lower.endsWith(".docx")) {
    const out = await mammoth.extractRawText({ buffer: buf }); // { value }
    return out.value || "";
  }
  if (lower.endsWith(".html") || lower.endsWith(".htm")) {
    const root = parseHTML(buf.toString("utf8"));
    return root.textContent || "";
  }
  if (lower.endsWith(".txt") || lower.endsWith(".md")) {
    return buf.toString("utf8");
  }
  return ""; // unsupported types are skipped
}

/* ---------- PG client (IPv4-pin + SSL) ---------- */
async function pgClientFromUrl(pgUrlString) {
  const u = new URL(pgUrlString); // postgresql://user:pass@host:port/db
  const host = u.hostname;
  const port = parseInt(u.port || "5432", 10);
  const database = decodeURIComponent(u.pathname.replace(/^\//, ""));
  const user = decodeURIComponent(u.username);
  const password = decodeURIComponent(u.password);

  const { address: ipv4 } = await new Promise((resolve, reject) =>
    dns.lookup(host, { family: 4 }, (err, addr, fam) =>
      err ? reject(err) : resolve({ address: addr, family: fam })
    )
  );

  return new Client({
    host: ipv4,
    port,
    database,
    user,
    password,
    ssl: { rejectUnauthorized: false },
    statement_timeout: 60000,
    query_timeout: 60000,
    connectionTimeoutMillis: 10000
  });
}

const db = await pgClientFromUrl(DATABASE_URL);
await db.connect();
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

async function embed(text) {
  const resp = await openai.embeddings.create({ model: EMBEDDINGS_MODEL, input: text });
  const vec = resp.data[0].embedding;
  if (vec.length !== EMBEDDING_DIM) throw new Error(`Embedding dim mismatch: got ${vec.length}, expected ${EMBEDDING_DIM}`);
  return `[${vec.join(",")}]`; // pgvector literal
}

async function upsertDocument({ title, author = null, published_at = null, tags = [], metadata = {}, source_uri = null }) {
  const q = await db.query(
    `INSERT INTO documents (title, author, published_at, tags, metadata, source_uri)
     VALUES ($1,$2,$3,$4,$5,$6) RETURNING id`,
    [title, author, published_at, tags, metadata, source_uri]
  );
  return q.rows[0].id;
}

async function insertChunk(doc_id, order_index, content, embeddingLiteral) {
  const token_count = countTokens(content);
  await db.query(
    `INSERT INTO doc_chunks (doc_id, order_index, content, token_count, embedding)
     VALUES ($1,$2,$3,$4,$5::vector)`,
    [doc_id, order_index, content, token_count, embeddingLiteral]
  );
}

/* ---------- Main ---------- */
async function main() {
  if (!fs.existsSync(FOLDER) || !fs.statSync(FOLDER).isDirectory()) {
    console.error(`Folder not found: ${FOLDER}`);
    process.exit(1);
  }
  const files = fs.readdirSync(FOLDER)
    .filter(f => /\.(pdf|docx|txt|md|html|htm)$/i.test(f));

  if (!files.length) {
    console.log("No supported files in folder:", FOLDER);
    process.exit(0);
  }

  console.log(`Found ${files.length} file(s). Ingesting into Supabase...`);
  for (const file of files) {
    const full = path.join(FOLDER, file);
    console.log(`\n→ ${file}`);
    const text = await extractTextFromFile(full);
    if (!text.trim()) { console.log("  (no text extracted, skipping)"); continue; }

    const title = path.basename(file);
    const doc_id = await upsertDocument({
      title,
      tags: DEFAULT_TAGS,
      metadata: { local_path: full },
      source_uri: null
    });

    const chunks = chunkText(text, MAX_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS);
    console.log(`  chunks: ${chunks.length}`);

    let order = 0;
    for (const ch of chunks) {
      const vec = await embed(ch);
      await insertChunk(doc_id, order++, ch, vec);
    }
    console.log(`  ✓ inserted doc_id=${doc_id}`);
  }

  await db.end();
  console.log("\nDone.");
}

main().catch(async (e) => {
  console.error("Ingest failed:", e);
  try { await db.end(); } catch {}
  process.exit(1);
});
