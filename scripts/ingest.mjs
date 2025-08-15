// scripts/ingest.mjs
import fs from "fs";
import path from "path";
import process from "process";
import { fileURLToPath } from "url";
import { Client } from "pg";
import dns from "dns";
import OpenAI from "openai";
import mammoth from "mammoth";
import { parse as parseHTML } from "node-html-parser";
import { encoding_for_model } from "tiktoken";
import { createClient as createSupabaseClient } from "@supabase/supabase-js";

if (typeof dns.setDefaultResultOrder === "function") dns.setDefaultResultOrder("ipv4first");

const EMBEDDINGS_MODEL = process.env.EMBEDDINGS_MODEL || "text-embedding-3-small";
const EMBEDDING_DIM = parseInt(process.env.EMBEDDING_DIM || "1536", 10);

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const supabase = createSupabaseClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE);

// ---- tiny utils ----
function enc() { return encoding_for_model("gpt-4o-mini"); }
function countTokens(text) { const e = enc(); const n = e.encode(text || "").length; e.free(); return n; }
function chunkText(text, maxTokens = 800, overlap = 100) {
  const e = enc(); const toks = e.encode(text || ""); const chunks = [];
  for (let i = 0; i < toks.length; i += Math.max(1, maxTokens - overlap)) {
    const slice = toks.slice(i, Math.min(i + maxTokens, toks.length));
    chunks.push(e.decode(slice));
    if (i + maxTokens >= toks.length) break;
  }
  e.free(); return chunks.length ? chunks : [text || ""];
}

async function embedLiteral(text) {
  const r = await openai.embeddings.create({ model: EMBEDDINGS_MODEL, input: text });
  const v = r.data[0].embedding;
  if (v.length !== EMBEDDING_DIM) throw new Error(`Embedding dim mismatch: got ${v.length}, expected ${EMBEDDING_DIM}`);
  return `[${v.join(",")}]`; // pgvector literal
}

// lazy pdf-parse (Actions runner can resolve ESM fine)
let _pdfParse = null;
async function getPdfParse() {
  if (_pdfParse) return _pdfParse;
  const mod = await import("pdf-parse/lib/pdf-parse.js").catch(async () => await import("pdf-parse"));
  _pdfParse = mod.default || mod;
  return _pdfParse;
}

async function readFileAsText(fp) {
  const buf = fs.readFileSync(fp);
  const lower = fp.toLowerCase();
  if (lower.endsWith(".pdf")) {
    const pdfParse = await getPdfParse();
    const out = await pdfParse(buf);
    return out.text || "";
  } else if (lower.endsWith(".docx")) {
    const out = await mammoth.extractRawText({ buffer: buf });
    return out.value || "";
  } else if (lower.endsWith(".html") || lower.endsWith(".htm")) {
    const root = parseHTML(buf.toString("utf8"));
    return root.textContent || "";
  } else if (lower.endsWith(".txt") || lower.endsWith(".md")) {
    return buf.toString("utf8");
  }
  throw new Error(`Unsupported file type: ${fp}`);
}

// IPv4-pinned PG client
async function pgClientFromUrl(pgUrlString) {
  const u = new URL(pgUrlString);
  const host = u.hostname;
  const { address: ipv4 } = await new Promise((resolve, reject) =>
    dns.lookup(host, { family: 4 }, (err, addr, fam) => err ? reject(err) : resolve({ address: addr, family: fam }))
  );
  return new Client({
    host: ipv4,
    port: parseInt(u.port || "5432", 10),
    database: decodeURIComponent(u.pathname.replace(/^\//, "")),
    user: decodeURIComponent(u.username),
    password: decodeURIComponent(u.password),
    ssl: { rejectUnauthorized: false },
  });
}

async function upsertDocument(db, { title, author = null, published_at = null, tags = [], metadata = {}, source_uri = null }) {
  const q = await db.query(
    `INSERT INTO documents (title, author, published_at, tags, metadata, source_uri)
     VALUES ($1,$2,$3,$4,$5,$6) RETURNING id`,
    [title, author, published_at, tags, metadata, source_uri]
  );
  return q.rows[0].id;
}

async function insertChunk(db, doc_id, order_index, content, embeddingLiteral) {
  const token_count = countTokens(content);
  await db.query(
    `INSERT INTO doc_chunks (doc_id, order_index, content, token_count, embedding)
     VALUES ($1,$2,$3,$4,$5::vector)`,
    [doc_id, order_index, content, token_count, embeddingLiteral]
  );
}

async function main() {
  // Inputs from workflow_dispatch
  const ROOT = process.env.INPUT_FOLDER || "docs";
  const TAGS = (process.env.INPUT_TAGS || "").split(",").map(s => s.trim()).filter(Boolean);

  const db = await pgClientFromUrl(process.env.DATABASE_URL);
  await db.connect();

  const files = fs.readdirSync(ROOT).filter(f =>
    /\.(pdf|docx|txt|md|html?)$/i.test(f)
  );

  if (!files.length) {
    console.log(`No files found in ${ROOT}.`);
    process.exit(0);
  }

  for (const name of files) {
    const filepath = path.join(ROOT, name);
    console.log(`\nIngesting: ${filepath}`);

    // Optional: store original file in Supabase Storage
    let uploadedPath = null;
    try {
      const buf = fs.readFileSync(filepath);
      const key = `${Date.now()}_${name}`;
      const { data, error } = await supabase
        .storage.from(process.env.SUPABASE_BUCKET)
        .upload(key, buf, { contentType: "application/octet-stream", upsert: true });
      if (error) throw error;
      uploadedPath = data.path;
      console.log(`Stored original in Supabase Storage: ${uploadedPath}`);
    } catch (e) {
      console.warn(`Storage upload failed (continuing): ${e.message}`);
    }

    const text = await readFileAsText(filepath);
    const title = path.parse(name).name;

    const doc_id = await upsertDocument(db, {
      title,
      tags: TAGS,
      metadata: { uploadedPath },
      source_uri: null
    });

    const pieces = chunkText(text, parseInt(process.env.MAX_CHUNK_TOKENS || "800",10), parseInt(process.env.CHUNK_OVERLAP_TOKENS || "100",10));
    let order = 0;
    for (const p of pieces) {
      const vec = await embedLiteral(p);
      await insertChunk(db, doc_id, order++, p, vec);
    }

    console.log(`✓ Ingested "${title}" → doc_id=${doc_id}, chunks=${pieces.length}`);
  }

  await db.end();
  console.log("\nAll done.");
}

main().catch(err => {
  console.error("INGEST ERROR:", err);
  process.exit(1);
});
