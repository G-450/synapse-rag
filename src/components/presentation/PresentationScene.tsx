"use client";

import { motion } from "framer-motion";
import {
  User, Brain, Database, Search, Cpu, FileText,
  Layers, Sparkles, ArrowDown, Zap, BarChart3, CheckCircle2,
  MessageSquare, GitBranch, ScanSearch, BookOpen, Binary,
  Gauge, Target, Shield, ArrowRight, Network, Workflow,
  type LucideIcon,
} from "lucide-react";

/* ─────────── helpers ─────────── */

const fadeUp = {
  initial: { opacity: 0, y: 24 },
  animate: { opacity: 1, y: 0 },
};

const stagger = {
  animate: { transition: { staggerChildren: 0.12 } },
};

function Badge({ children, color }: { children: React.ReactNode; color: string }) {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "6px",
        padding: "4px 14px",
        borderRadius: "9999px",
        fontSize: "11px",
        fontWeight: 600,
        letterSpacing: "1px",
        textTransform: "uppercase",
        background: `${color}15`,
        color: color,
        border: `1px solid ${color}30`,
      }}
    >
      {children}
    </span>
  );
}

function GlowCard({
  color,
  children,
  style,
}: {
  color: string;
  children: React.ReactNode;
  style?: React.CSSProperties;
}) {
  return (
    <motion.div
      variants={fadeUp}
      style={{
        background: `linear-gradient(135deg, ${color}08, ${color}04)`,
        border: `1px solid ${color}25`,
        borderRadius: "16px",
        padding: "24px",
        backdropFilter: "blur(12px)",
        position: "relative",
        overflow: "hidden",
        ...style,
      }}
    >
      <div
        style={{
          position: "absolute",
          top: "-50%",
          right: "-30%",
          width: "200px",
          height: "200px",
          borderRadius: "50%",
          background: `radial-gradient(circle, ${color}10, transparent 70%)`,
          pointerEvents: "none",
        }}
      />
      <div style={{ position: "relative", zIndex: 1 }}>{children}</div>
    </motion.div>
  );
}

function IconBox({ icon: Icon, color, size = 48 }: { icon: LucideIcon; color: string; size?: number }) {
  return (
    <div
      style={{
        width: `${size + 16}px`,
        height: `${size + 16}px`,
        borderRadius: "16px",
        background: `${color}15`,
        border: `1px solid ${color}40`,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        boxShadow: `0 0 30px ${color}20`,
      }}
    >
      <Icon size={size * 0.55} color={color} />
    </div>
  );
}

function SceneTitle({ title, subtitle, color }: { title: string; subtitle: string; color: string }) {
  return (
    <motion.div
      variants={fadeUp}
      style={{ textAlign: "center", marginBottom: "40px" }}
    >
      <Badge color={color}>{subtitle}</Badge>
      <h2
        style={{
          fontSize: "clamp(28px, 4vw, 48px)",
          fontWeight: 700,
          marginTop: "16px",
          lineHeight: 1.15,
          background: `linear-gradient(135deg, #fff 40%, ${color})`,
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
        }}
      >
        {title}
      </h2>
    </motion.div>
  );
}

function CodeBlock({ children, label }: { children: string; label?: string }) {
  return (
    <div
      style={{
        background: "rgba(0,0,0,0.5)",
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: "12px",
        overflow: "hidden",
      }}
    >
      {label && (
        <div
          style={{
            padding: "8px 16px",
            fontSize: "11px",
            fontWeight: 600,
            color: "rgba(255,255,255,0.4)",
            borderBottom: "1px solid rgba(255,255,255,0.06)",
            textTransform: "uppercase",
            letterSpacing: "1px",
          }}
        >
          {label}
        </div>
      )}
      <pre
        style={{
          padding: "16px 20px",
          margin: 0,
          fontSize: "13px",
          lineHeight: 1.7,
          color: "#94a3b8",
          fontFamily: "'Fira Code', 'Cascadia Code', monospace",
          overflowX: "auto",
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
        }}
      >
        {children}
      </pre>
    </div>
  );
}

/* ─────────── SCENES ─────────── */

function Scene0() {
  const pipelineSteps = [
    { icon: MessageSquare, label: "Query", color: "#38bdf8" },
    { icon: Binary, label: "Tokenize", color: "#818cf8" },
    { icon: Brain, label: "Embed", color: "#a78bfa" },
    { icon: Database, label: "Index", color: "#34d399" },
    { icon: Search, label: "Search", color: "#2dd4bf" },
    { icon: ScanSearch, label: "Rerank", color: "#22d3ee" },
    { icon: FileText, label: "Retrieve", color: "#4ade80" },
    { icon: Layers, label: "Augment", color: "#fbbf24" },
    { icon: Cpu, label: "LLM", color: "#f472b6" },
    { icon: Zap, label: "Stream", color: "#fb923c" },
    { icon: BarChart3, label: "Evaluate", color: "#c084fc" },
    { icon: Sparkles, label: "Output", color: "#f43f5e" },
  ];

  return (
    <motion.div variants={stagger} initial="initial" animate="animate" style={{ textAlign: "center" }}>
      <motion.div variants={fadeUp}>
        <Badge color="#a78bfa">Interactive Walkthrough</Badge>
      </motion.div>

      <motion.h1
        variants={fadeUp}
        style={{
          fontSize: "clamp(36px, 5vw, 64px)",
          fontWeight: 800,
          marginTop: "20px",
          lineHeight: 1.1,
          background: "linear-gradient(135deg, #fff 20%, #a78bfa 50%, #38bdf8 80%)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
        }}
      >
        How Synapse RAG Works
      </motion.h1>

      <motion.p
        variants={fadeUp}
        style={{
          fontSize: "18px",
          color: "#64748b",
          maxWidth: "600px",
          margin: "16px auto 0",
          lineHeight: 1.6,
        }}
      >
        A detailed visual walkthrough of Retrieval-Augmented Generation — from user query to grounded, evaluated output.
      </motion.p>

      {/* Pipeline Overview */}
      <motion.div
        variants={fadeUp}
        style={{
          display: "flex",
          flexWrap: "wrap",
          justifyContent: "center",
          gap: "8px",
          marginTop: "48px",
          maxWidth: "800px",
          marginLeft: "auto",
          marginRight: "auto",
        }}
      >
        {pipelineSteps.map((s, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3 + i * 0.06 }}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "6px",
              padding: "8px 14px",
              borderRadius: "10px",
              background: `${s.color}10`,
              border: `1px solid ${s.color}20`,
              fontSize: "12px",
              fontWeight: 500,
              color: s.color,
            }}
          >
            <s.icon size={14} />
            {s.label}
            {i < pipelineSteps.length - 1 && (
              <ArrowRight size={10} color="rgba(255,255,255,0.2)" style={{ marginLeft: "2px" }} />
            )}
          </motion.div>
        ))}
      </motion.div>

      <motion.p
        variants={fadeUp}
        style={{ fontSize: "14px", color: "#475569", marginTop: "32px" }}
      >
        Press <kbd style={{ padding: "2px 8px", borderRadius: "4px", background: "rgba(255,255,255,0.1)", border: "1px solid rgba(255,255,255,0.15)", fontSize: "12px" }}>→</kbd> or click <strong>Next Step</strong> to begin
      </motion.p>
    </motion.div>
  );
}

function Scene1() {
  return (
    <motion.div variants={stagger} initial="initial" animate="animate">
      <SceneTitle title="User Submits a Query" subtitle="Step 1 · Input" color="#38bdf8" />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px", maxWidth: "900px", margin: "0 auto" }}>
        <GlowCard color="#38bdf8">
          <IconBox icon={User} color="#38bdf8" />
          <h3 style={{ fontSize: "18px", fontWeight: 600, marginTop: "16px", color: "#e2e8f0" }}>
            Natural Language Input
          </h3>
          <p style={{ fontSize: "14px", color: "#94a3b8", marginTop: "8px", lineHeight: 1.6 }}>
            The user types a question in plain English. The system accepts any free-form natural language query — no special syntax required.
          </p>
        </GlowCard>

        <GlowCard color="#38bdf8">
          <div style={{ fontSize: "13px", color: "#64748b", marginBottom: "12px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "1px" }}>
            Example Query
          </div>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
            style={{
              padding: "20px",
              borderRadius: "12px",
              background: "rgba(56,189,248,0.08)",
              border: "1px solid rgba(56,189,248,0.2)",
            }}
          >
            <MessageSquare size={16} color="#38bdf8" style={{ marginBottom: "8px" }} />
            <p style={{ fontSize: "16px", color: "#e2e8f0", fontStyle: "italic", lineHeight: 1.5 }}>
              &ldquo;What are the termination clauses in the vendor agreement?&rdquo;
            </p>
          </motion.div>

          <div style={{ marginTop: "16px" }}>
            <CodeBlock label="API Request">{`POST /api/chat
{
  "messages": [
    { "role": "user",
      "content": "What are the termination
                  clauses?" }
  ],
  "documentId": "abc-123"  // optional
}`}</CodeBlock>
          </div>
        </GlowCard>
      </div>
    </motion.div>
  );
}

function Scene2() {
  const tokens = ["What", "are", "the", "termination", "clauses", "in", "the", "vendor", "agreement", "?"];

  return (
    <motion.div variants={stagger} initial="initial" animate="animate">
      <SceneTitle title="Text Preprocessing & Tokenization" subtitle="Step 2 · Tokenize" color="#818cf8" />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px", maxWidth: "900px", margin: "0 auto" }}>
        <GlowCard color="#818cf8">
          <IconBox icon={Binary} color="#818cf8" />
          <h3 style={{ fontSize: "18px", fontWeight: 600, marginTop: "16px", color: "#e2e8f0" }}>
            Breaking Down Language
          </h3>
          <p style={{ fontSize: "14px", color: "#94a3b8", marginTop: "8px", lineHeight: 1.6 }}>
            The raw text is split into <strong style={{ color: "#c4b5fd" }}>tokens</strong> — individual units the model can understand. This includes subword tokenization using the model&apos;s vocabulary.
          </p>
          <div style={{ marginTop: "16px", padding: "12px", borderRadius: "8px", background: "rgba(129,140,248,0.08)", border: "1px solid rgba(129,140,248,0.15)", fontSize: "13px", color: "#a5b4fc" }}>
            <strong>Model:</strong> Xenova/all-MiniLM-L6-v2<br />
            <strong>Vocab Size:</strong> ~30,522 tokens (WordPiece)<br />
            <strong>Max Sequence:</strong> 256 tokens
          </div>
        </GlowCard>

        <GlowCard color="#818cf8">
          <div style={{ fontSize: "13px", color: "#64748b", marginBottom: "16px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "1px" }}>
            Tokenization Result
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
            {tokens.map((token, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, scale: 0.5, y: 10 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                transition={{ delay: 0.3 + i * 0.08 }}
                style={{
                  padding: "8px 16px",
                  borderRadius: "8px",
                  background: "rgba(129,140,248,0.12)",
                  border: "1px solid rgba(129,140,248,0.3)",
                  fontSize: "14px",
                  fontFamily: "monospace",
                  color: "#c4b5fd",
                  fontWeight: 500,
                }}
              >
                {token}
              </motion.div>
            ))}
          </div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.2 }}
            style={{ marginTop: "20px" }}
          >
            <ArrowDown size={20} color="#818cf8" style={{ margin: "0 auto", display: "block" }} />
            <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", marginTop: "12px" }}>
              {[101, 2054, 2024, 1996, 22851, 14638, 1999, 1996, 17709, 4259, 1029, 102].map((id, i) => (
                <motion.span
                  key={i}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 1.4 + i * 0.04 }}
                  style={{
                    padding: "4px 10px",
                    borderRadius: "6px",
                    background: "rgba(99,102,241,0.15)",
                    fontSize: "11px",
                    fontFamily: "monospace",
                    color: "#a5b4fc",
                  }}
                >
                  {id}
                </motion.span>
              ))}
            </div>
            <p style={{ fontSize: "11px", color: "#64748b", marginTop: "8px" }}>
              Token IDs (including [CLS]=101 and [SEP]=102)
            </p>
          </motion.div>
        </GlowCard>
      </div>
    </motion.div>
  );
}

function Scene3() {
  const vectorVals = [0.0231, -0.4521, 0.1837, 0.7104, -0.0892, 0.3341, -0.5578, 0.2110, 0.0045, -0.3892, 0.6234, "..."];

  return (
    <motion.div variants={stagger} initial="initial" animate="animate">
      <SceneTitle title="Vector Embedding Generation" subtitle="Step 3 · Embed" color="#a78bfa" />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px", maxWidth: "900px", margin: "0 auto" }}>
        <GlowCard color="#a78bfa">
          <IconBox icon={Brain} color="#a78bfa" />
          <h3 style={{ fontSize: "18px", fontWeight: 600, marginTop: "16px", color: "#e2e8f0" }}>
            Semantic Encoding
          </h3>
          <p style={{ fontSize: "14px", color: "#94a3b8", marginTop: "8px", lineHeight: 1.6 }}>
            The tokenized input passes through a <strong style={{ color: "#c4b5fd" }}>6-layer Transformer encoder</strong> (MiniLM). Each layer applies self-attention and feed-forward networks to build a rich understanding of meaning.
          </p>
          <p style={{ fontSize: "14px", color: "#94a3b8", marginTop: "8px", lineHeight: 1.6 }}>
            The final hidden states are <strong style={{ color: "#c4b5fd" }}>mean-pooled</strong> across all tokens and <strong style={{ color: "#c4b5fd" }}>L2-normalized</strong> to produce a unit-length vector.
          </p>
          <CodeBlock label="embeddings.ts">{`const embedder = await PipelineSingleton
  .getInstance();
const output = await embedder(text, {
  pooling: 'mean',
  normalize: true
});
// → Float32Array[384]`}</CodeBlock>
        </GlowCard>

        <GlowCard color="#a78bfa">
          <div style={{ fontSize: "13px", color: "#64748b", marginBottom: "16px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "1px" }}>
            Output: 384-Dimensional Vector
          </div>

          {/* Neural network visualization */}
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "20px", marginBottom: "24px" }}>
            {[4, 6, 8, 6, 4].map((nodes, layerIdx) => (
              <motion.div
                key={layerIdx}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 + layerIdx * 0.1 }}
                style={{ display: "flex", flexDirection: "column", gap: "4px", alignItems: "center" }}
              >
                {Array.from({ length: nodes }).map((_, nodeIdx) => (
                  <motion.div
                    key={nodeIdx}
                    animate={{
                      boxShadow: [
                        "0 0 4px rgba(167,139,250,0.3)",
                        "0 0 12px rgba(167,139,250,0.6)",
                        "0 0 4px rgba(167,139,250,0.3)",
                      ],
                    }}
                    transition={{
                      duration: 1.5,
                      repeat: Infinity,
                      delay: (layerIdx * 0.15 + nodeIdx * 0.05),
                    }}
                    style={{
                      width: "10px",
                      height: "10px",
                      borderRadius: "50%",
                      background: "#a78bfa",
                    }}
                  />
                ))}
                <span style={{ fontSize: "9px", color: "#64748b", marginTop: "4px" }}>
                  L{layerIdx + 1}
                </span>
              </motion.div>
            ))}
          </div>

          <ArrowDown size={20} color="#a78bfa" style={{ margin: "0 auto 16px", display: "block" }} />

          <div style={{
            padding: "16px",
            borderRadius: "10px",
            background: "rgba(167,139,250,0.08)",
            border: "1px solid rgba(167,139,250,0.2)",
            fontFamily: "monospace",
            fontSize: "12px",
            color: "#c4b5fd",
            display: "flex",
            flexWrap: "wrap",
            gap: "4px",
          }}>
            <span style={{ color: "#64748b" }}>[</span>
            {vectorVals.map((v, i) => (
              <motion.span
                key={i}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.8 + i * 0.05 }}
              >
                {typeof v === "number" ? v.toFixed(4) : v}
                {i < vectorVals.length - 1 ? "," : ""}
              </motion.span>
            ))}
            <span style={{ color: "#64748b" }}>]</span>
          </div>
          <p style={{ fontSize: "11px", color: "#64748b", marginTop: "8px", textAlign: "center" }}>
            384 floating-point numbers capturing semantic meaning
          </p>
        </GlowCard>
      </div>
    </motion.div>
  );
}

function Scene4() {
  return (
    <motion.div variants={stagger} initial="initial" animate="animate">
      <SceneTitle title="Vector Database & Document Indexing" subtitle="Step 4 · Index" color="#34d399" />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px", maxWidth: "900px", margin: "0 auto" }}>
        <GlowCard color="#34d399">
          <IconBox icon={Database} color="#34d399" />
          <h3 style={{ fontSize: "18px", fontWeight: 600, marginTop: "16px", color: "#e2e8f0" }}>
            Pre-Indexed Knowledge Base
          </h3>
          <p style={{ fontSize: "14px", color: "#94a3b8", marginTop: "8px", lineHeight: 1.6 }}>
            During <strong style={{ color: "#6ee7b7" }}>ingestion</strong>, legal documents from the LegalBench-RAG corpus were chunked and embedded into the same 384-dimensional space.
          </p>
          <div style={{ marginTop: "16px" }}>
            <CodeBlock label="Ingestion Pipeline">{`1. Load LegalBench-RAG QA pairs
2. Group snippets by corpus_file
3. Create Document records in Neon
4. For each snippet:
   → generateEmbedding(text)
   → Store as vector(384) in Chunk`}</CodeBlock>
          </div>
        </GlowCard>

        <GlowCard color="#34d399">
          <div style={{ fontSize: "13px", color: "#64748b", marginBottom: "16px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "1px" }}>
            Database Schema
          </div>

          {[
            { table: "Document", fields: ["id (UUID)", "filename", "title", "source_corpus", "createdAt"], color: "#34d399" },
            { table: "Chunk", fields: ["id (UUID)", "document_id (FK)", "content (text)", "embedding vector(384)", "char_start, char_end"], color: "#2dd4bf" },
          ].map((t, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 + i * 0.2 }}
              style={{
                padding: "16px",
                borderRadius: "10px",
                background: `${t.color}08`,
                border: `1px solid ${t.color}20`,
                marginBottom: i === 0 ? "12px" : "0",
              }}
            >
              <div style={{ fontSize: "14px", fontWeight: 700, color: t.color, marginBottom: "8px", fontFamily: "monospace" }}>
                📋 {t.table}
              </div>
              {t.fields.map((f, j) => (
                <div key={j} style={{ fontSize: "12px", color: "#94a3b8", padding: "2px 0", fontFamily: "monospace" }}>
                  • {f}
                </div>
              ))}
            </motion.div>
          ))}

          <div style={{ marginTop: "12px", padding: "10px 14px", borderRadius: "8px", background: "rgba(52,211,153,0.08)", fontSize: "12px", color: "#6ee7b7" }}>
            <strong>Neon Postgres</strong> with <strong>pgvector</strong> extension for vector similarity
          </div>
        </GlowCard>
      </div>
    </motion.div>
  );
}

function Scene5() {
  const results = [
    { doc: "vendor_agreement.pdf", score: 0.934, highlight: true },
    { doc: "vendor_agreement.pdf", score: 0.891, highlight: true },
    { doc: "service_contract.pdf", score: 0.756, highlight: false },
    { doc: "nda_template.pdf", score: 0.623, highlight: false },
    { doc: "lease_agreement.pdf", score: 0.412, highlight: false },
  ];

  return (
    <motion.div variants={stagger} initial="initial" animate="animate">
      <SceneTitle title="Cosine Similarity Search" subtitle="Step 5 · Search" color="#2dd4bf" />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px", maxWidth: "900px", margin: "0 auto" }}>
        <GlowCard color="#2dd4bf">
          <IconBox icon={Search} color="#2dd4bf" />
          <h3 style={{ fontSize: "18px", fontWeight: 600, marginTop: "16px", color: "#e2e8f0" }}>
            Finding Nearest Neighbors
          </h3>
          <p style={{ fontSize: "14px", color: "#94a3b8", marginTop: "8px", lineHeight: 1.6 }}>
            The query vector is compared to every chunk embedding using <strong style={{ color: "#5eead4" }}>cosine distance</strong> (<code style={{ color: "#2dd4bf" }}>&lt;=&gt;</code> operator in pgvector). The closer to 1.0, the more semantically similar.
          </p>
          <CodeBlock label="SQL Query">{`SELECT c.id, c.content,
  1 - (c.embedding <=> $1::vector)
    AS similarity
FROM "Chunk" c
JOIN "Document" d ON d.id = c.document_id
WHERE c.embedding IS NOT NULL
ORDER BY c.embedding <=> $1::vector
LIMIT 20  -- fetch 4× for reranking`}</CodeBlock>
        </GlowCard>

        <GlowCard color="#2dd4bf">
          <div style={{ fontSize: "13px", color: "#64748b", marginBottom: "16px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "1px" }}>
            Initial Retrieval Results (Top 5 of 20)
          </div>
          {results.map((r, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 + i * 0.12 }}
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "12px 16px",
                borderRadius: "10px",
                background: r.highlight ? "rgba(45,212,191,0.1)" : "rgba(255,255,255,0.02)",
                border: `1px solid ${r.highlight ? "rgba(45,212,191,0.3)" : "rgba(255,255,255,0.06)"}`,
                marginBottom: "8px",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <FileText size={14} color={r.highlight ? "#2dd4bf" : "#475569"} />
                <span style={{ fontSize: "13px", color: r.highlight ? "#e2e8f0" : "#64748b" }}>
                  {r.doc}
                </span>
              </div>
              <div style={{
                padding: "4px 12px",
                borderRadius: "6px",
                background: r.score > 0.85 ? "rgba(45,212,191,0.15)" : r.score > 0.6 ? "rgba(251,191,36,0.1)" : "rgba(255,255,255,0.05)",
                fontSize: "13px",
                fontWeight: 600,
                fontFamily: "monospace",
                color: r.score > 0.85 ? "#2dd4bf" : r.score > 0.6 ? "#fbbf24" : "#64748b",
              }}>
                {(r.score * 100).toFixed(1)}%
              </div>
            </motion.div>
          ))}
          <p style={{ fontSize: "11px", color: "#475569", marginTop: "8px" }}>
            Over-fetching 4× the needed chunks (20 instead of 5) for cross-encoder reranking
          </p>
        </GlowCard>
      </div>
    </motion.div>
  );
}

function Scene6() {
  return (
    <motion.div variants={stagger} initial="initial" animate="animate">
      <SceneTitle title="Cross-Encoder Reranking" subtitle="Step 6 · Rerank" color="#22d3ee" />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px", maxWidth: "900px", margin: "0 auto" }}>
        <GlowCard color="#22d3ee">
          <IconBox icon={ScanSearch} color="#22d3ee" />
          <h3 style={{ fontSize: "18px", fontWeight: 600, marginTop: "16px", color: "#e2e8f0" }}>
            Two-Stage Retrieval
          </h3>
          <p style={{ fontSize: "14px", color: "#94a3b8", marginTop: "8px", lineHeight: 1.6 }}>
            A <strong style={{ color: "#67e8f9" }}>Bi-Encoder</strong> (Step 5) is fast but approximate. The <strong style={{ color: "#67e8f9" }}>Cross-Encoder</strong> compares query and document <em>together</em> through all attention layers for much higher accuracy.
          </p>
          <div style={{ marginTop: "16px", padding: "12px", borderRadius: "8px", background: "rgba(34,211,238,0.08)", border: "1px solid rgba(34,211,238,0.15)", fontSize: "13px", color: "#67e8f9" }}>
            <strong>Model:</strong> Xenova/ms-marco-MiniLM-L-6-v2<br />
            <strong>Type:</strong> Sequence Classification (relevance logit)<br />
            <strong>Input:</strong> [query, chunk_content] pairs
          </div>
          <CodeBlock label="cross-encoder.ts">{`const inputs = await tokenizer(query, {
  text_pair: chunk.content,
  padding: true,
  truncation: true
});
const output = await model(inputs);
const score = output.logits.data[0];`}</CodeBlock>
        </GlowCard>

        <GlowCard color="#22d3ee">
          <div style={{ fontSize: "13px", color: "#64748b", marginBottom: "16px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "1px" }}>
            Before vs After Reranking
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
            <div>
              <p style={{ fontSize: "11px", color: "#64748b", marginBottom: "8px", fontWeight: 600, textTransform: "uppercase" }}>
                Bi-Encoder Order
              </p>
              {["Chunk A (93.4%)", "Chunk B (89.1%)", "Chunk C (75.6%)", "Chunk D (62.3%)", "Chunk E (41.2%)"].map((c, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.2 + i * 0.08 }}
                  style={{ padding: "6px 10px", fontSize: "11px", color: "#94a3b8", borderLeft: "2px solid rgba(255,255,255,0.1)", marginBottom: "4px" }}
                >
                  #{i + 1} {c}
                </motion.div>
              ))}
            </div>
            <div>
              <p style={{ fontSize: "11px", color: "#67e8f9", marginBottom: "8px", fontWeight: 600, textTransform: "uppercase" }}>
                Cross-Encoder Reranked
              </p>
              {[
                { label: "Chunk B → #1", score: "8.42" },
                { label: "Chunk A → #2", score: "7.91" },
                { label: "Chunk D → #3", score: "5.23" },
                { label: "Chunk C → #4", score: "3.67" },
                { label: "Chunk E → #5", score: "1.02" },
              ].map((c, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.8 + i * 0.1 }}
                  style={{
                    padding: "6px 10px",
                    fontSize: "11px",
                    color: i < 3 ? "#67e8f9" : "#64748b",
                    borderLeft: `2px solid ${i < 3 ? "#22d3ee" : "rgba(255,255,255,0.1)"}`,
                    marginBottom: "4px",
                    background: i < 3 ? "rgba(34,211,238,0.05)" : "transparent",
                  }}
                >
                  {c.label} <span style={{ fontFamily: "monospace", fontSize: "10px" }}>({c.score})</span>
                </motion.div>
              ))}
            </div>
          </div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.5 }}
            style={{ marginTop: "16px", padding: "10px 14px", borderRadius: "8px", background: "rgba(34,211,238,0.1)", fontSize: "12px", color: "#67e8f9", textAlign: "center" }}
          >
            ✨ Reranking promotes more contextually relevant chunks that bi-encoder ranking might miss
          </motion.div>
        </GlowCard>
      </div>
    </motion.div>
  );
}

function Scene7() {
  return (
    <motion.div variants={stagger} initial="initial" animate="animate">
      <SceneTitle title="Retrieving Top-K Chunks" subtitle="Step 7 · Retrieve" color="#4ade80" />
      <div style={{ maxWidth: "900px", margin: "0 auto" }}>
        <motion.div variants={fadeUp} style={{ marginBottom: "24px", textAlign: "center" }}>
          <p style={{ fontSize: "15px", color: "#94a3b8", maxWidth: "600px", margin: "0 auto" }}>
            The final top-5 chunks are selected after reranking. Each chunk contains actual text from the legal documents, paired with its source metadata.
          </p>
        </motion.div>

        <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          {[
            { file: "vendor_agreement.pdf", snippet: "Section 14.2 — Either party may terminate this Agreement upon thirty (30) days' prior written notice...", score: "94.1%", rank: 1 },
            { file: "vendor_agreement.pdf", snippet: "In the event of material breach, the non-breaching party shall have the right to terminate immediately...", score: "91.3%", rank: 2 },
            { file: "vendor_agreement.pdf", snippet: "Upon termination, all confidential information must be returned or destroyed within ten (10) business days...", score: "87.8%", rank: 3 },
            { file: "service_contract.pdf", snippet: "The Service Provider may terminate for cause if Client fails to remit payment within sixty (60) days...", score: "72.4%", rank: 4 },
            { file: "vendor_agreement.pdf", snippet: "Notwithstanding the foregoing, obligations under Sections 8, 11, and 15 shall survive termination...", score: "68.9%", rank: 5 },
          ].map((chunk, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 + i * 0.15 }}
              style={{
                padding: "20px",
                borderRadius: "12px",
                background: i < 3 ? "rgba(74,222,128,0.05)" : "rgba(255,255,255,0.02)",
                border: `1px solid ${i < 3 ? "rgba(74,222,128,0.2)" : "rgba(255,255,255,0.06)"}`,
                display: "grid",
                gridTemplateColumns: "40px 1fr 80px",
                gap: "16px",
                alignItems: "start",
              }}
            >
              <div style={{
                width: "36px", height: "36px", borderRadius: "10px",
                background: i < 3 ? "rgba(74,222,128,0.15)" : "rgba(255,255,255,0.05)",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: "14px", fontWeight: 700,
                color: i < 3 ? "#4ade80" : "#64748b",
              }}>
                #{chunk.rank}
              </div>
              <div>
                <div style={{ fontSize: "12px", color: "#4ade80", fontWeight: 600, marginBottom: "6px" }}>
                  📄 {chunk.file}
                </div>
                <p style={{ fontSize: "13px", color: "#cbd5e1", lineHeight: 1.5, margin: 0 }}>
                  &ldquo;{chunk.snippet}&rdquo;
                </p>
              </div>
              <div style={{
                padding: "4px 12px", borderRadius: "6px",
                background: parseFloat(chunk.score) > 80 ? "rgba(74,222,128,0.12)" : "rgba(255,255,255,0.05)",
                fontSize: "13px", fontWeight: 600, fontFamily: "monospace", textAlign: "center",
                color: parseFloat(chunk.score) > 80 ? "#4ade80" : "#94a3b8",
              }}>
                {chunk.score}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}

function Scene8() {
  return (
    <motion.div variants={stagger} initial="initial" animate="animate">
      <SceneTitle title="Augmented Prompt Construction" subtitle="Step 8 · Augment" color="#fbbf24" />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px", maxWidth: "900px", margin: "0 auto" }}>
        <GlowCard color="#fbbf24">
          <IconBox icon={Layers} color="#fbbf24" />
          <h3 style={{ fontSize: "18px", fontWeight: 600, marginTop: "16px", color: "#e2e8f0" }}>
            Context Window Assembly
          </h3>
          <p style={{ fontSize: "14px", color: "#94a3b8", marginTop: "8px", lineHeight: 1.6 }}>
            The retrieved chunks are formatted with their source metadata and injected into the system prompt. This provides the LLM with <strong style={{ color: "#fcd34d" }}>grounded context</strong> so it can answer factually.
          </p>
          <CodeBlock label="formatContext()">{`[Source: vendor_agreement.pdf | 
 Relevance: 94.1%]
Section 14.2 — Either party may
terminate this Agreement...

---

[Source: vendor_agreement.pdf | 
 Relevance: 91.3%]
In the event of material breach...`}</CodeBlock>
        </GlowCard>

        <GlowCard color="#fbbf24">
          <div style={{ fontSize: "13px", color: "#64748b", marginBottom: "16px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "1px" }}>
            Final Prompt Structure
          </div>

          {[
            { label: "System Prompt", color: "#f472b6", content: "You are Synapse RAG, an expert legal contract analyst.\nAnswer ONLY using information in the Context.\nQuote exact clauses. Cite source documents." },
            { label: "Context (Retrieved)", color: "#fbbf24", content: "=== CONTEXT (Retrieved Contract Excerpts) ===\n\n[Source: vendor_agreement.pdf | 94.1%]\nSection 14.2 — Either party may terminate...\n\n---\n\n[Source: vendor_agreement.pdf | 91.3%]\nIn the event of material breach..." },
            { label: "User Message", color: "#38bdf8", content: "What are the termination clauses in the vendor agreement?" },
          ].map((section, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 + i * 0.2 }}
              style={{
                padding: "12px 16px",
                borderRadius: "10px",
                background: `${section.color}08`,
                borderLeft: `3px solid ${section.color}`,
                marginBottom: "10px",
              }}
            >
              <div style={{ fontSize: "11px", fontWeight: 700, color: section.color, marginBottom: "6px", textTransform: "uppercase", letterSpacing: "0.5px" }}>
                {section.label}
              </div>
              <pre style={{ fontSize: "11px", color: "#94a3b8", margin: 0, fontFamily: "monospace", whiteSpace: "pre-wrap", lineHeight: 1.5 }}>
                {section.content}
              </pre>
            </motion.div>
          ))}
        </GlowCard>
      </div>
    </motion.div>
  );
}

function Scene9() {
  return (
    <motion.div variants={stagger} initial="initial" animate="animate">
      <SceneTitle title="LLM Inference (Groq)" subtitle="Step 9 · Process" color="#f472b6" />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px", maxWidth: "900px", margin: "0 auto" }}>
        <GlowCard color="#f472b6">
          <IconBox icon={Cpu} color="#f472b6" />
          <h3 style={{ fontSize: "18px", fontWeight: 600, marginTop: "16px", color: "#e2e8f0" }}>
            Language Model Processing
          </h3>
          <p style={{ fontSize: "14px", color: "#94a3b8", marginTop: "8px", lineHeight: 1.6 }}>
            The augmented prompt is sent to <strong style={{ color: "#f9a8d4" }}>Meta Llama 3.1 8B Instant</strong> via the <strong style={{ color: "#f9a8d4" }}>Groq</strong> inference API. Groq&apos;s LPU hardware delivers ultra-low-latency inference.
          </p>
          <CodeBlock label="route.ts">{`const result = streamText({
  model: groq('llama-3.1-8b-instant'),
  system: augmentedSystem,
  messages: modelMessages,
  headers: {
    'X-Citations': encodeURIComponent(
      JSON.stringify(citations)
    ),
  },
});`}</CodeBlock>
          <div style={{ marginTop: "12px", padding: "10px 14px", borderRadius: "8px", background: "rgba(244,114,182,0.08)", fontSize: "12px", color: "#f9a8d4" }}>
            <strong>AI SDK:</strong> Vercel AI SDK v7 — <code>streamText()</code> for streaming responses
          </div>
        </GlowCard>

        <GlowCard color="#f472b6">
          <div style={{ fontSize: "13px", color: "#64748b", marginBottom: "16px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "1px" }}>
            Processing Visualization
          </div>

          {/* Attention layers diagram */}
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "8px", marginBottom: "24px" }}>
            {["Input Embedding", "Self-Attention × 32 layers", "Feed-Forward Networks", "Output Projection"].map((layer, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, scaleX: 0.5 }}
                animate={{ opacity: 1, scaleX: 1 }}
                transition={{ delay: 0.3 + i * 0.15 }}
                style={{
                  padding: "10px 24px",
                  borderRadius: "8px",
                  background: `rgba(244,114,182,${0.05 + i * 0.03})`,
                  border: "1px solid rgba(244,114,182,0.15)",
                  fontSize: "12px",
                  color: "#f9a8d4",
                  fontWeight: 500,
                  width: "100%",
                  textAlign: "center",
                }}
              >
                {layer}
              </motion.div>
            ))}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
            {[
              { label: "Model", value: "Llama 3.1 8B" },
              { label: "Provider", value: "Groq (LPU)" },
              { label: "Parameters", value: "8 Billion" },
              { label: "Latency", value: "~500ms TTFT" },
            ].map((stat, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1 + i * 0.1 }}
                style={{
                  padding: "12px",
                  borderRadius: "8px",
                  background: "rgba(255,255,255,0.02)",
                  border: "1px solid rgba(255,255,255,0.06)",
                  textAlign: "center",
                }}
              >
                <div style={{ fontSize: "10px", color: "#64748b", fontWeight: 600, textTransform: "uppercase", letterSpacing: "1px" }}>
                  {stat.label}
                </div>
                <div style={{ fontSize: "14px", color: "#e2e8f0", fontWeight: 600, marginTop: "4px" }}>
                  {stat.value}
                </div>
              </motion.div>
            ))}
          </div>
        </GlowCard>
      </div>
    </motion.div>
  );
}

function Scene10() {
  const words = [
    "The", "vendor", "agreement", "contains", "several", "key", "termination", "clauses:",
    "\n\n", "**1.", "Termination", "for", "Convenience**", "—", "Per", "Section", "14.2,",
    "either", "party", "may", "terminate", "with", "30", "days'", "written", "notice.",
    "\n\n", "**2.", "Termination", "for", "Cause**", "—", "Material", "breach", "allows",
    "immediate", "termination", "by", "the", "non-breaching", "party.",
    "\n\n", "**3.", "Post-Termination**", "—", "Sections", "8,", "11,", "and", "15", "survive."
  ];

  return (
    <motion.div variants={stagger} initial="initial" animate="animate">
      <SceneTitle title="Streaming Response Generation" subtitle="Step 10 · Stream" color="#fb923c" />
      <div style={{ maxWidth: "900px", margin: "0 auto" }}>
        <GlowCard color="#fb923c" style={{ padding: "32px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "20px" }}>
            <IconBox icon={Zap} color="#fb923c" size={36} />
            <div>
              <h3 style={{ fontSize: "16px", fontWeight: 600, color: "#e2e8f0", margin: 0 }}>Token-by-Token Streaming</h3>
              <p style={{ fontSize: "13px", color: "#94a3b8", margin: "4px 0 0" }}>
                Using <code style={{ color: "#fb923c" }}>toTextStreamResponse()</code> — tokens appear in real-time as the LLM generates them
              </p>
            </div>
          </div>

          <div style={{
            padding: "24px",
            borderRadius: "12px",
            background: "rgba(0,0,0,0.4)",
            border: "1px solid rgba(255,255,255,0.06)",
            minHeight: "200px",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
              <div style={{ width: "24px", height: "24px", borderRadius: "6px", background: "linear-gradient(135deg, #fb923c, #f472b6)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <Sparkles size={12} color="#fff" />
              </div>
              <span style={{ fontSize: "13px", fontWeight: 600, color: "#94a3b8" }}>Synapse RAG</span>
            </div>

            <div style={{ fontSize: "14px", color: "#cbd5e1", lineHeight: 1.8 }}>
              {words.map((word, i) => (
                <motion.span
                  key={i}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.5 + i * 0.04 }}
                >
                  {word === "\n\n" ? <><br /><br /></> : word + " "}
                </motion.span>
              ))}
              <motion.span
                animate={{ opacity: [1, 0, 1] }}
                transition={{ duration: 0.8, repeat: Infinity }}
                style={{ display: "inline-block", width: "2px", height: "16px", background: "#fb923c", marginLeft: "2px", verticalAlign: "text-bottom" }}
              />
            </div>
          </div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 2.5 }}
            style={{ marginTop: "16px", display: "flex", gap: "24px", justifyContent: "center" }}
          >
            {[
              { label: "Time to First Token", value: "~480ms" },
              { label: "Total Generation", value: "~1.2s" },
              { label: "Tokens Generated", value: "~85 tokens" },
            ].map((m, i) => (
              <div key={i} style={{ textAlign: "center" }}>
                <div style={{ fontSize: "10px", color: "#64748b", textTransform: "uppercase", letterSpacing: "1px" }}>{m.label}</div>
                <div style={{ fontSize: "16px", fontWeight: 700, color: "#fb923c", marginTop: "4px" }}>{m.value}</div>
              </div>
            ))}
          </motion.div>
        </GlowCard>
      </div>
    </motion.div>
  );
}

function Scene11() {
  const metrics = [
    { name: "Document Retrieval Match", value: 87.5, color: "#a78bfa", desc: "Did we retrieve the correct source document?" },
    { name: "Precision", value: 72.3, color: "#38bdf8", desc: "What fraction of retrieved content is relevant?" },
    { name: "Recall", value: 84.1, color: "#34d399", desc: "What fraction of the expected answer was retrieved?" },
    { name: "F1 Score", value: 77.7, color: "#fbbf24", desc: "Harmonic mean of Precision & Recall" },
  ];

  return (
    <motion.div variants={stagger} initial="initial" animate="animate">
      <SceneTitle title="Retrieval Evaluation (Automated)" subtitle="Step 11 · Evaluate" color="#c084fc" />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px", maxWidth: "900px", margin: "0 auto" }}>
        <GlowCard color="#c084fc">
          <IconBox icon={BarChart3} color="#c084fc" />
          <h3 style={{ fontSize: "18px", fontWeight: 600, marginTop: "16px", color: "#e2e8f0" }}>
            Quality Assurance
          </h3>
          <p style={{ fontSize: "14px", color: "#94a3b8", marginTop: "8px", lineHeight: 1.6 }}>
            Synapse RAG includes an automated evaluation pipeline that benchmarks retrieval quality against the <strong style={{ color: "#d8b4fe" }}>LegalBench-RAG</strong> ground-truth dataset.
          </p>
          <CodeBlock label="evaluate.ts">{`// Word-level overlap metric
function computeContentOverlap(
  textA: string, textB: string
): number {
  const wordsA = new Set(
    textA.toLowerCase()
      .split(/\\s+/)
      .filter(w => w.length > 2)
  );
  // Count intersection / |A|
}`}</CodeBlock>
          <div style={{ marginTop: "12px", padding: "10px 14px", borderRadius: "8px", background: "rgba(192,132,252,0.08)", fontSize: "12px", color: "#d8b4fe" }}>
            Run with: <code>npm run evaluate</code>
          </div>
        </GlowCard>

        <GlowCard color="#c084fc">
          <div style={{ fontSize: "13px", color: "#64748b", marginBottom: "20px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "1px" }}>
            Evaluation Metrics
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            {metrics.map((m, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 + i * 0.15 }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
                  <div>
                    <span style={{ fontSize: "14px", fontWeight: 600, color: "#e2e8f0" }}>
                      {m.name}
                    </span>
                    <p style={{ fontSize: "11px", color: "#64748b", margin: "2px 0 0" }}>{m.desc}</p>
                  </div>
                  <span style={{ fontSize: "18px", fontWeight: 700, color: m.color, fontFamily: "monospace" }}>
                    {m.value}%
                  </span>
                </div>
                <div style={{
                  height: "6px",
                  borderRadius: "3px",
                  background: "rgba(255,255,255,0.06)",
                  overflow: "hidden",
                }}>
                  <motion.div
                    initial={{ width: "0%" }}
                    animate={{ width: `${m.value}%` }}
                    transition={{ delay: 0.5 + i * 0.15, duration: 0.8, ease: "easeOut" }}
                    style={{
                      height: "100%",
                      borderRadius: "3px",
                      background: `linear-gradient(90deg, ${m.color}60, ${m.color})`,
                      boxShadow: `0 0 10px ${m.color}40`,
                    }}
                  />
                </div>
              </motion.div>
            ))}
          </div>
        </GlowCard>
      </div>
    </motion.div>
  );
}

function Scene12() {
  const pipelineSummary = [
    { icon: MessageSquare, label: "Query Input", time: "0ms", color: "#38bdf8" },
    { icon: Brain, label: "Embedding", time: "~120ms", color: "#a78bfa" },
    { icon: Search, label: "Vector Search", time: "~50ms", color: "#2dd4bf" },
    { icon: ScanSearch, label: "Reranking", time: "~200ms", color: "#22d3ee" },
    { icon: Layers, label: "Prompt Assembly", time: "~5ms", color: "#fbbf24" },
    { icon: Cpu, label: "LLM Inference", time: "~1200ms", color: "#f472b6" },
  ];

  return (
    <motion.div variants={stagger} initial="initial" animate="animate">
      <SceneTitle title="Complete RAG Output" subtitle="Final · Output" color="#f43f5e" />
      <div style={{ maxWidth: "900px", margin: "0 auto" }}>
        <GlowCard color="#f43f5e" style={{ padding: "32px", marginBottom: "24px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "20px" }}>
            <div style={{ width: "40px", height: "40px", borderRadius: "12px", background: "linear-gradient(135deg, #f43f5e, #a78bfa)", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <Sparkles size={20} color="#fff" />
            </div>
            <div>
              <h3 style={{ fontSize: "16px", fontWeight: 700, color: "#e2e8f0", margin: 0 }}>Grounded, Cited Response</h3>
              <p style={{ fontSize: "12px", color: "#94a3b8", margin: 0 }}>Answer is fully grounded in the retrieved documents — no hallucination</p>
            </div>
          </div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
            style={{
              padding: "20px",
              borderRadius: "12px",
              background: "rgba(0,0,0,0.3)",
              border: "1px solid rgba(255,255,255,0.08)",
              fontSize: "14px",
              color: "#cbd5e1",
              lineHeight: 1.8,
            }}
          >
            The vendor agreement contains several key termination clauses:
            <br /><br />
            <strong style={{ color: "#e2e8f0" }}>1. Termination for Convenience</strong> — Per Section 14.2, either party may terminate with 30 days&apos; written notice.
            <br /><br />
            <strong style={{ color: "#e2e8f0" }}>2. Termination for Cause</strong> — Material breach allows immediate termination by the non-breaching party.
            <br /><br />
            <strong style={{ color: "#e2e8f0" }}>3. Post-Termination Obligations</strong> — Confidential information must be returned within 10 business days, and Sections 8, 11, and 15 survive termination.
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8 }}
            style={{ display: "flex", gap: "8px", marginTop: "16px", flexWrap: "wrap" }}
          >
            {["vendor_agreement.pdf §14.2", "vendor_agreement.pdf §8.1", "service_contract.pdf §6.3"].map((cite, i) => (
              <span key={i} style={{
                padding: "4px 12px", borderRadius: "6px",
                background: "rgba(74,222,128,0.1)", border: "1px solid rgba(74,222,128,0.2)",
                fontSize: "11px", color: "#4ade80",
              }}>
                📎 {cite}
              </span>
            ))}
          </motion.div>
        </GlowCard>

        {/* End-to-end timeline */}
        <motion.div variants={fadeUp}>
          <div style={{ fontSize: "13px", color: "#64748b", marginBottom: "12px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "1px", textAlign: "center" }}>
            End-to-End Pipeline Timeline
          </div>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "4px", flexWrap: "wrap" }}>
            {pipelineSummary.map((s, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 1 + i * 0.1 }}
                style={{ display: "flex", alignItems: "center", gap: "4px" }}
              >
                <div style={{
                  padding: "6px 12px", borderRadius: "8px",
                  background: `${s.color}10`, border: `1px solid ${s.color}20`,
                  display: "flex", alignItems: "center", gap: "6px",
                  fontSize: "11px", color: s.color,
                }}>
                  <s.icon size={12} />
                  <span style={{ fontWeight: 500 }}>{s.label}</span>
                  <span style={{ fontFamily: "monospace", fontSize: "10px", opacity: 0.7 }}>{s.time}</span>
                </div>
                {i < pipelineSummary.length - 1 && (
                  <ArrowRight size={12} color="rgba(255,255,255,0.15)" />
                )}
              </motion.div>
            ))}
          </div>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 2 }}
            style={{ fontSize: "16px", fontWeight: 700, color: "#f43f5e", textAlign: "center", marginTop: "16px" }}
          >
            Total: ~1.6 seconds end-to-end ⚡
          </motion.p>
        </motion.div>
      </div>
    </motion.div>
  );
}

/* ─────────── MAIN COMPONENT ─────────── */

export default function PresentationScene({ step }: { step: number }) {
  const scenes: Record<number, React.ReactNode> = {
    0: <Scene0 />,
    1: <Scene1 />,
    2: <Scene2 />,
    3: <Scene3 />,
    4: <Scene4 />,
    5: <Scene5 />,
    6: <Scene6 />,
    7: <Scene7 />,
    8: <Scene8 />,
    9: <Scene9 />,
    10: <Scene10 />,
    11: <Scene11 />,
    12: <Scene12 />,
  };

  return <>{scenes[step] ?? scenes[0]}</>;
}
