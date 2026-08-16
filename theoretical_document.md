# Synapse RAG: A Complete Theoretical Explanation

> [!NOTE]
> **V2 Architecture Update**: Since this document was originally drafted, the Synapse RAG system has been upgraded with several key features:
> - **Vector Database Migration**: The system now utilizes **Qdrant** (`qdrant_client`) for vector storage and similarity search instead of pgvector, providing enhanced local performance and Python-native integration via LangChain.
> - **Advanced User Interface**: A new Glassmorphism dark-mode UI built with Tailwind CSS v4 and Framer Motion.
> - **Real-Time Streaming & Citations**: Utilizing the Vercel AI SDK Data Stream Protocol, the LLM (`llama-3.1-8b-instant` via Groq) now streams answers in real-time, accompanied by verifiable, interactive citation panels.
> 
> *The fundamental theoretical concepts regarding embeddings, HNSW, bi-encoders, and cross-encoders described below remain fully applicable to the new architecture.*

## How Every Process Works — From Raw Contracts to Grounded Legal Answers

---

## Table of Contents

1. [The Core Problem: Why Do We Need This System?](#1-the-core-problem)
2. [What is Retrieval-Augmented Generation (RAG)?](#2-what-is-rag)
3. [The Dataset: LegalBench-RAG](#3-the-dataset)
4. [Phase 1 — Data Ingestion Pipeline](#4-data-ingestion-pipeline)
   - 4.1 [Downloading the Dataset](#41-downloading-the-dataset)
   - 4.2 [Document Storage in PostgreSQL](#42-document-storage)
   - 4.3 [Text Chunking](#43-text-chunking)
   - 4.4 [Tokenization: Converting Text to Numbers](#44-tokenization)
   - 4.5 [Generating Vector Embeddings](#45-generating-embeddings)
   - 4.6 [Storing Vectors with pgvector](#46-storing-vectors)
   - 4.7 [Building the HNSW Index](#47-hnsw-index)
5. [Phase 2 — Query Processing and Retrieval](#5-query-processing)
   - 5.1 [How a User Query is Processed](#51-query-processing)
   - 5.2 [Dense Vector Retrieval (Bi-Encoder)](#52-dense-retrieval)
   - 5.3 [Cosine Similarity: Measuring Relevance](#53-cosine-similarity)
   - 5.4 [Re-Ranking with the Cross-Encoder](#54-cross-encoder-reranking)
   - 5.5 [The Two-Stage Pipeline: Why Both Models?](#55-two-stage-pipeline)
6. [Phase 3 — Answer Generation](#6-answer-generation)
   - 6.1 [Context Assembly](#61-context-assembly)
   - 6.2 [The System Prompt: Constraining the LLM](#62-system-prompt)
   - 6.3 [Streaming the Response](#63-streaming)
7. [Query Routing: Single vs. Multi-Document](#7-query-routing)
   - 7.1 [Single-Document Mode](#71-single-document)
   - 7.2 [Multi-Document Fan-Out Mode](#72-fan-out)
   - 7.3 [Why We Never Blend Chunks Across Documents](#73-why-no-blending)
8. [How the Deep Learning Models Work](#8-deep-learning-models)
   - 8.1 [What is a Transformer Encoder?](#81-transformer-encoder)
   - 8.2 [The Input Embedding Layer](#82-input-embedding)
   - 8.3 [Self-Attention: How Tokens Understand Context](#83-self-attention)
   - 8.4 [The Feed-Forward Network](#84-feed-forward)
   - 8.5 [How the Bi-Encoder Produces Sentence Embeddings](#85-bi-encoder)
   - 8.6 [How the Cross-Encoder Scores Relevance](#86-cross-encoder)
   - 8.7 [Knowledge Distillation: Making Models Smaller and Faster](#87-distillation)
   - 8.8 [ONNX Runtime: Running Models in the Browser](#88-onnx-runtime)
9. [The Database Architecture](#9-database-architecture)
   - 9.1 [Prisma ORM and Schema Design](#91-prisma)
   - 9.2 [The Neon Serverless Driver](#92-neon-driver)
   - 9.3 [The Vector Column and Index](#93-vector-column)
10. [Evaluation: How We Measure Success](#10-evaluation)
    - 10.1 [Document Retrieval Match (DRM)](#101-drm)
    - 10.2 [Precision, Recall, and F1](#102-precision-recall-f1)
    - 10.3 [What the Results Tell Us](#103-results-analysis)
11. [The User Interface](#11-user-interface)
    - 11.1 [Three-Panel Layout](#111-layout)
    - 11.2 [Real-Time Streaming](#112-streaming)
    - 11.3 [Citation Panel: Provenance and Trust](#113-citations)
12. [The Singleton Pattern: Efficient Model Loading](#12-singleton-pattern)
13. [End-to-End Walkthrough: A Complete Query Journey](#13-end-to-end)
14. [Limitations and Future Directions](#14-limitations)

---

## 1. The Core Problem: Why Do We Need This System? <a name="1-the-core-problem"></a>

Large Language Models (LLMs) like GPT-4, LLaMA, and Gemini have transformed natural language processing. They can summarize documents, translate languages, write code, and answer complex questions. However, they suffer from a fundamental problem called **hallucination**: the tendency to generate text that sounds correct and confident but is factually wrong.

In everyday conversation, hallucination might be a minor inconvenience. In the legal domain, it is catastrophic. A lawyer asking an LLM about the termination clause of a specific contract needs the exact text from that exact contract — not a plausible-sounding fabrication. If the LLM invents a clause that does not exist, or misrepresents the terms, the consequences could include financial penalties, breached agreements, or malpractice liability.

**The core problem Synapse RAG solves**: How do we get the power of LLMs to answer natural language questions about legal contracts while guaranteeing that every answer is grounded in actual contract text?

The answer is a technique called **Retrieval-Augmented Generation (RAG)**.

---

## 2. What is Retrieval-Augmented Generation (RAG)? <a name="2-what-is-rag"></a>

RAG is a technique that separates the "knowing" from the "generating." Instead of relying on an LLM's internal (and potentially outdated or incorrect) knowledge, we first **retrieve** the relevant information from a trusted database, and then ask the LLM to **generate** an answer using only that retrieved information.

Think of it like this: instead of asking a person to answer a question from memory (where they might misremember), you hand them the exact pages of a textbook and say, "Answer this question using only what's written here."

**The RAG process has three fundamental phases:**

### Phase 1: Ingestion (Offline, Done Once)
Before anyone asks a question, we must prepare the contract documents:
1. Take each legal contract and break it into smaller pieces called **chunks**.
2. Convert each chunk into a mathematical representation called a **vector embedding** — a list of 384 numbers that captures the meaning of the text.
3. Store these embeddings in a database with a special search index.

### Phase 2: Retrieval (Online, Per Query)
When a user asks a question:
1. Convert their question into the same kind of vector embedding.
2. Search the database to find the chunks whose embeddings are most similar to the question's embedding.
3. Re-rank the results using a more accurate (but slower) model.

### Phase 3: Generation (Online, Per Query)
1. Take the top retrieved chunks and insert them into the LLM's prompt as "context."
2. Instruct the LLM to answer the user's question using only the provided context.
3. Stream the answer back to the user in real-time.

The crucial insight is that the LLM never needs to "remember" the contracts. It only needs to read the relevant passages we hand it and generate a coherent answer from them. This eliminates hallucination at its root.

---

## 3. The Dataset: LegalBench-RAG <a name="3-the-dataset"></a>

To build and evaluate our system, we use the **LegalBench-RAG** dataset. This is a curated collection of legal question-answer pairs drawn from real contracts, where legal experts have manually identified the exact text spans that answer each question.

### What makes LegalBench-RAG special?

Unlike typical QA datasets that only tell you whether an answer is correct or not, LegalBench-RAG provides **exact character offsets** — the precise starting and ending character positions within the source document where the answer can be found. This allows us to measure retrieval accuracy at a granular level.

### The Four Sub-Corpora

The dataset spans four distinct areas of legal practice:

| Sub-Corpus | Domain | What It Contains |
|---|---|---|
| **CUAD** | Commercial contracts | Clause type extraction — identifying specific clause types like indemnification, termination, and non-compete |
| **ContractNLI** | Non-Disclosure Agreements | Hypothesis verification — determining if a given statement is supported by the NDA text |
| **MAUD** | Merger & Acquisition | Deal term analysis — extracting terms related to merger considerations, closing conditions, and representations |
| **PrivacyQA** | Privacy policies | Policy clause retrieval — finding specific privacy-related provisions |

### How the dataset is downloaded

A Python script uses the HuggingFace `datasets` library to download the dataset from the `amentaphd/legalbench-qa` repository. We select the first 50 question-answer pairs for our evaluation prototype. Each pair includes the question text, the expected answer text, the source document filename, and the exact character span locations within that document.

The raw data is saved as a JSON file (`data/legalbench.json`) that our ingestion pipeline reads.

---

## 4. Phase 1 — Data Ingestion Pipeline <a name="4-data-ingestion-pipeline"></a>

The ingestion pipeline transforms raw legal contracts into searchable vector representations. This is done once, before any user queries the system.

### 4.1 Downloading the Dataset <a name="41-downloading-the-dataset"></a>

The process begins with a Python script (`download_legalbench.py`) that:

1. Connects to the HuggingFace Hub, a public repository of machine learning datasets.
2. Downloads the `amentaphd/legalbench-qa` dataset, which contains legal QA pairs with ground truth answer spans.
3. Extracts 50 question-answer pairs along with their metadata (source corpus, file path, character spans).
4. Saves everything as a structured JSON file in the `data/` directory.

Each entry in the JSON file looks conceptually like this:
- **id**: A unique identifier for the QA pair
- **text**: The question text
- **metadata**: Contains the query, the expected answer, the source corpus file, and an array of snippets with exact file paths and character-span positions

### 4.2 Document Storage in PostgreSQL <a name="42-document-storage"></a>

Each unique contract file mentioned in the dataset is registered as a **Document** record in a PostgreSQL database. The database schema (defined using Prisma ORM) stores:

- **Document**: The contract file — its filename, a human-readable title, the source corpus it came from (CUAD, MAUD, etc.), and the creation timestamp.
- **Chunk**: A piece of the document — the actual text content, a vector embedding (384 numbers), the character start and end positions within the original document, and a foreign key linking it back to its parent Document.

The relationship is hierarchical: one Document has many Chunks. When a Document is deleted, all its Chunks are automatically deleted as well (cascade delete).

### 4.3 Text Chunking <a name="43-text-chunking"></a>

Legal contracts are typically long documents — often hundreds or thousands of words. Embedding an entire contract into a single vector would lose the fine-grained details that make specific clauses retrievable. Instead, we break each contract into smaller pieces called **chunks**.

**Why chunking matters**: If a user asks "What is the termination clause?", we want to retrieve just the termination clause — not the entire 50-page contract. Smaller chunks mean more precise retrieval.

**The chunking strategy used in Synapse RAG**:

In our ingestion pipeline, each unique answer snippet from the LegalBench-RAG dataset is treated as an individual chunk. This means each chunk corresponds to a specific passage that a legal expert identified as the answer to a question. This approach ensures that each chunk contains a semantically coherent legal concept.

Each chunk is stored with:
- The raw text content
- Its parent document ID
- A 384-dimensional vector embedding (computed in the next step)
- Character offset positions

### 4.4 Tokenization: Converting Text to Numbers <a name="44-tokenization"></a>

Before a Transformer model can process text, the text must be converted into numbers. This is done through **tokenization**, and specifically, Synapse RAG uses the **WordPiece** algorithm.

#### What is WordPiece?

WordPiece is a subword tokenization algorithm. Instead of treating each word as a single token (which would require an impossibly large vocabulary to cover every possible word) or each character as a token (which would lose word-level meaning), WordPiece splits text into meaningful subword units.

#### How it works, step by step:

1. **Lowercasing**: The input text is converted to lowercase (for uncased models). "The Seller" becomes "the seller".

2. **Whitespace splitting**: The text is split on spaces into individual words.

3. **Greedy longest-match**: Each word is compared against a vocabulary of 30,522 known tokens. The algorithm tries to match the longest possible token from left to right. If the entire word is in the vocabulary, it becomes a single token. If not, it finds the longest prefix that is in the vocabulary, outputs that as a token, and continues with the remainder.

4. **Continuation markers**: When a word is split into multiple tokens, all tokens except the first are prefixed with `##` to indicate they are continuations. For example:
   - "indemnification" → `in`, `##dem`, `##ni`, `##fi`, `##cation` (5 subword tokens)
   - "seller" → `seller` (1 token — it exists in the vocabulary)
   - "majeure" → `maj`, `##eure` (2 subword tokens)

5. **Special tokens**: Two special tokens are always added:
   - `[CLS]` (classification) is prepended at the very beginning. This token's output representation is used as a summary of the entire sequence.
   - `[SEP]` (separator) is appended at the end. It marks the boundary of the input sequence.

So the final tokenized sequence for "The Seller shall indemnify" would be:
```
[CLS] the seller shall in ##dem ##ni ##fy [SEP]
```

6. **ID mapping**: Each token is mapped to its integer ID in the vocabulary table. The model never sees text — it only sees sequences of integers.

#### Why WordPiece is effective for legal text:

Legal documents contain many specialized terms that may not appear in general-purpose vocabularies. WordPiece handles these gracefully by decomposing them into known subwords. The model can learn the meaning of unfamiliar compound words from the meanings of their constituent subwords — much like how a human can understand "indemnification" by recognizing the root "indemn" (to secure against loss).

### 4.5 Generating Vector Embeddings <a name="45-generating-embeddings"></a>

This is the heart of the ingestion process. Each text chunk is converted into a **vector embedding** — a list of 384 floating-point numbers that represents the semantic meaning of the text.

#### What is a vector embedding?

Imagine a 384-dimensional space (impossible to visualize, but mathematically straightforward). Each dimension represents some learned aspect of meaning. When we embed a piece of text, we place it at a specific point in this space. The key property is that **texts with similar meanings end up close together**, and **texts with different meanings end up far apart**.

For example:
- "The Seller shall indemnify the Buyer" and "The Vendor warrants to compensate the Purchaser" would be placed near each other because they express similar legal concepts.
- "The weather in Paris is sunny" would be placed far from both because it has nothing to do with indemnification.

#### How the embedding is generated:

1. The text chunk is tokenized using WordPiece (as described above).
2. The token IDs are fed into the Transformer encoder model (`all-MiniLM-L6-v2`).
3. The model processes the tokens through 6 layers of self-attention and feed-forward transformations.
4. The model outputs a 384-dimensional vector for every token in the input.
5. **Mean pooling** is applied: all token vectors are averaged (element-wise) to produce a single 384-dimensional vector that represents the entire chunk.
6. **L2 normalization** is applied: the vector is scaled so its magnitude (length) is exactly 1.0. This places it on the surface of a unit hypersphere, which makes cosine similarity computations more efficient.

The result is a single array of 384 numbers like `[0.0234, -0.1567, 0.0891, ..., 0.0445]`, where each number ranges roughly from -0.2 to +0.2 and the entire vector has a magnitude of exactly 1.0.

#### The Singleton Pattern for model loading:

Loading a Transformer model into memory takes several seconds and consumes significant RAM (~80MB). If we loaded a new model instance for every chunk we need to embed, the ingestion process would be extraordinarily slow. Instead, we use the **Singleton Pattern**: the model is loaded exactly once (on the first embedding request), and all subsequent requests reuse that same model instance. This is implemented by storing the model pipeline in a class-level static variable.

### 4.6 Storing Vectors with pgvector <a name="46-storing-vectors"></a>

Once the embedding is generated, it needs to be stored in the database alongside the chunk text. Standard SQL databases do not natively support vector data types or similarity searches. This is where **pgvector** comes in.

#### What is pgvector?

pgvector is an extension for PostgreSQL that adds:
1. A `vector` data type — a column that can store arrays of floating-point numbers with a fixed dimension.
2. Distance operators — special SQL operators for computing distances between vectors:
   - `<=>` for cosine distance
   - `<->` for Euclidean (L2) distance
   - `<#>` for inner product distance
3. Index types optimized for approximate nearest-neighbor search.

In our schema, the `Chunk` table has an `embedding` column of type `vector(384)` — a column that stores exactly 384 floating-point numbers per row. The embedding is stored as a JSON string that PostgreSQL automatically casts to the vector type.

### 4.7 Building the HNSW Index <a name="47-hnsw-index"></a>

To search through embeddings quickly, we need an index. The naive approach — computing the cosine similarity between the query embedding and every single chunk embedding in the database — works but becomes slow as the number of chunks grows. For 1,000 chunks, it's fine. For 1,000,000 chunks, it's unacceptably slow.

#### What is HNSW?

HNSW stands for **Hierarchical Navigable Small World**. It is a graph-based data structure for approximate nearest-neighbor search. Instead of comparing the query against every vector in the database, HNSW navigates a multi-layered graph to quickly find the most similar vectors.

#### How HNSW works, conceptually:

Think of it like searching for a specific building in a city:

1. **Top layer** (satellite view): A sparse map with only major landmarks. You quickly identify the general neighborhood.
2. **Middle layers** (street map): More detail. You navigate closer to the right block.
3. **Bottom layer** (walking around): Every building is visible. You find the exact one you're looking for.

In HNSW:
- Each layer is a graph where nodes are embedding vectors and edges connect similar vectors.
- The top layers are sparse (few nodes, long-range connections), allowing fast coarse navigation.
- The bottom layer is dense (all nodes), allowing precise final selection.
- A search starts at the top layer, greedily follows edges toward the nearest neighbor, then descends to the next layer and repeats, until it reaches the bottom layer where it performs a thorough local search.

#### Why "approximate"?

HNSW does not guarantee finding the absolute closest vector — it finds a very close approximation in much less time. For our use case, this trade-off is excellent: the top 5 results from HNSW are virtually always the same as the top 5 from an exhaustive search, but they are found in milliseconds instead of seconds.

---

## 5. Phase 2 — Query Processing and Retrieval <a name="5-query-processing"></a>

When a user types a question, the system must find the most relevant contract passages. This happens in two stages: fast approximate retrieval followed by precise re-ranking.

### 5.1 How a User Query is Processed <a name="51-query-processing"></a>

When the user submits a question through the chat interface:

1. The React frontend sends an HTTP POST request to the `/api/chat` endpoint.
2. The server extracts the latest user message from the conversation history.
3. The server determines the retrieval mode:
   - If the user has selected a specific document in the sidebar → **single-document mode**.
   - If no document is selected → **multi-document fan-out mode**.
4. The server calls the retrieval function with the query text and mode parameters.

### 5.2 Dense Vector Retrieval (Bi-Encoder) <a name="52-dense-retrieval"></a>

The first retrieval stage uses the **bi-encoder** model (`all-MiniLM-L6-v2`) to find the most relevant chunks.

#### What is a bi-encoder?

The bi-encoder architecture processes the query and each document chunk **independently** through the same Transformer model. "Bi" refers to the two separate encoding paths — one for the query, one for the document.

#### Why "independently" matters:

Because the document chunks are embedded during ingestion (offline, ahead of time), we do not need to re-process them when a query arrives. When the user asks a question, we only need to embed the query itself (a single forward pass through the model, taking about 5 milliseconds). We then compare this query embedding against the pre-computed chunk embeddings in the database using a simple mathematical operation (cosine similarity).

This is the fundamental speed advantage of the bi-encoder: it decouples query processing from document processing.

#### The retrieval SQL query:

The actual retrieval is performed by a SQL query that uses pgvector's `<=>` (cosine distance) operator:

```sql
SELECT c.id, c.content, c.document_id,
       1 - (c.embedding <=> query_embedding::vector) as similarity
FROM "Chunk" c
JOIN "Document" d ON d.id = c.document_id
WHERE c.embedding IS NOT NULL
ORDER BY c.embedding <=> query_embedding::vector
LIMIT 5
```

This query:
1. Computes the cosine distance between the query embedding and every chunk embedding.
2. Converts distance to similarity (1 − distance, so higher = more similar).
3. Sorts by distance (ascending, so the most similar chunks come first).
4. Returns the top 5 results.

If a specific document is selected, a `WHERE` clause is added to filter chunks by document ID.

### 5.3 Cosine Similarity: Measuring Relevance <a name="53-cosine-similarity"></a>

Cosine similarity measures the angle between two vectors, ignoring their magnitude. Two vectors pointing in the same direction have a cosine similarity of 1.0 (identical meaning). Two vectors pointing in opposite directions have a cosine similarity of -1.0 (opposite meaning). Two perpendicular vectors have a cosine similarity of 0.0 (unrelated meaning).

#### Why cosine similarity and not Euclidean distance?

In high-dimensional spaces, cosine similarity is more robust than Euclidean distance. Euclidean distance is affected by the magnitude (length) of vectors — two vectors with similar directions but different magnitudes would appear "far apart." Cosine similarity only considers the direction, which better captures semantic similarity.

Since we L2-normalize all embeddings (making their magnitude exactly 1.0), cosine similarity reduces to a simple dot product — the fastest possible comparison operation.

### 5.4 Re-Ranking with the Cross-Encoder <a name="54-cross-encoder-reranking"></a>

The bi-encoder retrieval is fast but has a limitation: because the query and document are processed independently, the model cannot capture fine-grained interactions between specific words in the query and specific words in the document. The cross-encoder fixes this.

#### What is a cross-encoder?

The cross-encoder (`ms-marco-MiniLM-L-6-v2`) takes the query and a document chunk as a single concatenated input:

```
[CLS] What is the termination clause? [SEP] The Buyer may terminate this Agreement upon... [SEP]
```

Both texts are processed together through the Transformer, meaning every self-attention layer can compute attention weights between query tokens and document tokens. The query word "termination" can directly attend to the document word "terminate," allowing the model to recognize this fine-grained lexical and semantic match.

#### How it scores relevance:

After the Transformer processes the concatenated input, the `[CLS]` token's output vector (384 dimensions) is fed into a linear classification head — a single layer that maps 384 dimensions down to 1 number. This number is a **relevance logit**: a higher score means more relevant, a lower score means less relevant.

#### How re-ranking works:

1. The bi-encoder retrieves the top 20 candidate chunks (4× more than the final 5 we need).
2. The cross-encoder scores each of these 20 chunks individually against the query.
3. The chunks are re-sorted by their cross-encoder scores (descending).
4. The top 5 re-ranked chunks are returned as the final result.

#### Why over-retrieve?

We retrieve 20 candidates but only return 5. This "over-retrieval" ensures that if the bi-encoder ranked a truly relevant chunk at position #15 (because its embedding wasn't perfectly aligned with the query), the cross-encoder has a chance to promote it to the top 5 based on deeper semantic analysis.

### 5.5 The Two-Stage Pipeline: Why Both Models? <a name="55-two-stage-pipeline"></a>

This is a natural question: if the cross-encoder is more accurate, why not just use it for everything?

The answer is computational cost:

| Property | Bi-Encoder | Cross-Encoder |
|---|---|---|
| How it works | Embeds query once, compares against pre-computed embeddings | Must process query+chunk together for each chunk |
| Speed per chunk | ~0.01ms (just a dot product) | ~5ms (full Transformer forward pass) |
| To search 1,000 chunks | ~10ms | ~5,000ms (5 seconds) |
| To search 1,000,000 chunks | ~10 seconds | ~83 minutes |
| Accuracy | Good (but no cross-attention) | Excellent (full cross-attention) |

The two-stage pipeline gives us the best of both worlds:
- **Stage 1** (Bi-Encoder): Cast a wide net cheaply. Narrow 1,000 chunks down to 20 candidates in ~5ms.
- **Stage 2** (Cross-Encoder): Apply expensive but precise analysis to just 20 candidates in ~100ms.

Total: ~105ms for state-of-the-art retrieval accuracy. Using the cross-encoder alone on 1,000 chunks would take 5 seconds — 50× slower.

---

## 6. Phase 3 — Answer Generation <a name="6-answer-generation"></a>

Once the most relevant chunks are retrieved and re-ranked, they are used to generate a grounded answer.

### 6.1 Context Assembly <a name="61-context-assembly"></a>

The retrieved chunks are formatted into a context string that the LLM can read. Each chunk is presented with metadata:

```
[Source: contract_name.txt | Relevance: 87.3%]
The Seller hereby represents and warrants that...

---

[Source: contract_name.txt | Relevance: 82.1%]
In the event of a material breach...
```

This formatting serves two purposes:
1. It tells the LLM which document each passage came from.
2. The relevance score gives the LLM a hint about which passages to prioritize.

### 6.2 The System Prompt: Constraining the LLM <a name="62-system-prompt"></a>

The LLM (Groq's LLaMA 3.1 8B Instant model) receives a carefully crafted system prompt with strict rules:

1. **Answer ONLY using information found in the Context.** The LLM must not draw on its pre-trained knowledge. If the answer isn't in the retrieved passages, it must say so.
2. **If the Context does not contain the information needed, explicitly refuse.** The model must respond with "I cannot answer this question based on the provided contract excerpts" rather than guessing.
3. **Quote exact clauses or passages when relevant.** The LLM should use quotation marks around text it pulls directly from the context.
4. **Mention the source document name.** This provides traceability.
5. **For multi-document queries, structure the response by document.** This ensures clarity when comparing across contracts.
6. **Use professional legal analysis tone.** The responses should read like a legal memo, not a casual conversation.

These constraints are what make the system "grounded." The LLM becomes a sophisticated reading comprehension engine that synthesizes information from the provided passages rather than generating from memory.

### 6.3 Streaming the Response <a name="63-streaming"></a>

Rather than waiting for the entire answer to be generated before showing it to the user, the system uses **streaming**. As the LLM generates each word, it is immediately sent to the frontend and displayed. This creates a responsive, "typing" experience where the user can begin reading the answer while it's still being generated.

The streaming is implemented using the Vercel AI SDK (`ai` package), which provides:
- Server-side: `streamText()` function that creates a streaming response.
- Client-side: `useChat()` hook that receives and renders the streamed text in real-time.

The response is sent as a standard HTTP response with chunked transfer encoding. Citations (the source chunks with their similarity scores) are attached as HTTP headers so the frontend can display them in a side panel.

---

## 7. Query Routing: Single vs. Multi-Document <a name="7-query-routing"></a>

Synapse RAG supports two distinct retrieval modes, chosen based on whether the user has selected a specific document.

### 7.1 Single-Document Mode <a name="71-single-document"></a>

When the user clicks a specific document in the sidebar:

- **Use case**: "What is the termination clause in this agreement?"
- **Behavior**: The retrieval SQL query includes a `WHERE document_id = ?` filter, constraining the search to chunks from only the selected document.
- **Result**: All retrieved chunks come from the same contract, ensuring the answer is scoped to that specific agreement.

This mode is ideal for deep analysis of a single contract.

### 7.2 Multi-Document Fan-Out Mode <a name="72-fan-out"></a>

When no document is selected ("All Documents" mode):

- **Use case**: "Which of these agreements have an uncapped liability clause?"
- **Behavior**: The system performs a **fan-out** retrieval:

  1. **Find the top documents**: Search across all chunks to identify which documents contain the most relevant passages. This is done by finding the best-matching chunk per document and sorting documents by their best chunk's similarity score.
  
  2. **Select the top 3 documents**: Take the 3 documents whose best chunks scored highest.
  
  3. **Retrieve per document**: For each of these 3 documents, independently retrieve the top 3 most relevant chunks.
  
  4. **Format by document**: The context is organized with clear document headers:
     ```
     === Document: merger_agreement.txt (MAUD) ===
     [Snippet 1 | Relevance: 91.2%]
     ...
     
     === Document: nda_2023.txt (ContractNLI) ===
     [Snippet 1 | Relevance: 85.7%]
     ...
     ```

### 7.3 Why We Never Blend Chunks Across Documents <a name="73-why-no-blending"></a>

A simpler approach would be to search globally, take the top 9 chunks regardless of which document they come from, and feed them all to the LLM as a single undifferentiated context. We deliberately avoid this because:

1. **Cross-contamination risk**: The LLM might combine clauses from different contracts, producing an answer that does not accurately represent any single agreement.
2. **Document dominance**: A long document with many chunks might dominate the results, crowding out shorter but equally relevant documents.
3. **Provenance clarity**: When chunks are grouped by document, the LLM can provide structured answers ("In Contract A... but in Contract B...") rather than a confusing blend.

The fan-out approach ensures each document is analyzed independently and the results are presented with clear attribution.

---

## 8. How the Deep Learning Models Work <a name="8-deep-learning-models"></a>

Both the bi-encoder and cross-encoder are based on the **Transformer encoder** architecture. This section explains how these models work at a conceptual level.

### 8.1 What is a Transformer Encoder? <a name="81-transformer-encoder"></a>

A Transformer encoder is a neural network architecture designed to convert a sequence of tokens (words or subwords) into a sequence of context-aware vector representations. "Context-aware" means that the representation of each token is influenced by every other token in the sequence.

For example, the word "bank" in "I went to the bank to deposit money" should have a different representation than "bank" in "I sat on the bank of the river." The Transformer encoder achieves this through its **self-attention mechanism**, which allows each token to "look at" every other token and adjust its own representation accordingly.

#### The overall structure of our Transformer (all-MiniLM-L6-v2):

1. **Input Embedding Layer**: Converts token IDs into 384-dimensional vectors.
2. **6 Transformer Blocks**: Each block refines the representations through self-attention and feed-forward processing.
3. **Pooling Layer**: Reduces the sequence of per-token vectors into a single vector.
4. **Normalization**: Scales the vector to unit length.

The model contains approximately 22.7 million parameters (learned numerical values) and occupies about 80MB of disk space.

### 8.2 The Input Embedding Layer <a name="82-input-embedding"></a>

The first step is converting each token ID into a meaningful vector. This is done by summing three separate embeddings:

#### Token Embedding
A lookup table with 30,522 rows (one per vocabulary word) and 384 columns. When we look up token ID 2003 (say, "the"), we get a 384-dimensional vector that represents the inherent meaning of "the" regardless of context. This is learned during training.

#### Position Embedding
A separate lookup table with 512 rows (one per possible position, up to position 512) and 384 columns. Position 0 gets one vector, position 1 gets a different vector, and so on. This tells the model where each token appears in the sequence, which is critical because the self-attention mechanism is otherwise order-agnostic — it treats the input as a set, not a sequence.

#### Segment Embedding
A tiny table with just 2 rows and 384 columns (Segment A and Segment B). In the bi-encoder, all tokens are Segment A. In the cross-encoder, query tokens are Segment A and document tokens are Segment B, allowing the model to distinguish which part of the input is the query and which is the passage.

These three vectors are added element-wise to produce the initial representation for each token. The result then passes through Layer Normalization (which stabilizes the values) and Dropout (which randomly zeros out some values during training to prevent overfitting).

### 8.3 Self-Attention: How Tokens Understand Context <a name="83-self-attention"></a>

Self-attention is the mechanism that gives Transformers their power. It allows each token to query every other token in the sequence and selectively aggregate information from them.

#### The intuition:

Imagine you're reading a legal clause: "The **Seller** hereby represents and warrants that **they** have full authority to enter into this Agreement."

To understand what "they" refers to, you need to look back at "Seller." Self-attention does exactly this — it computes how strongly each token should attend to every other token, then updates each token's representation based on a weighted combination of all tokens it attends to.

#### How it works:

Each token's vector is projected into three roles:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I have to offer?"
- **Value (V)**: "What information do I carry?"

The attention score between token A and token B is the dot product of A's Query with B's Key. High dot products mean high attention. These scores are normalized using softmax (converting them into probabilities that sum to 1), and the output for each token is a weighted sum of all tokens' Values, weighted by the attention probabilities.

#### Multi-head attention:

Rather than computing a single attention pattern, the model uses **12 parallel attention heads**. Each head independently learns to focus on different types of relationships:
- One head might learn syntactic relationships (subject-verb agreement).
- Another might learn semantic relationships (coreference — what "they" refers to).
- Another might learn positional patterns (nearby tokens).

The 384-dimensional space is split into 12 subspaces of 32 dimensions each. Each head operates in its own 32-dimensional subspace, and the results are concatenated back to 384 dimensions.

#### Residual connections:

After the attention computation, the output is added back to the original input (before attention). This "residual connection" prevents the gradients from vanishing during training of deep networks and allows the model to learn incremental refinements rather than complete transformations.

### 8.4 The Feed-Forward Network <a name="84-feed-forward"></a>

After self-attention, each token's representation passes through a two-layer feed-forward network (FFN):

1. **Expansion**: The 384-dimensional vector is projected to 1,536 dimensions (4× expansion).
2. **GELU Activation**: A non-linear activation function is applied. GELU (Gaussian Error Linear Unit) is similar to ReLU but smoother — it doesn't have a hard cutoff at zero, which provides better gradient flow.
3. **Projection**: The 1,536-dimensional vector is projected back to 384 dimensions.

The FFN acts as a "processing step" where the model can perform non-linear transformations on each token independently. While self-attention handles inter-token relationships (which tokens attend to which), the FFN handles intra-token processing (how to transform the information gathered by attention).

Another residual connection and Layer Normalization follow the FFN, completing one Transformer block. This block is repeated 6 times, each time further refining the representations.

### 8.5 How the Bi-Encoder Produces Sentence Embeddings <a name="85-bi-encoder"></a>

After the 6 Transformer blocks have processed the input, we have a 384-dimensional vector for each token. But we need a single vector for the entire text. This is where **pooling** comes in.

#### Mean Pooling

The bi-encoder uses **mean pooling**: it computes the element-wise average of all token vectors (excluding padding tokens). If the input has 15 tokens, the model outputs 15 vectors of 384 dimensions each, and mean pooling averages them into 1 vector of 384 dimensions.

Why mean pooling instead of just using the `[CLS]` token? Our experiments show that the `[CLS]` token and mean-pooled representations have a cosine similarity of only 0.405 — meaning they encode very different aspects of the input. Mean pooling captures distributed information across the entire sequence, while `[CLS]` concentrates on what the token learned during pre-training (which was optimized for next-sentence prediction, not sentence similarity). Research has consistently shown that mean pooling outperforms `[CLS]` pooling for sentence similarity tasks.

#### L2 Normalization

The mean-pooled vector is then divided by its L2 norm (Euclidean length), making its magnitude exactly 1.0. This places the embedding on the surface of a 384-dimensional unit hypersphere. After normalization, cosine similarity between any two embeddings reduces to a simple dot product, which is the fastest possible comparison operation.

### 8.6 How the Cross-Encoder Scores Relevance <a name="86-cross-encoder"></a>

The cross-encoder has the same Transformer architecture as the bi-encoder (6 layers, 384 dimensions, 12 heads) but is used differently:

1. **Input**: The query and document are concatenated with special tokens:
   `[CLS] query tokens [SEP] document tokens [SEP]`

2. **Segment embeddings**: Query tokens get Segment A (0), document tokens get Segment B (1). This tells the model which tokens belong to the query and which belong to the document.

3. **Full cross-attention**: Because both texts are in the same sequence, every self-attention layer can compute attention between query tokens and document tokens. The query token "termination" can directly attend to the document token "terminate" — a word-level interaction that the bi-encoder cannot perform.

4. **Classification head**: After the Transformer processes the concatenated input, only the `[CLS]` token's output is used. It passes through a linear layer (384 → 1) that produces a single number: the relevance logit. Higher numbers mean more relevant.

The cross-encoder was fine-tuned on the MS MARCO dataset — approximately 500,000 real Bing search queries paired with relevant and irrelevant passages. This training teaches the model to distinguish between passages that actually answer a question and passages that merely share some vocabulary with the question.

### 8.7 Knowledge Distillation: Making Models Smaller and Faster <a name="87-distillation"></a>

Both the bi-encoder and cross-encoder are **distilled** models. Distillation is a technique where a large, accurate model (the "teacher") trains a smaller, faster model (the "student") to mimic its behavior.

#### The teacher-student relationship:

- **Teacher**: BERT-base (110 million parameters, 12 layers, 768-dimensional hidden size). This model is highly accurate but slow.
- **Student**: MiniLM-L6-v2 (22.7 million parameters, 6 layers, 384-dimensional hidden size). This model is 5× faster and 5× smaller, while retaining approximately 95% of the teacher's accuracy.

#### How distillation works:

During training, the student model processes the same inputs as the teacher and is trained to match two things:

1. **Attention distributions**: The student learns to produce the same attention patterns as the teacher. If the teacher's attention head #3 focuses heavily on the subject of a sentence, the student is trained to do the same.

2. **Value-relation matrices**: The student learns to produce the same relationships between value representations. This captures not just where the model attends, but what information it extracts.

The key insight is that the teacher's learned behaviors contain more information than the original training data. By learning from the teacher's internal representations (not just the final outputs), the student acquires knowledge that would require much more training data to learn from scratch.

#### Why this matters for our system:

A 110M parameter model would take ~200ms per embedding and use ~420MB of memory. The distilled 22.7M parameter model takes ~40ms per embedding and uses ~80MB of memory. For a real-time legal Q&A system, this difference between responsive and sluggish.

### 8.8 ONNX Runtime: Running Models in the Browser and Server <a name="88-onnx-runtime"></a>

Our models run through the `@xenova/transformers` library, which uses **ONNX Runtime** as its backend.

#### What is ONNX?

ONNX (Open Neural Network Exchange) is a standardized format for representing machine learning models. A model trained in PyTorch (Python) can be exported to ONNX format and then executed in any language or environment that has an ONNX Runtime implementation — including JavaScript/Node.js.

#### Why this matters:

Without ONNX, we would need to run a separate Python server just to generate embeddings, adding complexity, latency, and deployment overhead. With ONNX Runtime in Node.js, the embedding model runs directly inside our Next.js server process. No Python. No separate microservice. No network calls for embeddings.

The `@xenova/transformers` library automatically:
1. Downloads the ONNX-format model files from HuggingFace on first use.
2. Caches them locally for subsequent runs.
3. Loads them into memory using the ONNX Runtime.
4. Provides a high-level API (`pipeline('feature-extraction', model_name)`) that handles tokenization, inference, and output parsing.

---

## 9. The Database Architecture <a name="9-database-architecture"></a>

### 9.1 Prisma ORM and Schema Design <a name="91-prisma"></a>

The database schema is defined using **Prisma**, an Object-Relational Mapping (ORM) tool that provides type-safe database access.

#### The data models:

1. **User**: Represents an authenticated user with email and hashed password.

2. **Document**: Represents a legal contract. Fields:
   - `id`: Unique identifier (UUID)
   - `filename`: The original file name (e.g., "merger_agreement.txt")
   - `title`: Human-readable title
   - `source_corpus`: Which sub-corpus it came from (CUAD, MAUD, etc.)
   - `category`: Optional categorization
   - `createdAt`: Timestamp

3. **Chunk**: A piece of a document with its embedding. Fields:
   - `id`: Unique identifier (UUID)
   - `document_id`: Foreign key to the parent Document
   - `content`: The raw text of the chunk
   - `embedding`: A `vector(384)` column storing the 384-dimensional embedding
   - `char_start`, `char_end`: Character offset positions in the original document

4. **QueryLog**: Records every question asked, for analytics and debugging.

5. **AnswerCitation**: Links a query response back to the specific chunks and documents that were cited.

6. **EvaluationRun**: Stores the configuration and metrics from each evaluation run.

#### Relationships:

- One Document has many Chunks (cascade delete — deleting a document removes all its chunks).
- One Document has many QueryLogs.
- One QueryLog has many AnswerCitations.
- One Chunk has many AnswerCitations.

### 9.2 The Neon Serverless Driver <a name="92-neon-driver"></a>

The database is hosted on **Neon**, a serverless PostgreSQL platform. Synapse RAG connects to Neon using the `@neondatabase/serverless` driver, which communicates with PostgreSQL over WebSockets instead of traditional TCP connections.

#### Why WebSockets?

In serverless environments (like Next.js API routes), each request may spin up in a new process with no persistent TCP connection to the database. Establishing a TCP connection to PostgreSQL takes 100-200ms, which is unacceptable overhead for every request. WebSocket connections through Neon's proxy are faster to establish and can be multiplexed.

The Prisma adapter (`@prisma/adapter-neon`) bridges Prisma's ORM interface with Neon's WebSocket driver, giving us type-safe queries that execute over efficient WebSocket connections.

For raw SQL queries (used in the retrieval functions), we use the `neon` SQL template tag directly, which provides parameterized query execution with automatic SQL injection protection.

### 9.3 The Vector Column and Index <a name="93-vector-column"></a>

The `embedding` column in the Chunk table is declared as `vector(384)` — a pgvector column that stores exactly 384 floating-point numbers. This column:

- Uses the `Unsupported("vector(384)")` type in Prisma because Prisma does not natively support pgvector types. This means vector operations must be performed using raw SQL queries rather than Prisma's query builder.
- Has an HNSW index configured for cosine distance, enabling sub-linear approximate nearest-neighbor search.
- Accepts embeddings as JSON-serialized arrays that PostgreSQL casts to the `vector` type.

---

## 10. Evaluation: How We Measure Success <a name="10-evaluation"></a>

To objectively measure how well our retrieval system works, we run a formal evaluation against the LegalBench-RAG ground truth.

### 10.1 Document Retrieval Match (DRM) <a name="101-drm"></a>

**Question**: "Did we find the right document?"

For each query in the evaluation set, we know which document contains the answer. DRM measures the fraction of queries where the correct document appears anywhere in our top-5 retrieved results.

- **Score achieved**: 38.0% (19 out of 50 queries)
- **Interpretation**: For 38% of queries, the correct document was among the top 5 retrieved results.

### 10.2 Precision, Recall, and F1 <a name="102-precision-recall-f1"></a>

These metrics operate at the word level, measuring overlap between the expected answer text and the text we actually retrieved.

#### Recall: "How much of the answer did we find?"

Recall measures what fraction of the ground-truth answer words appear in our retrieved chunks. If the expected answer is 100 words and our retrieved chunks contain 60 of those words, recall is 60%.

- **Score achieved**: 60.3%
- **Interpretation**: On average, our system retrieves 60% of the words in the ground-truth answer.

#### Precision: "How much of what we retrieved was relevant?"

Precision measures what fraction of the retrieved text actually corresponds to the ground-truth answer. If we retrieved 500 words but only 100 of them overlap with the expected answer, precision is 20%.

- **Score achieved**: 20.4%
- **Interpretation**: About 20% of the text we retrieve is part of the actual answer. The remaining 80% is surrounding context (which is not necessarily useless — it often provides additional relevant information, but it's not the exact answer span).

#### F1 Score: "The balanced measure"

F1 is the harmonic mean of precision and recall. It punishes systems that achieve high recall by retrieving everything (which would have low precision) or high precision by being overly selective (which would have low recall).

- **Score achieved**: 30.5%

### 10.3 What the Results Tell Us <a name="103-results-analysis"></a>

The evaluation reveals both strengths and areas for improvement:

**Strengths:**
- The recall of 60.3% shows that the system successfully finds a majority of the relevant information.
- The semantic search correctly handles paraphrasing — queries that use different vocabulary than the contract text can still find relevant passages.

**Areas for improvement:**
- The DRM of 38% suggests that for some queries, the most relevant document is ranked below position 5. Increasing the retrieval pool or adding keyword-based hybrid search could help.
- The precision of 20.4% indicates that chunks often contain more text than just the answer span. Finer-grained chunking strategies could improve precision.
- The vocabulary gap between user queries (informal, short) and contract text (formal, verbose) is the primary challenge. Techniques like query expansion or Hypothetical Document Embeddings (HyDE) could bridge this gap.

### 10.4 Advanced Evaluation Suites (LLM-as-a-Judge) <a name="104-advanced-eval"></a>

Beyond basic precision and recall, Synapse RAG employs automated LLM-as-a-Judge scripts to measure real-world reliability:

1. **Faithfulness Evaluation** (`evaluate_faithfulness.py`): An LLM acts as an impartial judge to score whether our final generated answer is completely supported by the retrieved citations. If the system invents a detail not present in the citations, it fails the faithfulness check. This provides a strict, automated **Hallucination Rate**.
2. **Multi-Document Benchmark** (`evaluate_multidoc.py`): Tests the system's ability to answer complex questions that require synthesizing information across multiple different contracts (e.g., comparing termination clauses between two NDAs), measuring **Citation Diversity** and **Synthesis Accuracy**.

---

## 11. The User Interface <a name="11-user-interface"></a>

The frontend is built with Next.js (React) and presents a three-panel layout designed for legal professionals.

### 11.1 Three-Panel Layout <a name="111-layout"></a>

#### Left Panel: Contract Library (Document Sidebar)
- Displays all ingested contracts, organized by corpus (CUAD, MAUD, ContractNLI, PrivacyQA).
- Each corpus section is collapsible and shows color-coded badges.
- Clicking a document enters **single-document mode**.
- An "All Documents" button at the top activates **multi-document mode**.
- A search bar filters documents by name or title.
- Shows the total document count and chunk count.

#### Center Panel: Chat Interface
- The primary interaction area where users type questions and receive answers.
- Displays a conversation history with message bubbles (user messages in blue, assistant messages in a dark card).
- Shows example queries on the empty state to guide new users.
- The input area adapts its placeholder text based on the selected mode ("Ask about this contract..." vs. "Ask across all contracts...").
- A pulsing glow animation on the bot icon indicates the system is ready.

#### Right Panel: Citation Panel
- Automatically appears when the system retrieves source passages.
- Shows each cited chunk with:
  - The source document filename
  - A similarity score (color-coded: green for high relevance, yellow for medium, red for low)
  - The actual text passage, truncated to 6 lines with overflow hidden
- Provides transparency and traceability — the user can verify exactly which contract text the answer was based on.

### 11.2 Real-Time Streaming <a name="112-streaming"></a>

The chat uses the Vercel AI SDK's `useChat` hook to manage the conversation state and handle streaming responses. When the LLM generates text, it appears word-by-word in the chat interface, creating a responsive, typewriter-like experience. A three-dot typing indicator animation shows while the system is processing.

### 11.3 Citation Panel: Provenance and Trust <a name="113-citations"></a>

After each response is generated, the frontend makes a separate request to the `/api/retrieve` endpoint to fetch the source citations. This is a deliberate architectural choice: the citations are fetched independently of the LLM response, so they represent the actual retrieval results (not what the LLM claims to have used).

Each citation shows:
- The document filename (with `.txt` extension removed for readability).
- A percentage similarity score, derived from the cosine similarity computation.
- The raw text of the chunk, allowing the user to verify the source.

---

## 12. The Singleton Pattern: Efficient Model Loading <a name="12-singleton-pattern"></a>

Loading a neural network model involves reading ~80MB of binary weights from disk, allocating memory, and initializing computational graphs. This process takes 2-5 seconds. In a web server that handles many requests, loading the model fresh for each request would be disastrous for performance.

Synapse RAG solves this using the **Singleton Pattern**:

For the **bi-encoder**:
- A class (`PipelineSingleton`) has a static `instance` variable initialized to `null`.
- On the first call to `getInstance()`, the model is loaded via the `pipeline()` function and stored in `instance`.
- All subsequent calls to `getInstance()` return the already-loaded model immediately.
- Since the class is a module-level singleton, it persists across requests in the same server process.

For the **cross-encoder**:
- The same pattern is used, but with separate static variables for the tokenizer and model (since the cross-encoder requires manual tokenization rather than using the pipeline API).
- An additional safeguard prevents race conditions: if the model is currently being initialized by one request and another request arrives, the second request awaits the same initialization promise rather than starting a second initialization.

This ensures both models are loaded exactly once, regardless of how many requests the server handles.

---

## 13. End-to-End Walkthrough: A Complete Query Journey <a name="13-end-to-end"></a>

Let us trace a complete query from the moment the user types it to the moment they see the answer.

### Step 1: User Input
The user has selected a merger agreement document in the sidebar and types: "What is the definition of Intervening Event?"

### Step 2: Frontend Processing
The React `Chat` component captures the input, stores it in `latestQuery.current`, clears the input field, and calls `sendMessage({ text: ... })`. The Vercel AI SDK sends a POST request to `/api/chat` with the message history and the `documentId`.

### Step 3: Server Receives the Request
The Next.js API route at `/api/chat` receives the POST request. It extracts the latest message text ("What is the definition of Intervening Event?") and the document ID.

### Step 4: Bi-Encoder Embedding
The query is passed to `generateEmbedding()`:
1. The Singleton ensures the `all-MiniLM-L6-v2` model is loaded (already loaded if not the first query).
2. The text is tokenized: `[CLS] what is the definition of inter ##ven ##ing event ? [SEP]`
3. The token IDs are fed through 6 Transformer layers.
4. Mean pooling reduces the 12 per-token vectors to a single 384-dimensional vector.
5. L2 normalization makes the vector unit-length.
6. The result is a `Float32Array` of 384 numbers, converted to a regular JavaScript array.

### Step 5: Database Search
The embedding is serialized as a JSON string and sent to PostgreSQL via the Neon serverless driver:
```sql
SELECT ... FROM "Chunk" c JOIN "Document" d ...
WHERE c.document_id = 'abc-123' AND c.embedding IS NOT NULL
ORDER BY c.embedding <=> '[0.023, -0.157, ...]'::vector
LIMIT 5
```
The pgvector HNSW index efficiently finds the 5 chunks whose embeddings are closest to the query embedding. This takes approximately 5ms.

### Step 6: Result Assembly
The 5 retrieved chunks are returned with their text content, document metadata, and cosine similarity scores. The chunks are formatted into a context string with source annotations.

### Step 7: LLM Prompt Assembly
The system prompt (with its strict grounding rules) is combined with the context string. The full message history is also included so the LLM can maintain conversational context.

### Step 8: Streaming Generation
The `streamText()` function sends the prompt to Groq's LLaMA 3.1 8B Instant API. Groq processes inference on custom LPU (Language Processing Unit) hardware, achieving extremely fast token generation. As each token is generated, it is streamed back to the client.

### Step 9: Frontend Rendering
The `useChat` hook receives each token and appends it to the current assistant message. The user sees the answer appearing word-by-word in the chat interface. The typing indicator disappears as soon as the first token arrives.

### Step 10: Citation Fetch
After the LLM finishes generating, the `onFinish` callback fires. The frontend sends a separate POST request to `/api/retrieve` with the same query, fetching the source citations. These are displayed in the right-side Citation Panel.

### Step 11: User Verification
The user reads the answer and can cross-reference it with the citations in the side panel. Each citation shows the exact passage from the contract, along with a relevance score. This allows the user to verify that the answer is truly grounded in the source material.

---

## 14. Limitations and Future Directions <a name="14-limitations"></a>

### Current Limitations

1. **Fixed-size chunking**: The current ingestion pipeline does not perform intelligent, semantics-aware chunking. Clause boundaries are not detected, which can split coherent legal provisions across chunks.

2. **No query expansion**: Short user queries may not contain enough vocabulary to match verbose contract language. The system relies entirely on the embedding model's ability to bridge vocabulary gaps.

3. **No hybrid search**: The system uses only dense (vector) retrieval. Adding sparse (keyword) retrieval with BM25 would help with exact-match queries (e.g., searching for a specific clause number like "Section 7.2").

4. **Evaluation scope**: The evaluation uses 50 QA pairs, which provides indicative but not statistically robust results. A larger evaluation set would give more confidence in the metrics.

5. **No fine-tuning**: Both the bi-encoder and cross-encoder use general-purpose pre-trained weights. Fine-tuning them on legal text would likely improve retrieval accuracy significantly.

### Future Directions

1. **Semantic Chunking**: Splitting documents at clause boundaries, paragraph breaks, or section headers to ensure each chunk contains a complete, coherent legal concept.

2. **Hypothetical Document Embeddings (HyDE)**: Having the LLM generate a hypothetical answer before searching, and using that hypothetical answer's embedding for retrieval. This bridges the vocabulary gap between short queries and verbose contracts.

3. **Hybrid Search with RRF**: Combining dense vector search with sparse BM25 keyword search, merging results using Reciprocal Rank Fusion.

4. **Domain-specific fine-tuning**: Fine-tuning the embedding model on legal corpora to better understand domain-specific terminology and concepts.

5. **Expanded evaluation**: Running evaluation on the full LegalBench-RAG dataset and incorporating RAGAS metrics (faithfulness, answer relevancy, context precision, context recall) for end-to-end generation quality assessment.
