# Synapse RAG: Project & Architecture Overview

This document provides a comprehensive, high-level explanation of the **Synapse RAG** project. It covers the core concepts, the dataset, and how all the technologies in our stack work together to create a powerful AI application.

---

## 1. What is this project?
**Synapse RAG** is an intelligent web application designed to analyze and answer complex questions about legal contracts. Instead of relying on a standard AI that might hallucinate (make up facts), this application uses **Retrieval-Augmented Generation (RAG)** to guarantee that its answers are strictly based on the real legal documents we feed into it.

## 2. How Does RAG Work?
LLMs (like ChatGPT or Gemini) are smart, but they don't know your specific private documents. **RAG (Retrieval-Augmented Generation)** solves this by giving the AI a "search engine" for your documents. 

Here is the three-step RAG process:
1. **Ingestion (The Setup):** We take a massive document (like a 50-page legal contract), break it down into small, digestible paragraphs (chunks), and convert those chunks into mathematical representations called **vector embeddings**. These vectors are stored in a database.
2. **Retrieval (The Search):** When a user asks a question (e.g., *"What is the termination clause?"*), the system converts that question into a vector and searches the database for the document chunks that are most mathematically similar to the question.
3. **Generation (The Answer):** The system takes the retrieved paragraphs and sends them to the LLM along with the user's question, essentially saying: *"Answer the user's question, but ONLY use the information provided in these paragraphs."*

## 3. The Dataset: CUAD v1
To build and test this system, we are using the **Contract Understanding Atticus Dataset (CUAD v1)**. 
- **What is it?** A massive, expert-annotated dataset of commercial legal contracts. It contains raw text files of contracts and thousands of labels pointing out specific legal clauses (like non-compete clauses, indemnification, etc.).
- **How we use it:** We will use the raw text contracts to populate our vector database. We will use the annotations to test our AI and make sure it is retrieving and generating the correct legal answers.

## 4. The Technology Stack
Here is how every piece of technology we chose fits into the puzzle:

- **The Front-End & Back-End: Next.js**
  Next.js is a React framework that allows us to build both the user interface (the front-end) and the API server (the back-end) in a single codebase. It will serve our highly-interactive, premium UI and handle the server-side logic of talking to the AI and database.

- **Authentication: NextAuth.js**
  We need to secure the application. NextAuth is an open-source library that handles user logins (passwords, Google, GitHub, etc.) completely for free. It will store user credentials securely in our database.

- **The Database: PostgreSQL + `pgvector`**
  PostgreSQL is an enterprise-grade database. We need it to store standard relational data (like our NextAuth Users and Passwords). However, because we are building a RAG app, we also need to store those mathematical **vector embeddings**. By adding the `pgvector` extension to PostgreSQL, it gains the ability to perform high-speed similarity searches on vectors. This means we only need **one** database for everything.

- **The ORM: Prisma**
  Prisma is a tool that sits between Next.js and PostgreSQL. Instead of writing raw, complex SQL queries, Prisma lets us interact with the database using simple, type-safe TypeScript code.

## 5. The User Journey (How it all flows)
1. A user visits the app and logs in securely using **NextAuth**.
2. Behind the scenes, our **Ingestion Script** has already processed the **CUAD v1** dataset, chunked it, and stored the vectors in **PostgreSQL**.
3. The user types a question into the beautiful **Next.js** interface: *"Are there any non-compete agreements?"*
4. Next.js converts that question into a vector embedding.
5. Next.js queries **PostgreSQL (`pgvector`)** to find the top 5 most relevant chunks from the CUAD contracts.
6. Next.js passes those 5 chunks to an LLM.
7. The LLM generates a precise, accurate answer based purely on the contracts and streams it back to the user's screen.
