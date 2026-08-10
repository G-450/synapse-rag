"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronRight, ChevronLeft, RefreshCcw, Home, Keyboard } from "lucide-react";
import Link from "next/link";
import PresentationScene from "@/components/presentation/PresentationScene";

const TOTAL_STEPS = 12;

const PIPELINE_STAGES = [
  { label: "Intro", color: "#ffffff" },
  { label: "Query", color: "#38bdf8" },
  { label: "Tokenize", color: "#818cf8" },
  { label: "Embed", color: "#a78bfa" },
  { label: "Index", color: "#34d399" },
  { label: "Search", color: "#2dd4bf" },
  { label: "Rerank", color: "#22d3ee" },
  { label: "Retrieve", color: "#4ade80" },
  { label: "Augment", color: "#fbbf24" },
  { label: "LLM", color: "#f472b6" },
  { label: "Stream", color: "#fb923c" },
  { label: "Evaluate", color: "#c084fc" },
  { label: "Output", color: "#f43f5e" },
];

export default function PresentationPage() {
  const [step, setStep] = useState(0);
  const [showKeys, setShowKeys] = useState(true);

  const handleNext = useCallback(() => {
    setStep((s) => Math.min(s + 1, TOTAL_STEPS));
  }, []);

  const handlePrev = useCallback(() => {
    setStep((s) => Math.max(s - 1, 0));
  }, []);

  const handleReset = useCallback(() => {
    setStep(0);
  }, []);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight" || e.key === " ") {
        e.preventDefault();
        handleNext();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        handlePrev();
      } else if (e.key === "r") {
        handleReset();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [handleNext, handlePrev, handleReset]);

  // Auto-hide keyboard hint
  useEffect(() => {
    const t = setTimeout(() => setShowKeys(false), 5000);
    return () => clearTimeout(t);
  }, []);

  return (
    <div
      style={{
        minHeight: "100vh",
        width: "100%",
        background: "#050508",
        color: "#fff",
        fontFamily: "'Inter', 'Segoe UI', system-ui, sans-serif",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        position: "relative",
      }}
    >
      {/* Background Gradient Orb */}
      <div
        style={{
          position: "fixed",
          top: "-20%",
          left: "50%",
          transform: "translateX(-50%)",
          width: "800px",
          height: "800px",
          borderRadius: "50%",
          background: `radial-gradient(circle, ${PIPELINE_STAGES[step].color}08 0%, transparent 70%)`,
          transition: "background 0.8s ease",
          pointerEvents: "none",
          zIndex: 0,
        }}
      />

      {/* Header */}
      <header
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          zIndex: 100,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "20px 32px",
          background: "linear-gradient(to bottom, rgba(5,5,8,0.95), transparent)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div
            style={{
              width: "36px",
              height: "36px",
              borderRadius: "50%",
              background: "linear-gradient(135deg, #38bdf8, #a78bfa)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              boxShadow: "0 0 20px rgba(56,189,248,0.4)",
            }}
          >
            <span style={{ fontWeight: 800, fontSize: "14px" }}>S</span>
          </div>
          <span style={{ fontSize: "18px", fontWeight: 600, color: "#e2e8f0" }}>
            Synapse RAG Pipeline
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          {/* Keyboard hint */}
          <AnimatePresence>
            {showKeys && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  fontSize: "12px",
                  color: "#64748b",
                }}
              >
                <Keyboard size={14} />
                <span>← → Arrow keys to navigate</span>
              </motion.div>
            )}
          </AnimatePresence>

          <Link
            href="/"
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              padding: "8px 16px",
              borderRadius: "9999px",
              background: "rgba(255,255,255,0.05)",
              border: "1px solid rgba(255,255,255,0.1)",
              color: "#94a3b8",
              textDecoration: "none",
              fontSize: "13px",
              transition: "all 0.2s",
            }}
          >
            <Home size={14} />
            Back to App
          </Link>
        </div>
      </header>

      {/* Main Scene Area */}
      <main
        style={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          position: "relative",
          zIndex: 1,
          padding: "80px 40px 160px",
        }}
      >
        <AnimatePresence mode="wait">
          <motion.div
            key={step}
            initial={{ opacity: 0, scale: 0.92, y: 30 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 1.05, y: -20 }}
            transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
            style={{ width: "100%", maxWidth: "1200px" }}
          >
            <PresentationScene step={step} />
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Bottom Controls Bar */}
      <footer
        style={{
          position: "fixed",
          bottom: 0,
          left: 0,
          right: 0,
          zIndex: 100,
          background: "linear-gradient(to top, rgba(5,5,8,0.98) 60%, transparent)",
          padding: "0 40px 28px",
        }}
      >
        {/* Pipeline Mini-map */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "4px",
            marginBottom: "20px",
          }}
        >
          {PIPELINE_STAGES.map((stage, i) => (
            <button
              key={i}
              onClick={() => setStep(i)}
              title={stage.label}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: "6px",
                background: "none",
                border: "none",
                cursor: "pointer",
                padding: "4px 2px",
              }}
            >
              <div
                style={{
                  height: "4px",
                  width: i === step ? "40px" : "20px",
                  borderRadius: "2px",
                  background: i <= step ? stage.color : "rgba(255,255,255,0.1)",
                  transition: "all 0.3s ease",
                  boxShadow: i === step ? `0 0 10px ${stage.color}60` : "none",
                }}
              />
              <span
                style={{
                  fontSize: "9px",
                  fontWeight: i === step ? 600 : 400,
                  color: i === step ? stage.color : "rgba(255,255,255,0.25)",
                  transition: "all 0.3s",
                  letterSpacing: "0.5px",
                  textTransform: "uppercase",
                }}
              >
                {stage.label}
              </span>
            </button>
          ))}
        </div>

        {/* Navigation Buttons */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "16px",
          }}
        >
          <button
            onClick={handleReset}
            disabled={step === 0}
            style={{
              padding: "10px",
              borderRadius: "50%",
              background: "rgba(255,255,255,0.05)",
              border: "1px solid rgba(255,255,255,0.1)",
              color: step === 0 ? "rgba(255,255,255,0.2)" : "#94a3b8",
              cursor: step === 0 ? "not-allowed" : "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              transition: "all 0.2s",
            }}
            title="Reset"
          >
            <RefreshCcw size={16} />
          </button>

          <button
            onClick={handlePrev}
            disabled={step === 0}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              padding: "10px 24px",
              borderRadius: "9999px",
              background: "rgba(255,255,255,0.05)",
              border: "1px solid rgba(255,255,255,0.1)",
              color: step === 0 ? "rgba(255,255,255,0.2)" : "#e2e8f0",
              cursor: step === 0 ? "not-allowed" : "pointer",
              fontSize: "14px",
              fontWeight: 500,
              transition: "all 0.2s",
            }}
          >
            <ChevronLeft size={18} />
            Previous
          </button>

          <button
            onClick={handleNext}
            disabled={step === TOTAL_STEPS}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              padding: "12px 32px",
              borderRadius: "9999px",
              background: step === TOTAL_STEPS ? "rgba(255,255,255,0.1)" : "#fff",
              border: "none",
              color: step === TOTAL_STEPS ? "rgba(255,255,255,0.3)" : "#0a0a0a",
              cursor: step === TOTAL_STEPS ? "not-allowed" : "pointer",
              fontSize: "15px",
              fontWeight: 600,
              boxShadow:
                step === TOTAL_STEPS
                  ? "none"
                  : "0 0 40px rgba(255,255,255,0.15)",
              transition: "all 0.2s",
            }}
          >
            {step === TOTAL_STEPS ? "Finished" : "Next Step"}
            {step !== TOTAL_STEPS && <ChevronRight size={18} />}
          </button>
        </div>

        {/* Step Counter */}
        <div
          style={{
            textAlign: "center",
            marginTop: "12px",
            fontSize: "12px",
            color: "rgba(255,255,255,0.25)",
          }}
        >
          {step} / {TOTAL_STEPS}
        </div>
      </footer>
    </div>
  );
}
