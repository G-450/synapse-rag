'use client';

import { BookOpen, X, FileText, BarChart3 } from 'lucide-react';

export interface Citation {
  chunk_id: string;
  document_id: string;
  filename: string;
  content: string;
  similarity: number;
}

interface CitationPanelProps {
  citations: Citation[];
  isOpen: boolean;
  onClose: () => void;
}

function getSimilarityColor(score: number): string {
  if (score >= 0.7) return 'var(--success)';
  if (score >= 0.5) return 'var(--warning)';
  return 'var(--error)';
}


export default function CitationPanel({
  citations,
  isOpen,
  onClose,
}: CitationPanelProps) {
  if (!isOpen) return null;

  return (
    <div
      className="h-full flex flex-col animate-slide-right"
      style={{ background: 'var(--surface)' }}
    >
      {/* Header */}
      <div
        className="px-4 py-4 border-b flex items-center justify-between"
        style={{ borderColor: 'var(--border)' }}
      >
        <div className="flex items-center gap-2">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ background: 'rgba(34, 197, 94, 0.1)' }}
          >
            <BookOpen size={16} style={{ color: 'var(--success)' }} />
          </div>
          <div>
            <h2
              className="text-sm font-semibold"
              style={{ color: 'var(--foreground)' }}
            >
              Sources
            </h2>
            <p
              className="text-xs"
              style={{ color: 'var(--foreground-faint)' }}
            >
              {citations.length} citations
            </p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded-md transition-colors hover:opacity-70"
          style={{ color: 'var(--foreground-faint)' }}
        >
          <X size={16} />
        </button>
      </div>

      {/* Citations List */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3 stagger-children">
        {citations.length === 0 ? (
          <div
            className="flex flex-col items-center justify-center h-full text-center p-4"
            style={{ color: 'var(--foreground-faint)' }}
          >
            <BookOpen size={32} className="opacity-20 mb-3" />
            <p className="text-sm">No citations yet</p>
            <p className="text-xs mt-1 opacity-60">
              Ask a question to see source passages
            </p>
          </div>
        ) : (
          citations.map((citation, i) => (
            <div
              key={`${citation.chunk_id}-${i}`}
              className="glass-card rounded-lg p-3 transition-all hover:border-opacity-20"
              style={{
                borderColor: getSimilarityColor(citation.similarity),
              }}
            >
              {/* Citation Header */}
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-1.5">
                  <FileText
                    size={12}
                    style={{ color: 'var(--foreground-faint)' }}
                  />
                  <span
                    className="text-[11px] font-medium truncate max-w-[140px]"
                    style={{ color: 'var(--foreground-muted)' }}
                    title={citation.filename}
                  >
                    {citation.filename?.replace('.txt', '') || 'Unknown'}
                  </span>
                </div>
                <div
                  className="flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[10px] font-semibold"
                  style={{
                    background: `color-mix(in srgb, ${getSimilarityColor(citation.similarity)} 12%, transparent)`,
                    color: getSimilarityColor(citation.similarity),
                  }}
                >
                  <BarChart3 size={10} />
                  {(citation.similarity * 100).toFixed(0)}% Match
                </div>
              </div>

              {/* Citation Text */}
              <p
                className="text-xs leading-relaxed"
                style={{
                  color: 'var(--foreground)',
                  display: '-webkit-box',
                  WebkitLineClamp: 6,
                  WebkitBoxOrient: 'vertical',
                  overflow: 'hidden',
                }}
              >
                &ldquo;{citation.content}&rdquo;
              </p>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
