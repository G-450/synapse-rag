'use client';

import { useState, useEffect } from 'react';
import {
  FileText,
  Search,
  ChevronDown,
  ChevronRight,
  Layers,
  Database,
  Loader2,
} from 'lucide-react';

interface Document {
  id: string;
  filename: string;
  title: string;
  source_corpus: string;
  chunk_count: number;
}

interface DocumentSidebarProps {
  selectedDocId: string | null;
  onSelectDocument: (docId: string | null) => void;
}

const CORPUS_LABELS: Record<string, string> = {
  cuad: 'CUAD',
  maud: 'MAUD',
  contractnli: 'ContractNLI',
  privacy_qa: 'PrivacyQA',
  'legalbench-rag': 'LegalBench',
};

const CORPUS_DESCRIPTIONS: Record<string, string> = {
  cuad: 'Commercial contracts',
  maud: 'Merger agreements',
  contractnli: 'Non-disclosure agreements',
  privacy_qa: 'Privacy policies',
  'legalbench-rag': 'Legal benchmark',
};

export default function DocumentSidebar({
  selectedDocId,
  onSelectDocument,
}: DocumentSidebarProps) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [grouped, setGrouped] = useState<Record<string, Document[]>>({});
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [expandedCorpora, setExpandedCorpora] = useState<Set<string>>(
    new Set()
  );

  useEffect(() => {
    fetch('/api/documents')
      .then((res) => res.json())
      .then((data) => {
        setDocuments(data.documents || []);
        setGrouped(data.grouped || {});
        // Expand all corpora by default
        setExpandedCorpora(new Set(Object.keys(data.grouped || {})));
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const toggleCorpus = (corpus: string) => {
    setExpandedCorpora((prev) => {
      const next = new Set(prev);
      if (next.has(corpus)) next.delete(corpus);
      else next.add(corpus);
      return next;
    });
  };

  const filteredGrouped = Object.entries(grouped).reduce(
    (acc, [corpus, docs]) => {
      const filtered = docs.filter(
        (d) =>
          d.filename.toLowerCase().includes(search.toLowerCase()) ||
          d.title.toLowerCase().includes(search.toLowerCase())
      );
      if (filtered.length > 0) acc[corpus] = filtered;
      return acc;
    },
    {} as Record<string, Document[]>
  );

  const totalDocs = documents.length;
  const totalChunks = documents.reduce((sum, d) => sum + d.chunk_count, 0);

  return (
    <div className="h-full flex flex-col" style={{ background: 'var(--surface)' }}>
      {/* Header */}
      <div
        className="px-4 py-4 border-b"
        style={{ borderColor: 'var(--border)' }}
      >
        <div className="flex items-center gap-2 mb-3">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ background: 'var(--accent-glow)' }}
          >
            <Database size={16} style={{ color: 'var(--accent)' }} />
          </div>
          <div>
            <h2
              className="text-sm font-semibold"
              style={{ color: 'var(--foreground)' }}
            >
              Contract Library
            </h2>
            <p
              className="text-xs"
              style={{ color: 'var(--foreground-faint)' }}
            >
              {totalDocs} docs · {totalChunks} chunks
            </p>
          </div>
        </div>

        {/* Search */}
        <div className="relative">
          <Search
            size={14}
            className="absolute left-3 top-1/2 -translate-y-1/2"
            style={{ color: 'var(--foreground-faint)' }}
          />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search contracts..."
            className="w-full pl-9 pr-3 py-2 text-xs rounded-lg border transition-all focus:outline-none"
            style={{
              background: 'var(--background)',
              borderColor: 'var(--border)',
              color: 'var(--foreground)',
            }}
          />
        </div>
      </div>

      {/* All Documents Button */}
      <div className="px-3 py-2">
        <button
          onClick={() => onSelectDocument(null)}
          className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg text-sm font-medium transition-all"
          style={{
            background:
              selectedDocId === null ? 'var(--accent-glow)' : 'transparent',
            color:
              selectedDocId === null
                ? 'var(--accent-hover)'
                : 'var(--foreground-muted)',
            border:
              selectedDocId === null
                ? '1px solid var(--accent)'
                : '1px solid transparent',
          }}
        >
          <Layers size={16} />
          All Documents
          <span
            className="ml-auto text-xs px-1.5 py-0.5 rounded-full"
            style={{
              background: 'var(--accent-soft)',
              color: 'var(--accent)',
            }}
          >
            Multi
          </span>
        </button>
      </div>

      {/* Document Tree */}
      <div className="flex-1 overflow-y-auto px-3 pb-3">
        {loading ? (
          <div className="space-y-3 mt-2">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="skeleton h-8 w-full" />
            ))}
          </div>
        ) : (
          <div className="space-y-1 stagger-children">
            {Object.entries(filteredGrouped).map(([corpus, docs]) => (
              <div key={corpus}>
                {/* Corpus header */}
                <button
                  onClick={() => toggleCorpus(corpus)}
                  className="w-full flex items-center gap-2 px-2 py-2 rounded-md text-xs font-semibold uppercase tracking-wider transition-colors hover:opacity-80"
                  style={{ color: 'var(--foreground-faint)' }}
                >
                  {expandedCorpora.has(corpus) ? (
                    <ChevronDown size={14} />
                  ) : (
                    <ChevronRight size={14} />
                  )}
                  <span
                    className={`corpus-badge corpus-${corpus} px-1.5 py-0.5 rounded text-[10px] font-bold`}
                  >
                    {CORPUS_LABELS[corpus] || corpus}
                  </span>
                  <span className="ml-auto opacity-60">{docs.length}</span>
                </button>

                {/* Documents under this corpus */}
                {expandedCorpora.has(corpus) && (
                  <div className="ml-4 space-y-0.5">
                    {docs.map((doc) => (
                      <button
                        key={doc.id}
                        onClick={() => onSelectDocument(doc.id)}
                        className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-xs transition-all group text-left"
                        style={{
                          background:
                            selectedDocId === doc.id
                              ? 'var(--accent-glow)'
                              : 'transparent',
                          color:
                            selectedDocId === doc.id
                              ? 'var(--accent-hover)'
                              : 'var(--foreground-muted)',
                        }}
                        title={doc.filename}
                      >
                        <FileText
                          size={13}
                          className="flex-shrink-0 opacity-50 group-hover:opacity-100 transition-opacity"
                        />
                        <span className="truncate">
                          {doc.title || doc.filename.replace('.txt', '')}
                        </span>
                        <span
                          className="ml-auto text-[10px] opacity-40 flex-shrink-0"
                        >
                          {doc.chunk_count}
                        </span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
