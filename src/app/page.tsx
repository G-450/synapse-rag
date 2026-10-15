'use client';

import { useState, useEffect, useMemo } from 'react';
import Chat from '@/components/chat';
import DocumentSidebar from '@/components/document-sidebar';
import CitationPanel from '@/components/citation-panel';
import type { Citation } from '@/components/citation-panel';
import { PanelLeftClose, PanelLeftOpen } from 'lucide-react';

export default function Home() {
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  // Selecting every contract is the same query as selecting none, and far
  // cheaper to run unscoped than as a filter over every id in the index.
  const [wholeLibrary, setWholeLibrary] = useState(false);
  const [selectedDocNames, setSelectedDocNames] = useState<string[]>([]);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [showCitations, setShowCitations] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);
  const [documents, setDocuments] = useState<
    Array<{ id: string; filename: string; title: string }>
  >([]);

  // Load documents to map IDs to names
  useEffect(() => {
    fetch('/api/documents')
      .then((res) => res.json())
      .then((data) => setDocuments(data.documents || []))
      .catch(() => {});
  }, []);

  // A map, not a scan per id: a whole-corpus selection resolves hundreds at once.
  const namesById = useMemo(
    () =>
      new Map(
        documents.map((doc) => [
          doc.id,
          doc.title || doc.filename?.replace(/\.[^.]+$/, '') || 'Selected document',
        ])
      ),
    [documents]
  );

  // The sidebar knows about fresh uploads the initial fetch never saw, so prefer its labels.
  const handleSelectionChange = (
    docIds: string[],
    names?: string[],
    coversWholeLibrary?: boolean
  ) => {
    setSelectedDocIds(docIds);
    setSelectedDocNames(
      docIds.map((id, i) => names?.[i] || namesById.get(id) || 'Selected document')
    );
    setWholeLibrary(Boolean(coversWholeLibrary));
    setCitations([]);
    setShowCitations(false);
  };

  const handleCitationsReceived = (newCitations: Citation[]) => {
    setCitations(newCitations);
    if (newCitations.length > 0) {
      setShowCitations(true);
    }
  };

  return (
    <div className="h-screen flex overflow-hidden" style={{ background: 'var(--background)' }}>
      {/* Sidebar Toggle (mobile) */}
      <button
        onClick={() => setShowSidebar(!showSidebar)}
        className="fixed top-4 left-4 z-50 p-2 rounded-lg lg:hidden"
        style={{
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          color: 'var(--foreground-muted)',
        }}
      >
        {showSidebar ? <PanelLeftClose size={18} /> : <PanelLeftOpen size={18} />}
      </button>

      {/* Left Sidebar — Document Library */}
      <div
        className={`
          ${showSidebar ? 'w-72' : 'w-0'} 
          flex-shrink-0 border-r transition-all duration-300 overflow-hidden
          fixed lg:relative z-40 h-full
        `}
        style={{ borderColor: 'var(--border)' }}
      >
        <DocumentSidebar
          selectedDocIds={selectedDocIds}
          onSelectionChange={handleSelectionChange}
        />
      </div>

      {/* Center — Chat */}
      <div className="flex-1 min-w-0">
        <Chat
          key={(!wholeLibrary && selectedDocIds.join(',')) || 'all-documents'}
          documentIds={wholeLibrary ? [] : selectedDocIds}
          documentNames={wholeLibrary ? [] : selectedDocNames}
          onCitationsReceived={handleCitationsReceived}
          onToggleCitations={() => setShowCitations(!showCitations)}
        />
      </div>

      {/* Right Panel — Citations */}
      <div
        className={`
          ${showCitations ? 'w-80' : 'w-0'} 
          flex-shrink-0 border-l transition-all duration-300 overflow-hidden
          ${showCitations ? 'fixed lg:relative right-0 z-40 h-full' : ''}
        `}
        style={{ borderColor: 'var(--border)' }}
      >
        <CitationPanel
          citations={citations}
          isOpen={showCitations}
          onClose={() => setShowCitations(false)}
        />
      </div>

      {/* Overlay for mobile when sidebar or citations are open */}
      {(showSidebar || showCitations) && (
        <div
          className="fixed inset-0 bg-black/40 z-30 lg:hidden"
          onClick={() => {
            setShowSidebar(false);
            setShowCitations(false);
          }}
        />
      )}
    </div>
  );
}
