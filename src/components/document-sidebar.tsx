'use client';

import { useState, useEffect, useMemo, useCallback, useRef, memo } from 'react';
import {
  FileText,
  Search,
  ChevronDown,
  ChevronRight,
  Layers,
  Database,
  Upload,
  Check,
  Trash2,
  Loader2,
  X,
} from 'lucide-react';
import UploadModal, { type UploadResult } from './upload-modal';

interface Document {
  id: string;
  filename: string;
  title: string;
  source_corpus: string;
  chunk_count: number;
}

interface DocumentSidebarProps {
  selectedDocIds: string[];
  /** `wholeLibrary` lets the page query every contract unscoped, which is far
   *  cheaper than filtering on an id list that covers the whole index anyway. */
  onSelectionChange: (
    docIds: string[],
    documentNames?: string[],
    wholeLibrary?: boolean
  ) => void;
}

const CORPUS_LABELS: Record<string, string> = {
  cuad: 'CUAD',
  maud: 'MAUD',
  contractnli: 'ContractNLI',
  privacy_qa: 'PrivacyQA',
  'legalbench-rag': 'LegalBench',
  'user-upload': 'User Uploads',
};

const UPLOAD_CORPUS = 'user-upload';
/** How long an armed delete waits for its confirming click before standing down. */
const CONFIRM_TIMEOUT_MS = 5000;

interface DocumentRowProps {
  doc: Document;
  isSelected: boolean;
  isDeleting: boolean;
  isPendingDelete: boolean;
  deletable: boolean;
  onToggle: (docId: string) => void;
  onSelectOnly: (docId: string) => void;
  onDelete: (docId: string) => void;
}

// The library runs to hundreds of contracts. Rendering every row again on each
// click made a single checkbox feel sluggish, so a row only re-renders when its
// own state changes — which needs the handlers below to keep their identity.
const DocumentRow = memo(function DocumentRow({
  doc,
  isSelected,
  isDeleting,
  isPendingDelete,
  deletable,
  onToggle,
  onSelectOnly,
  onDelete,
}: DocumentRowProps) {
  const label = doc.title || doc.filename;
  return (
    <div
      className="group flex items-center gap-2 rounded-md px-2 py-1.5 text-xs transition-all"
      style={{
        background: isSelected ? 'var(--accent-glow)' : 'transparent',
        color: isSelected ? 'var(--accent-hover)' : 'var(--foreground-muted)',
      }}
    >
      <button
        onClick={() => onToggle(doc.id)}
        className="flex h-4 w-4 flex-shrink-0 items-center justify-center rounded border transition-all"
        style={{
          borderColor: isSelected ? 'var(--accent)' : 'var(--border)',
          background: isSelected ? 'var(--accent)' : 'transparent',
        }}
        aria-label={
          isSelected
            ? `Remove ${label} from the selection`
            : `Add ${label} to the selection`
        }
        aria-pressed={isSelected}
      >
        {isSelected && <Check size={11} strokeWidth={3} color="#fff" />}
      </button>
      <button
        onClick={() => onSelectOnly(doc.id)}
        className="flex min-w-0 flex-1 items-center gap-2 text-left"
        title={`${doc.filename} — click to query this contract only`}
      >
        <FileText
          size={13}
          className="flex-shrink-0 opacity-50 transition-opacity group-hover:opacity-100"
        />
        <span className="truncate">
          {doc.title || doc.filename.replace('.txt', '')}
        </span>
        <span className="ml-auto flex-shrink-0 text-[10px] opacity-40">
          {doc.chunk_count}
        </span>
      </button>
      {deletable && (
        <button
          onClick={() => onDelete(doc.id)}
          disabled={isDeleting}
          className="flex-shrink-0 rounded p-1 opacity-0 transition-opacity group-hover:opacity-70 hover:opacity-100 focus:opacity-100 disabled:opacity-40"
          style={{
            color: 'var(--error)',
            opacity: isPendingDelete ? 1 : undefined,
          }}
          title={
            isPendingDelete
              ? 'Click again to delete this upload'
              : 'Delete this upload'
          }
          aria-label={`Delete ${label}`}
        >
          {isDeleting ? (
            <Loader2 size={12} className="animate-spin" />
          ) : isPendingDelete ? (
            <Check size={12} strokeWidth={3} />
          ) : (
            <Trash2 size={12} />
          )}
        </button>
      )}
    </div>
  );
});

export default function DocumentSidebar({
  selectedDocIds,
  onSelectionChange,
}: DocumentSidebarProps) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [grouped, setGrouped] = useState<Record<string, Document[]>>({});
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState('');
  const [search, setSearch] = useState('');
  const [showUpload, setShowUpload] = useState(false);
  const [expandedCorpora, setExpandedCorpora] = useState<Set<string>>(
    new Set()
  );
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  const [confirmClearUploads, setConfirmClearUploads] = useState(false);
  const [deleting, setDeleting] = useState<Set<string>>(new Set());
  const [deleteError, setDeleteError] = useState('');

  const selected = useMemo(() => new Set(selectedDocIds), [selectedDocIds]);

  const labelFor = (doc: Document) => doc.title || doc.filename.replace(/\.[^.]+$/, '');

  // Selecting a whole corpus commits hundreds of ids at once, so the labels are
  // looked up through a map. Scanning the library per id made that click crawl.
  const labelsById = useMemo(
    () => new Map(documents.map((doc) => [doc.id, labelFor(doc)])),
    [documents]
  );

  // Everything the stable row handlers below need to read at click time, without
  // taking a dependency on it — a changed dependency would re-render every row.
  const live = useRef<{
    documents: Document[];
    labelsById: Map<string, string>;
    selectedDocIds: string[];
    pendingDeleteId: string | null;
    onSelectionChange: DocumentSidebarProps['onSelectionChange'];
    removeDocuments: (docIds: string[]) => unknown;
  }>({
    documents,
    labelsById,
    selectedDocIds,
    pendingDeleteId,
    onSelectionChange,
    removeDocuments: () => {},
  });

  /** Push a selection upward together with display labels for the chat header. */
  const commitSelection = useCallback((docIds: string[], pool?: Document[]) => {
    const current = live.current;
    const labels = pool
      ? new Map(pool.map((doc) => [doc.id, labelFor(doc)]))
      : current.labelsById;
    current.onSelectionChange(
      docIds,
      docIds.map((id) => labels.get(id) || 'Selected document'),
      docIds.length > 0 && docIds.length >= current.documents.length
    );
  }, []);

  /** Pull the library straight from the index, so the tree mirrors what is stored. */
  const loadDocuments = async () => {
    const res = await fetch('/api/documents', { cache: 'no-store' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || data.error || 'Unable to load contracts');
    setDocuments(data.documents || []);
    setGrouped(data.grouped || {});
    return data as { documents: Document[]; grouped: Record<string, Document[]> };
  };

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const data = await loadDocuments();
        // Expand all corpora by default
        if (!cancelled) setExpandedCorpora(new Set(Object.keys(data.grouped || {})));
      } catch (error) {
        if (!cancelled) {
          setLoadError(error instanceof Error ? error.message : 'Unable to load contracts');
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Arming a delete used to be cancelled by the button's own onBlur, so the
  // confirming click only survived while the button kept focus. Browsers that
  // do not focus a button on mousedown disarmed it on every second click, and
  // the contract could never be deleted. Time the confirmation out instead.
  useEffect(() => {
    if (!pendingDeleteId) return;
    const timer = setTimeout(() => setPendingDeleteId(null), CONFIRM_TIMEOUT_MS);
    return () => clearTimeout(timer);
  }, [pendingDeleteId]);

  useEffect(() => {
    if (!confirmClearUploads) return;
    const timer = setTimeout(() => setConfirmClearUploads(false), CONFIRM_TIMEOUT_MS);
    return () => clearTimeout(timer);
  }, [confirmClearUploads]);

  const toggleCorpus = (corpus: string) => {
    setExpandedCorpora((prev) => {
      const next = new Set(prev);
      if (next.has(corpus)) next.delete(corpus);
      else next.add(corpus);
      return next;
    });
  };

  const handleUploaded = (results: UploadResult[]) => {
    if (results.length === 0) return;
    const uploaded: Document[] = results.map((result) => ({
      id: result.document_id,
      filename: result.filename,
      title: result.filename.replace(/\.[^.]+$/, ''),
      source_corpus: UPLOAD_CORPUS,
      chunk_count: result.chunks_created,
    }));
    const ids = new Set(uploaded.map((doc) => doc.id));

    setDocuments((current) => [...current.filter((item) => !ids.has(item.id)), ...uploaded]);
    setGrouped((current) => ({
      ...current,
      [UPLOAD_CORPUS]: [
        ...(current[UPLOAD_CORPUS] || []).filter((item) => !ids.has(item.id)),
        ...uploaded,
      ],
    }));
    setExpandedCorpora((current) => new Set([...current, UPLOAD_CORPUS]));
    // Query what was just uploaded, as a set.
    commitSelection(uploaded.map((doc) => doc.id), uploaded);
  };

  /** Click a row: make it the only selection. Checkbox: add/remove from the selection. */
  const selectOnly = useCallback(
    (docId: string) => commitSelection([docId]),
    [commitSelection]
  );

  const toggleDocument = useCallback(
    (docId: string) => {
      const next = new Set(live.current.selectedDocIds);
      if (next.has(docId)) next.delete(docId);
      else next.add(docId);
      commitSelection([...next]);
    },
    [commitSelection]
  );

  const toggleCorpusSelection = (docs: Document[]) => {
    const ids = docs.map((d) => d.id);
    const allSelected = ids.every((id) => selected.has(id));
    const next = new Set(selected);
    ids.forEach((id) => (allSelected ? next.delete(id) : next.add(id)));
    commitSelection([...next]);
  };

  /** Deletes exactly the ids handed in, so what is on screen is what goes. */
  const removeDocuments = async (docIds: string[]) => {
    if (docIds.length === 0) return;
    setDeleteError('');
    setDeleting((current) => new Set([...current, ...docIds]));
    try {
      const response = await fetch('/api/documents', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ documentIds: docIds }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || result.error || 'Delete failed');

      const removed = new Set<string>(result.deleted || docIds);
      setDocuments((current) => current.filter((doc) => !removed.has(doc.id)));
      setGrouped((current) =>
        Object.entries(current).reduce((acc, [key, docs]) => {
          const kept = docs.filter((doc) => !removed.has(doc.id));
          if (kept.length > 0) acc[key] = kept;
          return acc;
        }, {} as Record<string, Document[]>)
      );
      commitSelection(selectedDocIds.filter((id) => !removed.has(id)));
      setPendingDeleteId(null);
      setConfirmClearUploads(false);
      // Re-read the index: a row that survives the delete has to stay visible
      // rather than vanish locally and reappear on the next page load.
      await loadDocuments().catch(() => {});
    } catch (caught) {
      setDeleteError(caught instanceof Error ? caught.message : 'Delete failed');
    } finally {
      setDeleting((current) => {
        const next = new Set(current);
        docIds.forEach((id) => next.delete(id));
        return next;
      });
    }
  };

  // Refreshed after every render, so the stable handlers above always read the
  // current library, selection and pending delete without depending on them.
  useEffect(() => {
    live.current = {
      documents,
      labelsById,
      selectedDocIds,
      pendingDeleteId,
      onSelectionChange,
      removeDocuments,
    };
  });

  /** First click arms the confirmation, the second one deletes. */
  const handleDeleteClick = useCallback((docId: string) => {
    if (live.current.pendingDeleteId === docId) void live.current.removeDocuments([docId]);
    else setPendingDeleteId(docId);
  }, []);

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
  // The button is a toggle: it checks every contract, and clears them again.
  // Either way the next question runs against the whole library.
  const everythingSelected = totalDocs > 0 && selectedDocIds.length >= totalDocs;
  const toggleAllDocuments = () =>
    commitSelection(everythingSelected ? [] : documents.map((doc) => doc.id));
  const totalChunks = documents.reduce((sum, d) => sum + d.chunk_count, 0);
  const uploadCount = (grouped[UPLOAD_CORPUS] || []).length;
  const selectedCount = selectedDocIds.length;
  // No selection and every contract selected both query the whole library.
  const queryingEverything = selectedCount === 0 || everythingSelected;
  const clearingUploads = (grouped[UPLOAD_CORPUS] || []).some((doc) =>
    deleting.has(doc.id)
  );

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
        <button
          onClick={() => setShowUpload(true)}
          className="mt-2 flex w-full items-center justify-center gap-2 rounded-lg px-3 py-2 text-xs font-semibold transition-all"
          style={{ background: 'var(--accent)', color: '#fff' }}
        >
          <Upload size={14} /> Upload Contract
        </button>
      </div>

      {/* All Documents Button */}
      <div className="px-3 py-2">
        <button
          onClick={toggleAllDocuments}
          aria-pressed={everythingSelected}
          title={
            everythingSelected
              ? 'Deselect every contract'
              : 'Select every contract'
          }
          className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg text-sm font-medium transition-all"
          style={{
            background: queryingEverything ? 'var(--accent-glow)' : 'transparent',
            color: queryingEverything
              ? 'var(--accent-hover)'
              : 'var(--foreground-muted)',
            border: queryingEverything
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

        {/* Selection summary */}
        {selectedCount > 0 && (
          <div
            className="mt-2 flex items-center gap-2 rounded-lg px-3 py-2 text-xs"
            style={{
              background: 'var(--accent-soft)',
              color: 'var(--accent)',
            }}
          >
            <span className="font-semibold">
              {selectedCount} document{selectedCount === 1 ? '' : 's'} selected
            </span>
            <button
              onClick={() => commitSelection([])}
              className="ml-auto flex items-center gap-1 font-medium transition-opacity hover:opacity-70"
              title="Clear selection"
            >
              <X size={12} /> Clear
            </button>
          </div>
        )}

        {deleteError && (
          <p
            className="mt-2 px-1 text-xs"
            style={{ color: 'var(--error)' }}
            role="alert"
          >
            {deleteError}
          </p>
        )}
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
            {loadError && (
              <p className="px-2 py-3 text-xs" style={{ color: 'var(--error)' }} role="alert">
                {loadError}
              </p>
            )}
            {Object.entries(filteredGrouped).map(([corpus, docs]) => {
              const allSelected = docs.every((doc) => selected.has(doc.id));
              const someSelected = docs.some((doc) => selected.has(doc.id));
              return (
                <div key={corpus}>
                  {/* Corpus header */}
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => toggleCorpusSelection(docs)}
                      className="flex h-4 w-4 flex-shrink-0 items-center justify-center rounded border transition-all"
                      style={{
                        borderColor: someSelected ? 'var(--accent)' : 'var(--border)',
                        background: allSelected ? 'var(--accent)' : 'transparent',
                      }}
                      aria-label={
                        allSelected
                          ? `Deselect all ${CORPUS_LABELS[corpus] || corpus} documents`
                          : `Select all ${CORPUS_LABELS[corpus] || corpus} documents`
                      }
                    >
                      {allSelected ? (
                        <Check size={11} strokeWidth={3} color="#fff" />
                      ) : someSelected ? (
                        <span
                          className="h-0.5 w-2 rounded"
                          style={{ background: 'var(--accent)' }}
                        />
                      ) : null}
                    </button>
                    <button
                      onClick={() => toggleCorpus(corpus)}
                      className="flex flex-1 items-center gap-2 px-1 py-2 rounded-md text-xs font-semibold uppercase tracking-wider transition-colors hover:opacity-80"
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
                    {corpus === UPLOAD_CORPUS && uploadCount > 0 && (
                      <button
                        onClick={() =>
                          confirmClearUploads
                            ? removeDocuments(docs.map((doc) => doc.id))
                            : setConfirmClearUploads(true)
                        }
                        disabled={clearingUploads}
                        className="flex-shrink-0 rounded px-1.5 py-1 text-[10px] font-bold uppercase tracking-wider transition-opacity hover:opacity-100 disabled:opacity-40"
                        style={{
                          color: 'var(--error)',
                          opacity: confirmClearUploads ? 1 : 0.6,
                        }}
                        title={
                          docs.length === uploadCount
                            ? `Remove all ${uploadCount} uploaded contracts from the index`
                            : `Remove the ${docs.length} uploaded contracts matching this search`
                        }
                      >
                        {clearingUploads ? (
                          <Loader2 size={12} className="animate-spin" />
                        ) : confirmClearUploads ? (
                          `Delete ${docs.length}?`
                        ) : (
                          `Clear ${docs.length === uploadCount ? 'all' : `these ${docs.length}`}`
                        )}
                      </button>
                    )}
                  </div>

                  {/* Documents under this corpus */}
                  {expandedCorpora.has(corpus) && (
                    <div className="ml-4 space-y-0.5">
                      {docs.map((doc) => (
                        <DocumentRow
                          key={doc.id}
                          doc={doc}
                          isSelected={selected.has(doc.id)}
                          isDeleting={deleting.has(doc.id)}
                          isPendingDelete={pendingDeleteId === doc.id}
                          deletable={corpus === UPLOAD_CORPUS}
                          onToggle={toggleDocument}
                          onSelectOnly={selectOnly}
                          onDelete={handleDeleteClick}
                        />
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
      <UploadModal isOpen={showUpload} onClose={() => setShowUpload(false)} onUploaded={handleUploaded} />
    </div>
  );
}
