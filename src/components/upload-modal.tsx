'use client';

import { useRef, useState } from 'react';
import { Check, FileUp, Loader2, UploadCloud, X } from 'lucide-react';

export interface UploadResult {
  status: 'success';
  document_id: string;
  filename: string;
  chunks_created: number;
}

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUploaded: (results: UploadResult[]) => void;
}

type QueueStatus = 'pending' | 'uploading' | 'done' | 'error';

interface QueuedFile {
  id: string;
  file: File;
  status: QueueStatus;
  message?: string;
  chunks?: number;
}

const ACCEPTED = ['.pdf', '.docx', '.txt', '.md'];

const extensionOf = (name: string) =>
  name.slice(name.lastIndexOf('.')).toLowerCase();

export default function UploadModal({ isOpen, onClose, onUploaded }: UploadModalProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [queue, setQueue] = useState<QueuedFile[]>([]);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');

  if (!isOpen) return null;

  const addFiles = (candidates: FileList | File[] | null) => {
    const incoming = Array.from(candidates || []);
    if (incoming.length === 0) return;

    const rejected = incoming.filter((f) => !ACCEPTED.includes(extensionOf(f.name)));
    const accepted = incoming.filter((f) => ACCEPTED.includes(extensionOf(f.name)));

    setQueue((current) => {
      // Same name and size twice in a row is a re-pick, not a second contract.
      const existing = new Set(current.map((item) => `${item.file.name}:${item.file.size}`));
      const fresh = accepted
        .filter((f) => !existing.has(`${f.name}:${f.size}`))
        .map((f) => ({
          id: `${f.name}:${f.size}:${f.lastModified}`,
          file: f,
          status: 'pending' as QueueStatus,
        }));
      return [...current, ...fresh];
    });

    setError(
      rejected.length > 0
        ? `Skipped ${rejected.length === 1 ? rejected[0].name : `${rejected.length} files`} — only PDF, DOCX, TXT, or Markdown contracts are supported.`
        : ''
    );
  };

  const removeFile = (id: string) =>
    setQueue((current) => current.filter((item) => item.id !== id));

  const closeModal = () => {
    if (uploading) return;
    setQueue([]);
    setError('');
    onClose();
  };

  const uploadAll = async () => {
    const todo = queue.filter((item) => item.status !== 'done');
    if (todo.length === 0 || uploading) return;

    setUploading(true);
    setError('');
    const succeeded: UploadResult[] = [];

    // One at a time: parsing, OCR, and embedding are heavy on the Python backend.
    for (const item of todo) {
      setQueue((current) =>
        current.map((entry) =>
          entry.id === item.id ? { ...entry, status: 'uploading', message: undefined } : entry
        )
      );
      try {
        const body = new FormData();
        body.append('file', item.file);
        const response = await fetch('/api/upload', { method: 'POST', body });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || result.error || 'Upload failed');

        succeeded.push(result as UploadResult);
        setQueue((current) =>
          current.map((entry) =>
            entry.id === item.id
              ? { ...entry, status: 'done', chunks: (result as UploadResult).chunks_created }
              : entry
          )
        );
      } catch (caught) {
        setQueue((current) =>
          current.map((entry) =>
            entry.id === item.id
              ? {
                  ...entry,
                  status: 'error',
                  message: caught instanceof Error ? caught.message : 'Upload failed',
                }
              : entry
          )
        );
      }
    }

    setUploading(false);
    if (succeeded.length > 0) onUploaded(succeeded);

    const failed = todo.length - succeeded.length;
    if (failed === 0) {
      setQueue([]);
      onClose();
    } else {
      // Leave the modal open so the failures stay visible and can be retried.
      setError(`${failed} of ${todo.length} contracts could not be indexed.`);
    }
  };

  const pendingCount = queue.filter((item) => item.status !== 'done').length;
  const totalSize = queue.reduce((sum, item) => sum + item.file.size, 0);

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/60 p-4" role="dialog" aria-modal="true" aria-label="Upload contracts">
      <div className="w-full max-w-lg rounded-2xl border p-5 shadow-2xl" style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}>
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="font-semibold" style={{ color: 'var(--foreground)' }}>Upload Contracts</h2>
            <p className="text-xs" style={{ color: 'var(--foreground-faint)' }}>PDF, DOCX, TXT, or MD — up to 25 MB each</p>
          </div>
          <button onClick={closeModal} disabled={uploading} className="rounded-lg p-2 hover:opacity-70 disabled:opacity-40" aria-label="Close upload dialog">
            <X size={18} />
          </button>
        </div>

        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => { event.preventDefault(); setDragging(false); addFiles(event.dataTransfer.files); }}
          className="flex min-h-36 w-full flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 text-center transition-all"
          style={{ borderColor: dragging ? 'var(--accent)' : 'var(--border)', background: dragging ? 'var(--accent-glow)' : 'var(--background)' }}
        >
          {queue.length > 0 ? <FileUp size={34} style={{ color: 'var(--accent)' }} /> : <UploadCloud size={34} style={{ color: 'var(--foreground-faint)' }} />}
          <span className="mt-3 text-sm font-medium" style={{ color: 'var(--foreground)' }}>
            {queue.length === 0
              ? 'Drop contracts here'
              : `${queue.length} contract${queue.length === 1 ? '' : 's'} ready`}
          </span>
          <span className="mt-1 text-xs" style={{ color: 'var(--foreground-faint)' }}>
            {queue.length === 0
              ? 'or click to browse — you can pick several at once'
              : `${(totalSize / 1024 / 1024).toFixed(2)} MB total · click to add more`}
          </span>
        </button>
        <input
          ref={inputRef}
          className="hidden"
          type="file"
          multiple
          accept={ACCEPTED.join(',')}
          onChange={(event) => {
            addFiles(event.target.files);
            // Let the same file be picked again after it is removed from the queue.
            event.target.value = '';
          }}
        />

        {queue.length > 0 && (
          <ul className="mt-3 max-h-48 space-y-1 overflow-y-auto">
            {queue.map((item) => (
              <li
                key={item.id}
                className="flex items-center gap-2 rounded-lg px-2.5 py-2 text-xs"
                style={{ background: 'var(--background)' }}
              >
                <span className="flex h-4 w-4 flex-shrink-0 items-center justify-center">
                  {item.status === 'uploading' ? (
                    <Loader2 size={13} className="animate-spin" style={{ color: 'var(--accent)' }} />
                  ) : item.status === 'done' ? (
                    <Check size={13} strokeWidth={3} style={{ color: 'var(--success)' }} />
                  ) : item.status === 'error' ? (
                    <X size={13} strokeWidth={3} style={{ color: 'var(--error)' }} />
                  ) : (
                    <FileUp size={13} style={{ color: 'var(--foreground-faint)' }} />
                  )}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate" style={{ color: 'var(--foreground)' }}>
                    {item.file.name}
                  </span>
                  {item.status === 'error' && (
                    <span className="block truncate" style={{ color: 'var(--error)' }}>
                      {item.message}
                    </span>
                  )}
                </span>
                <span className="flex-shrink-0" style={{ color: 'var(--foreground-faint)' }}>
                  {item.status === 'done'
                    ? `${item.chunks} chunks`
                    : item.status === 'uploading'
                      ? 'Indexing…'
                      : `${(item.file.size / 1024 / 1024).toFixed(2)} MB`}
                </span>
                {!uploading && item.status !== 'done' && (
                  <button
                    onClick={() => removeFile(item.id)}
                    className="flex-shrink-0 rounded p-0.5 opacity-50 transition-opacity hover:opacity-100"
                    aria-label={`Remove ${item.file.name} from the upload queue`}
                  >
                    <X size={13} />
                  </button>
                )}
              </li>
            ))}
          </ul>
        )}

        {error && <p className="mt-3 text-xs" style={{ color: 'var(--error)' }}>{error}</p>}

        <div className="mt-5 flex justify-end gap-2">
          <button onClick={closeModal} disabled={uploading} className="rounded-lg px-4 py-2 text-sm disabled:opacity-40" style={{ color: 'var(--foreground-muted)' }}>
            {queue.some((item) => item.status === 'done') ? 'Done' : 'Cancel'}
          </button>
          <button onClick={uploadAll} disabled={pendingCount === 0 || uploading} className="flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium disabled:opacity-40" style={{ background: 'var(--accent)', color: '#fff' }}>
            {uploading && <Loader2 size={15} className="animate-spin" />}
            {uploading
              ? 'Parsing and embedding…'
              : `Upload and index${pendingCount > 1 ? ` (${pendingCount})` : ''}`}
          </button>
        </div>
      </div>
    </div>
  );
}
