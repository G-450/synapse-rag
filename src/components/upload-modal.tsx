'use client';

import { useRef, useState } from 'react';
import { FileUp, Loader2, UploadCloud, X } from 'lucide-react';

export interface UploadResult {
  status: 'success';
  document_id: string;
  filename: string;
  chunks_created: number;
}

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUploaded: (result: UploadResult) => void;
}

const ACCEPTED = ['.pdf', '.docx', '.txt', '.md'];

export default function UploadModal({ isOpen, onClose, onUploaded }: UploadModalProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');

  if (!isOpen) return null;

  const chooseFile = (candidate?: File) => {
    if (!candidate) return;
    const extension = candidate.name.slice(candidate.name.lastIndexOf('.')).toLowerCase();
    if (!ACCEPTED.includes(extension)) {
      setError('Choose a PDF, DOCX, TXT, or Markdown contract.');
      return;
    }
    setFile(candidate);
    setError('');
  };

  const upload = async () => {
    if (!file || uploading) return;
    setUploading(true);
    setError('');
    try {
      const body = new FormData();
      body.append('file', file);
      const response = await fetch('/api/upload', { method: 'POST', body });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || result.error || 'Upload failed');
      onUploaded(result as UploadResult);
      setFile(null);
      onClose();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/60 p-4" role="dialog" aria-modal="true" aria-label="Upload contract">
      <div className="w-full max-w-lg rounded-2xl border p-5 shadow-2xl" style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}>
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="font-semibold" style={{ color: 'var(--foreground)' }}>Upload Contract</h2>
            <p className="text-xs" style={{ color: 'var(--foreground-faint)' }}>PDF, DOCX, TXT, or MD up to 25 MB</p>
          </div>
          <button onClick={onClose} disabled={uploading} className="rounded-lg p-2 hover:opacity-70" aria-label="Close upload dialog">
            <X size={18} />
          </button>
        </div>

        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => { event.preventDefault(); setDragging(false); chooseFile(event.dataTransfer.files[0]); }}
          className="flex min-h-48 w-full flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 text-center transition-all"
          style={{ borderColor: dragging ? 'var(--accent)' : 'var(--border)', background: dragging ? 'var(--accent-glow)' : 'var(--background)' }}
        >
          {file ? <FileUp size={34} style={{ color: 'var(--accent)' }} /> : <UploadCloud size={34} style={{ color: 'var(--foreground-faint)' }} />}
          <span className="mt-3 text-sm font-medium" style={{ color: 'var(--foreground)' }}>{file?.name || 'Drop a contract here'}</span>
          <span className="mt-1 text-xs" style={{ color: 'var(--foreground-faint)' }}>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : 'or click to browse'}</span>
        </button>
        <input ref={inputRef} className="hidden" type="file" accept={ACCEPTED.join(',')} onChange={(event) => chooseFile(event.target.files?.[0])} />

        {uploading && <div className="mt-4 h-1.5 overflow-hidden rounded-full" style={{ background: 'var(--background)' }}><div className="h-full w-2/3 animate-pulse rounded-full" style={{ background: 'var(--accent)' }} /></div>}
        {error && <p className="mt-3 text-xs" style={{ color: 'var(--error)' }}>{error}</p>}

        <div className="mt-5 flex justify-end gap-2">
          <button onClick={onClose} disabled={uploading} className="rounded-lg px-4 py-2 text-sm" style={{ color: 'var(--foreground-muted)' }}>Cancel</button>
          <button onClick={upload} disabled={!file || uploading} className="flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium disabled:opacity-40" style={{ background: 'var(--accent)', color: '#fff' }}>
            {uploading && <Loader2 size={15} className="animate-spin" />}
            {uploading ? 'Parsing and embedding…' : 'Upload and index'}
          </button>
        </div>
      </div>
    </div>
  );
}
