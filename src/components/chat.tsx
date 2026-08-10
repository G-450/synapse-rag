'use client';

import { useState, useRef, useEffect } from 'react';
import { useChat } from '@ai-sdk/react';
import { Send, Bot, User, Loader2, BookOpen, Sparkles } from 'lucide-react';
import type { Citation } from './citation-panel';

interface ChatProps {
  documentId: string | null;
  documentName: string | null;
  onCitationsReceived: (citations: Citation[]) => void;
  onToggleCitations: () => void;
}

export default function Chat({
  documentId,
  documentName,
  onCitationsReceived,
  onToggleCitations,
}: ChatProps) {
  const { messages, sendMessage, status } = useChat({
    body: documentId ? { documentId } : undefined,
    onFinish: async () => {
      // Fetch citations after response completes
      if (latestQuery.current) {
        try {
          const res = await fetch('/api/retrieve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              query: latestQuery.current,
              documentId,
              limit: 5,
            }),
          });
          const data = await res.json();
          if (data.chunks) {
            onCitationsReceived(
              data.chunks.map((c: any) => ({
                chunk_id: c.id,
                document_id: c.document_id,
                filename: c.filename || '',
                content: c.content,
                similarity: c.similarity,
              }))
            );
          }
        } catch {
          // Silently fail citation fetch
        }
      }
    },
  });

  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const latestQuery = useRef<string>('');

  const isStreaming = status === 'streaming' || status === 'submitted';

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isStreaming) return;
    latestQuery.current = inputValue;
    sendMessage({ text: inputValue });
    setInputValue('');
  };

  // Extract text content from message parts
  const getMessageText = (msg: (typeof messages)[0]): string => {
    if (!msg.parts) return '';
    return msg.parts
      .filter((p): p is { type: 'text'; text: string } => p.type === 'text')
      .map((p) => p.text)
      .join('');
  };

  const modeLabel = documentId
    ? `Querying: ${documentName || 'Selected Document'}`
    : 'Multi-document mode (all contracts)';

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div
        className="px-5 py-4 border-b flex items-center justify-between flex-shrink-0"
        style={{ borderColor: 'var(--border)', background: 'var(--surface)' }}
      >
        <div>
          <div className="flex items-center gap-2">
            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center"
              style={{ background: 'var(--accent-glow)' }}
            >
              <Sparkles size={16} style={{ color: 'var(--accent)' }} />
            </div>
            <div>
              <h2
                className="text-sm font-semibold"
                style={{ color: 'var(--foreground)' }}
              >
                Synapse RAG
              </h2>
              <p
                className="text-xs"
                style={{ color: 'var(--foreground-faint)' }}
              >
                {modeLabel}
              </p>
            </div>
          </div>
        </div>
        <button
          onClick={onToggleCitations}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all"
          style={{
            background: 'var(--accent-soft)',
            color: 'var(--accent)',
            border: '1px solid var(--accent-glow)',
          }}
        >
          <BookOpen size={14} />
          Sources
        </button>
      </div>

      {/* Messages Area */}
      <div
        className="flex-1 overflow-y-auto p-5 space-y-5"
        style={{ background: 'var(--background)' }}
      >
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center animate-fade-in">
            <div
              className="w-16 h-16 rounded-2xl flex items-center justify-center mb-4 animate-pulse-glow"
              style={{ background: 'var(--accent-glow)' }}
            >
              <Bot
                size={28}
                style={{ color: 'var(--accent)', opacity: 0.6 }}
              />
            </div>
            <h3
              className="text-lg font-semibold mb-1"
              style={{ color: 'var(--foreground-muted)' }}
            >
              Legal Contract Intelligence
            </h3>
            <p
              className="text-sm text-center max-w-md"
              style={{ color: 'var(--foreground-faint)' }}
            >
              Ask questions about legal contracts. Select a specific document
              from the sidebar, or use multi-document mode to compare across
              contracts.
            </p>

            {/* Example queries */}
            <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-2 max-w-lg w-full">
              {[
                'What is the definition of "Intervening Event"?',
                'What are the termination conditions?',
                'Describe the merger consideration terms',
                'What representations does the seller make?',
              ].map((q) => (
                <button
                  key={q}
                  onClick={() => {
                    latestQuery.current = q;
                    sendMessage({ text: q });
                  }}
                  className="px-3 py-2.5 rounded-lg text-xs text-left transition-all glass-card hover:border-opacity-30"
                  style={{ color: 'var(--foreground-muted)' }}
                >
                  &ldquo;{q}&rdquo;
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, index) => (
          <div
            key={m.id}
            className={`flex gap-3 animate-fade-in ${
              m.role === 'user' ? 'justify-end' : 'justify-start'
            }`}
            style={{ animationDelay: `${index * 50}ms` }}
          >
            {m.role !== 'user' && (
              <div
                className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5"
                style={{ background: 'var(--accent-glow)' }}
              >
                <Bot size={16} style={{ color: 'var(--accent)' }} />
              </div>
            )}

            <div
              className="px-4 py-3 rounded-2xl max-w-[80%] text-sm leading-relaxed"
              style={
                m.role === 'user'
                  ? {
                      background: 'var(--accent)',
                      color: '#ffffff',
                      borderBottomRightRadius: '4px',
                    }
                  : {
                      background: 'var(--surface)',
                      color: 'var(--foreground)',
                      border: '1px solid var(--border)',
                      borderBottomLeftRadius: '4px',
                    }
              }
            >
              <div className="whitespace-pre-wrap">{getMessageText(m)}</div>
            </div>

            {m.role === 'user' && (
              <div
                className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5"
                style={{ background: 'var(--surface)' }}
              >
                <User
                  size={16}
                  style={{ color: 'var(--foreground-muted)' }}
                />
              </div>
            )}
          </div>
        ))}

        {/* Typing indicator */}
        {isStreaming && messages[messages.length - 1]?.role !== 'assistant' && (
          <div className="flex gap-3 justify-start animate-fade-in">
            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
              style={{ background: 'var(--accent-glow)' }}
            >
              <Loader2
                className="animate-spin"
                size={16}
                style={{ color: 'var(--accent)' }}
              />
            </div>
            <div
              className="px-4 py-3 rounded-2xl flex items-center gap-1.5"
              style={{
                background: 'var(--surface)',
                border: '1px solid var(--border)',
              }}
            >
              <span className="typing-dot w-1.5 h-1.5 rounded-full" style={{ background: 'var(--accent)' }} />
              <span className="typing-dot w-1.5 h-1.5 rounded-full" style={{ background: 'var(--accent)' }} />
              <span className="typing-dot w-1.5 h-1.5 rounded-full" style={{ background: 'var(--accent)' }} />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div
        className="px-5 py-4 border-t flex-shrink-0"
        style={{ borderColor: 'var(--border)', background: 'var(--surface)' }}
      >
        <form onSubmit={handleSubmit} className="flex gap-3">
          <input
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder={
              documentId
                ? `Ask about ${documentName || 'this contract'}...`
                : 'Ask across all contracts...'
            }
            className="flex-1 px-4 py-3 rounded-xl text-sm transition-all focus:outline-none"
            style={{
              background: 'var(--background)',
              color: 'var(--foreground)',
              border: '1px solid var(--border)',
            }}
            disabled={isStreaming}
          />
          <button
            type="submit"
            disabled={isStreaming || !inputValue.trim()}
            className="px-4 py-3 rounded-xl font-medium transition-all flex items-center gap-2 text-sm disabled:opacity-40 disabled:cursor-not-allowed"
            style={{
              background: 'var(--accent)',
              color: '#ffffff',
            }}
          >
            {isStreaming ? (
              <Loader2 className="animate-spin" size={18} />
            ) : (
              <Send size={18} />
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
