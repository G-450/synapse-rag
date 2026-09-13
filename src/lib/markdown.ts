const HTML_BREAK_PATTERN = /(?:\\?<br\s*\/?\s*>|&lt;br\s*\/?\s*&gt;)/gi;

/**
 * Convert the line-break tokens commonly emitted by chat models into
 * Markdown hard breaks. Other raw HTML remains escaped by react-markdown.
 */
export function normalizeAssistantMarkdown(value: string): string {
  return value
    .replace(/\r\n?/g, '\n')
    .replace(HTML_BREAK_PATTERN, '  \n');
}
