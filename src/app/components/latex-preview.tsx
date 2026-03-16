import { useMemo } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

interface LatexPreviewProps {
  latex: string;
  displayMode?: boolean;
  className?: string;
}

export function LatexPreview({ latex, displayMode = true, className = "" }: LatexPreviewProps) {
  const rendered = useMemo(() => {
    const cleaned = latex.replace(/\s+/g, " ").trim();
    if (!cleaned) return { html: "", error: null };

    try {
      return {
        html: katex.renderToString(cleaned, {
          displayMode,
          throwOnError: false,
          trust: true,
        }),
        error: null,
      };
    } catch (err: any) {
      return { html: "", error: err.message || "Cannot render LaTeX" };
    }
  }, [latex, displayMode]);

  if (!latex.trim()) {
    return (
      <div className={`text-muted-foreground text-sm italic ${className}`}>
        No LaTeX to preview
      </div>
    );
  }

  if (rendered.error) {
    return (
      <div className={`text-center space-y-2 ${className}`}>
        <div className="text-destructive text-sm">{rendered.error}</div>
        <code className="font-mono text-muted-foreground text-xs">{latex}</code>
      </div>
    );
  }

  return (
    <div
      className={`overflow-auto ${className}`}
      dangerouslySetInnerHTML={{ __html: rendered.html }}
    />
  );
}
