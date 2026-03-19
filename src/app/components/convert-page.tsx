import { useState, useEffect, useRef } from "react";
import { Button } from "@/app/components/ui/button";
import { Card, CardContent } from "@/app/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/app/components/ui/tabs";
import { Textarea } from "@/app/components/ui/textarea";
import { ScrollArea } from "@/app/components/ui/scroll-area";
import { Sheet, SheetContent, SheetTrigger } from "@/app/components/ui/sheet";
import { Skeleton } from "@/app/components/ui/skeleton";
import { Separator } from "@/app/components/ui/separator";
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from "@/app/components/ui/resizable";
import {
  Upload, Copy, Send, MessageSquare, Loader2, ImageIcon, X,
  ZoomIn, ZoomOut, RotateCcw, WrapText, Type, Minus, Plus,
} from "lucide-react";
import { Input } from "@/app/components/ui/input";
import { LatexPreview } from "@/app/components/latex-preview";
import { toast } from "sonner";

// ============================================================
// ImagePanel — Upload area with drag & drop
// ============================================================
interface ImageItem {
  id: string;
  url: string;
  file: File;
  latex: string;
  status: "pending" | "processing" | "done" | "error";
}

interface ImagePanelProps {
  images: ImageItem[];
  activeImageId: string | null;
  isProcessing: boolean;
  onUpload: (files: File[]) => void;
  onSelect: (id: string) => void;
  onRemove: (id: string) => void;
  onClearAll: () => void;
  compact?: boolean;
}

function ImagePanel({ images, activeImageId, isProcessing, onUpload, onSelect, onRemove, onClearAll, compact }: ImagePanelProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const activeImage = images.find((img) => img.id === activeImageId);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    if (files.length > 0) onUpload(files);
    event.target.value = "";
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files).filter((f) => f.type.startsWith("image/"));
    if (files.length > 0) onUpload(files);
  };

  const handleDragOver = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(false); };

  // Compact mode for tablet/mobile: horizontal strip with thumbnails
  if (compact) {
    return (
      <div
        className="flex items-center gap-2 p-2 border-b border-border bg-card overflow-x-auto"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        {/* Upload button */}
        <div
          onClick={() => !isProcessing && fileInputRef.current?.click()}
          className={`h-12 w-14 rounded-lg border-2 border-dashed flex items-center justify-center cursor-pointer flex-shrink-0 transition-all ${
            isDragging ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
          }`}
        >
          <Plus className="h-4 w-4 text-muted-foreground" />
        </div>
        {/* Thumbnail list */}
        {images.map((img) => (
          <div
            key={img.id}
            onClick={() => onSelect(img.id)}
            className={`h-12 w-16 rounded-lg border overflow-hidden flex-shrink-0 relative cursor-pointer transition-all ${
              img.id === activeImageId ? "ring-2 ring-primary border-primary" : "border-border hover:border-primary/50"
            }`}
          >
            <img src={img.url} alt="" className="w-full h-full object-cover" />
            {img.status === "processing" && (
              <div className="absolute inset-0 bg-background/60 flex items-center justify-center">
                <Loader2 className="h-3 w-3 text-primary animate-spin" />
              </div>
            )}
            {img.status === "done" && (
              <div className="absolute top-0.5 right-0.5 w-3 h-3 rounded-full bg-green-500 border border-background" />
            )}
            {img.status === "error" && (
              <div className="absolute top-0.5 right-0.5 w-3 h-3 rounded-full bg-red-500 border border-background" />
            )}
            <button
              onClick={(e) => { e.stopPropagation(); onRemove(img.id); }}
              className="absolute top-0 left-0 w-4 h-4 bg-black/60 text-white rounded-br flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity"
            >
              <X className="h-2.5 w-2.5" />
            </button>
          </div>
        ))}
        {images.length > 0 && (
          <Button variant="ghost" size="sm" onClick={onClearAll} className="h-7 text-[10px] flex-shrink-0 px-2">
            Clear all
          </Button>
        )}
        <input ref={fileInputRef} type="file" className="hidden" accept="image/*" multiple onChange={handleFileChange} />
      </div>
    );
  }

  // Full mode for desktop sidebar
  return (
    <div className="h-full flex flex-col p-3 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
          Images {images.length > 0 && <span className="text-primary">({images.length})</span>}
        </h3>
        {images.length > 0 && (
          <Button variant="ghost" size="sm" onClick={onClearAll} className="h-6 text-[10px] px-2">
            <X className="h-3 w-3 mr-1" /> Clear all
          </Button>
        )}
      </div>

      {/* Upload area */}
      <div
        onClick={() => !isProcessing && fileInputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`
          relative border-2 border-dashed rounded-xl p-3 text-center cursor-pointer
          transition-all duration-200 ease-in-out flex-shrink-0
          ${isDragging ? "border-primary bg-primary/5 scale-[1.02]" : "border-border hover:border-primary/50 hover:bg-muted/30"}
          ${isProcessing ? "pointer-events-none opacity-60" : ""}
        `}
      >
        <div className="space-y-1">
          <div className="mx-auto w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
            <Upload className="h-3.5 w-3.5 text-primary" />
          </div>
          <p className="text-[11px] font-medium">{isDragging ? "Drop here" : "Upload or drag & drop"}</p>
          <p className="text-[9px] text-muted-foreground">PNG, JPG, WEBP · Multiple files · Ctrl+V</p>
        </div>
        <input ref={fileInputRef} type="file" className="hidden" accept="image/*" multiple onChange={handleFileChange} />
      </div>

      {/* Image gallery */}
      {images.length > 0 ? (
        <ScrollArea className="flex-1 min-h-0">
          <div className="space-y-2 pr-2">
            {images.map((img) => (
              <div
                key={img.id}
                onClick={() => onSelect(img.id)}
                className={`relative rounded-lg border overflow-hidden cursor-pointer transition-all group ${
                  img.id === activeImageId
                    ? "ring-2 ring-primary border-primary"
                    : "border-border hover:border-primary/50"
                }`}
              >
                <img src={img.url} alt="" className="w-full h-auto object-contain max-h-32" />
                {/* Status overlay */}
                {img.status === "processing" && (
                  <div className="absolute inset-0 bg-background/50 flex items-center justify-center">
                    <Loader2 className="h-5 w-5 text-primary animate-spin" />
                  </div>
                )}
                {/* Status badge */}
                <div className="absolute top-1 left-1">
                  {img.status === "done" && (
                    <span className="text-[8px] px-1 py-0.5 rounded bg-green-500/90 text-white font-medium">Done</span>
                  )}
                  {img.status === "error" && (
                    <span className="text-[8px] px-1 py-0.5 rounded bg-red-500/90 text-white font-medium">Error</span>
                  )}
                  {img.status === "pending" && (
                    <span className="text-[8px] px-1 py-0.5 rounded bg-muted/90 text-muted-foreground font-medium">Pending</span>
                  )}
                </div>
                {/* Remove button */}
                <button
                  onClick={(e) => { e.stopPropagation(); onRemove(img.id); }}
                  className="absolute top-1 right-1 w-5 h-5 bg-black/60 text-white rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
            ))}
          </div>
        </ScrollArea>
      ) : (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center space-y-1">
            <ImageIcon className="mx-auto h-10 w-10 text-muted-foreground/20" />
            <p className="text-[10px] text-muted-foreground">No images</p>
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================
// Code Panel — Editable LaTeX with toolbar
// ============================================================
function CodePanel({
  latexCode, onCodeChange, isProcessing, fontSize, onFontSizeChange, wordWrap, onWordWrapToggle, onCopy,
}: {
  latexCode: string;
  onCodeChange: (v: string) => void;
  isProcessing: boolean;
  fontSize: number;
  onFontSizeChange: (v: number) => void;
  wordWrap: boolean;
  onWordWrapToggle: () => void;
  onCopy: () => void;
}) {
  return (
    <div className="flex-1 flex flex-col min-w-0 min-h-0 border-r border-border md:border-r-0 lg:border-r">
      {/* Toolbar */}
      <div className="flex items-center gap-1.5 px-3 py-1.5 border-b border-border bg-muted/30 flex-shrink-0">
        <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide mr-1">Code</span>
        <Separator orientation="vertical" className="h-4" />

        {/* Font size */}
        <div className="flex items-center gap-0.5">
          <Type className="h-3 w-3 text-muted-foreground" />
          <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => onFontSizeChange(Math.max(10, fontSize - 2))}>
            <Minus className="h-2.5 w-2.5" />
          </Button>
          <span className="text-[10px] font-mono w-5 text-center">{fontSize}</span>
          <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => onFontSizeChange(Math.min(24, fontSize + 2))}>
            <Plus className="h-2.5 w-2.5" />
          </Button>
        </div>

        <Separator orientation="vertical" className="h-4" />

        {/* Word wrap */}
        <Button
          variant={wordWrap ? "secondary" : "ghost"}
          size="icon"
          className="h-6 w-6"
          onClick={onWordWrapToggle}
          title="Word wrap"
        >
          <WrapText className="h-3 w-3" />
        </Button>

        <div className="flex-1" />

        {/* Copy */}
        <Button variant="ghost" size="sm" className="h-6 text-[10px] px-2" onClick={onCopy} disabled={!latexCode}>
          <Copy className="h-3 w-3 mr-1" /> Copy
        </Button>
      </div>

      {/* Editor */}
      <div className="flex-1 p-2 overflow-hidden">
        {isProcessing ? (
          <div className="space-y-2 p-2">
            <Skeleton className="h-3 w-3/4" />
            <Skeleton className="h-3 w-1/2" />
            <Skeleton className="h-3 w-2/3" />
          </div>
        ) : (
          <Textarea
            value={latexCode}
            onChange={(e) => onCodeChange(e.target.value)}
            placeholder="LaTeX code will appear here..."
            className="h-full font-mono resize-none bg-muted/20 border-muted"
            style={{
              fontSize: `${fontSize}px`,
              lineHeight: 1.6,
              whiteSpace: wordWrap ? "pre-wrap" : "pre",
              overflowWrap: wordWrap ? "break-word" : "normal",
            }}
          />
        )}
      </div>
    </div>
  );
}

// ============================================================
// Preview Panel — Rendered KaTeX with zoom
// ============================================================
function PreviewPanel({
  latexCode, isProcessing, zoom, onZoomChange,
}: {
  latexCode: string;
  isProcessing: boolean;
  zoom: number;
  onZoomChange: (v: number) => void;
}) {
  return (
    <div className="flex-1 flex flex-col min-w-0 min-h-0">
      {/* Toolbar */}
      <div className="flex items-center gap-1.5 px-3 py-1.5 border-b border-border bg-muted/30 flex-shrink-0">
        <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide mr-1">Preview</span>
        <Separator orientation="vertical" className="h-4" />

        <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => onZoomChange(Math.max(50, zoom - 25))}>
          <ZoomOut className="h-3 w-3" />
        </Button>
        <span className="text-[10px] font-mono w-8 text-center">{zoom}%</span>
        <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => onZoomChange(Math.min(200, zoom + 25))}>
          <ZoomIn className="h-3 w-3" />
        </Button>

        <Separator orientation="vertical" className="h-4" />

        <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => onZoomChange(100)} title="Reset zoom">
          <RotateCcw className="h-3 w-3" />
        </Button>

        <div className="flex-1" />

        {/* Zoom presets */}
        <div className="hidden sm:flex gap-0.5">
          {[75, 100, 150].map(z => (
            <Button
              key={z}
              variant={zoom === z ? "secondary" : "ghost"}
              size="sm"
              className="h-5 text-[9px] px-1.5"
              onClick={() => onZoomChange(z)}
            >
              {z}%
            </Button>
          ))}
        </div>
      </div>

      {/* Rendered preview */}
      <ScrollArea className="flex-1">
        <div className="min-h-full flex items-center justify-center p-4 md:p-6">
          {isProcessing ? (
            <div className="space-y-3 text-center">
              <Loader2 className="mx-auto h-6 w-6 text-primary animate-spin" />
              <p className="text-xs text-muted-foreground">Recognizing formula...</p>
            </div>
          ) : latexCode ? (
            <div
              className="transition-transform duration-200 origin-center"
              style={{ transform: `scale(${zoom / 100})` }}
            >
              <div className="bg-card border rounded-xl p-6 md:p-8 shadow-sm">
                <LatexPreview latex={latexCode} className="text-center" />
              </div>
            </div>
          ) : (
            <div className="text-center space-y-1">
              <p className="text-muted-foreground text-xs">Upload an image to see the preview</p>
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}

// ============================================================
// ChatPanel — Collapsible AI chat
// ============================================================
function ChatPanel() {
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Xin chào! Mình là MathSnap Assistant 🧮 Bạn cần hỗ trợ gì về toán học hôm nay?" }
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;
    const currentInput = inputValue;
    setMessages((prev) => [...prev, { role: "user", content: currentInput }]);
    setInputValue("");
    setIsLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: currentInput }),
      });
      const data = await response.json();
      if (data.error) {
        setMessages((prev) => [...prev, { role: "assistant", content: `Error: ${data.error}` }]);
      } else {
        setMessages((prev) => [...prev, { role: "assistant", content: data.content }]);
      }
    } catch {
      setMessages((prev) => [...prev, { role: "assistant", content: "Failed to connect to AI." }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="h-full flex flex-col overflow-hidden bg-card">
      <div className="p-3 border-b">
        <h3 className="font-medium text-sm">MathSnap Assistant</h3>
        <p className="text-[10px] text-muted-foreground">Powered by Llama 3.3 · Mini-CoMER</p>
      </div>
      <ScrollArea className="flex-1 p-3">
        {messages.map((m, i) => (
          <div key={i} className={`mb-2 flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`p-2.5 rounded-lg text-xs max-w-[85%] ${
              m.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted"
            }`}>
              {m.content}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start mb-2">
            <div className="bg-muted p-2.5 rounded-lg">
              <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
            </div>
          </div>
        )}
        <div ref={scrollRef} />
      </ScrollArea>
      <div className="p-2.5 border-t">
        <div className="flex gap-1.5">
          <Input
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder="Ask about math..."
            className="text-xs h-8"
            disabled={isLoading}
          />
          <Button onClick={handleSend} size="icon" className="h-8 w-8" disabled={isLoading}>
            <Send className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>
    </div>
  );
}

// ============================================================
// ConvertPage — Main page with responsive split layout
// ============================================================
export function ConvertPage() {
  const [images, setImages] = useState<ImageItem[]>([]);
  const [activeImageId, setActiveImageId] = useState<string | null>(null);
  const [latexCode, setLatexCode] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState("code");
  const [chatOpen, setChatOpen] = useState(false);

  // Toolbar state
  const [fontSize, setFontSize] = useState(14);
  const [wordWrap, setWordWrap] = useState(true);
  const [previewZoom, setPreviewZoom] = useState(100);

  const processImage = async (item: ImageItem) => {
    setImages((prev) => prev.map((img) => img.id === item.id ? { ...img, status: "processing" as const } : img));
    try {
      const formData = new FormData();
      formData.append("file", item.file);
      const response = await fetch("/api/predict", { method: "POST", body: formData });
      if (!response.ok) throw new Error("Backend not responding");
      const data = await response.json();
      setImages((prev) => prev.map((img) => img.id === item.id ? { ...img, latex: data.latex, status: "done" as const } : img));
      // If this is the active image, update the code panel
      setActiveImageId((currentId) => {
        if (currentId === item.id) setLatexCode(data.latex);
        return currentId;
      });
      toast.success(`Converted: ${item.file.name}`);
    } catch {
      setImages((prev) => prev.map((img) => img.id === item.id ? { ...img, status: "error" as const } : img));
      toast.error(`Failed: ${item.file.name}`);
    }
  };

  const handleImageUpload = async (files: File[]) => {
    const newItems: ImageItem[] = files.map((file) => ({
      id: `img-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      url: URL.createObjectURL(file),
      file,
      latex: "",
      status: "pending" as const,
    }));

    setImages((prev) => [...prev, ...newItems]);
    // Select the first new image if nothing selected
    if (!activeImageId) {
      setActiveImageId(newItems[0].id);
    }
    setActiveTab("code");
    setIsProcessing(true);

    // Process all images sequentially
    for (const item of newItems) {
      await processImage(item);
    }
    setIsProcessing(false);
    setActiveTab("preview");
  };

  const handleSelectImage = (id: string) => {
    setActiveImageId(id);
    const img = images.find((i) => i.id === id);
    if (img) setLatexCode(img.latex);
  };

  const handleRemoveImage = (id: string) => {
    setImages((prev) => {
      const updated = prev.filter((img) => img.id !== id);
      if (activeImageId === id) {
        const next = updated[0] || null;
        setActiveImageId(next?.id || null);
        setLatexCode(next?.latex || "");
      }
      return updated;
    });
  };

  const handleClearAll = () => {
    images.forEach((img) => URL.revokeObjectURL(img.url));
    setImages([]);
    setActiveImageId(null);
    setLatexCode("");
    setActiveTab("code");
  };

  const handleCopyLatex = () => {
    navigator.clipboard.writeText(latexCode);
    toast.success("LaTeX copied to clipboard!");
  };

  // Sync latexCode edits back to active image
  const handleCodeChange = (code: string) => {
    setLatexCode(code);
    if (activeImageId) {
      setImages((prev) => prev.map((img) => img.id === activeImageId ? { ...img, latex: code } : img));
    }
  };

  // Paste listener
  useEffect(() => {
    const handlePaste = (event: ClipboardEvent) => {
      const files: File[] = [];
      const items = event.clipboardData?.items;
      if (items) {
        for (let i = 0; i < items.length; i++) {
          if (items[i].type.indexOf("image") !== -1) {
            const file = items[i].getAsFile();
            if (file) files.push(file);
          }
        }
      }
      if (files.length > 0) handleImageUpload(files);
    };
    window.addEventListener("paste", handlePaste);
    return () => window.removeEventListener("paste", handlePaste);
  }, [activeImageId]);

  return (
    <div className="h-[calc(100vh-3.5rem)] pb-14 md:pb-0 flex flex-col overflow-hidden">

      {/* ===== MOBILE (< md): Compact upload + tabbed code/preview ===== */}
      <div className="md:hidden flex flex-col flex-1 min-h-0">
        {/* Compact image upload strip */}
        <ImagePanel
          images={images}
          activeImageId={activeImageId}
          isProcessing={isProcessing}
          onUpload={handleImageUpload}
          onSelect={handleSelectImage}
          onRemove={handleRemoveImage}
          onClearAll={handleClearAll}
          compact
        />

        {/* Tabbed code/preview */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col min-h-0">
          <div className="flex items-center px-3 py-1.5 border-b border-border bg-card">
            <TabsList className="h-7">
              <TabsTrigger value="code" className="text-[10px] h-6 px-2.5">Code</TabsTrigger>
              <TabsTrigger value="preview" className="text-[10px] h-6 px-2.5">Preview</TabsTrigger>
            </TabsList>
            <div className="flex-1" />
            <Button variant="ghost" size="sm" className="h-6 text-[10px] px-2" onClick={handleCopyLatex} disabled={!latexCode}>
              <Copy className="h-3 w-3 mr-1" /> Copy
            </Button>
          </div>

          <TabsContent value="code" className="flex-1 min-h-0 overflow-hidden">
            <div className="h-full p-2">
              {isProcessing ? (
                <div className="space-y-2 p-2">
                  <Skeleton className="h-3 w-3/4" />
                  <Skeleton className="h-3 w-1/2" />
                </div>
              ) : (
                <Textarea
                  value={latexCode}
                  onChange={(e) => handleCodeChange(e.target.value)}
                  placeholder="LaTeX code will appear here..."
                  className="h-full font-mono text-xs resize-none bg-muted/20 border-muted"
                  style={{ fontSize: `${fontSize}px` }}
                />
              )}
            </div>
          </TabsContent>

          <TabsContent value="preview" className="flex-1 min-h-0 overflow-auto">
            <div className="min-h-full flex items-center justify-center p-4">
              {isProcessing ? (
                <Loader2 className="h-6 w-6 text-primary animate-spin" />
              ) : latexCode ? (
                <div style={{ transform: `scale(${previewZoom / 100})` }} className="transition-transform">
                  <div className="bg-card border rounded-xl p-6 shadow-sm">
                    <LatexPreview latex={latexCode} className="text-center" />
                  </div>
                </div>
              ) : (
                <p className="text-muted-foreground text-xs">Upload an image to see preview</p>
              )}
            </div>
          </TabsContent>
        </Tabs>
      </div>

      {/* ===== TABLET (md to lg): Compact upload top + side-by-side code/preview ===== */}
      <div className="hidden md:flex lg:hidden flex-col flex-1 min-h-0">
        {/* Compact image upload strip */}
        <ImagePanel
          images={images}
          activeImageId={activeImageId}
          isProcessing={isProcessing}
          onUpload={handleImageUpload}
          onSelect={handleSelectImage}
          onRemove={handleRemoveImage}
          onClearAll={handleClearAll}
          compact
        />

        {/* Side-by-side code + preview (resizable) */}
        <ResizablePanelGroup direction="horizontal" className="flex-1 min-h-0">
          <ResizablePanel defaultSize={50} minSize={25}>
            <CodePanel
              latexCode={latexCode}
              onCodeChange={handleCodeChange}
              isProcessing={isProcessing}
              fontSize={fontSize}
              onFontSizeChange={setFontSize}
              wordWrap={wordWrap}
              onWordWrapToggle={() => setWordWrap(!wordWrap)}
              onCopy={handleCopyLatex}
            />
          </ResizablePanel>
          <ResizableHandle withHandle />
          <ResizablePanel defaultSize={50} minSize={25}>
            <PreviewPanel
              latexCode={latexCode}
              isProcessing={isProcessing}
              zoom={previewZoom}
              onZoomChange={setPreviewZoom}
            />
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>

      {/* ===== DESKTOP (≥ lg): Resizable panels ===== */}
      <ResizablePanelGroup
        key={chatOpen ? "chat-open" : "chat-closed"}
        direction="horizontal"
        className="hidden lg:flex flex-1 min-h-0"
      >
        {/* Left: Image sidebar */}
        <ResizablePanel defaultSize={chatOpen ? 16 : 18} minSize={10} maxSize={30}>
          <div className="h-full border-r border-border bg-card">
            <ImagePanel
              images={images}
              activeImageId={activeImageId}
              isProcessing={isProcessing}
              onUpload={handleImageUpload}
              onSelect={handleSelectImage}
              onRemove={handleRemoveImage}
              onClearAll={handleClearAll}
            />
          </div>
        </ResizablePanel>
        <ResizableHandle withHandle />

        {/* Center: Code panel */}
        <ResizablePanel defaultSize={chatOpen ? 32 : 41} minSize={20}>
          <CodePanel
            latexCode={latexCode}
            onCodeChange={handleCodeChange}
            isProcessing={isProcessing}
            fontSize={fontSize}
            onFontSizeChange={setFontSize}
            wordWrap={wordWrap}
            onWordWrapToggle={() => setWordWrap(!wordWrap)}
            onCopy={handleCopyLatex}
          />
        </ResizablePanel>
        <ResizableHandle withHandle />

        {/* Right: Preview panel */}
        <ResizablePanel defaultSize={chatOpen ? 32 : 41} minSize={20}>
          <PreviewPanel
            latexCode={latexCode}
            isProcessing={isProcessing}
            zoom={previewZoom}
            onZoomChange={setPreviewZoom}
          />
        </ResizablePanel>

        {/* Chat sidebar (resizable, collapsible) */}
        {chatOpen && (
          <>
            <ResizableHandle withHandle />
            <ResizablePanel defaultSize={20} minSize={15} maxSize={40}>
              <div className="h-full border-l border-border bg-card">
                <ChatPanel />
              </div>
            </ResizablePanel>
          </>
        )}
      </ResizablePanelGroup>

      {/* Chat toggle — Desktop */}
      <Button
        variant="outline"
        size="icon"
        onClick={() => setChatOpen(!chatOpen)}
        className="hidden lg:flex fixed bottom-6 right-6 z-40 h-11 w-11 rounded-full shadow-lg bg-card hover:bg-primary hover:text-primary-foreground transition-colors"
      >
        <MessageSquare className="h-4.5 w-4.5" />
      </Button>

      {/* Chat — Mobile/Tablet */}
      <Sheet>
        <SheetTrigger asChild>
          <Button variant="outline" size="icon" className="lg:hidden fixed bottom-18 right-4 z-40 h-11 w-11 rounded-full shadow-lg bg-card">
            <MessageSquare className="h-4.5 w-4.5" />
          </Button>
        </SheetTrigger>
        <SheetContent side="right" className="w-80 p-0">
          <ChatPanel />
        </SheetContent>
      </Sheet>
    </div>
  );
}
