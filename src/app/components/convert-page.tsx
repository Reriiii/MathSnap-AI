import { useState, useEffect, useRef } from "react";
import { Button } from "@/app/components/ui/button";
import { Card, CardContent } from "@/app/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/app/components/ui/tabs";
import { Textarea } from "@/app/components/ui/textarea";
import { ScrollArea } from "@/app/components/ui/scroll-area";
import { Sheet, SheetContent, SheetTrigger } from "@/app/components/ui/sheet";
import { Skeleton } from "@/app/components/ui/skeleton";
import { Separator } from "@/app/components/ui/separator";
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
interface ImagePanelProps {
  selectedImage: string | null;
  isProcessing: boolean;
  onUpload: (file: File) => void;
  onClear: () => void;
  compact?: boolean;
}

function ImagePanel({ selectedImage, isProcessing, onUpload, onClear, compact }: ImagePanelProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) onUpload(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) onUpload(file);
  };

  const handleDragOver = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(false); };

  // Compact mode for tablet: horizontal strip with thumbnail
  if (compact) {
    return (
      <div
        className="flex items-center gap-3 p-3 border-b border-border bg-card"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        {selectedImage ? (
          <>
            <div className="h-14 w-20 rounded-lg border bg-muted/20 overflow-hidden flex-shrink-0 relative">
              <img src={selectedImage} alt="Uploaded" className="w-full h-full object-contain" />
              {isProcessing && (
                <div className="absolute inset-0 bg-background/60 flex items-center justify-center">
                  <Loader2 className="h-4 w-4 text-primary animate-spin" />
                </div>
              )}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium truncate">Image uploaded</p>
              <p className="text-[10px] text-muted-foreground">Click to change or drag new image</p>
            </div>
            <Button variant="ghost" size="sm" onClick={onClear} className="h-7 text-xs flex-shrink-0">
              <X className="h-3 w-3 mr-1" /> Clear
            </Button>
          </>
        ) : (
          <div
            onClick={() => !isProcessing && fileInputRef.current?.click()}
            className={`flex-1 flex items-center gap-3 p-2 border-2 border-dashed rounded-lg cursor-pointer transition-all ${
              isDragging ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
            } ${isProcessing ? "pointer-events-none opacity-60" : ""}`}
          >
            {isProcessing ? (
              <Loader2 className="h-5 w-5 text-primary animate-spin flex-shrink-0" />
            ) : (
              <Upload className="h-5 w-5 text-primary flex-shrink-0" />
            )}
            <div className="min-w-0">
              <p className="text-xs font-medium">{isDragging ? "Drop here" : "Upload image or drag & drop"}</p>
              <p className="text-[10px] text-muted-foreground">PNG, JPG, WEBP · Ctrl+V to paste</p>
            </div>
          </div>
        )}
        <input ref={fileInputRef} type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
      </div>
    );
  }

  // Full mode for desktop sidebar
  return (
    <div className="h-full flex flex-col p-3 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Input Image</h3>
        {selectedImage && (
          <Button variant="ghost" size="sm" onClick={onClear} className="h-6 text-[10px] px-2">
            <X className="h-3 w-3 mr-1" /> Clear
          </Button>
        )}
      </div>

      <div
        onClick={() => !isProcessing && fileInputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`
          relative border-2 border-dashed rounded-xl p-4 text-center cursor-pointer
          transition-all duration-200 ease-in-out
          ${isDragging ? "border-primary bg-primary/5 scale-[1.02]" : "border-border hover:border-primary/50 hover:bg-muted/30"}
          ${isProcessing ? "pointer-events-none opacity-60" : ""}
        `}
      >
        {isProcessing ? (
          <div className="space-y-2">
            <Loader2 className="mx-auto h-6 w-6 text-primary animate-spin" />
            <p className="text-xs font-medium text-primary">Processing...</p>
          </div>
        ) : (
          <div className="space-y-2">
            <div className="mx-auto w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
              <Upload className="h-4 w-4 text-primary" />
            </div>
            <p className="text-xs font-medium">{isDragging ? "Drop here" : "Upload or drag & drop"}</p>
            <p className="text-[10px] text-muted-foreground">Ctrl+V to paste</p>
            <div className="flex gap-1 justify-center">
              {["PNG", "JPG", "WEBP"].map((fmt) => (
                <span key={fmt} className="text-[9px] px-1 py-0.5 rounded bg-muted text-muted-foreground">{fmt}</span>
              ))}
            </div>
          </div>
        )}
        <input ref={fileInputRef} type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
      </div>

      {selectedImage ? (
        <div className="flex-1 min-h-0 overflow-auto">
          <div className="relative rounded-lg border bg-muted/20 overflow-hidden">
            <img src={selectedImage} alt="Uploaded" className="w-full h-auto object-contain" />
            {isProcessing && (
              <div className="absolute inset-0 bg-background/50 flex items-center justify-center">
                <Loader2 className="h-6 w-6 text-primary animate-spin" />
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center space-y-1">
            <ImageIcon className="mx-auto h-10 w-10 text-muted-foreground/20" />
            <p className="text-[10px] text-muted-foreground">No image</p>
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
    { role: "assistant", content: "Hi! How can I help with your math today?" }
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
      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=AIzaSyCwdM4RB-wUA-xxtCYEwideetEegHdNuIk`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            contents: [{ parts: [{ text: "Answer this math question concisely: " + currentInput }] }],
          }),
        }
      );
      const data = await response.json();
      const aiText = data.candidates?.[0]?.content?.parts?.[0]?.text || "No response.";
      setMessages((prev) => [...prev, { role: "assistant", content: aiText }]);
    } catch {
      setMessages((prev) => [...prev, { role: "assistant", content: "Failed to connect to AI." }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="h-full flex flex-col overflow-hidden bg-card">
      <div className="p-3 border-b">
        <h3 className="font-medium text-sm">Math Assistant</h3>
        <p className="text-[10px] text-muted-foreground">Powered by Gemini</p>
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
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [latexCode, setLatexCode] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState("code");
  const [chatOpen, setChatOpen] = useState(false);

  // Toolbar state
  const [fontSize, setFontSize] = useState(14);
  const [wordWrap, setWordWrap] = useState(true);
  const [previewZoom, setPreviewZoom] = useState(100);

  const handleImageUpload = async (file: File) => {
    const imageUrl = URL.createObjectURL(file);
    setSelectedImage(imageUrl);
    setIsProcessing(true);
    setActiveTab("code");

    try {
      const formData = new FormData();
      formData.append("file", file);
      const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
      const response = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error("Backend not responding");
      const data = await response.json();
      setLatexCode(data.latex);
      setActiveTab("preview");
      toast.success("Converted successfully!");
    } catch (error) {
      console.error("Upload error:", error);
      toast.error("Failed to process image. Is the backend running on port 8000?");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleClear = () => {
    setSelectedImage(null);
    setLatexCode("");
    setActiveTab("code");
  };

  const handleCopyLatex = () => {
    navigator.clipboard.writeText(latexCode);
    toast.success("LaTeX copied to clipboard!");
  };

  // Paste listener
  useEffect(() => {
    const handlePaste = (event: ClipboardEvent) => {
      const items = event.clipboardData?.items;
      if (items) {
        for (let i = 0; i < items.length; i++) {
          if (items[i].type.indexOf("image") !== -1) {
            const file = items[i].getAsFile();
            if (file) handleImageUpload(file);
          }
        }
      }
    };
    window.addEventListener("paste", handlePaste);
    return () => window.removeEventListener("paste", handlePaste);
  }, []);

  return (
    <div className="h-[calc(100vh-3.5rem)] pb-14 md:pb-0 flex flex-col overflow-hidden">

      {/* ===== MOBILE (< md): Compact upload + tabbed code/preview ===== */}
      <div className="md:hidden flex flex-col flex-1 min-h-0">
        {/* Compact image upload strip */}
        <ImagePanel
          selectedImage={selectedImage}
          isProcessing={isProcessing}
          onUpload={handleImageUpload}
          onClear={handleClear}
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
                  onChange={(e) => setLatexCode(e.target.value)}
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
          selectedImage={selectedImage}
          isProcessing={isProcessing}
          onUpload={handleImageUpload}
          onClear={handleClear}
          compact
        />

        {/* Side-by-side code + preview */}
        <div className="flex-1 flex min-h-0">
          <CodePanel
            latexCode={latexCode}
            onCodeChange={setLatexCode}
            isProcessing={isProcessing}
            fontSize={fontSize}
            onFontSizeChange={setFontSize}
            wordWrap={wordWrap}
            onWordWrapToggle={() => setWordWrap(!wordWrap)}
            onCopy={handleCopyLatex}
          />
          <PreviewPanel
            latexCode={latexCode}
            isProcessing={isProcessing}
            zoom={previewZoom}
            onZoomChange={setPreviewZoom}
          />
        </div>
      </div>

      {/* ===== DESKTOP (≥ lg): Sidebar image + side-by-side code/preview ===== */}
      <div className="hidden lg:flex flex-1 min-h-0">
        {/* Left: Image sidebar */}
        <div className="w-64 xl:w-72 border-r border-border bg-card flex-shrink-0">
          <ImagePanel
            selectedImage={selectedImage}
            isProcessing={isProcessing}
            onUpload={handleImageUpload}
            onClear={handleClear}
          />
        </div>

        {/* Center + Right: Code + Preview side-by-side */}
        <div className="flex-1 flex min-h-0 min-w-0">
          <CodePanel
            latexCode={latexCode}
            onCodeChange={setLatexCode}
            isProcessing={isProcessing}
            fontSize={fontSize}
            onFontSizeChange={setFontSize}
            wordWrap={wordWrap}
            onWordWrapToggle={() => setWordWrap(!wordWrap)}
            onCopy={handleCopyLatex}
          />
          <PreviewPanel
            latexCode={latexCode}
            isProcessing={isProcessing}
            zoom={previewZoom}
            onZoomChange={setPreviewZoom}
          />
        </div>

        {/* Chat sidebar (collapsible) */}
        {chatOpen && (
          <div className="w-72 border-l border-border bg-card flex-shrink-0 animate-in slide-in-from-right-5 duration-200">
            <ChatPanel />
          </div>
        )}
      </div>

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
