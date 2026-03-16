import { useState, useEffect, useRef } from "react";
import { Button } from "@/app/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/app/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/app/components/ui/tabs";
import { Textarea } from "@/app/components/ui/textarea";
import { ScrollArea } from "@/app/components/ui/scroll-area";
import { Sheet, SheetContent, SheetTrigger } from "@/app/components/ui/sheet";
import { Skeleton } from "@/app/components/ui/skeleton";
import {
  Upload, Copy, Send, Menu, MessageSquare, Loader2, ImageIcon, X
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
}

function ImagePanel({ selectedImage, isProcessing, onUpload, onClear }: ImagePanelProps) {
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

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  return (
    <div className="h-full flex flex-col p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-muted-foreground">Input Image</h3>
        {selectedImage && (
          <Button variant="ghost" size="sm" onClick={onClear} className="h-7 text-xs">
            <X className="h-3 w-3 mr-1" /> Clear
          </Button>
        )}
      </div>

      {/* Upload zone */}
      <div
        onClick={() => !isProcessing && fileInputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`
          relative border-2 border-dashed rounded-xl p-6 text-center cursor-pointer
          transition-all duration-200 ease-in-out
          ${isDragging
            ? "border-primary bg-primary/5 scale-[1.02]"
            : "border-border hover:border-primary/50 hover:bg-muted/30"
          }
          ${isProcessing ? "pointer-events-none opacity-60" : ""}
        `}
      >
        {isProcessing ? (
          <div className="space-y-3">
            <Loader2 className="mx-auto h-8 w-8 text-primary animate-spin" />
            <p className="text-sm font-medium text-primary">Processing...</p>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="mx-auto w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
              <Upload className="h-5 w-5 text-primary" />
            </div>
            <div>
              <p className="text-sm font-medium">
                {isDragging ? "Drop image here" : "Click to upload or drag & drop"}
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                Paste from clipboard: Ctrl + V
              </p>
            </div>
            <div className="flex gap-1.5 justify-center">
              {["PNG", "JPG", "WEBP"].map((fmt) => (
                <span key={fmt} className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
                  {fmt}
                </span>
              ))}
            </div>
          </div>
        )}
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          accept="image/*"
          onChange={handleFileChange}
        />
      </div>

      {/* Image preview */}
      {selectedImage && (
        <div className="flex-1 min-h-0 overflow-auto">
          <div className="relative rounded-lg border bg-muted/20 overflow-hidden">
            <img
              src={selectedImage}
              alt="Uploaded"
              className="w-full h-auto object-contain"
            />
            {isProcessing && (
              <div className="absolute inset-0 bg-background/50 flex items-center justify-center">
                <Loader2 className="h-8 w-8 text-primary animate-spin" />
              </div>
            )}
          </div>
        </div>
      )}

      {!selectedImage && (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center space-y-2">
            <ImageIcon className="mx-auto h-12 w-12 text-muted-foreground/30" />
            <p className="text-xs text-muted-foreground">No image selected</p>
          </div>
        </div>
      )}
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
      <div className="p-4 border-b">
        <h3 className="font-medium text-sm">Math Assistant</h3>
        <p className="text-xs text-muted-foreground">Powered by Gemini</p>
      </div>
      <ScrollArea className="flex-1 p-4">
        {messages.map((m, i) => (
          <div key={i} className={`mb-3 flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`p-3 rounded-lg text-sm max-w-[85%] ${
                m.role === "user"
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted"
              }`}
            >
              {m.content}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start mb-3">
            <div className="bg-muted p-3 rounded-lg">
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            </div>
          </div>
        )}
        <div ref={scrollRef} />
      </ScrollArea>
      <div className="p-3 border-t">
        <div className="flex gap-2">
          <Input
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder="Ask about math..."
            className="text-sm"
            disabled={isLoading}
          />
          <Button onClick={handleSend} size="icon" disabled={isLoading}>
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}

// ============================================================
// ConvertPage — Main page
// ============================================================
export function ConvertPage() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [latexCode, setLatexCode] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState("code");
  const [chatOpen, setChatOpen] = useState(false);
  const [leftPanelOpen, setLeftPanelOpen] = useState(false);

  // Upload handler — shared between ImagePanel and paste
  const handleImageUpload = async (file: File) => {
    const imageUrl = URL.createObjectURL(file);
    setSelectedImage(imageUrl);
    setIsProcessing(true);
    setActiveTab("code");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("http://localhost:8000/predict", {
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

  const handleCopyLatex = () => {
    navigator.clipboard.writeText(latexCode);
    toast.success("LaTeX copied to clipboard!");
  };

  return (
    <div className="h-[calc(100vh-4rem)] pb-16 md:pb-0 flex flex-col lg:flex-row overflow-hidden">
      {/* Left Panel — Desktop */}
      <div className="hidden lg:flex w-72 xl:w-80 border-r border-border bg-card flex-shrink-0">
        <ImagePanel
          selectedImage={selectedImage}
          isProcessing={isProcessing}
          onUpload={handleImageUpload}
          onClear={handleClear}
        />
      </div>

      {/* Left Panel — Mobile */}
      <Sheet open={leftPanelOpen} onOpenChange={setLeftPanelOpen}>
        <SheetTrigger asChild>
          <Button variant="outline" size="icon" className="lg:hidden fixed top-20 left-4 z-40 shadow-lg bg-card">
            <Menu className="h-4 w-4" />
          </Button>
        </SheetTrigger>
        <SheetContent side="left" className="w-80 p-0">
          <ImagePanel
            selectedImage={selectedImage}
            isProcessing={isProcessing}
            onUpload={(file) => {
              handleImageUpload(file);
              setLeftPanelOpen(false);
            }}
            onClear={handleClear}
          />
        </SheetContent>
      </Sheet>

      {/* Center — Editor */}
      <div className="flex-1 flex flex-col min-w-0">
        <Card className="flex-1 rounded-none border-0 flex flex-col">
          <CardHeader className="border-b border-border py-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium">LaTeX Editor</CardTitle>
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleCopyLatex}
                  disabled={!latexCode}
                  className="h-8 text-xs"
                >
                  <Copy className="h-3.5 w-3.5 mr-1.5" /> Copy
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="flex-1 p-0 flex flex-col overflow-hidden">
            <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
              <TabsList className="mx-4 mt-2 w-fit">
                <TabsTrigger value="code" className="text-xs">LaTeX Code</TabsTrigger>
                <TabsTrigger value="preview" className="text-xs">Preview</TabsTrigger>
              </TabsList>

              <TabsContent value="code" className="flex-1 p-4 overflow-hidden">
                {isProcessing ? (
                  <div className="space-y-3">
                    <Skeleton className="h-4 w-3/4" />
                    <Skeleton className="h-4 w-1/2" />
                    <Skeleton className="h-4 w-2/3" />
                  </div>
                ) : (
                  <Textarea
                    value={latexCode}
                    onChange={(e) => setLatexCode(e.target.value)}
                    placeholder="LaTeX code will appear here after uploading an image..."
                    className="h-full font-mono text-sm resize-none bg-muted/20 border-muted"
                  />
                )}
              </TabsContent>

              <TabsContent value="preview" className="flex-1 overflow-auto">
                <div className="min-h-full flex items-center justify-center p-8">
                  {isProcessing ? (
                    <div className="space-y-4 text-center">
                      <Loader2 className="mx-auto h-8 w-8 text-primary animate-spin" />
                      <p className="text-sm text-muted-foreground">Recognizing formula...</p>
                    </div>
                  ) : latexCode ? (
                    <div className="w-full max-w-2xl mx-auto">
                      <div className="bg-card border rounded-xl p-8 shadow-sm">
                        <LatexPreview latex={latexCode} className="text-center" />
                      </div>
                      <p className="text-center text-xs text-muted-foreground mt-3">
                        <code className="font-mono bg-muted px-1.5 py-0.5 rounded">{latexCode}</code>
                      </p>
                    </div>
                  ) : (
                    <div className="text-center space-y-2">
                      <p className="text-muted-foreground text-sm">Upload an image to see the preview</p>
                    </div>
                  )}
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>

      {/* Right — Chat Toggle Button */}
      <Button
        variant="outline"
        size="icon"
        onClick={() => setChatOpen(!chatOpen)}
        className="hidden lg:flex fixed bottom-6 right-6 z-40 h-12 w-12 rounded-full shadow-lg bg-card hover:bg-primary hover:text-primary-foreground transition-colors"
      >
        <MessageSquare className="h-5 w-5" />
      </Button>

      {/* Right — Chat Sidebar (collapsible) */}
      {chatOpen && (
        <div className="hidden lg:flex w-80 border-l border-border bg-card flex-shrink-0 animate-in slide-in-from-right-5 duration-200">
          <div className="w-full">
            <ChatPanel />
          </div>
        </div>
      )}

      {/* Chat — Mobile */}
      <Sheet>
        <SheetTrigger asChild>
          <Button variant="outline" size="icon" className="lg:hidden fixed bottom-20 right-4 z-40 h-12 w-12 rounded-full shadow-lg bg-card">
            <MessageSquare className="h-5 w-5" />
          </Button>
        </SheetTrigger>
        <SheetContent side="right" className="w-80 p-0">
          <ChatPanel />
        </SheetContent>
      </Sheet>
    </div>
  );
}
