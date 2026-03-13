import { useState, useEffect, useRef } from "react";
import { Button } from "@/app/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/app/components/ui/card";
import { Badge } from "@/app/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/app/components/ui/tabs";
import { Textarea } from "@/app/components/ui/textarea";
import { ScrollArea } from "@/app/components/ui/scroll-area";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/app/components/ui/sheet";
import {
  Upload, ZoomIn, ZoomOut, RotateCw, Copy, Download, FileText,
  Image as ImageIcon, MessageSquare, CheckCircle2, Send, Menu
} from "lucide-react";
import { Input } from "@/app/components/ui/input";

// 1. Khai báo Interface chuẩn cho ImagePanel
interface ImagePanelProps {
  mockImages: { id: number; name: string }[];
  selectedImage: string | null;
  onImageSelect: (id: string | null) => void;
  setLatexCode: (code: string) => void;
  isMobile?: boolean;
}

// 2. Component Chính: ConvertPage
export function ConvertPage() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [latexCode, setLatexCode] = useState("\\int_0^\\infty e^{-x^2} \\, dx = \\frac{\\sqrt{\\pi}}{2}");
  const [chatOpen, setChatOpen] = useState(false);
  const [leftPanelOpen, setLeftPanelOpen] = useState(false);

  const mockImages = [
    { id: 1, name: "equation1.jpg" },
    { id: 2, name: "formula2.jpg" },
    { id: 3, name: "integral3.jpg" }
  ];

  const suggestedPrompts = ["Explain this formula", "Fix LaTeX errors", "What does this symbol mean?"];

  // HÀM XỬ LÝ CHUNG CHO CẢ UPLOAD VÀ PASTE
  const handleImageUpload = async (file: File) => {
    const imageUrl = URL.createObjectURL(file);
    setSelectedImage(imageUrl);

    try {
      const formData = new FormData();
      formData.append('file', file);
      console.log("Đang xử lý ảnh...");
      
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error("Backend Python không phản hồi");
      const data = await response.json();
      setLatexCode(data.latex);
    } catch (error) {
      console.error("Lỗi xử lý ảnh:", error);
      alert("Lỗi rồi Dan ơi, kiểm tra Backend Python (cổng 8000) chưa?");
    }
  };

  // LẮNG NGHE SỰ KIỆN PASTE
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

  const handleCopyLatex = () => navigator.clipboard.writeText(latexCode);

  return (
    <div className="h-[calc(100vh-4rem)] md:h-[calc(100vh-4rem)] pb-16 md:pb-0 flex flex-col lg:flex-row overflow-hidden">
      {/* Sidebar - Desktop */}
      <div className="hidden lg:block w-80 xl:w-96 border-r border-border bg-card">
        <ImagePanel
          mockImages={mockImages}
          selectedImage={selectedImage}
          onImageSelect={setSelectedImage}
          setLatexCode={setLatexCode}
        />
      </div>

      {/* Sidebar - Mobile */}
      <Sheet open={leftPanelOpen} onOpenChange={setLeftPanelOpen}>
        <SheetTrigger asChild>
          <Button variant="outline" size="icon" className="lg:hidden fixed top-20 left-4 z-40 shadow-lg">
            <Menu className="h-4 w-4" />
          </Button>
        </SheetTrigger>
        <SheetContent side="left" className="w-80 p-0">
          <ImagePanel
            mockImages={mockImages}
            selectedImage={selectedImage}
            onImageSelect={(id) => { setSelectedImage(id); setLeftPanelOpen(false); }}
            setLatexCode={setLatexCode}
            isMobile
          />
        </SheetContent>
      </Sheet>

      {/* Editor Center */}
      <div className="flex-1 flex flex-col min-w-0">
        <Card className="flex-1 rounded-none border-0 border-b lg:border-r flex flex-col">
          <CardHeader className="border-b border-border">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">LaTeX Editor</CardTitle>
              <div className="flex gap-2">
                <Button size="sm" variant="outline" onClick={handleCopyLatex}><Copy className="h-4 w-4 mr-2" /> Copy</Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="flex-1 p-0 flex flex-col">
            <Tabs defaultValue="code" className="flex-1 flex flex-col">
              <TabsList className="px-4 border-b">
                <TabsTrigger value="code">LaTeX Code</TabsTrigger>
                <TabsTrigger value="preview">Preview</TabsTrigger>
              </TabsList>
              <TabsContent value="code" className="flex-1 p-4">
                <Textarea
                  value={latexCode}
                  onChange={(e) => setLatexCode(e.target.value)}
                  className="h-full font-mono text-sm resize-none"
                />
              </TabsContent>
              <TabsContent value="preview" className="flex-1 p-4 flex items-center justify-center bg-muted/30">
                <div className="text-2xl font-serif text-center">{latexCode}</div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>

      {/* Chat Sidebar */}
      <div className="hidden xl:block w-80 border-l border-border bg-card">
        <ChatPanel suggestedPrompts={suggestedPrompts} />
      </div>
    </div>
  );
}

// 3. Component Xử lý Ảnh: ImagePanel
function ImagePanel({ mockImages, selectedImage, onImageSelect, setLatexCode, isMobile }: ImagePanelProps) {
  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    onImageSelect(URL.createObjectURL(file));

    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error("Backend error");
      const data = await response.json();
      setLatexCode(data.latex);
    } catch (error) {
      console.error(error);
      alert("Chưa kết nối được Backend Python Dan ơi!");
    }
  };

  return (
    <div className="h-full flex flex-col p-4 space-y-4">
      <div
        onClick={() => document.getElementById('fileInput')?.click()}
        className="border-2 border-dashed rounded-lg p-6 text-center cursor-pointer hover:border-primary"
      >
        <Upload className="mx-auto h-8 w-8 text-muted-foreground" />
        <p className="mt-2 text-sm font-medium">Click to upload or Paste image (Ctrl + V)</p>
        <Input id="fileInput" type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
      </div>
      {selectedImage && (
        <div className="mt-4">
          <p className="text-xs mb-2">Preview:</p>
          <img src={selectedImage} alt="Selected" className="max-w-full rounded border" />
        </div>
      )}
    </div>
  );
}

// 4. Component Chat: ChatPanel
function ChatPanel({ suggestedPrompts }: { suggestedPrompts: string[] }) {
  const [messages, setMessages] = useState([{ role: "assistant", content: "Hi! How can I help with your math today?" }]);
  const [inputValue, setInputValue] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => { scrollRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  const handleSend = async () => {
    if (!inputValue.trim()) return;
    const currentInput = inputValue;
    setMessages(prev => [...prev, { role: "user", content: currentInput }]);
    setInputValue("");

    try {
      // Đã đổi sang gemini-1.5-flash cho ổn định
      const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=AIzaSyCwdM4RB-wUA-xxtCYEwideetEegHdNuIk`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ contents: [{ parts: [{ text: "Hãy trả lời câu hỏi toán học sau: " + currentInput }] }] })
      });
      const data = await response.json();
      const aiText = data.candidates?.[0]?.content?.parts?.[0]?.text || "AI not responding.";
      setMessages(prev => [...prev, { role: "assistant", content: aiText }]);
    } catch (e) {
      setMessages(prev => [...prev, { role: "assistant", content: "Lỗi API Gemini rồi ông ơi!" }]);
    }
  };

  return (
    <div className="h-full flex flex-col overflow-hidden bg-card">
      <div className="p-4 border-b font-medium">Math Assistant</div>
      <ScrollArea className="flex-1 p-4">
        {messages.map((m, i) => (
          <div key={i} className={`mb-4 flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`p-3 rounded-lg text-sm ${m.role === "user" ? "bg-primary text-white" : "bg-muted"}`}>{m.content}</div>
          </div>
        ))}
        <div ref={scrollRef} />
      </ScrollArea>
      <div className="p-4 border-t">
        <div className="flex gap-2">
          <Input value={inputValue} onChange={e => setInputValue(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleSend()} placeholder="Ask me..." />
          <Button onClick={handleSend} size="icon"><Send className="h-4 w-4" /></Button>
        </div>
      </div>
    </div>
  );
}