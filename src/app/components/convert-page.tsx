import { Button } from "@/app/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/app/components/ui/card";
import { Badge } from "@/app/components/ui/badge";
import OpenAI from "openai";
import { useState, useEffect, useRef } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/app/components/ui/tabs";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { Textarea } from "@/app/components/ui/textarea";
import { ScrollArea } from "@/app/components/ui/scroll-area";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/app/components/ui/sheet";
import {
  Upload,
  ZoomIn,
  ZoomOut,
  RotateCw,
  Crop,
  SunDim,
  Copy,
  Download,
  FileText,
  Image as ImageIcon,
  MessageSquare,
  CheckCircle2,
  AlertCircle,
  Send,
  Menu
} from "lucide-react";
import { Input } from "@/app/components/ui/input";

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

  const suggestedPrompts = [
    "Explain this formula",
    "Fix LaTeX errors",
    "What does this symbol mean?",
    "Convert to align environment"
  ];

  const handleCopyLatex = () => {
    navigator.clipboard.writeText(latexCode);
    // Toast notification would go here
  };

  return (
    <div className="h-[calc(100vh-4rem)] md:h-[calc(100vh-4rem)] pb-16 md:pb-0 flex flex-col lg:flex-row overflow-hidden">
      {/* Left Panel - Image Upload & Viewer (Desktop) */}
      <div className="hidden lg:block w-80 xl:w-96 border-r border-border bg-card">
        <ImagePanel
          mockImages={mockImages}
          selectedImage={selectedImage}
          onImageSelect={setSelectedImage}
        />
      </div>

      {/* Left Panel - Mobile Drawer */}
      <Sheet open={leftPanelOpen} onOpenChange={setLeftPanelOpen}>
        <SheetTrigger asChild>
          <Button
            variant="outline"
            size="icon"
            className="lg:hidden fixed top-20 left-4 z-40 shadow-lg"
          >
            <Menu className="h-4 w-4" />
          </Button>
        </SheetTrigger>
        <SheetContent side="left" className="w-80 p-0">
          <ImagePanel
            mockImages={mockImages}
            selectedImage={selectedImage}
            onImageSelect={(id) => {
              setSelectedImage(id);
              setLeftPanelOpen(false);
            }}
            isMobile
          />
        </SheetContent>
      </Sheet>

      {/* Center Panel - LaTeX Editor */}
      <div className="flex-1 flex flex-col min-w-0">
        <Card className="flex-1 rounded-none border-0 border-b lg:border-r flex flex-col">
          <CardHeader className="border-b border-border">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">LaTeX Editor</CardTitle>
              <div className="flex gap-2">
                <Button size="sm" variant="outline" onClick={handleCopyLatex}>
                  <Copy className="h-4 w-4 mr-2" />
                  <span className="hidden sm:inline">Copy</span>
                </Button>
                <Button size="sm" variant="outline">
                  <Download className="h-4 w-4 mr-2" />
                  <span className="hidden sm:inline">Download</span>
                </Button>
                <Button size="sm" className="bg-accent text-accent-foreground">
                  <FileText className="h-4 w-4 mr-2" />
                  <span className="hidden sm:inline">Export</span>
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="flex-1 p-0 flex flex-col">
            <Tabs defaultValue="code" className="flex-1 flex flex-col">
              <TabsList className="w-full justify-start rounded-none border-b border-border bg-muted/50 px-4">
                <TabsTrigger value="code">LaTeX Code</TabsTrigger>
                <TabsTrigger value="preview">Rendered Preview</TabsTrigger>
              </TabsList>
              <TabsContent value="code" className="flex-1 m-0 p-4">
                <div className="space-y-4 h-full flex flex-col">
                  <div className="flex items-center gap-2 text-sm">
                    <Badge variant="outline" className="bg-success/10 text-success border-success/20">
                      <CheckCircle2 className="h-3 w-3 mr-1" />
                      No errors
                    </Badge>
                    <span className="text-muted-foreground">Auto-saved</span>
                  </div>
                  <Textarea
                    value={latexCode}
                    onChange={(e) => setLatexCode(e.target.value)}
                    className="flex-1 font-mono text-sm resize-none"
                    placeholder="LaTeX code will appear here..."
                  />
                </div>
              </TabsContent>
              <TabsContent value="preview" className="flex-1 m-0 p-4">
                <div className="h-full flex items-center justify-center bg-muted/30 rounded-lg">
                  <div className="text-center space-y-4">
                    <div className="text-4xl font-serif">
                      ∫₀^∞ e⁻ˣ² dx = √π/2
                    </div>
                    <p className="text-sm text-muted-foreground">Rendered formula preview</p>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>

      {/* Right Panel - Chatbot (Desktop) */}
      <div className="hidden xl:block w-80 border-l border-border bg-card">
        <ChatPanel suggestedPrompts={suggestedPrompts} />
      </div>

      {/* Floating Chat Button (Mobile & Tablet) */}
      <Sheet open={chatOpen} onOpenChange={setChatOpen}>
        <SheetTrigger asChild>
          <Button
            size="icon"
            className="xl:hidden fixed bottom-20 md:bottom-6 right-6 z-40 h-14 w-14 rounded-full shadow-lg bg-accent text-accent-foreground"
          >
            <MessageSquare className="h-6 w-6" />
          </Button>
        </SheetTrigger>
        <SheetContent side="right" className="w-full sm:w-96 p-0">
          <ChatPanel suggestedPrompts={suggestedPrompts} isMobile />
        </SheetContent>
      </Sheet>
    </div>
  );
}

// Image Panel Component
interface ImagePanelProps {
  mockImages: { id: number; name: string }[];
  selectedImage: string | null;
  onImageSelect: (id: string | null) => void;
  isMobile?: boolean;
}

function ImagePanel({ mockImages, selectedImage, onImageSelect, isMobile }: ImagePanelProps) {
  // Hàm xử lý khi chọn file từ máy tính
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const imageUrl = URL.createObjectURL(file);
      onImageSelect(imageUrl); // Lưu link ảnh vào state selectedImage của cha
    }
  };

  return (
    <div className="h-full flex flex-col">
      {isMobile && (
        <SheetHeader className="p-4 border-b border-border">
          <SheetTitle>Image Upload</SheetTitle>
        </SheetHeader>
      )}
      <ScrollArea className="flex-1">
        <div className="p-4 space-y-4">

          {/* Upload Box - Đã kích hoạt Click để chọn file */}
          <div
            onClick={() => document.getElementById('fileInput')?.click()}
            className="border-2 border-dashed border-border rounded-lg p-6 text-center space-y-3 hover:border-primary/50 transition-colors cursor-pointer"
          >
            <div className="flex justify-center">
              {/* Nếu đã chọn ảnh thì hiện ảnh nhỏ (thumbnail) ở đây, không thì hiện icon Upload */}
              {selectedImage && selectedImage.startsWith('blob:') ? (
                <img src={selectedImage} alt="Preview" className="h-16 w-16 object-cover rounded border" />
              ) : (
                <Upload className="h-8 w-8 text-muted-foreground" />
              )}
            </div>
            <div>
              <p className="text-sm font-medium">Click to browse image</p>
              <p className="text-xs text-muted-foreground mt-1">PNG, JPG up to 10MB</p>
            </div>
            {/* Input này ẩn đi, chỉ dùng để nhận file */}
            <Input
              id="fileInput"
              type="file"
              className="hidden"
              accept="image/*"
              onChange={handleFileChange}
            />
          </div>

          {/* Image List */}
          <div className="space-y-2">
            <h4 className="text-sm font-medium">Samples / Uploaded</h4>
            <div className="space-y-2">
              {mockImages.map((img) => (
                <div
                  key={img.id}
                  onClick={() => onImageSelect(img.id.toString())}
                  className={`p-3 rounded-lg border cursor-pointer transition-colors ${selectedImage === img.id.toString()
                    ? "border-primary bg-primary/5"
                    : "border-border hover:border-primary/50"
                    }`}
                >
                  <div className="flex items-center gap-3">
                    <div className="w-12 h-12 bg-muted rounded flex items-center justify-center flex-shrink-0">
                      <ImageIcon className="h-5 w-5 text-muted-foreground" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{img.name}</p>
                      <p className="text-xs text-muted-foreground">Sample</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Image Viewer Tools - Sẽ hiện khi có ảnh được chọn */}
          {selectedImage && (
            <Card className="overflow-hidden border-primary/20">
              <CardHeader className="pb-3 bg-muted/50">
                <CardTitle className="text-sm flex justify-between items-center">
                  Image Preview
                  <Button variant="ghost" size="sm" className="h-6 text-xs" onClick={() => onImageSelect(null)}>Remove</Button>
                </CardTitle>
              </CardHeader>
              <CardContent className="p-0 border-t border-border">
                {/* Hiển thị ảnh lớn ở đây để bà soi công thức */}
                <div className="bg-black/5 flex items-center justify-center p-2 min-h-[150px]">
                  <img
                    src={selectedImage.startsWith('blob:') ? selectedImage : `https://via.placeholder.com/300?text=${selectedImage}`}
                    alt="Current selection"
                    className="max-w-full h-auto rounded shadow-sm"
                  />
                </div>
                <div className="p-3 grid grid-cols-2 gap-2">
                  <Button variant="outline" size="sm" className="h-8 text-xs"><ZoomIn className="h-3 w-3 mr-1" /> Zoom In</Button>
                  <Button variant="outline" size="sm" className="h-8 text-xs"><ZoomOut className="h-3 w-3 mr-1" /> Zoom Out</Button>
                  <Button variant="outline" size="sm" className="h-8 text-xs" className="col-span-2"><RotateCw className="h-3 w-3 mr-1" /> Rotate</Button>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}

// Chat Panel Component
interface ChatPanelProps {
  suggestedPrompts: string[];
  isMobile?: boolean;
}

function ChatPanel({ suggestedPrompts, isMobile }: ChatPanelProps) {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Hi Dan! I'm your Math Assistant. I can help explain formulas, fix LaTeX errors, or answer questions about mathematical notation."
    }
  ]);
  const [inputValue, setInputValue] = useState("");

  // 1. Tạo Ref để làm mỏ neo cuộn trang
  const scrollRef = useRef<HTMLDivElement>(null);

  // 2. Tự động cuộn xuống mỗi khi có tin nhắn mới
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const handleSend = async () => {
    if (!inputValue.trim()) return;

    const currentInput = inputValue;
    setMessages(prev => [...prev, { role: "user", content: currentInput }]);
    setInputValue("");

    try {
      // Dùng Key sạch ông vừa tạo nhé
      const API_KEY = "AIzaSyCwdM4RB-wUA-xxtCYEwideetEegHdNuIk";

      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${API_KEY}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            contents: [
              {
                parts: [
                  {
                    text: `Bạn là 'MathSnap AI Assistant' - một chuyên gia về Toán học và LaTeX. 
                    Nhiệm vụ của bạn là giúp sinh viên FPTU giải đáp thắc mắc. 
                    Nếu câu hỏi liên quan đến công thức, hãy giải thích chi tiết ý nghĩa các ký hiệu. 
                    Sử dụng ngôn ngữ thân thiện, chuyên nghiệp.
                    Câu hỏi hiện tại: ${currentInput}`
                  }
                ]
              }
            ]
          })
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error?.message || "Lỗi kết nối AI");
      }

      const aiText = data.candidates?.[0]?.content?.parts?.[0]?.text || "AI không trả lời.";

      setMessages(prev => [...prev, { role: "assistant", content: aiText }]);

    } catch (error: any) {
      console.error(error);
      setMessages(prev => [
        ...prev,
        { role: "assistant", content: "Có lỗi khi gọi AI: " + error.message }
      ]);
    }
  };

  return (
    // Thêm overflow-hidden để cố định khung chat không bị đùn toàn trang
    <div className="h-full flex flex-col bg-card overflow-hidden">

      {/* Header - Cố định ở trên nhờ shrink-0 */}
      <div className="p-4 border-b border-border shrink-0">
        <h3 className="font-medium">Math Assistant</h3>
        <p className="text-xs text-muted-foreground mt-1">Ask questions about your formulas</p>
      </div>

      {/* Messages - flex-1 để chiếm trọn không gian giữa và có scroll riêng */}
      <ScrollArea className="flex-1 w-full overflow-y-auto">
        <div className="p-4 space-y-4">
          {messages.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              <div className={`max-w-[85%] rounded-lg p-3 ${msg.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted"}`}>
                <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
              </div>
            </div>
          ))}
          {/* Mỏ neo để useEffect nhìn vào và cuộn xuống */}
          <div ref={scrollRef} />
        </div>
      </ScrollArea>

      {/* Phần chân trang - Luôn dính ở đáy nhờ shrink-0 */}
      <div className="shrink-0 border-t border-border bg-card">
        {/* Suggested Prompts */}
        <div className="p-4 pb-2 space-y-3">
          <p className="text-xs text-muted-foreground">Suggested prompts:</p>
          <div className="flex flex-wrap gap-2">
            {suggestedPrompts.map((prompt, idx) => (
              <Button
                key={idx}
                variant="outline"
                size="sm"
                className="text-xs h-auto py-1.5"
                onClick={() => { setInputValue(prompt); }}
              >
                {prompt}
              </Button>
            ))}
          </div>
        </div>

        {/* Input Field */}
        <div className="p-4 pt-2">
          <div className="flex gap-2">
            <Input
              placeholder="Ask a question..."
              className="flex-1"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            />
            <Button onClick={handleSend} size="icon" className="bg-accent text-accent-foreground shrink-0">
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}