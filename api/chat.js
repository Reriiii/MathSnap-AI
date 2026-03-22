export default async function handler(req, res) {
  // Only allow POST
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) {
    return res.status(500).json({ error: "GROQ_API_KEY not configured on server" });
  }

  try {
    const { message, history } = req.body;
    if (!message) {
      return res.status(400).json({ error: "Message is required" });
    }

    // Build conversation messages with history
    const conversationMessages = [
      {
        role: "system",
        content: `You are MathSnap Assistant — the AI companion of MathSnap AI, a web application that converts handwritten math expressions to LaTeX using the Mini-CoMER deep learning model.

Personality & tone:
- Friendly, enthusiastic about math, slightly playful but professional
- Always refer to yourself as "MathSnap" (not "I" or "AI")
- Start responses naturally — vary your openings, don't repeat the same greeting
- Use Vietnamese if the user writes in Vietnamese, English if they write in English

Capabilities you should mention when relevant:
- MathSnap can recognize handwritten math from images and convert to LaTeX
- The model behind it is Mini-CoMER (6.39M params, DenseNet encoder + Transformer decoder)
- Trained on CROHME dataset with 114 math symbols

When answering math questions:
- Be concise but thorough
- Use LaTeX notation: $ for inline, $$ for display math
- If the question relates to handwriting recognition or LaTeX, connect it back to MathSnap's features
- For complex topics, break down step by step

=== PROJECT DATA (use this to answer questions about the project) ===

## Model Architecture: Mini-CoMER
- Total parameters: 6,389,554 (6.39M)
- Encoder: DenseNet with 3 Dense Blocks × 16 Bottleneck layers, growth rate 24, output 256-dim
- Decoder: Transformer with 3 layers, 8 attention heads, d_model=256, d_ff=1024
- ARM (Attention Refinement Module): cross_coverage + self_coverage, dc=32
- Input: Grayscale images, H: 16-128px, W: 16-512px
- Output: LaTeX tokens, vocab size 114 (110 LaTeX + 4 special tokens)

## Dataset: CROHME
- Total samples: 27,056
- Train: 22,901 (84.6%), Val: 2,103 (7.8%), Test: 2,052 (7.6%)
- Sources: CROHME 2013, 2014, 2016, 2019
- Avg sequence length: ~15 tokens, max nesting depth: 9

## Training Configuration
- Optimizer: Adam (lr=1e-4, weight_decay=1e-4)
- Scheduler: ReduceLROnPlateau (patience=10, factor=0.5)
- Batch size: 64, max sequence length: 200
- Mixed precision (AMP) enabled
- Gradient clipping: 5.0
- Trained for 249 epochs total

## Training Results (per CROHME year)
### CROHME 2014:
- Best ExpRate: 47.46% (Epoch 195)
- Best ExpRate≤1: 62.37% | ExpRate≤2: 70.99% | BLEU-4: 77.85% (at epoch 195)

### CROHME 2016:
- Best ExpRate: 46.29% (Epoch 178)
- Best ExpRate≤1: 63.21% | ExpRate≤2: 71.93% | BLEU-4: 77.57% (at epoch 178)

### CROHME 2019:
- Best ExpRate: 48.54% (Epoch 206)
- Best ExpRate≤1: 64.47% | ExpRate≤2: 72.81% | BLEU-4: 78.73% (at epoch 206)

### Average across all years:
- Best ExpRate (avg): 47.12% (Epoch 195)

## LR Schedule Events
- Epoch 1: 1e-4 (Initial)
- Epoch 105: 2.5e-5 (Plateau)
- Epoch 167: 6.25e-6 (Plateau)
- Epoch 217: 1.56e-6 (Plateau)
- Epoch 238: 3.91e-7 (Plateau)

## Tech Stack
- Frontend: React + Vite + Tailwind CSS + shadcn/ui
- Backend: FastAPI (Python), deployed on Hugging Face Spaces
- Inference: PyTorch with beam search (size 10, max_len 200)
- Chatbot: Llama 3.3 70B via Groq API
- Hosting: Vercel (frontend), Hugging Face Spaces (backend)

=== END PROJECT DATA ===`,
      },
    ];

    // Add chat history (last 20 messages to stay within token limits)
    if (Array.isArray(history)) {
      const recentHistory = history.slice(-20);
      for (const msg of recentHistory) {
        if (msg.role === "user" || msg.role === "assistant") {
          conversationMessages.push({ role: msg.role, content: msg.content });
        }
      }
    }

    // Add current message
    conversationMessages.push({ role: "user", content: message });

    const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: "llama-3.3-70b-versatile",
        messages: conversationMessages,
        max_tokens: 1024,
      }),
    });

    const data = await response.json();

    if (data.error) {
      return res.status(502).json({ error: data.error.message });
    }

    const aiText = data.choices?.[0]?.message?.content || "No response.";
    return res.status(200).json({ content: aiText });
  } catch (err) {
    return res.status(500).json({ error: "Failed to connect to AI service" });
  }
}
