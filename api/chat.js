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
    const { message } = req.body;
    if (!message) {
      return res.status(400).json({ error: "Message is required" });
    }

    const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: "llama-3.3-70b-versatile",
        messages: [
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
- For complex topics, break down step by step`,
          },
          { role: "user", content: message },
        ],
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
