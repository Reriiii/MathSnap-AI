export const config = {
  api: {
    bodyParser: false,
  },
};

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const apiUrl = process.env.API_URL;
  if (!apiUrl) {
    return res.status(500).json({ error: "API_URL not configured on server" });
  }

  try {
    // Forward the raw request body to HF Space
    const chunks = [];
    for await (const chunk of req) {
      chunks.push(chunk);
    }
    const body = Buffer.concat(chunks);

    const response = await fetch(`${apiUrl}/predict`, {
      method: "POST",
      headers: {
        "content-type": req.headers["content-type"],
      },
      body,
    });

    if (!response.ok) {
      const text = await response.text();
      return res.status(response.status).json({ error: text });
    }

    const data = await response.json();
    return res.status(200).json(data);
  } catch (err) {
    return res.status(500).json({ error: "Failed to connect to backend" });
  }
}
