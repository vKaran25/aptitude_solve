# ⚡ AI Question Solver (Groq + OpenAI GPT-OSS 20B)

Upload photos of questions and get AI-powered solutions streamed live at ultra-fast speeds using **Groq** and **`openai/gpt-oss-20b`**.

---

## 🚀 Run Locally

### 1. Get a Free Groq API Key
1. Go to [https://console.groq.com/keys](https://console.groq.com/keys)
2. Sign in with Google / GitHub
3. Click **"Create API Key"** and copy it (starts with `gsk_...`)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Your API Key
In `.env` or in the app's sidebar:
```
GROQ_API_KEY=gsk_your_key_here
```

### 4. Run the App
```bash
streamlit run app.py
```

---

## 💡 How It Works

1. **Vision Reading**: Since `openai/gpt-oss-20b` is a text-only reasoning model, Groq's multimodal vision model (`qwen/qwen3.8-27b`) transcribes the question/math from the photo.
2. **Deep Solving**: **`openai/gpt-oss-20b`** receives the question and streams the step-by-step reasoning and solution live into your UI.
3. **Sidebar Chat**: **`openai/gpt-oss-20b`** maintains context memory from all solved questions so you can chat and ask follow-ups.

---

## ☁️ Deploy on Streamlit Community Cloud (Free)

1. Push this folder to a GitHub repository.
2. Go to [https://share.streamlit.io](https://share.streamlit.io) → New App.
3. In **Advanced Settings → Secrets**, add:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```
4. Click **Deploy**!
