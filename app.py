import os
import io
import base64
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# ── Env & imports ─────────────────────────────────────────────────────────────
load_dotenv()

try:
    from groq import Groq
except ImportError:
    st.error("Run: pip install groq")
    st.stop()

MODEL = "qwen/qwen3.8-27b"

SOLVE_SYSTEM_PROMPT = """You are an expert exam tutor. Carefully examine the provided image.

STEP 1 — Count: Identify ALL questions in the image. There may be 1 or many.

STEP 2 — For each question, use this exact format:

---
**Q[n]: [One-line description of what this question is asking]**

[If the question has multiple-choice options (A/B/C/D):]
- **A)** [option text] — [brief note: correct or why wrong]
- **B)** [option text] — [brief note: correct or why wrong]
- **C)** [option text] — [brief note: correct or why wrong]
- **D)** [option text] — [brief note: correct or why wrong]

✅ **Answer: [Letter]** — [1–2 sentence explanation of the concept]

[If the question is numerical or descriptive:]
**Step 1:** ...
**Step 2:** ...
**Step 3:** ...

✅ **Final Answer: [value/result]** — [1–2 sentence explanation]

---

RULES:
- Separate every question with --- (three dashes)
- Always start with Q1, Q2, Q3 ...
- Always mark the correct MCQ answer with ✅ on its own line
- Always end with a brief concept explanation
- If the image is blurry or unclear, say so clearly under Q1
"""

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Question Solver",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide top padding */
    .block-container { padding-top: 1.5rem; }

    /* Title */
    h1 { font-size: 1.8rem !important; font-weight: 700; margin-bottom: 0 !important; }

    /* Answer cards */
    .answer-card {
        border: 1px solid #2d2d2d;
        border-left: 4px solid #f97316;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 8px;
        background: #0f0f0f;
    }

    /* Image file label */
    .file-label {
        font-size: 0.78rem;
        color: #888;
        margin-bottom: 6px;
    }

    /* Divider between images */
    hr { margin: 2rem 0; border-color: #1f1f1f; }

    /* Hide streamlit branding clutter */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("chat_messages", []),
    ("solved_answers", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_client():
    api_key = (
        os.getenv("GROQ_API_KEY", "")
        or st.session_state.get("_api_key", "")
    )
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass
    return Groq(api_key=api_key) if api_key else None


def file_to_base64(file) -> tuple[str, str]:
    """Returns (base64_string, mime_type)."""
    file.seek(0)
    ext = file.name.lower().rsplit(".", 1)[-1]
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
    return base64.b64encode(file.read()).decode("utf-8"), mime


def stream_solve(client, file, custom_prompt: str):
    """
    Send one image to qwen/qwen3.8-27b and stream the answer.
    Returns the full accumulated text.
    """
    b64, mime = file_to_base64(file)
    system = custom_prompt.strip() if custom_prompt.strip() else SOLVE_SYSTEM_PROMPT

    placeholder = st.empty()
    full_text = ""

    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                        {
                            "type": "text",
                            "text": "Identify and solve all questions in this image.",
                        },
                    ],
                },
            ],
            stream=True,
            temperature=0.2,
        )
        for chunk in stream:
            delta = (
                chunk.choices[0].delta.content
                if chunk.choices and chunk.choices[0].delta
                else ""
            )
            if delta:
                full_text += delta
                placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text)
    except Exception as e:
        full_text = f"⚠️ Error: {e}"
        placeholder.error(full_text)

    return full_text


def stream_chat(client, user_msg: str):
    """Stream a chat reply using solved-answer memory."""
    memory = "\n\n---\n\n".join(
        f"[{it['filename']}]\n{it['answer']}"
        for it in st.session_state.solved_answers[-6:]
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful exam tutor. "
                "Use the solved questions below as memory when answering follow-up questions.\n\n"
                + (f"=== Solved Questions ===\n{memory}" if memory else "")
            ),
        }
    ]
    for m in st.session_state.chat_messages[-8:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_msg})

    placeholder = st.empty()
    full_reply = ""

    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True,
            temperature=0.3,
        )
        for chunk in stream:
            delta = (
                chunk.choices[0].delta.content
                if chunk.choices and chunk.choices[0].delta
                else ""
            )
            if delta:
                full_reply += delta
                placeholder.markdown(full_reply + "▌")
        placeholder.markdown(full_reply)
    except Exception as e:
        full_reply = f"⚠️ {e}"
        placeholder.error(full_reply)

    return full_reply


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # API key
    env_key = os.getenv("GROQ_API_KEY", "")
    if not env_key:
        try:
            env_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass

    if env_key:
        st.success("✅ API key loaded")
    else:
        typed_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Free key → console.groq.com/keys",
        )
        if typed_key:
            st.session_state["_api_key"] = typed_key
            os.environ["GROQ_API_KEY"] = typed_key

    # Custom prompt (collapsed by default)
    with st.expander("✏️ Custom Solve Prompt", expanded=False):
        custom_prompt = st.text_area(
            "Override the default solving instructions",
            value="",
            height=120,
            placeholder="Leave blank to use the default structured prompt.",
            label_visibility="collapsed",
        )

    st.divider()

    # Chat
    st.markdown("#### 💬 Chat")
    st.caption("Ask follow-ups about solved questions")

    chat_box = st.container(height=400)
    with chat_box:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_input = st.chat_input("Ask anything…")
    if user_input:
        client = get_client()
        if not client:
            st.warning("Add your Groq API key above.")
        else:
            st.session_state.chat_messages.append(
                {"role": "user", "content": user_input}
            )
            with chat_box:
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.chat_message("assistant"):
                    reply = stream_chat(client, user_input)
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": reply}
            )

    if st.session_state.chat_messages:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.title("⚡ Question Solver")
st.caption(f"Model: `{MODEL}` via Groq — upload question photos and get answers instantly")

uploaded = st.file_uploader(
    "Select question images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

has_key = bool(
    os.getenv("GROQ_API_KEY") or st.session_state.get("_api_key")
)

if uploaded:
    c1, c2 = st.columns([1, 6])
    with c1:
        solve = st.button("🚀 Solve", type="primary", use_container_width=True)
    with c2:
        if st.button("Clear results", use_container_width=False):
            st.session_state.solved_answers = []
            st.rerun()

    if not has_key:
        st.warning("Paste your Groq API key in the sidebar to start.")

    # ── Solve ──────────────────────────────────────────────────────────────
    if solve and has_key:
        client = get_client()
        st.session_state.solved_answers = []
        bar = st.progress(0, text="Starting…")

        for i, f in enumerate(uploaded):
            bar.progress(
                int(i / len(uploaded) * 100),
                text=f"Solving {i + 1}/{len(uploaded)}: {f.name}",
            )

            img_col, ans_col = st.columns([1, 2])

            with img_col:
                st.markdown(f'<p class="file-label">{f.name}</p>', unsafe_allow_html=True)
                f.seek(0)
                st.image(Image.open(f), use_container_width=True)

            with ans_col:
                answer = stream_solve(client, f, custom_prompt if "custom_prompt" in dir() else "")

            st.session_state.solved_answers.append(
                {"filename": f.name, "answer": answer}
            )
            st.divider()

        bar.progress(100, text=f"✅ Done — {len(uploaded)} image(s) solved")

    # ── Show previous results ──────────────────────────────────────────────
    elif st.session_state.solved_answers:
        st.markdown("#### Previous Results")
        for i, item in enumerate(st.session_state.solved_answers):
            with st.expander(f"{i + 1}. {item['filename']}", expanded=False):
                st.markdown(item["answer"])

else:
    # Empty state
    st.markdown("")
    st.info(
        "📂 **Upload question images above** and click **Solve**.\n\n"
        "- Works with 1 or 200+ images\n"
        "- One image can contain multiple questions — all are detected and solved\n"
        "- MCQ options are listed clearly with the correct answer marked ✅\n"
        "- Use the **chat sidebar** to ask follow-up questions"
    )
    if st.session_state.solved_answers:
        for i, item in enumerate(st.session_state.solved_answers):
            with st.expander(f"{i + 1}. {item['filename']}", expanded=False):
                st.markdown(item["answer"])