import streamlit as st
import datetime
import requests
from PIL import Image
import pytesseract
import io

st.set_page_config(page_title="ChatGPT UI (Ollama + OCR)", layout="wide")

# -------------------- Session State --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "saved_chats" not in st.session_state:
    st.session_state.saved_chats = []
if "active_chat" not in st.session_state:
    st.session_state.active_chat = None
if "theme" not in st.session_state:
    st.session_state.theme = "Light"
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "ocr_texts" not in st.session_state:
    st.session_state.ocr_texts = []
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []
if "preview_image" not in st.session_state:
    st.session_state.preview_image = None  # For enlarged image preview

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "llama2"

# -------------------- Theme --------------------
def apply_theme(theme):
    if theme == "Dark":
        st.markdown(
            """
            <style>
            body { background-color: #0e1117; color: white; }
            div.stApp { background-color: #0e1117; color: white; }
            </style>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            body { background-color: white; color: black; }
            div.stApp { background-color: white; color: black; }
            </style>
            """, unsafe_allow_html=True
        )

apply_theme(st.session_state.theme)

# -------------------- Helper Functions --------------------
def build_prompt(history):
    lines = []
    if st.session_state.ocr_texts:
        combined_ocr = "\n\n".join(st.session_state.ocr_texts)
        lines.append(f"The following text was extracted from uploaded images:\n{combined_ocr}\n")
    for m in history:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    lines.append("Assistant:")
    return "\n".join(lines)

def query_ollama_generate(prompt):
    try:
        payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
        if resp.ok:
            data = resp.json()
            if "response" in data:
                return data["response"]
            if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
                return data["message"]["content"]
            return str(data)
        else:
            return f"⚠️ Ollama error {resp.status_code}: {resp.text}"
    except Exception as e:
        return f"⚠️ Exception contacting Ollama: {e}"

def save_current_chat():
    if st.session_state.messages:
        title = st.session_state.messages[0]["content"][:30] + "..." if st.session_state.messages else "Untitled Chat"
        if st.session_state.active_chat is None:
            st.session_state.saved_chats.append({
                "title": title,
                "messages": list(st.session_state.messages)
            })
            st.session_state.active_chat = len(st.session_state.saved_chats) - 1
        else:
            st.session_state.saved_chats[st.session_state.active_chat]["messages"] = list(st.session_state.messages)
            st.session_state.saved_chats[st.session_state.active_chat]["title"] = title

def send_message(user_text=None):
    if user_text is None:
        user_text = st.session_state.user_input.strip()
    if not user_text:
        return
    st.session_state.messages.append({
        "role": "user",
        "content": user_text,
        "time": datetime.datetime.now().strftime("%H:%M")
    })
    st.session_state.user_input = ""
    st.session_state.messages.append({
        "role": "assistant",
        "content": "⏳ Thinking...",
        "time": datetime.datetime.now().strftime("%H:%M")
    })
    history = [m for m in st.session_state.messages if m["role"] in ("user", "assistant")]
    prompt = build_prompt(history)
    with st.spinner("Getting reply from Ollama..."):
        reply = query_ollama_generate(prompt)
    st.session_state.messages[-1] = {
        "role": "assistant",
        "content": reply,
        "time": datetime.datetime.now().strftime("%H:%M")
    }
    save_current_chat()

# -------------------- Sidebar --------------------
st.sidebar.title("⚙️ Menu")
st.session_state.theme = st.sidebar.radio("Choose Theme:", ["Light", "Dark"])
apply_theme(st.session_state.theme)

if st.sidebar.button("➕ New Chat"):
    save_current_chat()
    st.session_state.messages = []
    st.session_state.active_chat = None
    st.session_state.ocr_texts = []
    st.session_state.uploaded_images = []
    st.session_state.preview_image = None

search_query = st.sidebar.text_input("🔍 Search chats")
st.sidebar.subheader("💾 Chats")
filtered_chats = [(i, chat) for i, chat in enumerate(st.session_state.saved_chats) if search_query.lower() in chat["title"].lower()]
for i, chat in filtered_chats:
    if st.sidebar.button(chat["title"], key=f"chat_{i}"):
        save_current_chat()
        st.session_state.messages = list(chat["messages"])
        st.session_state.active_chat = i

# -------------------- Chat Container --------------------
st.title("💬 ChatGPT - How can I help you...?")

if st.session_state.messages:
    for msg in st.session_state.messages:
        role = "🧑 You" if msg["role"] == "user" else "🤖 Assistant"
        st.write(f"**{role}:** {msg['content']} ({msg['time']})")
else:
    st.info("Start a new conversation by typing below 👇")

# -------------------- Image Upload + OCR + Clickable Preview --------------------
uploaded_images = st.file_uploader("Upload Image(s) 📷", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_images:
    # ✅ Clear previous images and OCR text to prevent duplicates
    st.session_state.uploaded_images = []
    st.session_state.ocr_texts = []

    for img in uploaded_images:
        image = Image.open(img)
        st.session_state.uploaded_images.append(image)
        extracted_text = pytesseract.image_to_string(image).strip()
        if extracted_text:
            st.session_state.ocr_texts.append(extracted_text)

    st.success(f"✅ {len(uploaded_images)} image(s) processed successfully! OCR text stored internally.")

# Show image thumbnails (only once per upload)
if st.session_state.uploaded_images:
    st.markdown("### 🖼️ Uploaded Images")
    cols = st.columns(len(st.session_state.uploaded_images))
    for i, image in enumerate(st.session_state.uploaded_images):
        with cols[i]:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            btn = st.button(f"🖼️ Preview {i+1}")
            st.image(buf.getvalue(), width=120)
            if btn:
                st.session_state.preview_image = image

# Show enlarged image if selected
if st.session_state.preview_image:
    st.markdown("### 🔍 Image Preview (Click Close to return)")
    st.image(st.session_state.preview_image, use_container_width=True)
    if st.button("❌ Close Preview"):
        st.session_state.preview_image = None

# -------------------- OCR Text --------------------
if st.session_state.ocr_texts:
    combined_text = "\n\n".join(st.session_state.ocr_texts)
    st.info(
        f"🧾 Extracted Text from Images:\n{combined_text[:500]}..." 
        if len(combined_text) > 500 else 
        f"🧾 Extracted Text from Images:\n{combined_text}"
    )

# -------------------- Message Input --------------------
st.text_input("Type your message:", key="user_input", on_change=send_message)
st.button("Send", on_click=send_message)
