#app.py
import streamlit as st
import uuid
import subprocess
from PIL import Image
import pytesseract

# -------------------
# Function to stream Ollama LLaMA2 responses
# -------------------
def stream_ollama(prompt, model="llama2"):
    """
    Streams response from Ollama (LLaMA2) word by word.
    """
    try:
        process = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Send prompt
        process.stdin.write(prompt)
        process.stdin.close()

        # Stream words
        for line in process.stdout:
            yield line.strip()

        process.stdout.close()
        process.wait()

    except Exception as e:
        yield f"⚠️ Could not connect to Ollama: {e}"

# -------------------
# OCR function with language support
# -------------------
def extract_text_from_image(uploaded_file, lang_code="eng"):
    """
    Performs OCR on the uploaded image with the given language code.
    """
    try:
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image, lang=lang_code)
        return text.strip()
    except Exception as e:
        return f"⚠️ OCR failed: {e}"

# -------------------
# Initialize session state
# -------------------
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

# -------------------
# Sidebar controls
# -------------------
st.sidebar.title("Chat Menu")

# ✅ New Chat button: resets input state too
if st.sidebar.button("New Chat"):
    new_id = str(uuid.uuid4())[:8]
    st.session_state.chats[new_id] = {"name": "Untitled Chat", "messages": []}
    st.session_state.current_chat = new_id
    # Reset file upload, input, OCR text, etc.
    if "file_uploader" in st.session_state:
        del st.session_state["file_uploader"]
    if "ocr_text" in st.session_state:
        del st.session_state["ocr_text"]
    st.rerun()

if st.sidebar.button("Clear Current Chat"):
    if st.session_state.current_chat:
        st.session_state.chats[st.session_state.current_chat]["messages"] = []
        st.session_state.chats[st.session_state.current_chat]["name"] = "Untitled Chat"
        st.rerun()

if st.sidebar.button("Delete Current Chat"):
    if st.session_state.current_chat:
        del st.session_state.chats[st.session_state.current_chat]
        st.session_state.current_chat = None
        st.rerun()

if st.sidebar.button("Delete All Chats"):
    st.session_state.chats = {}
    st.session_state.current_chat = None
    st.rerun()

# Current chat display
st.sidebar.subheader("Current Chat")
if st.session_state.current_chat:
    st.sidebar.write(st.session_state.chats[st.session_state.current_chat]["name"])
else:
    st.sidebar.write("No active chat")

# Chat history display
st.sidebar.subheader("Chat History")
for chat_id, chat_data in st.session_state.chats.items():
    if chat_id != st.session_state.current_chat:
        if st.sidebar.button(chat_data["name"]):
            st.session_state.current_chat = chat_id
            st.rerun()

# -------------------
# Main Chat Window
# -------------------
st.title("🧠 Simple Chatbot (Ollama + OCR + 🌐 Multilingual)")

if st.session_state.current_chat is None:
    st.info("Click 'New Chat' in the sidebar to start.")
else:
    chat_id = st.session_state.current_chat
    chat_data = st.session_state.chats[chat_id]

    st.subheader(chat_data["name"])
    st.markdown("**Model:** llama2")

    # Display conversation for this chat ONLY
    for role, text in chat_data["messages"]:
        st.chat_message(role).markdown(text)

    # =====================
    # 🔸 OCR Image Upload with Language Selection
    # =====================
    st.write("### 📝 OCR Section")
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_image = st.file_uploader("📷 Upload an image", type=["png", "jpg", "jpeg"], key=f"file_uploader_{chat_id}")
    with col2:
        lang_choice = st.selectbox(
            "Language",
            options={
                "eng": "English",
                "hin": "Hindi",
                "tam": "Tamil",
                "tel": "Telugu",
                "kan": "Kannada",
                "mal": "Malayalam",
                "ben": "Bengali",
                "mar": "Marathi",
                "guj": "Gujarati",
                "eng+hin": "English + Hindi (Mixed)"
            }.items(),
            format_func=lambda x: x[1],
            key=f"lang_select_{chat_id}"
        )
        lang_code = lang_choice[0]  # get code like 'eng'

    ocr_text = None
    if uploaded_image:
        with st.spinner("🔍 Extracting text from image..."):
            ocr_text = extract_text_from_image(uploaded_image, lang_code=lang_code)
        if ocr_text:
            st.success("✅ Text extracted from image:")
            st.text_area("Extracted Text", ocr_text, height=150, key=f"ocr_text_{chat_id}")

    # =====================
    # 💬 Chat input (text or OCR text)
    # =====================
    user_input = st.chat_input("Type your message...") or (ocr_text if uploaded_image and ocr_text else None)

    if user_input:
        # Add and display user message
        chat_data["messages"].append(("user", user_input))
        st.chat_message("user").markdown(user_input)

        # Name chat after first input
        if chat_data["name"] == "Untitled Chat":
            chat_data["name"] = user_input[:30]

        # Stream assistant response
        response_text = ""
        with st.chat_message("assistant"):
            placeholder = st.empty()
            for chunk in stream_ollama(user_input):
                response_text += " " + chunk
                placeholder.markdown(response_text + "▌")
            placeholder.markdown(response_text)

        # Save assistant response
        chat_data["messages"].append(("assistant", response_text.strip()))
