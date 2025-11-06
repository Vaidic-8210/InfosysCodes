# app.py
import streamlit as st
import uuid
import speech_recognition as sr
import ollama
import PyPDF2
import pandas as pd
from PIL import Image
import pytesseract
import io
import base64
import re
import time
import traceback

# ------------------------------- #
# CONFIG & SETUP
# ------------------------------- #
st.set_page_config(page_title="CodeGene", layout="wide")

# Path to tesseract - adjust if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ------------------------------- #
# HELPER: LOAD CSS
# ------------------------------- #
def load_css(file_name: str = "styles.css"):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")

# ------------------------------- #
# UTILITY FUNCTIONS
# ------------------------------- #
def new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {"title": "New Chat", "messages": []}
    st.session_state.current_chat = chat_id

def looks_like_code(text: str) -> bool:
    """Detect if text looks like programming code."""
    code_keywords = [
        "def ", "class ", "import ", "return ", "if ", "for ", "while ",
        "print(", "=", ":", "(", ")", "{", "}", ";"
    ]
    return sum(k in text for k in code_keywords) >= 3

def clean_ocr_code(text: str) -> str:
    """Normalize OCR output to cleaner code format."""
    replacements = {
        "“": '"', "”": '"', "‘": "'", "’": "'", "—": "-",
        "•": "-", "´": "'"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # normalize whitespace/indentation
    text = re.sub(r'^[ \t]+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+\n', '\n', text)
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E]', '', text)
    return text.strip()

def process_file(uploaded_file):
    """Process non-image files immediately (text/pdf/csv)."""
    try:
        if uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8")
        
        elif uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            pdf_text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
            
            # ✨ Summarize PDF intelligently
            with st.spinner("🤖 Summarizing PDF content..."):
                summary = call_ollama_once(
                    system_prompt="You are CodeGene AI, an expert document summarizer.",
                    user_prompt=f"Summarize the following PDF in concise bullet points:\n\n{pdf_text}",
                    model_name="llama2:latest"
                )

                # Return BOTH user upload message + assistant summary properly
                current_chat = st.session_state.chats[st.session_state.current_chat]
                current_chat["messages"].append({
                    "role": "user",
                    "content": f"📄 Uploaded file: {uploaded_file.name}"
                })
                current_chat["messages"].append({
                    "role": "assistant",
                    "content": f"**Summary of {uploaded_file.name}:**\n\n{summary}"
                })

                if current_chat["title"] == "New Chat":
                    current_chat["title"] = f"PDF: {uploaded_file.name}"

            return None  # Prevent double message

        elif uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            return df.to_string()
    except Exception as e:
        return f"File processing error: {e}"
    return "Unsupported file type"

def perform_ocr(image_file):
    """Extract text from an uploaded image file using pytesseract."""
    try:
        image_file.seek(0)
        image = Image.open(image_file).convert("RGB")
        custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        text = pytesseract.image_to_string(image, config=custom_config)
        # Quick cleanup of common OCR substitutions
        text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
        text = text.replace("•", "-").replace("\t", "    ")
        
        # If result seems very short, try alternative psm
        if len(text.strip()) < 10:
            alt_config = r'--oem 3 --psm 11'
            alt_text = pytesseract.image_to_string(image, config=alt_config)
            alt_text = alt_text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
            if len(alt_text.strip()) > len(text.strip()):
                text = alt_text

        return text.strip()
    except Exception as e:
        return f"OCR failed: {e}"

def image_to_base64(image_file):
    image_file.seek(0)
    image = Image.open(image_file).convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def extract_ollama_message(response_obj):
    """
    Extract assistant content from an Ollama response object (safe).
    Returns string content or str(response_obj) fallback.
    """
    try:
        # If it's an object with message attribute:
        if hasattr(response_obj, "message") and response_obj.message is not None:
            # Some SDKs have .message.content
            msg = response_obj.message
            if hasattr(msg, "content"):
                return msg.content or ""
            # maybe msg itself is dict-like
            try:
                return str(msg)
            except Exception:
                pass
        # If it's a dict-like response
        if isinstance(response_obj, dict):
            m = response_obj.get("message", {})
            if isinstance(m, dict):
                return m.get("content", "") or ""
            # sometimes nested
            return str(response_obj)
        # fallback to string
        return str(response_obj)
    except Exception:
        return str(response_obj)

def call_ollama_once(system_prompt, user_prompt, model_name="llama2:latest"):
    """
    Calls Ollama without streaming (single response) to avoid streaming-event logs.
    Returns assistant text (string) or raises exception.
    """
    try:
        resp = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # do NOT pass stream=True to avoid event objects
        )
        content = extract_ollama_message(resp)
        return content
    except Exception as e:
        # Re-raise so callers can catch and display errors
        raise


# ------------------------------- #
# SESSION INITIALIZATION
# ------------------------------- #
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
if "account_open" not in st.session_state:
    st.session_state.account_open = False
if "page" not in st.session_state:
    st.session_state.page = "Chat"
if "last_file" not in st.session_state:
    st.session_state.last_file = None

# image handling flags
if "pending_image" not in st.session_state:
    st.session_state.pending_image = None          # holds uploaded image file (temp)
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None     # avoid re-setting same upload on rerun
if "processing" not in st.session_state:
    st.session_state.processing = False

# create first chat if none
if not st.session_state.current_chat:
    new_chat()

# ------------------------------- #
# SIDEBAR
# ------------------------------- #
with st.sidebar:
    with st.container():
        if st.button("✏️ New Chat"):
            new_chat()

        search_query = st.text_input("🔍 Search...")

        st.markdown("### GENIEs")
        if st.button("Tools"):
            st.session_state.page = "Tools"

        st.markdown("### Recents")
        for chat_id, chat in reversed(list(st.session_state.chats.items())):
            if search_query.strip() == "" or search_query.lower() in chat["title"].lower():
                if st.button(chat["title"], key=chat_id):
                    st.session_state.current_chat = chat_id
                    st.session_state.page = "Chat"

    with st.container():
        st.markdown('<div class="account-footer">', unsafe_allow_html=True)
        if st.button("👤 Account"):
            st.session_state.account_open = not st.session_state.account_open
        if st.session_state.account_open:
            st.info("**User:** demo_user@example.com")
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------- #
# MAIN PAGE: CHAT
# ------------------------------- #
if st.session_state.page == "Chat":
    current_chat = st.session_state.chats[st.session_state.current_chat]

    st.markdown("## CodeGene")

    # Messages container
    st.markdown('<div class="messages-container">', unsafe_allow_html=True)
    for msg in current_chat["messages"]:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        if role == "user":
            html = f'<div class="chat-row user"><div class="chat-bubble user-msg">{content}</div></div>'
        else:
            html = f'<div class="chat-row bot"><div class="chat-bubble bot-msg">{content}</div></div>'
        st.markdown(html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat input container (fixed)
    st.markdown('<div class="stChatInputContainer">', unsafe_allow_html=True)
    col_input, col_voice, col_file = st.columns([10, 1, 1])

    # chat_input returns None normally; when user submits, returns string (possibly empty)
    with col_input:
        prompt = st.chat_input("Type your message...")

    with col_voice:
        voice_input = st.button("🎤", help="Voice input (speech-to-text)")

    # file_uploader in the input area (small browse button)
    with col_file:
        uploaded_file = st.file_uploader(
            "Upload file",
            type=["txt", "pdf", "csv", "jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="chat_file_uploader"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # handle voice recognition (only when pressed)
    if voice_input:
        try:
            st.info("🎙️ Listening... Speak now")
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            try:
                recognized = recognizer.recognize_google(audio)
                # set prompt so the rest of the flow handles it (chat_input won't have a value)
                prompt = recognized
                st.success(f"Recognized: {recognized}")
            except sr.UnknownValueError:
                st.error("Could not understand audio")
            except sr.RequestError:
                st.error("Speech recognition service unavailable")
        except Exception as e:
            st.error(f"Microphone error: {e}")

    # -------------------------------
    # FILE UPLOAD HANDLING (images wait; others processed immediately)
    # -------------------------------
    if uploaded_file is not None:
        # If new upload (different name), save into session_state.pending_image and track name
        if uploaded_file.name != st.session_state.get("last_uploaded_name"):
            st.session_state.last_uploaded_name = uploaded_file.name
            if uploaded_file.type.startswith("image/"):
                # keep the uploaded file object in pending_image (it stays across reruns)
                st.session_state.pending_image = uploaded_file
                st.info(f"📸 Image '{uploaded_file.name}' is ready. Type a question or press Enter to send.")
            else:
                # Non-image: process immediately and append to chat
                file_content = process_file(uploaded_file)
                current_chat["messages"].append({
                    "role": "user",
                    "content": f"📄 Uploaded file: {uploaded_file.name}\n\n{file_content[:1500]}"
                })
                # we processed a file so clear sentinel to avoid processing again
                st.session_state.last_uploaded_name = None

    # show a small preview next to input (like ChatGPT): display only when pending_image exists
    # Only show preview if not processing any request
    if prompt is not None:
        if st.session_state.processing:
            st.warning("Already processing a previous request. Please wait.")
        else:
            st.session_state.processing = True
            try:
                user_text = prompt.strip()
                has_image = st.session_state.pending_image is not None

                # Case 1: Image + Text
                if has_image:
                    image_file = st.session_state.pending_image
                    img_name = image_file.name

                    # --- OCR step ---
                    with st.spinner("🔍 Extracting text from image..."):
                        ocr_text = perform_ocr(image_file)
                        ocr_text = clean_ocr_code(ocr_text)
                    st.info(f"🧾 OCR Extracted Text:\n\n{ocr_text[:1000]}")  # preview for debugging

                    # Detect if it looks like code
                    is_code = looks_like_code(ocr_text)

                    # 3) Detect explicit user intent keywords (fix / explain / write / optimize)
                    user_intent = "explain"  # default
                    ut_lower = user_text.lower() if user_text else ""
                    if any(k in ut_lower for k in ["fix", "error", "bug", "correct", "repair"]):
                        user_intent = "fix"
                    elif any(k in ut_lower for k in ["explain", "what does", "describe", "meaning"]):
                        user_intent = "explain"
                    elif any(k in ut_lower for k in ["write", "implement", "create", "build", "solve"]):
                        user_intent = "write"
                    elif any(k in ut_lower for k in ["optimize", "improve", "refactor"]):
                        user_intent = "optimize"

                    # --- Construct AI prompt including OCR text ---
                    if is_code:
                        full_prompt = f"""
                    You are CodeGene AI, a highly skilled AI programmer and code mentor.
                    INPUT (from image OCR):
                    {ocr_text}

                    USER MESSAGE / INTENT:
                    {user_text or '(no text provided)'}
                    Detected intent: {user_intent}
                    

                    The following text was extracted from an image using OCR. It may be a Python code snippet, 
                    assignment, or programming question. You must decide the correct intent automatically.

                    Follow these rules:

                    1. 🧩 If the code looks complete and correct → 
                    Explain what it does line-by-line and describe its logic and output.

                    2. 🛠️ If the code has syntax or logic errors → 
                    Fix the code fully and provide the corrected version inside:
                        ```python
                        # Corrected Code Here
                        ```
                    Then, explain what was wrong and how you fixed it.

                    3. 🧠 If the text is a question asking for code (e.g., "write a function that...") → 
                    Write a complete, efficient, and readable solution using best practices, 
                    followed by a detailed explanation.

                    4. 💬 Always include a clear **final explanation** after any code.

                    5. 🧾 Never skip showing the corrected or written code in ```python ... ``` format.

                    Here is the extracted text or code to analyze:
                    {ocr_text}
                    """
                    else:
                        full_prompt = f"""
                        You are CodeGene AI, a helpful assistant.
                        The user uploaded an image named '{img_name}' and asked:
                        {user_text}

                        Extracted text from the image:
                        {ocr_text}
                        Determine automatically whether the content is:
                        - Code (Python, Java, JS, etc.)
                        - A question about code
                        - A general query

                        If it is code:
                        - Correct and explain it if broken.
                        - Explain it line-by-line if correct.
                        - Provide rewritten or optimized code if the user asks "improve" or "optimize".
                        If it is not code:
                        - Just answer clearly and precisely in human language.

                        Output should always include:
                        - ```python``` block when returning code
                        - Bullet points or numbered explanation
                        - Concise summary at the end
                        """

                    # --- Add to chat visually ---
                    img_b64 = image_to_base64(image_file)
                    st.session_state.chats[st.session_state.current_chat]["messages"].append({
                        "role": "user",
                        "content": f"<img src='data:image/png;base64,{img_b64}' "
                                f"style='max-width:200px;border-radius:10px;margin-bottom:8px; display:block;'/>{user_text}"
                    })

                    # Auto-update chat title for image uploads
                    if current_chat["title"] == "New Chat":
                        current_chat["title"] = f"Image: {img_name}"

                    # --- Clear the pending image immediately ---
                    st.session_state.pending_image = None

                    # --- Get AI response ---
                    with st.spinner("🤖 Generating answer..."):
                        answer = call_ollama_once(
                            system_prompt=(
                                    "You are CodeGene AI, an expert programming assistant. "
                                    "When a user asks to fix code, you MUST output a fully corrected and runnable version "
                                    "inside triple backticks labeled with the language (e.g., ```python). "
                                    "You MUST fix all logical, syntax, and runtime errors. "
                                    "Do NOT include greetings, intros, or meta text. "
                                    "After the code, give a concise bullet-point explanation of what was fixed. "
                                    "Do NOT repeat the user’s code or text."
                            ),
                            user_prompt=full_prompt,
                            model_name="llama2:latest"
                        )


                    st.session_state.chats[st.session_state.current_chat]["messages"].append({
                        "role": "assistant",
                        "content": answer
                    })

                # Case 2: Text-only message
                else:
                    st.session_state.chats[st.session_state.current_chat]["messages"].append({
                        "role": "user",
                        "content": user_text
                    })

                    # 🔹 Auto-update chat title
                    if current_chat["title"] == "New Chat" and user_text:
                        current_chat["title"] = user_text[:40] + ("..." if len(user_text) > 40 else "")

                    with st.spinner("🤖 Generating answer..."):
                        answer = call_ollama_once(
                            system_prompt="system_prompt",
                            user_prompt=user_text,
                            model_name="llama2:latest"
                        )


                    st.session_state.chats[st.session_state.current_chat]["messages"].append({
                        "role": "assistant",
                        "content": answer
                    })

                st.session_state.processing = False
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error while processing image or question:\n{e}")
                st.session_state.processing = False


# ------------------------------- #
# TOOLS SECTION
# ------------------------------- #
elif st.session_state.page == "Tools":
    st.title("🛠️ Tools")
    tool = st.radio("Select a tool", ["Image Generator", "Deep Research"])
    if st.button("Open Tool"):
        if tool == "Image Generator":
            st.session_state.page = "ImageGen"
        elif tool == "Deep Research":
            st.session_state.page = "Research"

elif st.session_state.page == "ImageGen":
    st.title("🖼️ Image Generator")
    img_prompt = st.text_input("Enter prompt for image generation")
    if st.button("Generate Image"):
        st.info(f"Image generated for: {img_prompt} (placeholder)")

elif st.session_state.page == "Research":
    st.title("🔎 Deep Research")
    query = st.text_area("Enter your research query", height=150)
    model_choice = st.radio("Select model", ["llama2:latest"], index=0)
    if st.button("Run Research"):
        if not query.strip():
            st.warning("⚠️ Please enter a query first.")
        else:
            try:
                assistant_text = call_ollama_once(
                    system_prompt="You are a deep research assistant. Provide a detailed, factual, structured answer.",
                    user_prompt=query,
                    model_name=model_choice
                )
                st.markdown("**Assistant:**")
                st.markdown(assistant_text)
            except Exception as e:
                st.error(f"Error calling Ollama: {e}\n{traceback.format_exc()}")

#style.css
/* Adjust the main app container to account for the fixed chat input */ 
.stApp { 
    padding-bottom: 7rem; 
} 
/* ============ Messages area ============ */ 
.messages-container { 
    padding: 2rem 3rem; 
    padding-bottom: 8rem; /* ensure chat input doesn't overlap messages */ 
    overflow-y: auto; 
    box-sizing: border-box; 
} 
/* Each row is a flex row — this is what allows proper left/right alignment 
*/ 
.chat-row { 
    display: flex; 
    width: 100%; 
    margin: 6px 0; 
    box-sizing: border-box; 
} 
.chat-row.user { justify-content: flex-end; } 
.chat-row.bot  { justify-content: flex-start; } 
 
/* Bubble */ 
.chat-bubble { 
    max-width: 70%; 
    padding: 10px 14px; 
    border-radius: 14px; 
    font-size: 0.95rem; 
    line-height: 1.3; 
    word-wrap: break-word; 
    box-shadow: none; 
} 
.user-msg { 
    background-color: #dbefff; 
    border-bottom-right-radius: 6px; 
    text-align: right; 
} 
.bot-msg { 
    background-color: #f1f1f1; 
    border-bottom-left-radius: 6px; 
    text-align: left; 
} 
/* Sidebar Styling */ 
[data-testid="stSidebar"] { 
    display: flex; 
    flex-direction: column; 
} 
[data-testid="stSidebarNav"] { 
    flex-grow: 1; 
    overflow-y: auto; 
    padding-bottom: 2rem; 
} 
.account-footer { 
    padding: 1rem; 
    border-top: 1px solid rgba(22, 23, 26, 0.1); 
    background-color: #f0f2f6; 

 
} 
.stButton>button { 
    background-color: transparent !important; 
    border: none !important; 
    box-shadow: none !important; 
    text-align: left; 
    padding: 0.5rem 0.5rem; 
    width: 100%; 
    color: #000; 
    font-size: 1rem; 
    transition: background-color 0.1s ease; 
} 
.stButton>button:hover, .stButton>button:focus { 
    background-color: rgba(22, 23, 26, 0.05) !important; 
    outline: none !important; 
} 
/* Main Page Styling */ 
[data-testid="stAppViewBlockContainer"] h3 { 
    padding: 0 !important; 
    margin: 0 !important; 
} 
 
/* Chat Input Styling */ 
.stChatInputContainer { 
    position: fixed; 
    bottom: 0; 
    left: 250px; 
    width: calc(100% - 250px); 
    background-color: #fff; 
    padding: 1rem; 
    border-top: 1px solid rgba(22, 23, 26, 0.1); 
    z-index: 999; 
    box-sizing: border-box; 
    display: flex; 
    align-items: center; 
    gap: 10px; 
} 
/* ============ File uploader: hide drag text but keep Browse button 
============ */ 
/* The dropzone container */ 
/* Hide drag-and-drop text */ 
div[data-testid="stFileUploaderDropzoneInstructions"] { 
    display: none !important; 
} 
 
/* Hide limit text */ 
div[data-testid="stFileUploaderDetails"] { 
    display: none !important; 
} 
 
 
/* remove large instruction paragraph if present */ 
div[data-testid="stFileUploaderDropzone"] p { 
    display: none !important; 
} 
 
/* make the button small and inline */ 
div[data-testid="stFileUploaderDropzone"] button { 
    margin-left: 6px !important; 
    padding: 6px 10px !important; 
    height: 40px !important; 
 
 
} 
/* ------------------------------ 
   REMOVE uploader background completely ------------------------------ */ 
/* Outer uploader container */ 
div[data-testid="stFileUploader"] { 
    background: transparent !important; 
    border: none !important; 
    box-shadow: none !important; 
    padding: 0 !important; 
    margin: 0 !important; 
    display: inline-flex !important; 
    align-items: center !important; 
} 
 
/* Dropzone area */ 
div[data-testid="stFileUploaderDropzone"] { 
    background: transparent !important; 
    border: none !important; 
    box-shadow: none !important; 
    padding: 0 !important; 
    margin: 0 !important; 
} 
 
/* Inner section that sometimes adds bg */ 
div[data-testid="stFileUploader"] section { 
    background: transparent !important; 
    border: none !important; 
    box-shadow: none !important; 
    padding: 0 !important; 
    margin: 0 !important; 
} 
 
/* Target *all* uploader children to be transparent */ 
div[data-testid="stFileUploader"] * { 
    background: transparent !important; 
    border: none !important; 
    box-shadow: none !important; 
} 
 
/* Style Browse Files button like mic button */ 
div[data-testid="stFileUploader"] button { 
    background-color: white !important; 
    border: 1px solid rgba(0, 0, 0, 0.2) !important; 
    border-radius: 6px !important; 
    padding: 0.4rem 0.6rem !important; 
    font-size: 1rem !important; 
    cursor: pointer !important; 
    transition: background-color 0.2s ease; 
} 
 
/* Hover effect */ 
div[data-testid="stFileUploader"] button:hover { 
    background-color: rgba(22, 23, 26, 0.05) !important; 
} 
/* Keep it inline with mic button */ 
div[data-testid="stFileUploader"] { 
    display: inline-flex !important; 
    align-items: center !important; 
    margin-left: 4px !important;  /* small gap from mic */
