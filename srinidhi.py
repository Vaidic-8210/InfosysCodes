#app.py
import os, json, asyncio 
from datetime import datetime 
import streamlit as st 
import ollama 
from httpx import ConnectError 
from PIL import Image 
import pytesseract 
import fitz  # PyMuPDF 
 
# ----------------- CONFIG ----------------- 
st.set_page_config(page_title="Chatbot", page_icon="") 
HISTORY_DIR = "history" 
MODEL = "tinyllama" 
MAX_KEEP = 10 
MAX_NAME = 20 
USER, BOT = "user", "assistant" 
os.makedirs(HISTORY_DIR, exist_ok=True) 
 
# Optional: set Tesseract path on Windows (safe no-op elsewhere) 
try: 
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract
OCR\tesseract.exe" 
except Exception: 
    pass 
 
# ----------------- STATE ----------------- 
def ensure_state(): 
    ss = st.session_state 
    ss.setdefault("session_name", f"New Chat {datetime.now().strftime('%H
%M')}") 
    ss.setdefault("messages", []) 
    ss.setdefault("first_message", True) 
    ss.setdefault("rename_target", None) 
    ss.setdefault("menu_open", {}) 
    ss.setdefault("file", None) 
    ss.setdefault("input_text", "") 
    ss.setdefault("context_used", False) 
 
ensure_state() 
 
# ----------------- STORAGE ----------------- 
@st.cache_data(ttl=3600) 
def list_sessions(): 
    try: 
        files = [f for f in os.listdir(HISTORY_DIR) if f.endswith(".json")] 
        files.sort(key=lambda f: os.path.getctime(os.path.join(HISTORY_DIR, 
f)), reverse=True) 
        return [os.path.splitext(f)[0] for f in files] 
    except OSError: 
        return [] 
 
def load_session(name: str): 
    path = os.path.join(HISTORY_DIR, f"{name}.json") 
    try: 
        with open(path, "r", encoding="utf-8") as f: 

 
            return json.load(f) 
    except Exception: 
        return [] 
 
def save_session(name: str, messages): 
    path = os.path.join(HISTORY_DIR, f"{name}.json") 
    try: 
        with open(path, "w", encoding="utf-8") as f: 
            json.dump(messages, f, indent=2, ensure_ascii=False) 
        st.cache_data.clear() 
    except Exception as e: 
        st.error(f"Save error: {e}") 
 
def sanitize_name(s: str) -> str: 
    s2 = "".join(c for c in s if c.isalnum() or c in (" ", "_", "
")).strip()[:MAX_NAME] 
    return s2 or "Untitled" 
 
# ----------------- OCR/EXTRACT ----------------- 
def extract_from_pdf(file) -> str: 
    try: 
        text = "" 
        with fitz.open(stream=file.read(), filetype="pdf") as doc: 
            for p in doc: 
                text += p.get_text() 
        return text.strip() 
    except Exception as e: 
        return f"" 
 
def extract_from_image(file) -> str: 
    try: 
        img = Image.open(file) 
        return pytesseract.image_to_string(img).strip() 
    except Exception: 
        return "" 
 
def get_context_once() -> str: 
    ss = st.session_state 
    if not ss.file or ss.context_used: 
        return "" 
    name = ss.file.name.lower() 
    ctx = "" 
    if name.endswith(".pdf"): 
        ctx = extract_from_pdf(ss.file) 
    elif any(name.endswith(ext) for ext in (".png", ".jpg", ".jpeg")): 
        ctx = extract_from_image(ss.file) 
    ss.context_used = True 
    return ctx 
 
# ----------------- MODEL ----------------- 
def stream_reply(messages): 
    try: 
        for chunk in ollama.chat(model=MODEL, messages=messages[
MAX_KEEP:], stream=True): 
            yield chunk["message"]["content"] 
    except ConnectError as e: 
        st.error(f"Ollama not reachable: {e}") 
        yield "" 
    except ollama.ResponseError as e: 
        st.error(f"Ollama error: {e.error}") 
        yield "" 
 
 
 
# ----------------- ACTIONS ----------------- 
def on_send(): 
    ss = st.session_state 
    prompt = ss.input_text.strip() 
    if not prompt: 
        return 
    # Title from first user message 
    if ss.first_message: 
        ss.session_name = sanitize_name(prompt) 
        ss.first_message = False 
 
    # Build contextualized last message if file context available once 
    ctx = get_context_once() 
    final = (f"You are an assistant that answers based on the provided 
context. " 
             f"Do not repeat the context; only answer the question.\n\n" 
             f"CONTEXT:\n---\n{ctx}\n---\n\nQUESTION: {prompt}") if ctx 
else prompt 
 
    # Append user, call model with modified last content 
    ss.messages.append({"role": USER, "content": prompt}) 
    tmp = ss.messages[:-1] + [{"role": USER, "content": final}] 
 
    full = "" 
    with st.spinner("Thinking..."): 
        for piece in stream_reply(tmp): 
            full += piece 
 
    if full: 
        ss.messages.append({"role": BOT, "content": full}) 
        save_session(ss.session_name, ss.messages) 
 
    ss.input_text = "" 
 
def on_new_chat(): 
    st.session_state.session_name = f"New Chat 
{datetime.now().strftime('%H-%M')}" 
    st.session_state.messages = [] 
    st.session_state.first_message = True 
    st.session_state.rename_target = None 
    st.session_state.menu_open = {} 
    st.session_state.file = None 
    st.session_state.input_text = "" 
    st.session_state.context_used = False 
 
def on_choose_session(name: str): 
    st.session_state.session_name = name 
    st.session_state.messages = load_session(name) 
    st.session_state.first_message = False 
    st.session_state.rename_target = None 
    st.session_state.file = None 
    st.session_state.context_used = False 
 
def on_delete(name: str): 
    try: 
        os.remove(os.path.join(HISTORY_DIR, f"{name}.json")) 
        st.cache_data.clear() 
        if st.session_state.session_name == name: 
            on_new_chat() 
    except Exception as e: 

 
        st.error(f"Delete error: {e}") 
 
def on_rename(old: str, new: str): 
    new2 = sanitize_name(new) 
    src = os.path.join(HISTORY_DIR, f"{old}.json") 
    dst = os.path.join(HISTORY_DIR, f"{new2}.json") 
    if os.path.exists(dst): 
        st.error(f"Chat '{new2}' exists.") 
        return 
    try: 
        os.rename(src, dst) 
        st.cache_data.clear() 
        st.session_state.rename_target = None 
        if st.session_state.session_name == old: 
            st.session_state.session_name = new2 
    except Exception as e: 
        st.error(f"Rename error: {e}") 
 
def on_file_upload(): 
    if st.session_state.uploader: 
        st.session_state.file = st.session_state.uploader 
        st.session_state.context_used = False 
 
def on_clear_file(): 
    st.session_state.file = None 
    st.session_state.context_used = False 
    st.rerun() 
 
# ----------------- UI ----------------- 
st.title("Ollama Chatbot ") 
 
with st.sidebar: 
    st.header("New Chat") 
    if st.button("➕ New Chat"): 
        on_new_chat() 
        st.rerun() 
 
    st.markdown("---") 
    st.subheader("Saved Chats") 
 
    for name in list_sessions(): 
        row = st.container() 
        with row: 
            c1, c2 = st.columns([0.85, 0.15]) 
            with c1: 
                if st.button(name, key=f"load_{name}"): 
                    on_choose_session(name) 
                    st.rerun() 
            with c2: 
                if st.button("⋮", key=f"opt_{name}"): 
                    cur = st.session_state.menu_open.get(name, False) 
                    st.session_state.menu_open = {k: False for k in 
st.session_state.menu_open} 
                    st.session_state.menu_open[name] = not cur 
                    st.rerun() 
 
            if st.session_state.menu_open.get(name, False): 
                c3, c4 = st.columns(2) 
                with c3: 
                    if st.button("Rename", key=f"rn_{name}"): 
                        st.session_state.rename_target = name 

 
                        st.session_state.menu_open[name] = False 
                        st.rerun() 
                with c4: 
                    if st.button("Delete", key=f"del_{name}"): 
                        on_delete(name) 
                        st.rerun() 
 
        if st.session_state.rename_target == name: 
            new = st.text_input("New name", value=name, key=f"in_{name}") 
            cc1, cc2 = st.columns(2) 
            with cc1: 
                if st.button("Save", key=f"sv_{name}"): 
                    on_rename(name, new) 
                    st.rerun() 
            with cc2: 
                if st.button("Cancel", key=f"cx_{name}"): 
                    st.session_state.rename_target = None 
                    st.rerun() 
 
st.subheader(f"Current Chat: {st.session_state.session_name}") 
 
for m in st.session_state.messages: 
    with st.chat_message(m["role"]): 
        st.markdown(m["content"]) 
 
if st.session_state.file: 
    with st.container(): 
        colA, colB = st.columns([0.85, 0.15]) 
        with colA: 
            if getattr(st.session_state.file, "type", 
"").startswith("image/"): 
                st.image(st.session_state.file, caption="Image attached") 
            elif getattr(st.session_state.file, "type", "") == 
"application/pdf": 
                st.info(f" PDF attached: `{st.session_state.file.name}`") 
        with colB: 
            st.button("Clear", use_container_width=True, 
on_click=on_clear_file) 
 
# Composer 
box = st.container() 
with box: 
    up, txt, send = st.columns([1.5, 7, 1.5]) 
    with up: 
        with st.popover(""): 
            st.file_uploader( 
                "Upload image or PDF", 
                type=["jpg", "jpeg", "png", "pdf"], 
                key="uploader", 
                label_visibility="collapsed", 
                on_change=on_file_upload, 
            ) 
    with txt: 
        st.text_input("Type a message...", key="input_text", 
label_visibility="collapsed") 
    with send: 
        st.button("➤", use_container_width=True, on_click=on_send) 
