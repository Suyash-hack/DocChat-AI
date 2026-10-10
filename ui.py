import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
st.set_page_config(page_title="DocChat AI", page_icon="✦", layout="wide", initial_sidebar_state="expanded")

st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root{--bg:#090b14;--panel:#111421;--border:rgba(167,178,220,.14);--text:#f2f4ff;--muted:#a2a9c4;--purple:#9b8cff;--cyan:#6de4f2}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}.stApp{background:radial-gradient(ellipse at 10% 0%,rgba(111,78,230,.17),transparent 34%),radial-gradient(ellipse at 95% 15%,rgba(46,190,211,.09),transparent 30%),var(--bg);color:var(--text)}
[data-testid="stHeader"]{background:rgba(9,11,20,.72)}[data-testid="stSidebar"]{background:linear-gradient(180deg,#101322,#0d101b);border-right:1px solid var(--border)}.block-container{max-width:1440px;padding-top:1.6rem;padding-bottom:5rem}
h1,h2,h3,h4{font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.035em;color:var(--text)!important}p,label,li{color:#d9def2}
.brand{display:flex;align-items:center;gap:11px;margin:.2rem 0 1.6rem}.brand-mark{width:42px;height:42px;border-radius:14px;display:flex;align-items:center;justify-content:center;color:white;font-size:23px;font-weight:700;background:linear-gradient(135deg,#8d76ff,#6250d9 48%,#3fc9dc);box-shadow:0 8px 28px #775bff40}.brand-name{font-family:'Space Grotesk';color:#f6f7ff;font-weight:700;font-size:1.2rem;line-height:1.15}.brand-subtitle{color:#929bb9;font-size:.76rem;margin-top:4px}
.hero-shell{position:relative;overflow:hidden;padding:2rem;border-radius:26px;border:1px solid rgba(170,164,255,.19);background:radial-gradient(circle at 86% 20%,#5bd5e71f,transparent 25%),radial-gradient(circle at 75% 110%,#8161ff33,transparent 35%),linear-gradient(135deg,#181b30f5,#0e121ff5);box-shadow:0 18px 70px #0003;margin-bottom:1.25rem}.eyebrow{color:var(--cyan);text-transform:uppercase;letter-spacing:.15em;font-size:.72rem;font-weight:700;margin-bottom:.55rem}.hero-title{font-family:'Space Grotesk';font-size:clamp(2.2rem,4vw,3.65rem);font-weight:700;letter-spacing:-.065em;line-height:1.04;margin:0;color:#f7f8ff}.gradient-text{background:linear-gradient(90deg,#b6aaff,#8d9dff 45%,#76e7f1);-webkit-background-clip:text;background-clip:text;color:transparent}.hero-copy{color:#a8b0cc;font-size:1rem;line-height:1.7;max-width:680px;margin-top:1rem}.hero-orb{position:absolute;width:190px;height:190px;right:5%;top:12%;border-radius:50%;border:1px solid #9185ff38;box-shadow:0 0 0 24px #9185ff09,0 0 0 48px #6de4f206,inset 0 0 55px #7866ff14;pointer-events:none}.hero-orb:after{content:'✦';position:absolute;top:36%;left:40%;font-size:3rem;color:#c3beffb3;text-shadow:0 0 35px #8f7dffcc}
.metric-card{border:1px solid var(--border);background:linear-gradient(145deg,#171b2cf2,#10131ff2);border-radius:18px;padding:17px 18px;min-height:100px}.metric-label{color:#929bb9;font-size:.78rem;font-weight:600}.metric-value{color:#f4f5ff;font-family:'Space Grotesk';font-size:1.7rem;font-weight:700;margin-top:7px;line-height:1.1}.section-heading{font-family:'Space Grotesk';font-size:1.2rem;font-weight:600;color:#f0f2ff;margin:1.55rem 0 .75rem}.section-caption{color:#929bb9;font-size:.88rem;margin-top:-.4rem;margin-bottom:1rem}.document-chip{padding:10px 12px;margin:7px 0;border:1px solid var(--border);background:#ffffff06;border-radius:12px;color:#dfe3f8;font-size:.84rem;overflow-wrap:anywhere}.document-chip small{display:block;color:#8f99b9;margin-top:4px}.status-pill{display:inline-flex;align-items:center;gap:7px;padding:6px 10px;border-radius:999px;border:1px solid #6de4f233;background:#6de4f212;color:#a4f0f5;font-size:.75rem;font-weight:600}.status-dot{width:6px;height:6px;background:#69e3d0;border-radius:50%;box-shadow:0 0 10px #69e3d0a6}
div[data-testid="stChatMessage"]{background:#121625b8;border:1px solid var(--border);border-radius:18px;padding:1rem 1.1rem;margin-bottom:.85rem}div[data-testid="stChatInput"]{border:1px solid #978bff59;border-radius:18px;background:#111422f5;box-shadow:0 8px 30px #0003}div[data-testid="stChatInput"] textarea{color:#f4f5ff!important}.stButton>button{border:1px solid #978bff3d;border-radius:12px;background:#816fff14;color:#e9e7ff;transition:all .18s;min-height:42px}.stButton>button:hover{border-color:#6de4f28c;background:#816fff26;color:white;transform:translateY(-1px)}.stButton>button[kind="primary"]{background:linear-gradient(100deg,#7964ef,#5c73e8);border:0;color:white;font-weight:700}
[data-testid="stFileUploader"]{border:1px dashed #978bff59;border-radius:16px;padding:8px;background:#816fff09}[data-testid="stFileUploaderDropzone"]{background:#ffffff05;border:0}.stAlert{border-radius:14px}hr{border-color:var(--border)}[data-testid="stExpander"]{border:1px solid var(--border);border-radius:13px;background:#ffffff05}footer{visibility:hidden}
@media(max-width:800px){.hero-shell{padding:1.5rem}.hero-orb{opacity:.25;right:-65px}.hero-title{font-size:2.3rem}}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def normalize_response(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text").strip()
    return str(content).strip()

def build_document_index(files_data):
    documents, errors, names = [], [], []
    for filename, file_bytes in files_data:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file_bytes)
                temp_path = temp_file.name
            pages = PyPDFLoader(temp_path).load()
            for page in pages:
                page.metadata["source"] = filename
                page.metadata["page"] = int(page.metadata.get("page", 0)) + 1
            readable = [p for p in pages if p.page_content and p.page_content.strip()]
            if not readable:
                errors.append(f"{filename}: no selectable text found. Scanned PDFs may need OCR.")
                continue
            documents.extend(readable)
            names.append(filename)
        except Exception as exc:
            errors.append(f"{filename}: {exc}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
    if not documents:
        return None, {"documents": 0, "pages": 0, "chunks": 0, "names": [], "errors": errors}
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120, add_start_index=True)
    chunks = splitter.split_documents(documents)
    if not chunks:
        return None, {"documents": len(names), "pages": len(documents), "chunks": 0, "names": names, "errors": errors}
    return FAISS.from_documents(chunks, load_embeddings()), {"documents": len(names), "pages": len(documents), "chunks": len(chunks), "names": names, "errors": errors}

def get_answer(question, vectorstore, api_key):
    retrieved = vectorstore.as_retriever(search_kwargs={"k": 6}).invoke(question)
    if not retrieved:
        return "I couldn't find relevant information in the indexed documents. Try rephrasing your question.", []
    context = "\n\n---\n\n".join(
        f"[Source {i}: {doc.metadata.get('source', 'Unknown')}, page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for i, doc in enumerate(retrieved, 1)
    )
    prompt = f"""You are DocChat AI, a careful assistant answering from the user's documents.
Use only the supplied context; do not invent unsupported facts. If the answer is not present, say so clearly.
Use clear structure and add citations like [filename.pdf, p. 2]. Only use filenames and page numbers shown in the context.
If asked for a full-document summary, clarify that retrieved passages may not cover every page.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}"""
    model = ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=api_key, temperature=0.2, max_retries=2)
    answer = normalize_response(model.invoke(prompt).content)
    sources, seen = [], set()
    for doc in retrieved:
        source, page = doc.metadata.get("source", "Unknown"), doc.metadata.get("page", "?")
        key = (source, page, doc.page_content[:100])
        if key not in seen:
            seen.add(key)
            sources.append({"source": source, "page": page, "excerpt": doc.page_content.strip()})
    return answer, sources

def export_conversation(messages):
    lines = ["# DocChat AI — Conversation", ""]
    for message in messages:
        lines.extend([f"## {'You' if message['role'] == 'user' else 'DocChat AI'}", "", message["content"], ""])
        if message.get("sources"):
            lines.extend(["**Retrieved source excerpts**", ""])
            for source in message["sources"]:
                excerpt = source["excerpt"].replace("\n", " ")
                lines.append(f"- **{source['source']} — page {source['page']}**: {excerpt}")
            lines.append("")
    return "\n".join(lines)

if "messages" not in st.session_state: st.session_state.messages = []
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
if "index_stats" not in st.session_state: st.session_state.index_stats = {"documents": 0, "pages": 0, "chunks": 0, "names": [], "errors": []}
if "active_signature" not in st.session_state: st.session_state.active_signature = None
if "pending_question" not in st.session_state: st.session_state.pending_question = None

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">✦</div><div><div class="brand-name">DocChat AI</div><div class="brand-subtitle">Your intelligent document space</div></div></div>', unsafe_allow_html=True)
    st.markdown("#### Your knowledge base")
    st.caption("Upload one or more PDFs to get started.")
    uploads = st.file_uploader("Drop PDFs here", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    files_data = [(file.name, file.getvalue()) for file in uploads] if uploads else []
    source_signature = "|".join(f"{name}:{hashlib.sha256(data).hexdigest()}" for name, data in files_data)
    signature = hashlib.sha256(source_signature.encode()).hexdigest() if files_data else None
    if signature != st.session_state.active_signature:
        st.session_state.active_signature = signature
        st.session_state.vectorstore = None
        st.session_state.messages = []
        st.session_state.index_stats = {"documents": 0, "pages": 0, "chunks": 0, "names": [], "errors": []}
    if files_data and st.session_state.vectorstore is None:
        with st.spinner("Reading PDFs and building your searchable knowledge base..."):
            st.session_state.vectorstore, st.session_state.index_stats = build_document_index(files_data)
    stats = st.session_state.index_stats
    st.markdown("---")
    st.markdown("#### Workspace status")
    if st.session_state.vectorstore is not None:
        st.markdown('<span class="status-pill"><span class="status-dot"></span> Knowledge base ready</span>', unsafe_allow_html=True)
    elif files_data:
        st.warning("No searchable text was found in the selected PDFs.")
    else:
        st.caption("Waiting for your first PDF.")
    if stats["names"]:
        st.markdown("**Indexed documents**")
        for name in stats["names"]:
            st.markdown(f'<div class="document-chip">📄 {name}<small>Included in your knowledge base</small></div>', unsafe_allow_html=True)
    if stats["errors"]:
        with st.expander("File processing notes"):
            for error in stats["errors"]: st.warning(error)
    st.markdown("---")
    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()
    if st.button("↻ Rebuild document index", use_container_width=True, disabled=not files_data):
        st.session_state.vectorstore = None
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown('<div style="color:#8791b2;font-size:.78rem;line-height:1.65"><b style="color:#cbd1ec">Built for focused reading</b><br>PDF retrieval · Source excerpts · Gemini answers<br><br>Your API key belongs in your local <code>.env</code> file or your host’s Secrets settings.</div>', unsafe_allow_html=True)

st.markdown('''<div class="hero-shell"><div class="hero-orb"></div><div class="eyebrow">✦ YOUR AI READING COMPANION</div><div class="hero-title">Your documents.<br><span class="gradient-text">One smart conversation.</span></div><div class="hero-copy">Upload your PDFs, explore important details, and get grounded answers with source references — without digging through every page manually.</div><div style="margin-top:1.2rem"><span class="status-pill"><span class="status-dot"></span> Document intelligence workspace</span></div></div>''', unsafe_allow_html=True)
stats = st.session_state.index_stats
metric_cols = st.columns(3)
for col, (label, value, sublabel) in zip(metric_cols, [("PDFs indexed", stats["documents"], "Your uploaded files"), ("Text pages", stats["pages"], "Pages with readable text"), ("Search chunks", stats["chunks"], "Passages ready for retrieval")]):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div style="color:#7f89a9;font-size:.76rem;margin-top:7px">{sublabel}</div></div>', unsafe_allow_html=True)

if st.session_state.vectorstore is None:
    st.markdown('<div class="section-heading">Start with a document</div><div class="section-caption">Upload a PDF in the left sidebar. Once indexing finishes, ask questions about its content.</div>', unsafe_allow_html=True)
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown('<div style="text-align:center;padding:2rem 1rem;border:1px dashed #978bff47;background:#ffffff05;border-radius:22px"><div style="font-size:2.5rem">📚</div><div style="font-family:Space Grotesk;font-size:1.15rem;color:#f0f2ff;font-weight:600">Your knowledge base starts here</div><div style="color:#969fbe;font-size:.88rem;max-width:370px;margin:.55rem auto 0;line-height:1.6">Add lecture notes, reports, research papers, manuals, or any text-based PDF.</div></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="section-heading">Ask your documents</div><div class="section-caption">Choose a prompt to get started, or type your own question below.</div>', unsafe_allow_html=True)
    suggestions = [("✧", "Summarize key points", "Summarize the key points from the uploaded documents."), ("⌕", "Find important concepts", "What are the most important concepts I should understand?"), ("≋", "Create study notes", "Create structured study notes from the relevant document content."), ("?", "Generate practice questions", "Generate 10 practice questions with answers based on the uploaded documents.")]
    cols = st.columns(4)
    for col, (symbol, title, prompt) in zip(cols, suggestions):
        with col:
            if st.button(f"{symbol}  {title}", key=f"suggest_{title}", use_container_width=True):
                st.session_state.pending_question = prompt

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"], avatar="✨" if message["role"] == "assistant" else "🙂"):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"⌕ View retrieved sources ({len(message['sources'])})"):
                st.caption("Passages retrieved from your PDFs for this answer.")
                for j, source in enumerate(message["sources"], 1):
                    st.markdown(f"**{j}. {source['source']} · Page {source['page']}**")
                    st.write(source["excerpt"])
                    if j < len(message["sources"]): st.markdown("---")
        if message["role"] == "assistant":
            st.download_button("Download this answer", data=message["content"], file_name=f"docchat_answer_{i+1}.md", mime="text/markdown", key=f"download_answer_{i}")

question_from_input = st.chat_input("Ask anything about your PDFs…", disabled=st.session_state.vectorstore is None)
question = question_from_input or st.session_state.pending_question
if question:
    st.session_state.pending_question = None
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(question)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        answer, sources = "Your Gemini API key was not found. Add `GEMINI_API_KEY=your_key` to your local `.env` file, then restart Streamlit.", []
    else:
        with st.chat_message("assistant", avatar="✨"):
            with st.spinner("Searching your documents and preparing an answer..."):
                try:
                    answer, sources = get_answer(question, st.session_state.vectorstore, api_key)
                except Exception as exc:
                    answer, sources = ("I couldn't complete that request. Check your Gemini API key, model availability, internet connection, and terminal error.\n\n**Technical detail:** `" + str(exc) + "`"), []
            st.markdown(answer)
            if sources:
                with st.expander(f"⌕ View retrieved sources ({len(sources)})"):
                    st.caption("Retrieved passages supplied to the model.")
                    for j, source in enumerate(sources, 1):
                        st.markdown(f"**{j}. {source['source']} · Page {source['page']}**")
                        st.write(source["excerpt"])
                        if j < len(sources): st.markdown("---")
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    st.rerun()

if st.session_state.messages:
    st.markdown("---")
    export_col, note_col = st.columns([1, 2])
    with export_col:
        st.download_button("⇩ Export conversation", data=export_conversation(st.session_state.messages), file_name="docchat_conversation.md", mime="text/markdown", use_container_width=True)
    with note_col:
        st.caption("Answers use retrieved passages. For important work, verify inline citations against the source excerpts and original PDF.")
st.markdown('<div style="text-align:center;color:#6f7897;font-size:.76rem;padding-top:2.2rem">DOCCHAT AI · READ LESS, UNDERSTAND MORE</div>', unsafe_allow_html=True)
