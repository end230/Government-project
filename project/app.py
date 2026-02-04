import os
import streamlit as st
from docx import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Arabic-friendly multilingual embeddings (requires sentence-transformers)
# If you installed langchain-huggingface, you can switch to:
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


# -----------------------------
# Settings
# -----------------------------
DOCS_FOLDER = "Data"     # put your .docx files here
PERSIST_DIR = "db"       # where Chroma stores its data
TOP_K = 10               # retrieve how many chunks
SHOW_K = 4               # show how many results
CHUNK_SIZE = 2200        # bigger chunks to include tables/lists
CHUNK_OVERLAP = 300      # overlap to avoid cutting important parts


SYSTEM_PROMPT = """
أنت مساعد بحث قانوني.
بدون نموذج لغوي (LLM) ستقوم فقط بـ:
- إرجاع المقاطع الأكثر صلة من المستندات.
- تجنب تكرار نفس المقطع.
- لو السؤال تحية/كلام عام: اطلب من المستخدم سؤالاً متعلقاً بالمستندات.
- لو لا توجد نتائج: أخبر المستخدم أنه لا يوجد نص مطابق.
"""


def is_smalltalk(q: str) -> bool:
    q = q.strip().lower()
    small = [
        "hi", "hello", "how are you", "hey",
        "السلام عليكم", "ازيك", "ازىك", "كيف حالك", "مرحبا", "اهلا"
    ]
    return any(s in q for s in small)


def read_docx(path: str) -> str:
    doc = Document(path)
    # collect non-empty paragraphs
    lines = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(lines)


def build_vectorstore(folder=DOCS_FOLDER, persist_dir=PERSIST_DIR):
    # 1) read all docx
    texts = []
    metadatas = []

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    for fname in os.listdir(folder):
        # only .docx, skip Word temp/lock files
        if not fname.lower().endswith(".docx"):
            continue
        if fname.startswith("~$"):
            continue

        full_path = os.path.join(folder, fname)

        # if file path doesn't exist / broken shortcut
        if not os.path.isfile(full_path):
            continue

        content = read_docx(full_path)
        if content.strip():
            texts.append(content)
            metadatas.append({"source": fname})

    if not texts:
        raise ValueError(f"No .docx files found (or all empty) in folder: {folder}")

    # 2) split into chunks (bigger chunks are better for legal tables/lists)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = []
    chunk_meta = []

    # We'll also store the raw chunks per source to support "before/after" context
    raw_chunks_by_source = {}

    for text, meta in zip(texts, metadatas):
        parts = splitter.split_text(text)

        # keep for "context around"
        raw_chunks_by_source[meta["source"]] = parts

        for j, p in enumerate(parts):
            chunks.append(p)
            chunk_meta.append({**meta, "chunk_id": j})

    # 3) embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 4) vector store (Chroma)
    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=chunk_meta,
        persist_directory=persist_dir,
    )

    # Chroma persists automatically in newer versions; no need for vectordb.persist()

    return vectordb, raw_chunks_by_source


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Arabic Word RAG (No LLM)", layout="wide")
st.title("📚 Arabic Word RAG Chatbot (بدون LLM حالياً)")
st.caption("⚙️ هذا الإصدار يعرض المقاطع الأكثر صلة من ملفات Word. لاحقاً سنضيف LLM لتلخيص وإعطاء إجابة نهائية مباشرة.")

# Build / Load DB once
if "vectordb" not in st.session_state:
    with st.spinner("جاري تحميل المستندات وبناء قاعدة البحث (أول مرة قد تأخذ وقتاً)..."):
        vectordb, raw_chunks_by_source = build_vectorstore()
        st.session_state.vectordb = vectordb
        st.session_state.raw_chunks_by_source = raw_chunks_by_source

if "history" not in st.session_state:
    st.session_state.history = []


# Show system prompt for clarity
with st.expander("🧠 تعليمات النظام (System Prompt)"):
    st.write(SYSTEM_PROMPT.strip())

question = st.chat_input("اكتب سؤالك هنا...")

if question:
    st.session_state.history.append(("user", question))

    # handle small talk
    if is_smalltalk(question):
        answer = "مرحباً 👋 اسأل سؤالاً متعلقاً بالمستندات (مثلاً: ما نص المادة 36؟)."
        st.session_state.history.append(("assistant", answer))
    else:
        retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": TOP_K})
        docs = retriever.invoke(question)

        # deduplicate by normalized text
        seen = set()
        unique_docs = []
        for d in docs:
            txt_norm = " ".join(d.page_content.split())
            if txt_norm in seen:
                continue
            seen.add(txt_norm)
            unique_docs.append(d)
        docs = unique_docs

        if not docs:
            answer = "لم أجد نصاً مطابقاً أو قريباً في المستندات."
            st.session_state.history.append(("assistant", answer))
        else:
            # Build an organized answer (no LLM: we show relevant parts)
            answer = "✅ **أقرب مقاطع من المستندات:**\n\n"
            for i, d in enumerate(docs[:SHOW_K], start=1):
                src = d.metadata.get("source", "unknown")
                cid = d.metadata.get("chunk_id", None)
                answer += f"### {i}) المصدر: {src}\n"
                answer += d.page_content.strip() + "\n\n"
                if cid is not None:
                    answer += f"*(chunk_id: {cid})*\n\n"

            st.session_state.history.append(("assistant", answer))


# Render chat history
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.write(msg)

# Optional: show full context (before/after) for the top retrieved doc
if st.session_state.history:
    # Find last assistant message and if it had docs, allow user to explore context
    st.divider()
    st.subheader("🔎 عرض النص بالكامل + سياق قبل/بعد (أفضل نتيجة)")

    # We'll rerun retrieval only if the last user message exists
    last_user = None
    for r, m in reversed(st.session_state.history):
        if r == "user":
            last_user = m
            break

    if last_user and not is_smalltalk(last_user):
        retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": TOP_K})
        docs = retriever.invoke(last_user)

        # dedup again
        seen = set()
        unique_docs = []
        for d in docs:
            txt_norm = " ".join(d.page_content.split())
            if txt_norm in seen:
                continue
            seen.add(txt_norm)
            unique_docs.append(d)
        docs = unique_docs

        if docs:
            top = docs[0]
            src = top.metadata.get("source", "unknown")
            cid = top.metadata.get("chunk_id", None)

            st.write(f"**أفضل نتيجة من:** `{src}`")

            with st.expander("📄 عرض النص الكامل للجزء المسترجع"):
                st.write(top.page_content)

            # before/after context (if available)
            all_chunks = st.session_state.get("raw_chunks_by_source", {}).get(src, [])
            if cid is not None and all_chunks:
                before = all_chunks[cid - 1] if cid - 1 >= 0 else ""
                after = all_chunks[cid + 1] if cid + 1 < len(all_chunks) else ""

                with st.expander("🧩 عرض السياق (قبل/بعد) لاستكمال الجداول والقوائم"):
                    if before:
                        st.markdown("**قبل:**")
                        st.write(before)
                    st.markdown("**الجزء الأساسي:**")
                    st.write(top.page_content)
                    if after:
                        st.markdown("**بعد:**")
                        st.write(after)
        else:
            st.write("لا توجد نتائج لعرض السياق.")
    else:
        st.write("اكتب سؤالاً متعلقاً بالمستندات لعرض السياق.")
