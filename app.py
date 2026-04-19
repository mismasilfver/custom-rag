import logging
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from st_copy import st_copy

from constants import SUPPORTED_EXTENSIONS
from project_manager import ProjectManager
from rag_engine import RAGEngine, sources_contain_garbled

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.container(border=True):
    st.title("📚 RAG Document Q&A")
    st.caption("Query your documents using local LLMs")

# Supported file types for the uploader (derived from constants, .md excluded)
SUPPORTED_TYPES = [
    ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS) if ext != ".md"
]


# Initialize Project Manager and migrate legacy data if any
@st.cache_resource
def get_project_manager():
    pm = ProjectManager()
    if pm.migrate_legacy_data():
        logger.info("Migrated legacy data to 'default' project")
    # Ensure at least 'default' project exists
    if not pm.list_projects():
        pm.create_project("default")
    return pm


pm = get_project_manager()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_project" not in st.session_state:
    projects = pm.list_projects()
    st.session_state.current_project = projects[0] if projects else "default"


@st.cache_resource
def get_engine(project_name):
    """Get or create the RAGEngine instance for a specific project."""
    paths = pm.get_project_paths(project_name)
    if not paths:
        st.error(f"Project '{project_name}' not found!")
        return None
    return RAGEngine(data_dir=paths["data_dir"], chroma_dir=paths["chroma_dir"])


def get_chat_history_path(project_name):
    """Return the chat history JSON path for a project."""
    paths = pm.get_project_paths(project_name)
    return paths["chat_history_path"] if paths else None


def render_project_section():
    """Render the project selection and creation UI."""
    with st.sidebar.container(border=True):
        st.subheader("📁 Projects")

        projects = pm.list_projects()

        # Project selector
        if not projects:
            st.warning("No projects found.")
            return

        # Display project count metric
        st.metric("Total Projects", len(projects))

        # Ensure current project is valid
        if st.session_state.current_project not in projects:
            st.session_state.current_project = projects[0]

        selected_project = st.selectbox(
            "Select Project",
            projects,
            index=projects.index(st.session_state.current_project),
            label_visibility="collapsed",
        )

        # Handle project switch
        if selected_project != st.session_state.current_project:
            old_engine = get_engine(st.session_state.current_project)
            old_path = get_chat_history_path(st.session_state.current_project)
            if old_engine and old_path:
                old_engine.clear_chat_history(old_path)
            st.session_state.current_project = selected_project
            st.session_state.messages = []  # Clear chat history on switch
            st.rerun()

        # Create new project
        with st.expander("➕ New Project", expanded=False):
            # Use a counter to force widget reset after creation
            if "new_proj_key" not in st.session_state:
                st.session_state.new_proj_key = 0

            new_project_name = st.text_input(
                "Project Name",
                key=f"new_proj_name_{st.session_state.new_proj_key}",
                placeholder="Enter project name...",
            )
            if st.button("Create", key="create_proj_btn", use_container_width=True):
                if new_project_name:
                    if pm.create_project(new_project_name):
                        st.toast(f"✅ Project '{new_project_name}' created", icon="📁")
                        st.session_state.current_project = new_project_name
                        st.session_state.messages = []
                        # Increment key to force new widget instance (clears field)
                        st.session_state.new_proj_key += 1
                        st.rerun()
                    else:
                        st.error("Invalid name or project already exists.")
                else:
                    st.warning("Please enter a name.")

    st.sidebar.divider()


def render_source_references(sources):
    """Render source references in an expander.

    Args:
        sources: List of source dicts with number, file_name, page_label, snippet
    """
    with st.expander(f"📚 Source references ({len(sources)})"):
        for i, source in enumerate(sources):
            with st.container(border=True):
                page_info = (
                    f", Page {source['page_label']}" if source.get("page_label") else ""
                )
                label = f"**[{source['number']}] {source['file_name']}**{page_info}"
                st.markdown(label)
                snippet = source["snippet"]
                if len(snippet) > 200:
                    display_snippet = snippet[:200] + "..."
                else:
                    display_snippet = snippet
                st.caption(display_snippet)
            if i < len(sources) - 1:
                st.markdown("")


def render_chat_section(engine, chat_history_path):
    """Render the chat interface with conversation history and source citations."""
    st.divider()
    st.subheader("💬 Chat with your documents")

    # Show onboarding hint only when there is no conversation yet
    if not st.session_state.messages:
        with st.container(border=True):
            st.info(
                "**Chat history is preserved across page reloads.** "
                "Each message you send is saved to your project folder, so you can "
                "pick up right where you left off after refreshing the browser. "
                'Follow-up questions like *"tell me more"* or *"expand on point 2"* '
                "work naturally — the assistant always has the full conversation "
                "context. Use **🗑️ Clear chat history** below to start a fresh "
                "conversation."
            )

    # Display chat history in a scrollable container
    chat_container = st.container(height=400, border=True)
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                # Copy button for assistant messages
                if message["role"] == "assistant":
                    col_msg, col_copy = st.columns([10, 1])
                    with col_msg:
                        st.markdown(message["content"])
                    with col_copy:
                        st_copy(message["content"], key=f"copy_{i}")
                else:
                    st.markdown(message["content"])
                # Show timestamp if available
                if "timestamp" in message:
                    time_str = message["timestamp"].strftime("%H:%M")
                    st.caption(f"🕐 {time_str}")
                # Show source references in expandable section if available
                if "sources" in message and message["sources"]:
                    render_source_references(message["sources"])

    # Chat input - must be outside container to stay at bottom
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to history with timestamp
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "timestamp": datetime.now()}
        )

        # Display user message
        with chat_container:
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(prompt)
                time_str = datetime.now().strftime("%H:%M")
                st.caption(f"🕐 {time_str}")

        # Generate and display assistant response
        with chat_container:
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    try:
                        result = engine.chat(prompt, chat_history_path)
                        answer = result["answer"]
                        sources = result["sources"]

                        st.markdown(answer)

                        # Display timestamp for this response
                        time_str = datetime.now().strftime("%H:%M")
                        st.caption(f"🕐 {time_str}")

                        # Copy button for this response
                        st_copy(answer, key="copy_new_response")

                        # Display source references
                        if sources:
                            render_source_references(sources)
                            if sources_contain_garbled(sources):
                                st.session_state.garbled_detected = True

                        # Store assistant response in history with timestamp
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": sources,
                                "timestamp": datetime.now(),
                            }
                        )
                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.error(error_msg)
                        time_str = datetime.now().strftime("%H:%M")
                        st.caption(f"🕐 {time_str}")
                        # Copy button for error message
                        st_copy(error_msg, key="copy_error")
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": error_msg,
                                "sources": [],
                                "timestamp": datetime.now(),
                            }
                        )

    # Garbled sources warning + reindex offer
    if st.session_state.get("garbled_detected"):
        st.warning(
            "⚠️ Some source snippets appear garbled due to font encoding issues. "
            "Would you like to re-index using PDF→Markdown conversion "
            "for better results?"
        )
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Yes, re-index", key="reindex_markdown_btn"):
                with st.spinner("Converting PDFs to Markdown and re-indexing..."):
                    engine.reindex_with_markdown()
                st.session_state.garbled_detected = False
                st.session_state.messages = []
                st.success(
                    "Re-indexing complete. "
                    "Your queries should now show readable sources."
                )
                st.rerun()
        with col2:
            if st.button("No, keep current index", key="keep_index_btn"):
                st.session_state.garbled_detected = False
                st.rerun()

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear chat history", key="clear_chat"):
            engine.clear_chat_history(chat_history_path)
            st.session_state.messages = []
            st.toast("🗑️ Chat history cleared", icon="🧹")
            st.rerun()


def render_ollama_section(engine):
    """Render the Ollama management section in the sidebar."""
    with st.sidebar.container(border=True):
        st.subheader("🤖 Ollama")

        is_running = engine.check_ollama()

        # Status display with color-coded indicator
        status_emoji = "🟢" if is_running else "🔴"
        status_text = "Running" if is_running else "Stopped"
        st.markdown(f"**Status:** {status_emoji} {status_text}")

        col1, col2 = st.columns([1, 1])
        with col1:
            if is_running:
                if st.button("⏹️ Stop", key="stop_ollama", use_container_width=True):
                    engine.stop_ollama()
                    st.toast("⏹️ Ollama stopped", icon="🛑")
                    st.rerun()
            else:
                if st.button("▶️ Start", key="start_ollama", use_container_width=True):
                    with st.status("Starting Ollama..."):
                        success = engine.start_ollama()
                    if success:
                        st.toast("▶️ Ollama started", icon="🚀")
                    else:
                        st.error("Failed to start Ollama")
                    st.rerun()

        if is_running:
            models = engine.list_models()
            if models:
                current_model = engine.model_name
                current_index = (
                    models.index(current_model) if current_model in models else None
                )
                if current_index is None:
                    st.warning(
                        f"Configured model **{current_model}** not found. "
                        "Select a model below."
                    )
                    current_index = 0

                # Display model count metric
                st.metric("Available Models", len(models))

                selected = st.selectbox(
                    "Select Model",
                    models,
                    index=current_index,
                    label_visibility="collapsed",
                )
                if selected != current_model:
                    engine.set_model(selected)
                    st.toast(f"🤖 Switched to {selected}", icon="🔄")
            else:
                st.warning("No models found. Run `ollama pull` to add models.")


def render_file_section(engine):
    """Render the file management section in the sidebar."""
    with st.sidebar.container(border=True):
        st.subheader("📄 Documents")

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload files",
            type=SUPPORTED_TYPES,
            accept_multiple_files=True,
            key="file_uploader",
        )

        if uploaded_files:
            file_info = []
            for uploaded_file in uploaded_files:
                # Save to temp location and upload
                suffix = f"_{uploaded_file.name}"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    # Pass tuple of (temp_path, original_filename)
                    file_info.append((tmp.name, uploaded_file.name))

            with st.status(f"Processing {len(file_info)} file(s)...") as status:
                engine.upload_files(file_info)
                label = f"✅ Indexed {len(file_info)} file(s)"
                status.update(label=label, state="complete")
            st.toast(f"📄 Uploaded {len(file_info)} file(s)", icon="📤")

        # List current files with remove buttons (filter out hidden files)
        current_files = [f for f in engine.list_data_files() if not f.startswith(".")]

        # Display file count metric
        st.metric("Documents", len(current_files))

        # Initialize session state for delete confirmation
        if "confirm_delete_file" not in st.session_state:
            st.session_state.confirm_delete_file = None

        if not current_files:
            st.caption("No documents uploaded yet")
        else:
            with st.container(height=200):
                for filename in current_files:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.caption(f"📄 {filename}")
                    with col2:
                        if st.button("🗑️", key=f"remove_{filename}"):
                            st.session_state.confirm_delete_file = filename
                            st.rerun()

                    # Show confirmation if this file is pending deletion
                    if st.session_state.confirm_delete_file == filename:
                        st.warning(f"Delete **{filename}**?")
                        col_confirm, col_cancel = st.columns(2)
                        with col_confirm:
                            if st.button("✅ Yes", key=f"confirm_delete_{filename}"):
                                Path(engine.data_dir, filename).unlink()
                                st.toast(f"🗑️ Deleted {filename}", icon="🗑️")
                                st.session_state.confirm_delete_file = None
                                st.rerun()
                        with col_cancel:
                            if st.button("❌ No", key=f"cancel_delete_{filename}"):
                                st.session_state.confirm_delete_file = None
                                st.rerun()

        # Indexing actions
        st.divider()
        st.markdown("**Indexing:**")

        if st.button("🔄 Reindex", key="reindex_btn", use_container_width=True):
            with st.status("Reindexing documents...") as status:
                try:
                    engine.rebuild_index()
                    status.update(label="✅ Reindex complete", state="complete")
                    st.toast("🔄 Documents reindexed", icon="✅")
                except Exception as e:
                    status.update(label=f"❌ Error: {e}", state="error")

        if st.button("⚠️ Reset Everything", key="reset_btn", use_container_width=True):
            st.warning("Delete all documents and index?")
            if st.button("Yes, reset", key="confirm_reset", use_container_width=True):
                with st.status("Resetting...") as status:
                    engine.reset()
                    status.update(label="✅ Reset complete", state="complete")
                st.toast("⚠️ All data reset", icon="🗑️")
                st.rerun()


def main():
    render_project_section()

    if st.session_state.current_project:
        engine = get_engine(st.session_state.current_project)
        if not engine:
            return

        render_ollama_section(engine)
        st.sidebar.divider()
        render_file_section(engine)

        if not engine.check_ollama():
            st.error("🚫 Ollama is not running. Please start it from the sidebar.")
            return

        chat_history_path = get_chat_history_path(st.session_state.current_project)

        # Rehydrate display messages from disk after a page reload
        if not st.session_state.messages and chat_history_path:
            st.session_state.messages = engine.load_chat_messages(chat_history_path)

        # Check if documents are indexed
        try:
            engine.ensure_index()
            render_chat_section(engine, chat_history_path)
        except Exception:
            st.warning(
                "📄 Add documents and click **Index** in the sidebar to get started."
            )
    else:
        st.warning("Please create or select a project from the sidebar.")


if __name__ == "__main__":
    main()
