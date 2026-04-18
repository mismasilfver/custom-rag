import logging
import tempfile

import streamlit as st

from project_manager import ProjectManager
from rag_engine import RAGEngine

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="RAG UI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📚 RAG Document Q&A")

# Supported file types for the uploader
SUPPORTED_TYPES = ["pdf", "doc", "docx", "txt"]


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


def render_project_section():
    """Render the project selection and creation UI."""
    st.sidebar.header("📁 Projects")

    projects = pm.list_projects()

    # Project selector
    if not projects:
        st.sidebar.warning("No projects found.")
        return

    # Ensure current project is valid
    if st.session_state.current_project not in projects:
        st.session_state.current_project = projects[0]

    selected_project = st.sidebar.selectbox(
        "Select Project",
        projects,
        index=projects.index(st.session_state.current_project),
    )

    # Handle project switch
    if selected_project != st.session_state.current_project:
        st.session_state.current_project = selected_project
        st.session_state.messages = []  # Clear chat history on switch
        st.rerun()

    # Create new project
    with st.sidebar.expander("➕ New Project"):
        new_project_name = st.text_input("Project Name", key="new_proj_name")
        if st.button("Create", key="create_proj_btn"):
            if new_project_name:
                if pm.create_project(new_project_name):
                    st.success(f"Project '{new_project_name}' created!")
                    st.session_state.current_project = new_project_name
                    st.session_state.messages = []
                    st.rerun()
                else:
                    st.error("Invalid name or project already exists.")
            else:
                st.warning("Please enter a name.")

    st.sidebar.markdown("---")


def render_source_references(sources):
    """Render source references in an expander.

    Args:
        sources: List of source dicts with number, file_name, page_label, snippet
    """
    with st.expander("📚 Source references"):
        for source in sources:
            page_info = (
                f", Page {source['page_label']}" if source.get("page_label") else ""
            )
            label = f"**[{source['number']}] {source['file_name']}**{page_info}"
            st.markdown(label)
            st.caption(source["snippet"])
            st.markdown("---")


def render_chat_section(engine):
    """Render the chat interface with conversation history and source citations."""
    st.markdown("---")
    st.subheader("💬 Chat with your documents")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Show source references in expandable section if available
            if "sources" in message and message["sources"]:
                render_source_references(message["sources"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = engine.query_with_sources(prompt)
                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)

                    # Display source references
                    if sources:
                        render_source_references(sources)

                    # Store assistant response in history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_msg,
                            "sources": [],
                        }
                    )

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear chat history", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()


def render_ollama_section(engine):
    """Render the Ollama management section in the sidebar."""
    st.sidebar.header("🤖 Ollama")

    is_running = engine.check_ollama()

    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        st.write("**Status:**", ":green[Running]" if is_running else ":red[Stopped]")

    with col2:
        if is_running:
            if st.button("⏹️ Stop", key="stop_ollama"):
                engine.stop_ollama()
                st.rerun()
        else:
            if st.button("▶️ Start", key="start_ollama"):
                with st.spinner("Starting Ollama..."):
                    success = engine.start_ollama()
                if success:
                    st.success("Ollama started!")
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
                st.sidebar.warning(
                    f"Configured model **{current_model}** not found in Ollama. "
                    "Select a model below."
                )
                current_index = 0
            selected = st.sidebar.selectbox(
                "**Model:**",
                models,
                index=current_index,
            )
            if selected != current_model:
                engine.set_model(selected)
                st.sidebar.success(f"Switched to {selected}")
        else:
            st.sidebar.warning("No models found. Run `ollama pull` to add models.")


def render_file_section(engine):
    """Render the file management section in the sidebar."""
    st.sidebar.header("📄 Documents")

    # File uploader
    uploaded_files = st.sidebar.file_uploader(
        "Upload files",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        key="file_uploader",
    )

    if uploaded_files:
        file_paths = []
        for uploaded_file in uploaded_files:
            # Save to temp location and upload
            suffix = f"_{uploaded_file.name}"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                file_paths.append(tmp.name)

        engine.upload_files(file_paths)
        st.sidebar.success(f"Uploaded {len(file_paths)} file(s)")
        st.rerun()

    # List current files with remove buttons (filter out hidden files like .gitkeep)
    st.sidebar.markdown("**Current files:**")
    current_files = [f for f in engine.list_data_files() if not f.startswith(".")]

    # Initialize session state for delete confirmation
    if "confirm_delete_file" not in st.session_state:
        st.session_state.confirm_delete_file = None

    if not current_files:
        st.sidebar.caption("No documents in data folder")
    else:
        for filename in current_files:
            col1, col2 = st.sidebar.columns([4, 1])
            with col1:
                st.caption(f"📄 {filename}")
            with col2:
                if st.button("🗑️", key=f"remove_{filename}"):
                    st.session_state.confirm_delete_file = filename
                    st.rerun()

            # Show confirmation if this file is pending deletion
            if st.session_state.confirm_delete_file == filename:
                st.sidebar.warning(f"Delete **{filename}**?")
                col_confirm, col_cancel = st.sidebar.columns(2)
                with col_confirm:
                    if st.button("✅ Yes", key=f"confirm_delete_{filename}"):
                        from pathlib import Path

                        Path(engine.data_dir, filename).unlink()
                        st.session_state.confirm_delete_file = None
                        st.rerun()
                with col_cancel:
                    if st.button("❌ No", key=f"cancel_delete_{filename}"):
                        st.session_state.confirm_delete_file = None
                        st.rerun()

    # Indexing actions
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Indexing:**")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔍 Index", key="index_btn"):
            with st.spinner("Indexing documents..."):
                try:
                    engine.ensure_index()
                    st.sidebar.success("Indexed!")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")

    with col2:
        if st.button("🔄 Reindex", key="reindex_btn"):
            with st.spinner("Reindexing documents..."):
                try:
                    engine.rebuild_index()
                    st.sidebar.success("Reindexed!")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")

    if st.button("⚠️ Reset Everything", key="reset_btn"):
        st.sidebar.warning("This will delete all documents and index. Are you sure?")
        if st.button("Yes, reset everything", key="confirm_reset"):
            with st.spinner("Resetting..."):
                engine.reset()
            st.sidebar.success("Reset complete!")
            st.rerun()


def main():
    render_project_section()

    if st.session_state.current_project:
        engine = get_engine(st.session_state.current_project)
        if not engine:
            return

        render_ollama_section(engine)
        st.sidebar.markdown("---")
        render_file_section(engine)

        if not engine.check_ollama():
            st.error("🚫 Ollama is not running. Please start it from the sidebar.")
            return

        # Check if documents are indexed
        try:
            engine.ensure_index()
            render_chat_section(engine)
        except Exception:
            st.warning(
                "📄 Add documents and click **Index** in the sidebar to get started."
            )
    else:
        st.warning("Please create or select a project from the sidebar.")


if __name__ == "__main__":
    main()
