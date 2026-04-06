# Coding Agent Guidelines
# This file controls how the AI coding assistant interacts with your codebase.
# Place this in your project root and the agent will follow these rules.

## Core Principles
- Always take TDD approach to every change
- Break down each change to as atomic as possible steps
- Practice clean code practices
- Follow SOLID principles
- For tests follow DAMP (Descriptive, Accessible, Minimal, Precise) principles
- Follow Thoughtworks sensible defaults

## Debugging & Learning
- Encourage reading and understanding error messages instead of just fixing issues
- Help identify patterns in mistakes to improve debugging skills
- Suggest different approaches instead of leading to one specific solution
- Guide toward using console.log(), browser dev tools, and other debugging techniques

## Code Style
- Use meaningful variable and function names
- Keep functions small and focused on a single responsibility
- Avoid code duplication (DRY principle)
- Write self-documenting code; minimize comments unless explaining "why"
- Follow the existing code style in the project

## Testing
- Write tests before implementing features (TDD)
- Ensure all tests pass before committing changes
- Add regression tests when fixing bugs
- Aim for high test coverage on critical paths

## Commits & Changes
- Make minimal, focused changes
- One logical change per commit
- Explain the reasoning behind changes, not just what changed
- Never delete or modify tests without explicit direction
- Use Conventional Commits style for commit messages

## Communication
- Be direct and concise
- Use second person for user, first person for self
- No acknowledgment phrases like "Great idea!" or "You're absolutely right!"
- Reference specific files and line numbers when discussing code
- Explain the "why" behind recommendations

## Before Making Changes
1. Understand the existing code structure
2. Verify the root cause of issues (don't treat symptoms)
3. Consider if a single-line fix would suffice
4. Check for existing patterns in the codebase
5. Ensure you have a clear plan before editing

## Dependencies
- Prefer standard library solutions when sufficient
- Only add dependencies when necessary.
- Before adding new dependency, verify with user to avoid supply chain risks
- Verify compatibility with existing dependency management
- Never hardcode API keys or secrets

## Security
- Follow secure coding practices
- Never expose secrets in code
- Validate all inputs
- Be cautious with external requests and file system operations

## Project-Specific Rules
# Custom RAG with Ollama - Project Guidelines

## Preferred Frameworks/Libraries
- **Core Framework**: LlamaIndex (llama-index-core >=0.14.0)
- **LLM/Embeddings**: Ollama integration via llama-index-llms-ollama and llama-index-embeddings-ollama
- **Vector Store**: ChromaDB (llama-index-vector-stores-chroma)
- **Web UI**: Streamlit (>=1.30.0)
- **PDF Processing**: PyPDF (>=4.0)
- **Testing**: pytest with pytest-mock for mocking

## Architecture Patterns
- **RAGEngine Class**: Central abstraction with lazy initialization - no side effects on import
- **Lazy Loading**: Cache expensive resources (_index, _llm, _embed_model) until first access
- **Separation of Concerns**: 
  - rag_engine.py = Core RAG logic
  - app.py = Streamlit web interface
  - custom-rag.py = CLI wrapper
- **Session State**: Use Streamlit's st.session_state for UI state management
- **Private Attributes**: Use underscore prefix for internal/lazy-loaded state (_ollama_process, _index)

## Naming Conventions
- **Files**: snake_case (rag_engine.py, custom-rag.py)
- **Classes**: PascalCase (RAGEngine)
- **Functions/Variables**: snake_case
- **Private Attributes**: _prefix for internal state (_index, _llm, _embed_model)
- **Constants**: UPPER_SNAKE_CASE (SUPPORTED_TYPES)

## Testing Standards
- **Structure**: tests/unit/ for unit tests, tests/integration/ for integration tests
- **Fixtures**: Use conftest.py for shared fixtures (tmp_data_dir, tmp_chroma_dir, mock_ollama_responses)
- **Mocking**: Mock external dependencies (Ollama HTTP calls, subprocess.Popen)
- **DAMP Tests**: Descriptive test names, accessible fixtures, minimal setup, precise assertions
- **Test Classes**: Group by functionality (TestRAGEngineCheckOllama, TestRAGEngineFileManagement)
- **Temporary Resources**: Use tmp_path for test isolation

## Code Patterns to Follow
- **Lazy Initialization Pattern**:
  ```python
  @property
  def _get_llm(self):
      if self._llm is None:
          self._llm = OllamaLLM(model=self.model_name)
      return self._llm
  ```
- **Null-Safe Cleanup**: Check existence before operations (if self._ollama_process:)
- **Error Handling**: Try/except with user-friendly messages in UI, detailed logging for debugging
- **Logging**: Use logging module with structured format, not print statements

## Security & Privacy
- **No Telemetry**: Keep .streamlit/config.toml with telemetry disabled
- **No Hardcoded Secrets**: Never commit API keys or credentials
- **File Validation**: Reject unsupported file types in upload_files()
- **Safe Deletion**: Confirmation dialogs before destructive operations (reset, file delete)

## Documentation Requirements
- **Docstrings**: All public methods must have docstrings explaining purpose and behavior
- **README**: Keep README.md updated with features, usage, and project structure
- **Comments**: Explain "why", not "what"
- **Type Hints**: Optional but encouraged for public methods

## Development Workflow
1. Write tests first (TDD)
2. Mock external dependencies in unit tests
3. Use integration tests for end-to-end flows
4. Ensure all tests pass before committing
5. Update README for new features
6. Keep dependencies minimal - verify with user before adding
