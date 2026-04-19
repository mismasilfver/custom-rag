"""Shared constants for the RAG system."""

# Supported file extensions for document processing
SUPPORTED_EXTENSIONS = frozenset({".pdf", ".doc", ".docx", ".txt", ".md"})

# Chat engine configuration
CHAT_TOKEN_LIMIT = 3000
CHAT_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions exclusively based on "
    "the provided document context. If the answer is not found in the documents, "
    "say so clearly. Cite sources using numbered references like [1], [2], etc."
)

# Prompt template for citation-aware query responses
CITATION_PROMPT_TEMPLATE = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question. Please cite the sources of your information "
    "using numbered references like [1], [2], etc. Place citations "
    "immediately after the relevant information.\n"
    "At the end of your response, include a 'References' section listing "
    "each cited source with its number.\n\n"
    "Question: {query_str}\n"
    "Answer: "
)
