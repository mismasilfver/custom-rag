---
description: Review files for refactoring opportunities
---

Review the given files for refactoring opportunities.
When analyzing code, you will:

**FOCUS AREAS (in order of priority):**
1. **Unclear Names**: Identify variables, functions, types, and modules with ambiguous, misleading, or non-descriptive names. Propose specific, intention-revealing alternatives.
2. **Duplicated Code**: Detect repeated logic, similar patterns, and redundant implementations. Suggest consolidation strategies and abstraction opportunities.
3. **Unnecessary Elements**: Find dead code, unused variables, redundant conditions, over-engineered solutions, and complexity that doesn't add value.
4. **Missing Abstractions**: Identify opportunities for interfaces, base classes, utility functions, or design patterns that would simplify the code structure.
5. **Self documenting code**: Identify opportunities for self documenting code through better naming and structure over comments.


**ANALYSIS METHODOLOGY:**
- Prioritize changes by impact: high-impact, low-risk changes first
- Distinguish between style preferences and genuine structural issues
- Respect existing patterns and conventions unless they're genuinely problematic

**OUTPUT FORMAT:**
Structure your response as:

## Critical Analysis

### 🏷️ Unclear Names
[List specific naming issues with line references and proposed alternatives]

### 🔄 Duplicated Code
[Identify repeated patterns with suggestions for consolidation]

### 🗑️ Unnecessary Elements
[Point out redundant or over-complex code with simplification proposals]

### 🏗️ Missing Abstractions
[Suggest interfaces, patterns, or utilities that would improve structure]

## Refactoring Recommendations

### Priority 1: High Impact, Low Risk
[Most important changes that are safe to implement]

### Priority 2: Structural Improvements
[Larger refactoring that would significantly improve the codebase]

### Priority 3: Nice-to-Have
[Optional improvements for future consideration]

**QUALITY STANDARDS:**
- Be specific: provide exact line numbers, variable names, and concrete suggestions
- Be constructive: explain WHY each change would improve the code
- Be practical: consider implementation effort and risk
- Be respectful: acknowledge good patterns while suggesting improvements
- Follow project-specific guidelines from AGENTS.md when available

**CONSTRAINTS:**
- Never suggest removing comments unless they're actively misleading
- Respect the existing architecture unless fundamental changes are necessary
- Consider the project's maturity level and development phase
- Prioritize maintainability over performance optimizations unless performance is critical
