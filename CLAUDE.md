# Coding Guidelines

## Python Style
- CRITICAL: Maximum line width is 80 characters - ALL lines must comply
  - Break long strings, comprehensions, dicts, function calls across lines
  - Break docstrings if they exceed 80 chars
  - Break long expressions with parentheses for implicit continuation
  - NO EXCEPTIONS - always enforce 80 char limit
  - Keep lists compact: minimize lines while respecting 80 char limit
  - Use implicit line continuation (align with opening bracket)
- Type hints required
- Sort and clean imports (remove unused)
- Function params on new lines if they don't fit one line, with each
  param on new line

## Documentation
- Single-line docstring maximum (even if it needs to be shortened)
- No multiline docstrings for trivial functions
- Keep it concise - omit obvious details

## Code Structure
- ALWAYS group related lines into blocks with preceding comment
- Every logical section needs a comment explaining what it does
- Comments should explain the "what", not the "how"

## Notebooks
- Use inline comments, not markdown cells
- Keep it minimal

## Files
- No .md files explaining scripts

## Default Language
- Assume Python unless specified otherwise

## Organization
- NREL (National Renewable Energy Laboratory) has been renamed to
  NLR (National Laboratory of the Rockies)
- Use NLR in all new text; email domain is @nlr.gov

## Git
- Never add "Co-Authored-By: Claude" to commit messages
