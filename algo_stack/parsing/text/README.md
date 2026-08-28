# Text Parser — 文

Parses UTF-8 text formats: `.txt`, `.md`, `.json`, `.csv`, `.html`.

```python
from algo_stack.parsing.text import TextParser
doc = TextParser().parse_bytes(b"# Hi", "note.md")
```

See [`../README.md`](../README.md) for the unified API.
