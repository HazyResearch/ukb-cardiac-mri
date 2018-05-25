## Coral Static Analysis

A super simple prototype of performing basic static analysis on Python Coral labeling functions.

To use, write a labeling function in Python:

```python
def is_large(width, height):
    if width * height > 100:
        return True
    else:
        return False
```

Then, you can use this library to convert the Python function into a Coral AST, and ask questions
about it:

```python
>>> import coral_sa as csa
>>> coral_ast = csa.convert_to_coral_ast(is_square)
>>> vocabs = csa.extract_vocabularies(coral_ast)
>>> print vocabs
['width', 'height', 'width|height']
```

### Setup

```bash
pip install meta
pip install ast
```

And close this repo.

### Limitations

There are several limitations:

* All Python data types are unimplemented; dictionaries are likely important in labeling functions,
  but the decompiler this package uses doesn't support them :(

* Random binary ops, comparison ops, etc. aren't supported; I just added what I needed to get my
  tests to run, but it shouldn't be too difficult to add more stuff.

* There's currently no interesting analysis being done on the constructed tree. The
  `extract_vocabularies` function shows one way of doing something with the trees; this tree uses
the `walk` call to apply a function to each AST node; the function checks if the node is vocabulary,
and  adds it to a list. 
