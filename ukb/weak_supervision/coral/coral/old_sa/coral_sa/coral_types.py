
class Type(object):
    pass

class LabelingFunctionType(Type):
    def __str__(self):
        return "LF"

class VocabType(Type):
    def __init__(self, deps):
        # This vocabulary is dependent on the given list of vocabulary types.
        self.deps = list(set(deps))

    def __str__(self):
        return "|".join(self.deps)

class LabelType(Type):
    def __str__(self):
        return "label"

# Hmm...need to think about this a little.
class Threshold(Type):
    pass

class PythonBasicExpressionType(Type):
    # Encapsulates all the Python types we don't care about, e.g.,
    # booleans, integers, lists, classes, and so forth....
    pass
