
import coral_types as ct

# A basic node in an AST.
# This is meant to be subclassed.
class Expression(object):
    def __repr__(self):
        return str(self)

    def children(self):
        return NotImplementedError

    def walk(self, f):
        """
        Walks the AST, applying the function f to each node in pre-order.
        """
        for child in self.children():
            child.walk(f)
        f(self)


    def __str__(self):
        return "???"


class LabelingFunction(Expression):
    def __init__(self, body, vocabs):
        # A list of subexpressions.
        self.body = body
        # A list of vocabularies, identified via the vocab name.
        # Each latent vocabulary is derived from these vocabularies.
        self.vocabs = vocabs
        self.ty = ct.LabelingFunctionType

    def children(self):
        return self.body

    def __str__(self):
        if isinstance(self.body, list):
            return str(self.body)
        else:
            return str(self.body)

## Literals

class VocabLiteral(Expression):
    def __init__(self, name):
        self.name = name
        # The type is just the name.
        self.ty = ct.VocabType([self.name])

    def children(self):
        return []

    def __str__(self):
        return self.name

class TrueLabelLiteral(Expression):
    def __init__(self):
        self.ty = ct.LabelType()

    def children(self):
        return []

    def __str__(self):
        return "true"

class FalseLabelLiteral(Expression):
    def __init__(self):
        self.ty = ct.LabelType()

    def children(self):
        return []

    def __str__(self):
        return "false"


class AbstainLabelLiteral(Expression):
    def __init__(self):
        self.ty = ct.LabelType()

    def children(self):
        return []

    def __str__(self):
        return "abstain"

class PythonLiteral(Expression):
    """
    A catch-all for sub-expressions we don't care about.
    """
    def __init__(self, value):
        self.value = value
        self.ty = ct.PythonBasicExpressionType()

    def children(self):
        return []

    def __str__(self):
        return str(self.value)

## Etc Expressions important to Coral.

class IfThen(Expression):
    def __init__(self, cond, true_branch):
        self.cond = cond
        self.true_branch = true_branch
        self.ty = ct.PythonBasicExpressionType()

    def children(self):
        return [self.cond, self.true_branch]

    def __str__(self):
        return "if ({}) then ({})".format(self.cond, self.true_branch)

class Return(Expression):
    def __init__(self, value):
        self.value = value
        # Has no sensible return type.
        self.ty = ct.PythonBasicExpressionType()

    def children(self):
        return [self.value]

    def __str__(self):
        return "return {}".format(self.value)

## Basic Binary Operations

class BinOp(Expression):
    def __init__(self, left, right):
        self.left = left
        self.right = right

        left_isvocab = isinstance(self.left.ty, ct.VocabType)
        right_isvocab = isinstance(self.right.ty, ct.VocabType)
        if  left_isvocab and right_isvocab:
            # This is a new latent vocabulary composed of existing vocabularies.
            self.ty = ct.VocabType(self.left.ty.deps + self.right.ty.deps)
        elif left_isvocab:
            self.ty = ct.VocabType(self.left.ty.deps)
        elif rght_isvocab:
            self.ty = ct.VocabType(self.right.ty.deps)
        else:
            self.ty = ct.PythonBasicExpressionType()

    def children(self):
        return [self.left, self.right]

    def printer(self, operator, prefix=False):
        if prefix:
            return "{}({}, {})".format(operator, self.left, self.right)
        else:
            return "({} {} {})".format(self.left, operator, self.right)

class Add(BinOp):
    def __str__(self):
        return self.printer("+")

class Subtract(BinOp):
    def __str__(self):
        return self.printer("-")

class Multiply(BinOp):
    def __str__(self):
        return self.printer("*")

class Divide(BinOp):
    def __str__(self):
        return self.printer("/")

class Equal(BinOp):
    def __str__(self):
        return self.printer("==")

class GreaterThan(BinOp):
    def __str__(self):
        return self.printer(">")

class GreaterThanOrEqual(BinOp):
    def __str__(self):
        return self.printer(">=")

class LessThan(BinOp):
    def __str__(self):
        return self.printer("<")

class LessThanOrEqual(BinOp):
    def __str__(self):
        return self.printer("<=")

## Meta doesn't handle dictionaries, but they're probably important for labeling functions...
# TODO
