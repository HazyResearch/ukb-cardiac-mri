
import ast
import meta

import coral_ast as cast
import coral_types as ct

# Public API

# Turn this on for printing.
verbose = False

def vprint(a):
    global verbose
    if verbose:
        print a


def convert_to_coral_ast(func):
    """
    Converts a Python function to a Coral AST.
    """

    # A Python AST.
    tree = meta.decompiler.decompile_func(func)
    
    vprint(ast.dump(tree))
    coral_tree = _generate_coral_ast(tree)

    vprint(coral_tree)
    return coral_tree

# Ghetto a.f.
vocabs = []
def extract_vocabularies(coral_tree):
    global vocabs
    assert isinstance(coral_tree, cast.LabelingFunction)
    vocabs = [str(v) for v in coral_tree.vocabs]
    def _extract_vocabs(node):
        global vocabs
        if isinstance(node.ty, ct.VocabType):
            vocabs.append(str(node.ty))

    coral_tree.walk(_extract_vocabs)
    # Uniquify
    return list(set(list(vocabs)))

### Private Stuff.

def _generate_coral_ast(node, names={}):
    """
    Generates a Coral AST given a Python AST.
    """
    if isinstance(node, ast.FunctionDef):
        args = [name.id for name in node.args.args]
        for arg in args:
            names[arg] = cast.VocabLiteral(arg)
        body = [_generate_coral_ast(b, names) for b in node.body]
        expr = cast.LabelingFunction(body, args)
        return expr
    if isinstance(node, ast.Return):
        return cast.Return(_generate_coral_ast(node.value, names))
    if isinstance(node, ast.If):
        cond = _generate_coral_ast(node.test, names)
        true_branch = _generate_coral_ast(node.body[0], names)
        expr = cast.IfThen(cond, true_branch)
        vprint(expr)
        return expr
    if isinstance(node, ast.Compare):
        left = _generate_coral_ast(node.left, names)
        right = _generate_coral_ast(node.comparators[0], names)
        op = node.ops[0]
        if isinstance(op, ast.Eq):
            expr = cast.Equal(left, right)
            vprint(expr)
            return expr
        elif isinstance(op, ast.Gt):
            expr = cast.GreaterThan(left, right)
            vprint(expr)
            return expr
        elif isinstance(op, ast.Lt):
            expr = cast.LessThan(left, right)
            vprint(expr)
            return expr
        elif isinstance(op, ast.LtE):
            expr = cast.LessThanOrEqual(left, right)
            vprint(expr)
            return expr
        elif isinstance(op, ast.GtE):
            expr = cast.GreaterThanOrEqual(left, right)
            vprint(expr)
            return expr
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            expr = cast.Add(_generate_coral_ast(node.left, names),  _generate_coral_ast(node.right,
                names))
        elif isinstance(node.op, ast.Mult):
            expr = cast.Multiply(_generate_coral_ast(node.left, names),
                    _generate_coral_ast(node.right, names))
        if isinstance(node.op, ast.Sub):
            expr = cast.Subtract(_generate_coral_ast(node.left, names),
                    _generate_coral_ast(node.right, names))
        vprint(expr)
        return expr
    if isinstance(node, ast.Name):
        if node.id == "True":
            expr = cast.TrueLabelLiteral()
        elif node.id == "False":
            expr = cast.FalseLabelLiteral()
        elif node.id == "None":
            expr = cast.AbstainLabelLiteral()
        else:
            expr = names[node.id]
        vprint(expr)
        return expr
    if isinstance(node, ast.Num):
        return cast.PythonLiteral(node.n)

