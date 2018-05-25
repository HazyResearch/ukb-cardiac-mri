
import coral_sa as csa

def is_square(width, height):
    if width == height:
        return True
    else:
        return False

def is_large(width, height):
    if width * height > 100:
        return True
    else:
        return False

def is_small(width, height):
    if width * height < 10:
        return True
    # Just for testing.
    elif width * height <= 30:
        return True
    elif width * height + height * height >= 30:
        return True
    else:
        return False

tests = [is_square, is_large, is_small]

coral_ast = csa.convert_to_coral_ast(is_small)
vocabs_used = csa.extract_vocabularies(coral_ast)

print "AST:", coral_ast
print "Vocabs used:", vocabs_used


