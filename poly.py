from pyparsing import (
    Word,
    nums,
    alphas,
    Optional,
    Combine,
    oneOf,
    ZeroOrMore,
    OneOrMore,
    Group,
    Literal,
    CharsNotIn
)
import re
from fractions import Fraction as Q
import unittest


class PolyParser:
    def __init__(self, field):
        self.field = field
        integer = Word(nums)
        real = Combine(integer + Optional('.' + integer))
        variable = Combine(Word(alphas, exact=1) + Optional(integer))
        sign = oneOf('+ -')
        power = Group(variable) + Optional(Group('^' + integer))
        monomial = ZeroOrMore(Group(power))
        coefficient = real | Literal('(') + CharsNotIn('()') + ')'
        term = Group(Optional(coefficient)) + Group(monomial)
        self.polynomial = Group(Optional(sign) + term) + \
            ZeroOrMore(Group(sign + term))

    def parse(self, s):
        term_nodes = self.polynomial.parseString(s, parseAll=True)
        terms = [self.parse_term(term_node) for term_node in term_nodes]
        return terms

    def parse_term(self, term_node):
        coefficient = None
        monomial = None

        # no sign, possible in first term
        if len(term_node) == 2:
            coefficient = self.parse_coefficient(term_node[0])
            monomial = self.parse_monomial(term_node[1])
        elif len(term_node) == 3:
            sign = term_node[0]

            if sign == '+':
                coefficient = self.parse_coefficient(term_node[1])
            elif sign == '-':
                coefficient = -self.parse_coefficient(term_node[1])

            monomial = self.parse_monomial(term_node[2])

        return (coefficient, monomial)

    def parse_coefficient(self, coefficient_node):
        if len(coefficient_node) == 1:
            return self.field.parse(coefficient_node[0])
        elif len(coefficient_node) == 3:
            return self.field.parse(coefficient_node[1])
        elif len(coefficient_node) == 0:
            return self.field.unit

    def parse_monomial(self, monomial_node):
        powers = [self.parse_power(power_node) for power_node in monomial_node]
        return powers

    def parse_power(self, power_node):
        variable = None
        exponent = None

        if len(power_node) == 1:
            variable = power_node[0][0]
            exponent = 1
        if len(power_node) == 2:
            variable = power_node[0][0]
            exponent = int(power_node[1][1])

        return (variable, exponent)


class MonomialOrder(list):
    def __add__(self, other):
        if not isinstance(other, MonomialOrder):
            raise TypeError(
                'cannot add ' + type(other).__name__ + ' to monomial order')
        if type(self) != type(other):
            raise TypeError('cannot add monomial orders ' +
                            type(self).__name__ + ' and ' + type(other).__name__)
        if len(self) != len(other):
            raise Exception('monomial orders ' + str(self) +
                            ' and ' + str(other) + ' differ in length')

        return type(self)([a + b for a, b in zip(self, other)])

    def __hash__(self):
        return hash(str(self))


class LexOrder(MonomialOrder):
    def __lt__(self, other):
        for a, b in zip(self, other):
            if a < b:
                return True
            elif a > b:
                return False
        return False


class GrlexOrder(MonomialOrder):
    def __lt__(self, other):
        if sum(self) < sum(other):
            return True
        elif sum(self) > sum(other):
            return False

        for a, b in zip(self, other):
            if a < b:
                return True
            elif a > b:
                return False

        return False


class GrevlexOrder(MonomialOrder):
    def __lt__(self, other):
        if sum(self) < sum(other):
            return True
        elif sum(self) > sum(other):
            return False

        for a, b in list(zip(self, other))[::-1]:
            if a > b:
                return True
            elif a < b:
                return False

        return False


class SortedList(list):
    def __init__(self, items, key=lambda x: x, combine=lambda x, y: x, already_sorted=False):
        self.key = key
        self.combine = combine

        if already_sorted:
            self += items
        else:
            for item in items:
                self.insert(item)

    def remove_none(self):
        i = 0
        while i < len(self):
            if self[i] is None:
                del self[i]
            else:
                i += 1

    def insert(self, item):
        k = self.key(item)
        for i in range(0, len(self) + 1):
            if i == len(self) or k < self.key(self[i]):
                super().insert(i, item)
                break
            elif k == self.key(self[i]):
                self[i] = self.combine(self[i], item)
                break

        self.remove_none()

    def merge(self, items):
        l = []
        i = 0
        j = 0

        while i < len(self) and j < len(items):
            x = self[i]
            y = items[j]
            kx = self.key(x)
            ky = self.key(y)

            if kx == ky:
                l.append(self.combine(x, y))
                i += 1
                j += 1
            elif kx < ky:
                l.append(x)
                i += 1
            else:
                l.append(y)
                j += 1

        if i < len(self):
            l += self[i:]
        elif j < len(items):
            l += items[j:]

        sl = SortedList(l, key=self.key, combine=self.combine,
                        already_sorted=True)
        sl.remove_none()
        return sl


class Poly:
    def __init__(self, terms, poly_ring):
        def combine(term1, term2):
            coefficient = term1[0] + term2[0]
            if coefficient == 0:
                return None
            else:
                monomial = term1[1]
                term = (coefficient, monomial)
                return term

        self.terms = SortedList(terms, key=lambda t: t[1], combine=combine)
        self.poly_ring = poly_ring

    def __eq__(self, other):
        if not isinstance(other, Poly):
            return False

        return self.poly_ring == other.poly_ring and self.terms == other.terms

    def to_str(self, latex=False):
        terms_str = ''

        if self.terms == []:
            return '0'

        for index, (coefficient, exponents) in enumerate(self.terms[::-1]):
            monomial_str = ''
            all_zero = True

            for variable, power in zip(self.poly_ring.variables, exponents):
                if latex and len(variable) > 1:
                    variable = variable[0] + '_' + variable[1:]

                if power == 0:
                    continue
                elif power == 1:
                    monomial_str += variable
                    all_zero = False
                else:
                    if latex:
                        monomial_str += variable + '^{' + str(power) + '}'
                    else:
                        monomial_str += variable + '^' + str(power)

                    all_zero = False

            coefficient_str, needs_parens = self.poly_ring.field.to_str(
                coefficient, latex)

            sign = None

            if coefficient == self.poly_ring.field.unit and not all_zero:
                coefficient_str = ''
                sign = '+'
            elif coefficient == -self.poly_ring.field.unit and not all_zero:
                coefficient_str = ''
                sign = '-'
            elif needs_parens:
                coefficient_str = '(' + coefficient_str + ')'
                sign = '+'
            else:
                sign = self.poly_ring.field.sign(coefficient)

                if sign is None:
                    sign = '+'

            if index == 0:
                if sign == '+':
                    terms_str += coefficient_str + monomial_str
                elif sign == '-' and coefficient_str != '':
                    coefficient_str, _ = self.poly_ring.field.to_str(
                        -coefficient, latex)
                    terms_str += sign + coefficient_str + monomial_str
                else:
                    terms_str += sign + coefficient_str + monomial_str
            else:
                if sign == '-' and coefficient_str != '':
                    coefficient_str, _ = self.poly_ring.field.to_str(
                        -coefficient, latex)
                terms_str += ' ' + sign + ' ' + coefficient_str + monomial_str

        if latex:
            terms_str = '$' + terms_str + '$'

        return terms_str

    __str__ = to_str
    __repr__ = to_str

    def _repr_latex_(self):
        return self.to_str(latex=True)

    def __add__(self, other):
        if not isinstance(other, Poly):
            raise TypeError(
                'cannot add ' + str(type(other).__name__) + ' to polynomial')
        if self.poly_ring != other.poly_ring:
            raise Exception('cannot add polynomial in ' + str(self.poly_ring) +
                            ' to polynomial in ' + str(other.poly_ring))

        terms = self.terms.merge(other.terms)
        return Poly(terms, self.poly_ring)

    def __sub__(self, other):
        return self + self.poly_ring('-1') * other

    def __mul__(self, other):
        if not isinstance(other, Poly):
            raise TypeError(
                'cannot add ' + str(type(other).__name__) + ' to polynomial')
        if self.poly_ring != other.poly_ring:
            raise Exception('cannot add polynomial in ' + str(self.poly_ring) +
                            ' to polynomial in ' + str(other.poly_ring))

        terms = []

        for t1 in self.terms:
            for t2 in other.terms:
                coefficient = t1[0] * t2[0]
                monomial = t1[1] + t2[1]
                term = (coefficient, monomial)
                terms.append(term)

        return Poly(terms, self.poly_ring)

    def lt(self):
        term = self.terms[-1]
        return Term(term, self.poly_ring)

    def lc(self):
        return self.terms[-1][0]

    def lm(self):
        return self.terms[-1][1]
    
    def get_terms(self):
        return [Term(term, self.poly_ring) for term in self.terms]

    def __truediv__(self, other):
        f = self
        gs = None
        single = False

        if isinstance(other, Poly):
            gs = [other]
            single = True
        elif isinstance(other, list):
            gs = other

            for g in gs:
                if not isinstance(g, Poly):
                    raise TypeError('cannot divide polynomial by ' +
                                    type(other).__name__)
        else:
            raise TypeError('cannot divide polynomial by ' +
                            type(other).__name__)

        for g in gs:
            if self.poly_ring != g.poly_ring:
                raise Exception('cannot divide polynomial in ' +
                                str(self.poly_ring) + ' by polynomial in ' + str(other.poly_ring))

        qs = [self.poly_ring('0')] * len(gs)
        p = f
        r = self.poly_ring('0')

        while p != self.poly_ring('0'):
            any_divide = False
            for i in range(0, len(gs)):
                g = gs[i]
                if g.lt().divides(p.lt()):
                    qs[i] += (p.lt() / g.lt())
                    p -= (p.lt() / g.lt()) * g
                    any_divide = True
            if not any_divide:
                r += p.lt()
                p -= p.lt()

        if single:
            return qs[0], r
        else:
            return qs, r


class Term(Poly):
    def __init__(self, term, poly_ring):
        super().__init__([term], poly_ring)

    def coefficient(self):
        return self.terms[0][0]

    def monomial(self):
        return self.terms[0][1]

    def divides(self, other):
        if not isinstance(other, Term):
            raise TypeError('cannot call divides with ' + type(other).__name__)

        if self.poly_ring != other.poly_ring:
            raise Exception('cannot call divides with term in ' +
                            str(self.poly_ring) + ' and term in ' + str(other.poly_ring))

        for i, j in zip(self.monomial(), other.monomial()):
            if i > j:
                return False

        return True

    def __truediv__(self, other):
        if not isinstance(other, Term):
            raise TypeError('cannot divide polynomial by ' +
                            type(other).__name__)

        if self.poly_ring != other.poly_ring:
            raise Exception('cannot divide term in ' +
                            str(self.poly_ring) + ' by term in ' + str(other.poly_ring))

        coefficient = self.coefficient() / other.coefficient()
        monomial = self.poly_ring.monomial_order(
            [a - b for a, b in zip(self.monomial(), other.monomial())])
        term = (coefficient, monomial)
        return Term(term, self.poly_ring)


class Field:
    zero = None
    unit = None

    @ staticmethod
    def parse(element_str):
        pass

    @ staticmethod
    def sign(element):
        return None

    @ staticmethod
    def to_str(element, latex=False):
        pass


class RationalField(Field):
    zero = Q(0)
    unit = Q(1)
    _integer = Word(nums)
    _sign = Literal('-') | '+'
    _rational = Combine(Optional(_sign) + _integer) + Optional('/' + _integer)

    @ staticmethod
    def parse(element_str):
        node = RationalField._rational.parseString(element_str, parseAll=True)

        if len(node) == 1:
            return Q(int(node[0]))
        elif len(node) == 3:
            return Q(int(node[0]), int(node[2]))

    @ staticmethod
    def sign(element):
        if element > 0:
            return '+'
        elif element < 0:
            return '-'

    @ staticmethod
    def to_str(element, latex=False):
        if element.denominator == 1:
            return str(element.numerator), False
        else:
            if latex:
                return r'\frac{' + str(element.numerator) + r'}{' + str(element.denominator) + '}', False
            else:
                return str(element.numerator) + ' / ' + str(element.denominator), True

    @ staticmethod
    def field_to_str(latex=False):
        if latex:
            return r'\mathbb{Q}'
        else:
            return 'Q'


class RealField(Field):
    zero = 0
    unit = 1

    @ staticmethod
    def parse(element_str):
        return float(element_str)

    @ staticmethod
    def sign(element):
        if element > 0:
            return '+'
        elif element < 0:
            return '-'

    @ staticmethod
    def to_str(element, latex=False):
        if element.is_integer():
            return str(int(element)), False
        else:
            return str(element), False

    @ staticmethod
    def field_to_str(latex=False):
        if latex:
            return r'\mathbb{R}'
        else:
            return 'R'


class ComplexField(Field):
    zero = 0
    unit = 1
    _integer = Word(nums)
    _real = Combine(_integer + Optional('.' + _integer))
    _sign = Literal('-') | '+'
    _complex = Combine(Optional(_sign) + _real) + Literal('i') | Combine(Optional(
        _sign) + _real) + Optional(Combine(_sign + _real, adjacent=False) + Literal('i'))

    @ staticmethod
    def parse(element_str):
        node = ComplexField._complex.parseString(element_str, parseAll=True)
        if len(node) == 1:
            return complex(float(node[0]))
        elif len(node) == 2:
            return complex(0, float(node[0]))
        elif len(node) == 3:
            return complex(float(node[0]), float(node[1]))

    @ staticmethod
    def to_str(element, latex=False):
        real_str = None

        if element.real.is_integer():
            real_str = str(int(element.real))
        else:
            real_str = str(element.real)

        imag_str = None
        imag = element.imag

        sign = ' + '

        if imag == 0:
            return real_str, False
        elif imag < 0:
            imag = -imag
            sign = ' - '

        if imag.is_integer():
            imag_str = str(int(imag))
        else:
            imag_str = str(imag)

        return real_str + sign + imag_str + 'i', True

    @ staticmethod
    def field_to_str(latex=False):
        if latex:
            return r'\mathbb{C}'
        else:
            return 'C'


class PolyRing:
    def __init__(self, variables, field=RealField, monomial_order=LexOrder):
        self.poly_parser = PolyParser(field)
        self.field = field
        self.variables = variables
        self.monomial_order = monomial_order

    def __call__(self, terms_str):
        terms_node = self.poly_parser.parse(terms_str)
        terms = []

        for coefficient, monomial_node in terms_node:
            if coefficient == self.field.zero:
                continue

            monomial = [0] * len(self.variables)

            for variable, exponent in monomial_node:
                variable_index = self.variables.index(variable)
                monomial[variable_index] = exponent

            monomial = self.monomial_order(monomial)
            term = (coefficient, monomial)
            terms.append(term)

        return Poly(terms, self)

    def to_str(self, latex=False):
        if latex:
            return '$' + self.field.field_to_str(latex) + '[' + ', '.join(self.variables) + ']$'
        else:
            return self.field.field_to_str(latex) + '[' + ', '.join(self.variables) + ']'

    __str__ = to_str
    __repr__ = to_str

    def _repr_latex_(self):
        return self.to_str(latex=True)


class TestPoly(unittest.TestCase):
    def test_parse(self):
        R = PolyRing(['x', 'y', 'z'], field=RationalField)
        self.assertEqual(
            set(R('xy-(-1)xz+z^22y^241+(-2/11)z-(3/11)x').terms),
            set([
                (Q(1), LexOrder([1, 1, 0])),
                (Q(1), LexOrder([1, 0, 1])),
                (Q(1), LexOrder([0, 241, 22])),
                (Q(-2, 11), LexOrder([0, 0, 1])),
                (Q(-3, 11), LexOrder([1, 0, 0]))
            ])
        )

        R = PolyRing(['x', 'y', 'z'], field=ComplexField)
        self.assertEqual(
            set(R('xy-(-1.2i)xz+0.3z^22y^241+(-2 + 11i)z-(3 -11i)x+3xy').terms),
            set([
                (4, LexOrder([1, 1, 0])),
                (1.2j, LexOrder([1, 0, 1])),
                (0.3, LexOrder([0, 241, 22])),
                (-2 + 11j, LexOrder([0, 0, 1])),
                (-3 + 11j, LexOrder([1, 0, 0]))
            ])
        )

    def test_add(self):
        R = PolyRing(['x', 'y', 'z'], field=RationalField)
        p = R('2x+x^2-x^3y')
        q = R('x-x^2+27+2yx^3')
        r = R('3x+27+x^3y')
        self.assertEqual(p + q, r)

    def test_mul(self):
        R = PolyRing(['x', 'y', 'z'], field=RationalField)
        p = R('2x+x^2-x^3y')
        q = R('x-x^2+27+2yx^3')
        r = R('54 x + 29 x^2 - x^3 - x^4 - 27 x^3 y + 3 x^4 y + 3 x^5 y - 2 x^6 y^2')
        self.assertEqual(p * q, r)
        self.assertNotEqual(p * q, r + R('1'))


if __name__ == '__main__':
    unittest.main()
