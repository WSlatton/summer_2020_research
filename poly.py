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


class LexOrder(list):
    def __lt__(self, other):
        for a, b in zip(self, other):
            if a < b:
                return True
            elif a > b:
                return False
        return False


class Poly:
    def __init__(self, terms, poly_ring):
        self.terms = terms
        self.poly_ring = poly_ring

    def to_str(self, latex=False):
        terms_str = ''

        for index, (coefficient, exponents) in enumerate(self.terms):
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
                    monomial_str += variable + '^' + str(power)
                    all_zero = False

            if coefficient == self.poly_ring.field.unit and not all_zero:
                coefficient_str = ''
                needs_parens = False
            else:
                coefficient_str, needs_parens = self.poly_ring.field.to_str(
                    coefficient, latex)

            sign = None

            if coefficient_str == '':
                sign = '+'
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
                elif sign == '-':
                    coefficient_str, _ = self.poly_ring.field.to_str(
                        -coefficient, latex)
                    terms_str += sign + coefficient_str + monomial_str
            else:
                if sign == '-':
                    coefficient_str, _ = self.poly_ring.field.to_str(
                        -coefficient, latex)
                terms_str += ' ' + sign + ' ' + coefficient_str + monomial_str

        if latex:
            terms_str = '$' + terms_str + '$'

        return terms_str

    __str__ = to_str

    __repr__ = __str__

    def _repr_latex_(self):
        return self.to_str(latex=True)


class Field:
    zero = None
    unit = None

    @staticmethod
    def parse(element_str):
        pass

    @staticmethod
    def sign(element):
        return None

    @staticmethod
    def to_str(element, latex=False):
        pass


class RationalField(Field):
    zero = Q(0)
    unit = Q(1)
    _integer = Word(nums)
    _sign = Literal('-') | '+'
    _rational = Combine(Optional(_sign) + _integer) + Optional('/' + _integer)

    @staticmethod
    def parse(element_str):
        node = RationalField._rational.parseString(element_str, parseAll=True)

        if len(node) == 1:
            return Q(int(node[0]))
        elif len(node) == 3:
            return Q(int(node[0]), int(node[2]))

    @staticmethod
    def sign(element):
        if element > 0:
            return '+'
        elif element < 0:
            return '-'

    @staticmethod
    def to_str(element, latex=False):
        if element.denominator == 1:
            return str(element.numerator), False
        else:
            if latex:
                return r'\frac{' + str(element.numerator) + r'}{' + str(element.denominator) + '}', False
            else:
                return str(element.numerator) + ' / ' + str(element.denominator), True


class RealField(Field):
    zero = 0
    unit = 1

    @staticmethod
    def parse(element_str):
        return float(element_str)

    @staticmethod
    def sign(element):
        if element > 0:
            return '+'
        elif element < 0:
            return '-'

    @staticmethod
    def to_str(element, latex=False):
        if element.is_integer():
            return str(int(element)), False
        else:
            return str(element), False


class ComplexField(Field):
    zero = 0
    unit = 1
    _integer = Word(nums)
    _real = Combine(_integer + Optional('.' + _integer))
    _sign = Literal('-') | '+'
    _complex = Combine(Optional(_sign) + _real) + \
        Optional(Combine(_sign + _real, adjacent=False) + Literal('i'))

    @staticmethod
    def parse(element_str):
        node = ComplexField._complex.parseString(element_str, parseAll=True)
        if len(node) == 1:
            return complex(float(node[0]))
        elif len(node) == 3:
            return complex(float(node[0]), float(node[1]))

    @staticmethod
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

            term = (coefficient, monomial)
            terms.append(term)

        return Poly(terms, self)
