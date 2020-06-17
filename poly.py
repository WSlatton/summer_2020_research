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


class PolyParser:
    def __init__(self, field_parser, field_unit):
        self.field_parser = field_parser
        self.field_unit = field_unit
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
        term_nodes = self.polynomial.parseString(s)
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
            return self.field_parser(coefficient_node[0])
        elif len(coefficient_node) == 3:
            return self.field_parser(coefficient_node[1])
        elif len(coefficient_node) == 0:
            return self.field_unit

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

    def __str__(self):
        terms_str = ''

        for index, (coefficient, exponents) in enumerate(self.terms):
            monomial_str = ''
            all_zero = True

            for variable, power in zip(self.poly_ring.variables, exponents):
                if power == 0:
                    continue
                elif power == 1:
                    monomial_str += variable
                    all_zero = False
                else:
                    monomial_str += variable + '^' + str(power)
                    all_zero = False

            if coefficient == self.poly_ring.field_unit and not all_zero:
                coefficient_str = ''
            else:
                coefficient_str = str(coefficient)

            real_match = re.fullmatch(r'(\+|-)?\d+(\.\d*)?', coefficient_str)
            sign = None

            if coefficient_str == '':
                sign = '+'
            elif real_match is None:
                if coefficient_str[0] != '(' or coefficient_str[-1] != ')':
                    coefficient_str = '(' + coefficient_str + ')'
                sign = '+'
            else:
                if real_match.group(1) is None:
                    sign = '+'
                else:
                    sign = real_match.group(1)
                    coefficient_str = coefficient_str[1:]

            if index == 0:
                if sign == '+':
                    terms_str += coefficient_str + monomial_str
                else:
                    terms_str += sign + coefficient_str + monomial_str
            else:
                terms_str += ' ' + sign + ' ' + coefficient_str + monomial_str

        return terms_str

    __repr__ = __str__


class PolyRing:
    def __init__(self, variables, field_parser=float, field_unit=1, monomial_order=LexOrder):
        self.poly_parser = PolyParser(field_parser, field_unit)
        self.field_unit = field_unit
        self.variables = variables
        self.monomial_order = monomial_order

    def __call__(self, terms_str):
        terms_node = self.poly_parser.parse(terms_str)
        terms = []

        for coefficient, monomial_node in terms_node:
            monomial = [0] * len(self.variables)

            for variable, exponent in monomial_node:
                variable_index = self.variables.index(variable)
                monomial[variable_index] = exponent

            if coefficient == '':
                coefficient = self.field_unit

            term = (coefficient, monomial)
            terms.append(term)

        return Poly(terms, self)
