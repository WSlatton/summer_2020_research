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

class PolyParser:
    def __init__(self, field_parser = float, variables = None):
        self.field_parser = field_parser
        integer = Word(nums)
        real = Combine(integer + Optional('.' + integer))
        variable = Combine(Word(alphas, exact=1) + Optional(integer))
        sign = oneOf('+ -')
        power = Group(variable) + Optional(Group('^' + integer))
        monomial = ZeroOrMore(Group(power))
        coefficient = real | Literal('(') + CharsNotIn('()') + ')'
        term = Group(coefficient) + Group(monomial)
        self.polynomial = Group(Optional(sign) + term) + ZeroOrMore(Group(sign + term))
    
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