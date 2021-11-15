import re
import sre_parse
import sre_constants
import sre_compile
from collections import namedtuple
from typing import List, Union, Text, Type, Optional

pat = r"^o[^a-z][a-zA-Z]n[^ABC]p*p.*t[^a]i.{1,3}o[0-9]{0, 10}n.+?al/(?P<arg1>\d+)/✂️/(?:(?P<arg2>[a-b])/)?/([0-9]+?)/(?P<arg3>\\d+)/(?:(?P<arg4>[c-d])/)?$"

pat = r"^option[^a-zA-Z].*LOL.+test.{1,4}[a-f0-9]{2,4} - (?P<arg1>\d+) (\d+)$"
pat = r"^test .* .+ .{1,4} [a-f0-9]{2,4} (?P<arg1>\d+) (\d+)$"
pat = r"^(?P<arg1>\d){1,4}(?:(?P<arg2>\d))(?P<arg3>\d)$"
# print("*" * 32)
# pat = r'^[^@]+@[^@]+\.[^@]+$'
# print(pat)


tree = sre_parse.parse(pat)
# tree.dump()

groupdict = {}
for k, v in tree.state.groupdict.items():
    groupdict[v] = k


Position = namedtuple("Position", ("start", "by", "end"))


class Token:
    __slots__ = ("start_position",)

    def __new__(cls, position):
        instance = super().__new__(cls)
        instance.start_position = position
        return instance

    def __str__(self):
        return "..."

    def __repr__(self):
        value = str(self)
        return f"<{self.__class__.__name__!s} {value!r} ({self.position!s})>"

    @property
    def position(self):
        start = self.start_position
        advance_by = len(str(self))
        end = start + advance_by
        return Position(start=start, by=advance_by, end=end)

    def describe(self):
        return repr(self)

    def simplify(self):
        return "..."

    def generate(self):
        return "generate"


class Literal(Token):
    __slots__ = ("value", "sre_type", "start_position")

    def __new__(cls, value, position):
        instance = super().__new__(cls, position)
        instance.value = value
        instance.sre_type = sre_constants.LITERAL
        return instance

    def __str__(self):
        if self.value in sre_parse.SPECIAL_CHARS:
            return f"\\{self.value}"
        return self.value

    def __len__(self):
        return self.value

    def describe(self):
        return f"character {self.value!r}"

    simplify = __str__
    generate = __str__


class Beginning(Token):
    __slots__ = ("value", "sre_type", "start_position")

    def __new__(cls, position):
        instance = super().__new__(cls, position)
        instance.value = '^'
        instance.sre_type = sre_constants.AT_BEGINNING
        return instance

    def __str__(self):
        return self.value

    def describe(self):
        return f"anchor to beginning"


class End(Token):
    __slots__ = ("value", "sre_type", "start_position")

    def __new__(cls, position):
        instance = super().__new__(cls, position)
        instance.value = '$'
        instance.sre_type = sre_constants.AT_END
        return instance

    def __str__(self):
        return self.value

    def describe(self):
        return f"anchor to end"


class Anything(Token):
    __slots__ = ("value", "sre_type", "start_position")

    def __new__(cls, position):
        instance = super().__new__(cls, position)
        instance.value = '.'
        instance.sre_type = sre_constants.ANY
        return instance

    def __str__(self):
        return self.value

    def describe(self):
        return "anything"


class NegatedLiteral(Literal):
    __slots__ = ()

    def __new__(cls, value, position):
        instance = super().__new__(cls, value, position)
        return instance

    def __str__(self):
        return f"[^{self.value}]"

    def describe(self):
        return f"anything other than {self.value!r}"


class Range(Token):
    __slots__ = ("start", "end")

    def __new__(cls, start, end, position):
        instance = super().__new__(cls, position)
        instance.start = start
        instance.end = end
        return instance

    def __str__(self):
        start = chr(self.start)
        end = chr(self.end)
        return f"{start}-{end}"

    def describe(self):
        if (self.end - self.start) > 10:
            first = chr(self.start)
            last = chr(self.end)
            chars = ", ".join(chr(i) for i in range(self.start + 1, self.start + 6, 1))
            return f"any of {first!r}, ... {chars!s}, ... {last!r}"
        chars = "', '".join(chr(i) for i in range(self.start, self.end, 1))
        return f"any of '{chars!s}'"

    simplify = __str__


class Repeat(Token):
    __slots__ = ("min", "max", "value")

    def __new__(cls, min, max, value, position):
        instance = super().__new__(cls, position)
        instance.min = min
        instance.max = max
        instance.value = value
        return instance

    def __str__(self):
        value = "".join(str(v) for v in self.value)
        minmax = f"{{{self.min},{self.max}}}"
        if self.min == self.max:
            minmax = f"{{{self.min}}}"
        # Actually using MAXREPEAT as a number, in {0,4294967295} results in an
        # OverflowError, so consider the one lower than it as maximum, too
        if self.max in (int(sre_constants.MAXREPEAT), int(sre_constants.MAXREPEAT)-1):
            if self.min == 0:
                minmax = "*"
            elif self.min == 1:
                minmax = "+"
            else:
                minmax = f"{{{self.min},}}"
        return f"{value}{minmax}"

    def describe(self):
        values = " ".join(v.describe() for v in self.value)
        if values == ".":
            values = "anything"
        if self.max == sre_constants.MAXREPEAT:
            if self.min == 0:
                template = "{value} (optional, any number of times)"
            else:
                template = "{value} ({min} or more times)"
        else:
            template = "{value} (between {min} and {max} times)"
        return template.format(value=values, min=self.min, max=self.max)

    # def __repr__(self):
    #     self.
    #     return f'<{self.__class__.__name__!s} {value!r}>'


class Category(Token):
    __slots__ = ("value",)

    def __new__(cls, value, position):
        instance = super().__new__(cls, position)
        instance.value = value
        return value

    def __str__(self):
        return "test"


class In(Token):
    __slots__ = ("value",)

    def __new__(cls, value, position):
        instance = super().__new__(cls, position)
        instance.value = value
        return instance

    def __str__(self):
        value = "".join(str(sub) for sub in self.value)
        return f"[{value!s}]"

    def describe(self):
        parts = []
        for part in self.value:
            parts.append(part.describe())
        return " or ".join(parts)


class NegatedIn(In):
    def __str__(self):
        value = "".join(str(sub) for sub in self.value)
        return f"[^{value!s}]"


class SubPattern(Token):
    __slots__ = ("name", "number", "value")

    def __new__(cls, name, num, value, position):
        instance = super().__new__(cls, position)
        instance.name = name
        instance.number = num
        instance.value = value
        return instance

    def __str__(self):
        if self.name:
            return f"(?P<{self.name!s}>{self.value!s})"
        return f"({self.value!s})"

    def describe(self):
        if self.name:
            return f"group (named {self.name!r}) capturing: {self.value!s}"
        else:
            return f"group (number {self.number!s}) capturing {self.value!s}"


# Literal = namedtuple("Literal", ("value", "position"))
# NegatedLiteral = namedtuple("NegatedLiteral", ("value", "position"))
LiteralGroup = namedtuple("LiteralGroup", ("value", "start", "end"))
Negated = namedtuple("Negated", ("value"))
# Range = namedtuple("Range", ("start", "end", "position"))
NegatedRange = namedtuple("Range", ("start", "end", "position"))
# Beginning = namedtuple("Beginning", ('value', 'position'))
# End = namedtuple("End", ('value', 'position'))
# Repeat = namedtuple("Repeat", ("start", "end", "value", "position"))
# In = namedtuple("In", ("value", "position"))
# MaxRepeat = namedtuple("MaxRepeat", ("start", "end", "value", "position"))


class Reparser:
    __slots__ = (
        "pattern",
        "named_groups_by_name",
        "named_groups_by_number",
        "positions",
        "seen_tokens",
    )

    pattern: sre_parse.SubPattern
    seen_tokens: List[Token]

    def __init__(self, pattern: Union[sre_parse.SubPattern, str, re.Pattern]):
        if isinstance(pattern, sre_parse.SubPattern):
            self.pattern = pattern
        elif isinstance(pattern, str):
            self.pattern = sre_parse.parse(pattern)
        elif isinstance(pattern, re.Pattern):
            self.pattern = sre_parse.parse(pattern.pattern, flags=pattern.flags)
        self.named_groups_by_name = {
            k: v for k, v in self.pattern.state.groupdict.items()
        }
        self.named_groups_by_number = {
            v: k for k, v in self.pattern.state.groupdict.items()
        }
        self.positions = [0]
        self.seen_tokens = []

    def __repr__(self):
        return f'<{self.__class__.__name__} pattern: {self.pattern.data!r}>'

    @property
    def last_token(self) -> Optional[Token]:
        if not self.seen_tokens:
            return None
            return self.seen_tokens[-1]

    @property
    def current_position(self) -> Position:
        last = self.last_token
        if not last:
            return Position(start=0, by=0, end=0)
        return last.position

    def parse(self):
        from pprint import pprint

        handled_nodes = []
        handled_positions = []
        handled_reprs = []
        # for d in self.pattern.data:
        #     pprint(d)
        final_nodes = self._continue_parsing(
            self.pattern, handled_nodes, handled_positions, handled_reprs
        )
        # print(final_nodes)
        # print("".join(str(node) for node in final_nodes))
        # print("\n".join(node.describe() for node in final_nodes))
        # print("".join(node.generate() for node in final_nodes))

        return final_nodes

    def _parse_subcomponent(self):
        pass

    def _continue_parsing(
        self, nodes, handled_nodes: List, handled_positions: List, handled_reprs: List
    ):
        final_nodes = []
        for node in nodes:
            result = self._parse_node(
                node, handled_nodes, handled_positions, handled_reprs
            )
            final_nodes.append(result)
            self.seen_tokens.append(result)
        return final_nodes

    def _parse_node(
        self, node, handled_nodes: List, handled_positions: List, handled_reprs: List
    ):
        op, av = node
        # print(node)
        if op is sre_constants.AT:
            return self._at(op, av, handled_nodes, handled_positions, handled_reprs)
            parsed_node, positions, repr = self._at(op, av)
            handled_nodes.append(parsed_node)
            handled_positions.append(positions)
            handled_reprs.append(repr)
            self.positions.append(positions.end)
        elif op is sre_constants.NOT_LITERAL:
            return self._not_literal(
                op, av, handled_nodes, handled_positions, handled_reprs
            )
            parsed_node, positions, repr = self._not_literal(op, av)
            handled_nodes.append(parsed_node)
            handled_positions.append(positions)
            handled_reprs.append(repr)
            self.positions.append(positions.end)
        elif op is sre_constants.LITERAL:
            return self._literal(
                op, av, handled_nodes, handled_positions, handled_reprs
            )
            parsed_node, positions, repr = self._literal(op, av)
            handled_nodes.append(parsed_node)
            handled_positions.append(positions)
            handled_reprs.append(repr)
            self.positions.append(positions.end)
        elif op is sre_constants.MIN_REPEAT:
            return self._min_repeat(
                op, av, handled_nodes, handled_positions, handled_reprs
            )
            parsed_node, positions, repr = self._min_repeat(
                op, av, handled_nodes, handled_positions, handled_reprs
            )
            handled_nodes.append(parsed_node)
            handled_positions.append(positions)
            handled_reprs.append(repr)
            # self.positions.append(positions.end)
        elif op is sre_constants.MAX_REPEAT:
            return self._max_repeat(
                op, av, handled_nodes, handled_positions, handled_reprs
            )
            parsed_node, positions, repr = self._max_repeat(
                op, av, handled_nodes, handled_positions, handled_reprs
            )
            handled_nodes.append(parsed_node)
            handled_positions.append(positions)
            handled_reprs.append(repr)
            # self.positions.append(positions.end)
        elif op is sre_constants.SUBPATTERN:
            return self._subpattern(
                op, av, handled_nodes, handled_positions, handled_reprs
            )
            parsed_node, positions, repr = self._subpattern(
                op, av, handled_nodes, handled_positions, handled_reprs
            )
            handled_nodes.append(parsed_node)
            handled_positions.append(positions)
            handled_reprs.append(repr)
        elif op is sre_constants.ANY:
            return self._any(op, av, handled_nodes, handled_positions, handled_reprs)
        elif op is sre_constants.IN:
            return self._in(op, av, handled_nodes, handled_positions, handled_reprs)
        elif op is sre_constants.RANGE:
            return self._range(op, av, handled_nodes, handled_positions, handled_reprs)
        elif op is sre_constants.CATEGORY:
            return self._category(
                op, av, handled_nodes, handled_positions, handled_reprs
            )
        else:
            raise ValueError(f"unexpected {op}: {av}")

    # handle promotion of literals into literalgroups

    # from pprint import pprint
    # pprint(handled_nodes)
    # pprint(handled_positions)
    # pprint(handled_reprs)
    # pprint(self.positions)

    def _category(self, op, av, *args, **kwargs):
        if av is sre_constants.CATEGORY_DIGIT:
            return Category("\d", self.current_position.end)
        raise ValueError(f"unexpected {op}: {av}")

    def _any(self, op, av, *args, **kwargs):
        return Anything(self.current_position.end)
        return (None, None, None)

    def _range(
        self, op, av, handled_nodes: List, handled_positions: List, handled_reprs: List
    ):
        first, last = av
        span = Range(first, last, self.current_position.end)
        return span
        handled_nodes.append(span)
        #         start_position = positions[-1] + len(first)
        #         chars.append(first)
        #         positions.append(start_position)
        #
        #         position = positions[-1] + len(first)
        #         chars.append('-')
        #         positions.append(position)
        #
        #         end_position = positions[-1] + len(last)
        #         chars.append(last)
        #         positions.append(end_position)
        return (None, None, None)

    def _in(
        self, op, av, handled_nodes: List, handled_positions: List, handled_reprs: List
    ):
        negated: bool = av[0][0] is sre_constants.NEGATE
        cls: Union[Type[In], Type[NegatedIn]] = In
        if negated:
            av = av[1:]
            cls = NegatedIn
        # multiple_ops = len({subop for subop, subav in av}) > 1
        # if not multiple_ops:
        #
        some_stuff = self._continue_parsing(
            av, handled_nodes, handled_positions, handled_reprs
        )
        return cls(some_stuff, self.current_position.end)
        handled_nodes.append(node)
        return (None, None, None)
        # chars.append('[')
        # position = positions[-1] + 1
        # positions.append(position)
        #
        # if negated:
        #     av.pop(0)
        #     chars.append('^')
        #     position = positions[-1] + 1
        #     positions.append(position)
        # for i, j in av:
        #     print(i, j)
        #     if i is sre_constants.LITERAL:
        #         byte = chr(j)
        #         position = positions[-1] + len(byte)
        #         chars.append(byte)
        #         positions.append(position)
        #         segments.append(Negated(Literal(byte, position)))
        #     elif i is sre_constants.RANGE:
        #         first = chr(j[0])
        #         last = chr(j[1])
        #
        #         start_position = positions[-1] + len(first)
        #         chars.append(first)
        #         positions.append(start_position)
        #
        #         position = positions[-1] + len(first)
        #         chars.append('-')
        #         positions.append(position)
        #
        #         end_position = positions[-1] + len(last)
        #         chars.append(last)
        #         positions.append(end_position)
        #
        #         if negated:
        #             segments.append(
        #                 Negated(Range(first, last, (start_position, end_position))))
        #         else:
        #             segments.append(Range(first, last, (start_position, end_position)))

    def _subpattern(
        self, op, av, handled_nodes: List, handled_positions: List, handled_reprs: List
    ):
        group_number, x, y, subpattern = av
        group_name = self.named_groups_by_number.get(group_number, "")
        some_stuff = self._continue_parsing(
            subpattern, handled_nodes, handled_positions, handled_reprs
        )
        if len(some_stuff) == 1:
            some_stuff = some_stuff.pop()
        return SubPattern(
            group_name, group_number, some_stuff, self.current_position.end
        )
        return (None, None, None)

    def _min_repeat(
        self, op, av, handled_nodes: List, handled_positions: List, handled_reprs: List
    ):
        min_num, max_num, subpattern = av
        some_stuff = self._continue_parsing(
            subpattern, handled_nodes, handled_positions, handled_reprs
        )
        # if len(some_stuff) == 1:
        #     some_stuff = some_stuff.pop(0)
        Repeat(int(min_num), int(max_num), some_stuff, 1)
        return (None, None, None)

    def _max_repeat(
        self, op, av, handled_nodes: List, handled_positions: List, handled_reprs: List
    ):
        min_num, max_num, subpattern = av
        some_stuff = self._continue_parsing(
            subpattern, handled_nodes, handled_positions, handled_reprs
        )
        # if len(some_stuff) == 1:
        #     some_stuff = some_stuff.pop(0)
        return Repeat(int(min_num), int(max_num), some_stuff, 1)
        return (None, None, None)

    def _at(self, op, av, *args, **kwargs):
        if av is sre_constants.AT_BEGINNING:
            return self._at_beginning(op, av)
        elif av is sre_constants.AT_END:
            return self._at_end(op, av)
        else:
            raise ValueError(f"unexpected {op}: {av}")

    def _at_beginning(self, op, av, **kwargs):
        return Beginning(self.current_position.end)
        advance_from = self.start_position
        advance_by = 1
        advance_to = advance_from + advance_by
        bit = "^"
        return (
            Beginning("^", advance_from),
            Position(advance_from, advance_by, advance_to),
            bit,
        )

    def _at_end(self, op, av):
        return End(self.current_position.end)
        advance_from = self.start_position
        advance_by = 1
        advance_to = advance_from + advance_by
        bit = "$"
        return (
            End("$", advance_from),
            Position(advance_from, advance_by, advance_to),
            bit,
        )

    def _not_literal(self, op, av, *a, **kw):
        return NegatedLiteral(chr(av), self.current_position.end)
        # bit =
        bit_length = len(bit)
        # [^] + character length
        advance_from = self.start_position
        advance_by = 3 + bit_length
        advance_to = advance_from + advance_by
        return (
            NegatedLiteral(bit, advance_from),
            Position(advance_from, advance_by, advance_to),
            f"[^{bit}]",
        )

    def _literal(self, op, av, *args, **kwargs):
        bit = chr(av)
        return Literal(bit, position=self.current_position.end)
        bit_length = len(bit)
        advance_from = self.start_position
        advance_by = bit_length
        advance_to = advance_from + advance_by
        return (
            Literal(bit, advance_from),
            Position(advance_from, advance_by, advance_to),
            bit,
        )


#
#
# def _dump(parsed_tree: sre_parse.SubPattern, groups: dict):
#     dumper = Reparser()
#     positions = [0]
#     chars = []
#     treebits = []
#     segments = []
#     for node in parsed_tree:
#         op, av = node
#         if op is sre_constants.LITERAL:
#             byte = chr(av)
#             position = positions[-1] + len(byte)
#             chars.append(byte)
#             positions.append(position)
#             segments.append(Literal(byte, position))
#         elif op is sre_constants.NOT_LITERAL:
#             dumper.not_literal(op, av)
#             chars.append('^')
#             position = positions[-1] + 1
#             positions.append(position)
#
#             byte = chr(av)
#             position = positions[-1] + len(byte)
#             chars.append(byte)
#             positions.append(position)
#             segments.append(Negated(Literal(byte, position)))
#         elif op is sre_constants.IN:
#             negated = av[0][0] is sre_constants.NEGATE
#             chars.append('[')
#             position = positions[-1] + 1
#             positions.append(position)
#
#             if negated:
#                 av.pop(0)
#                 chars.append('^')
#                 position = positions[-1] + 1
#                 positions.append(position)
#             for i, j in av:
#                 print(i, j)
#                 if i is sre_constants.LITERAL:
#                     byte = chr(j)
#                     position = positions[-1] + len(byte)
#                     chars.append(byte)
#                     positions.append(position)
#                     segments.append(Negated(Literal(byte, position)))
#                 elif i is sre_constants.RANGE:
#                     first = chr(j[0])
#                     last = chr(j[1])
#
#                     start_position = positions[-1] + len(first)
#                     chars.append(first)
#                     positions.append(start_position)
#
#                     position = positions[-1] + len(first)
#                     chars.append('-')
#                     positions.append(position)
#
#                     end_position = positions[-1] + len(last)
#                     chars.append(last)
#                     positions.append(end_position)
#
#                     if negated:
#                         segments.append(Negated(Range(first, last, (start_position, end_position))))
#                     else:
#                         segments.append(Range(first, last, (start_position, end_position)))
#
#                 else:
#                     import pdb; pdb.set_trace()
#             chars.append(']')
#             position = positions[-1] + 1
#             positions.append(position)
#         # elif op in (sre_constants.MAX_REPEAT, sre_constants.MIN_REPEAT):
#         #     i, j, subtree = av
#         #     # print(subtree)
#         #     s, suboffsets = _dump(subtree, groupdict)
#         # elif op == sre_constants.SUBPATTERN:
#         #     # print(op, av)
#         #     # # groupnum, subtree = av
#         #     groupnum, add_flags, del_flags, subtree = av
#         #     # print(subtree)
#         #     s, suboffsets = _dump(subtree, groupdict)
#             # if groupnum in groupdict:
#             #     name = groupdict[groupnum]
#     # print(positions)
#     # print(chars)
#     print(segments)
#     print("".join(chars))
#     return treebits, chars
#
# # _dump(tree, groupdict)

# dumpee = Reparser(tree)
# dumpee.parse()


def parse(pattern):
    parser = Reparser(pattern)
    results = parser.parse()
    return results, parser


parse(pat)


if __name__ == "__main__":
    import unittest
    import sys

    class CountingTestResult(unittest.TextTestResult):
        def addSubTest(self, test, subtest, outcome):
            # handle failures calling base class
            super().addSubTest(test, subtest, outcome)
            # add to total number of tests run
            self.testsRun += 1

    class Tests(unittest.TestCase):
        InOut = namedtuple("InOut", ("raw", "expected"))
        parameters = (
            # empty/blank
            InOut(r"^$", "^$"),
            # literal runs
            InOut(r"^testing$", "^testing$"),
            # multiple spaces (specifically only space!)
            InOut(r" +", " +"),
            # simple range; vowels
            InOut(r'[aeiou]', '[aeiou]'),
            # collapsing ranges
            InOut(r'[abcdefgh]', '[aeiou]'),
            # hex ranges
            InOut(r"[\x20-\x7E]", " +"),
            # unicode ranges
            InOut(r"[\u03A0-\u03FF]", " +"),
            # quacks like an email
            InOut(r"^[^@]+@[^@]+\.[^@]+$", "^[^@]+@[^@]+\.[^@]+$"),
            # non-ascii alphanumeric
            InOut(r"^[^a-zA-Z0-9]$", "^[^a-zA-Z0-9]$"),
            # positive integers
            InOut(r"^[1-9]+[0-9]*$", "^[1-9]+[0-9]*$"),
            # positive decimals
            InOut(r"(^\d*\.?\d*[0-9]+\d*$)|(^[0-9]+\d*\.\d*$)", "^[^a-zA-Z0-9]$"),
            # positive or negative decimals with `,` as a separator
            InOut(r"-?\d+(,\d*)?", "^[^a-zA-Z0-9]$"),
            # ISO-2 country code(ish); 84094 or 84094-1234
            InOut(r"[A-Z][A-Z]", "[A-Z][A-Z]"),
            # social security; ###-##-####
            InOut(r"[0-9]\{3\}-[0-9]\{2\}-[0-9]\{4\}", "[0-9]{3}-[0-9]{2}-[0-9]{4}"),
            # uuid
            InOut(
                r"[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}",
                "[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}",
            ),
            # dates; yyyy/mm/dd
            InOut(r'^\d{4}/(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])$',
                  ''),
            # ipv4
            InOut('^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$',
                  '')
        )

        def _test_parsing(self):
            for i, (raw, expected) in enumerate(self.parameters, start=0):
                with self.subTest(msg=raw):
                    # if raw == '[abcdefgh]':
                    #     pass
                    re.compile(raw)
                    nodes = parse(raw)
                    output = "".join(str(n) for n in nodes)
                    self.assertEqual(output, expected)
                    # self.assertSequenceEqual(output, expected)

        def assertRawMatches(self, raw, expected):
            try:
                re.compile(raw)
            except re.error:
                self.fail(f"Invalid regex: {raw!r}")
            nodes, parser = parse(raw)
            output = "".join(str(n) for n in nodes)
            self.assertEqual(output, expected)
            print(raw, '==', expected)

        def test_repeats(self):
            raw, expected = "^a{3}$", "^a{3}$"
            self.assertRawMatches(raw, expected)
            raw, expected = "^a{3,}$", "^a{3,}$"
            self.assertRawMatches(raw, expected)
            raw, expected = "^a{3,10}$", "^a{3,10}$"
            self.assertRawMatches(raw, expected)
            raw, expected = "^a{0,10}$", "^a{0,10}$"
            self.assertRawMatches(raw, expected)
            maximum = int(sre_constants.MAXREPEAT)-1
            raw, expected = f"^a{{0,{maximum}}}$", f"^a*$"
            self.assertRawMatches(raw, expected)
            raw, expected = f"^a{{1,{maximum}}}$", f"^a+$"
            self.assertRawMatches(raw, expected)
            raw, expected = "^[^@]$", "^[^@]$"
            self.assertRawMatches(raw, expected)

        def test_iso2_codes(self):
            """ISO-2 country code(ish); GB, US etc"""
            raw, expected = r"[A-Z][A-Z]", "[A-Z][A-Z]"
            self.assertRawMatches(raw, expected)

        def test_zipcodeish(self):
            """zip code(ish?); 84094 or 84094-1234"""
            raw, expected = r"[0-9]{5}(-[0-9]{4})?", "[0-9]{5}(-[0-9]{4})?"
            self.assertRawMatches(raw, expected)

        def test_emailish(self):
            raw, expected = (r"^[^@]+@[^@]+\.[^@]+$", "^[^@]+@[^@]+\.[^@]+$")
            self.assertRawMatches(raw, expected)

    unittest.main(
        module=sys.modules[__name__],
        # testRunner=unittest.TextTestRunner(resultclass=CountingTestResult),
        verbosity=2,
        catchbreak=True,
        tb_locals=True,
        buffer=False,
    )

