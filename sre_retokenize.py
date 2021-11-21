import functools
import random
import re
import sre_parse
import sre_constants
import sre_compile
import string
from collections import namedtuple
from itertools import chain
from typing import List, Union, Text, Type, Optional

# pat = r"^o[^a-z][a-zA-Z]n[^ABC]p*p.*t[^a]i.{1,3}o[0-9]{0, 10}n.+?al/(?P<arg1>\d+)/✂️/(?:(?P<arg2>[a-b])/)?/([0-9]+?)/(?P<arg3>\\d+)/(?:(?P<arg4>[c-d])/)?$"
#
# pat = r"^option[^a-zA-Z].*LOL.+test.{1,4}[a-f0-9]{2,4} - (?P<arg1>\d+) (\d+)$"
# pat = r"^test .* .+ .{1,4} [a-f0-9]{2,4} (?P<arg1>\d+) (\d+)$"
# pat = r"^(?P<arg1>\d){1,4}(?:(?P<arg2>\d))(?P<arg3>\d)$"


# tree = sre_parse.parse(pat)
# tree.dump()
#
# groupdict = {}
# for k, v in tree.state.groupdict.items():
#     groupdict[v] = k


Position = namedtuple("Position", ("start", "by", "end"))


class Token:
    __slots__ = ("start_position",)

    position: int

    def __new__(cls, position: int):
        instance = super().__new__(cls)
        instance.start_position = position
        return instance

    def __str__(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't implement __str__"
        )

    def __repr__(self):
        value = str(self)
        return f"<{self.__class__.__name__!s} {value!r}>"
        return f"<{self.__class__.__name__!s} {value!r} ({self.position!s})>"

    @property
    def position(self) -> Position:
        start = self.start_position
        advance_by = len(str(self))
        end = start + advance_by
        return Position(start=start, by=advance_by, end=end)

    def describe(self):
        return repr(self)

    def simplify(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't implement simplify"
        )

    def generate(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't implement generate"
        )

    def generate_nonmatch(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't implement generate_nonmatch"
        )


class Literal(Token):
    __slots__ = ("value", "sre_types", "start_position", "is_numbers", "is_ascii_letters")

    value: str

    def __new__(cls, value: int, position: int):
        instance = super().__new__(cls, position)
        instance.value = chr(value)
        # instance.is_numbers = value >= 49 and value <= 57
        # instance.is_ascii_letters = value >= 65 and value <= 122
        instance.sre_types = {sre_constants.LITERAL}
        return instance

    def __str__(self):
        # if self.value in sre_parse.SPECIAL_CHARS:
        #     return f"\{self.value}"
        return self.value

    def __len__(self):
        return len(self.value)

    def describe(self):
        return f"character {self.value!r}"

    simplify = __str__

    def generate(self):
        return str(self)

    def generate_nonmatch(self):
        """
        Return the original fixed value.

        So it can be seen that the overall match format is unchanged, don't mutate
        a static string like like abc-def into 123|456
        """
        return str(self)
        initial = current = ord(self.value)
        if self.is_ascii_letters:
            choose = functools.partial(random.choice, seq=[ord(x) for x in string.ascii_letters])
        elif self.is_numbers:
            choose = functools.partial(random.choice, seq=[ord(x) for x in string.digits])
        else:
            choose = functools.partial(random.randint, 33, 1000)
        while current == initial:
            current = choose()
        return chr(current)


class AsciiAlphabetLiteral(Literal):
    pass

class NumericLiteral(Literal):
    pass

class Beginning(Token):
    __slots__ = ("value", "sre_type", "start_position")

    value: str

    def __new__(cls, position: int):
        instance = super().__new__(cls, position)
        instance.value = "^"
        instance.sre_type = sre_constants.AT_BEGINNING
        return instance

    def __str__(self):
        return self.value

    def describe(self):
        return f"anchor to beginning"

    def generate(self):
        return ""

    def generate_nonmatch(self):
        return ""


class End(Token):
    __slots__ = ("value", "sre_type", "start_position")

    value: str

    def __new__(cls, position: int):
        instance = super().__new__(cls, position)
        instance.value = "$"
        instance.sre_type = sre_constants.AT_END
        return instance

    def __str__(self):
        return self.value

    def describe(self):
        return f"anchor to end"

    def generate(self):
        return ""

    def generate_nonmatch(self):
        return ""


class Anything(Token):
    __slots__ = ("value", "sre_type", "start_position")

    value: str

    def __new__(cls, position: int):
        instance = super().__new__(cls, position)
        instance.value = "."
        instance.sre_type = sre_constants.ANY
        return instance

    def __str__(self):
        return self.value

    def describe(self):
        return "anything"


class NegatedLiteral(Literal):
    __slots__ = ("value", "sre_type", "start_position", "is_numbers", "is_ascii_letters")

    def __new__(cls, value: int, position: int):
        instance = super().__new__(cls, value, position)
        return instance

    def __str__(self):
        return f"[^{self.value}]"

    def describe(self):
        return f"anything other than {self.value!r}"

    def generate(self):
        initial = current = self.value
        choices = string.digits + string.ascii_letters + string.punctuation
        while current == initial:
            current = random.choice(choices)
        return current

        bad = self.value
        current = self.value
        top_end = range(0x110000)[-1]  # chr(1114112) causes an error giving us this.
        while current == bad:
            current = random.randint(33, top_end)
        return chr(current)

    def generate_nonmatch(self):
        return self.value


class Range(Token):
    __slots__ = ("start", "end", "is_numbers", "is_ascii_letters")

    def __new__(cls, start: int, end: int, position: int):
        instance = super().__new__(cls, position)
        instance.start = start
        instance.end = end
        instance.is_numbers = start >= 49 and end <= 57
        instance.is_ascii_letters = start >= 65 and end <=122
        return instance

    def __str__(self) -> str:
        start = chr(self.start)
        end = chr(self.end)
        return f"[{start}-{end}]"

    def describe(self) -> str:
        if (self.end - self.start) > 10:
            first = chr(self.start)
            last = chr(self.end)
            chars = ", ".join(chr(i) for i in range(self.start + 1, self.start + 6, 1))
            return f"any of {first!r}, ... {chars!s}, ... {last!r}"
        chars = "', '".join(chr(i) for i in range(self.start, self.end, 1))
        return f"any of '{chars!s}'"

    def generate(self):
        return chr(random.randint(self.start, self.end))

    def generate_nonmatch(self) -> str:
        # if self.is_numbers:
        #     choices = ()
        #     # If it's something like 3-7, try and pick from 0-3 and 8-9 by
        #     # preference, because those values are more specific/local to highlight.
        #     if self.start > 49:
        #         choices += tuple(chr(x) for x in range(49, self.start))
        #     if self.end < 57:
        #         choices += tuple(chr(x) for x in range(self.end, 57))
        #
        #     if not choices:
        #         choices = tuple(string.ascii_letters)
        #
        #     return random.choice(choices)
        # elif self.is_ascii_letters:
        #     return random.choice(string.digits)
        # This'll do for now...
        current = self.start
        top_end = range(0x110000)[-1]  # chr(1114112) causes an error giving us this.
        # ascii + extened to 255 ... easier to read currently for my testing purposes
        # than a super long value like \U00047739
        while current >= self.start and current <= self.end:
            current = random.randint(32, 255)
        return chr(current)

    @classmethod
    def from_chars(cls, start: str, end: str):
        return cls(start=ord(start), end=ord(end), position=0)


class NumericRange(Range):
    def generate_nonmatch(self) -> str:
        # If we're in a numeric range rather than a numeric subrange,
        # it's the whole range 0-9, take a letter instead.
        choices = tuple(string.ascii_letters)
        return random.choice(choices)


class NumericSubRange(NumericRange):

    def generate_nonmatch(self) -> str:
        choices = ()
        # If it's something like 3-7, try and pick from 0-3 and 8-9 by
        # preference, because those values are more specific/local to highlight.
        if self.start > 48:
            choices += tuple(chr(x) for x in range(48, self.start))
        if self.end < 57:
            choices += tuple(chr(x) for x in range(self.end, 57))
        if not choices:
            choices = tuple(string.ascii_letters)
        return random.choice(choices)

class AsciiAlphabetRange(Range):
    def generate_nonmatch(self) -> str:
        # A-Z
        choices = ()
        if self.start > 65:
            choices += tuple(chr(x) for x in range(65, self.start))
        if self.end < 90:
            choices += tuple(chr(x) for x in range(self.end, 90))

        if self.start > 97:
            choices += tuple(chr(x) for x in range(97, self.start))
        if self.end < 122:
            choices += tuple(chr(x) for x in range(self.end, 122))

        if not choices:
            choices = tuple(string.digits)
        return random.choice(choices)

class HexAlphabetRange(Range):
    def generate_nonmatch(self) -> str:
        hex = {'a', 'b', 'c', 'd', 'e', 'f', 'A', 'B', 'C', 'D', 'E', 'F'}
        choices = tuple(x for x in string.ascii_letters if x not in hex)
        return random.choice(choices)

class Repeat(Token):
    __slots__ = ("min", "max", "value")

    def __new__(cls, min: int, max: int, value, position: int):
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
        if self.max in (int(sre_constants.MAXREPEAT), int(sre_constants.MAXREPEAT) - 1):
            if self.min == 0:
                minmax = "*"
            elif self.min == 1:
                minmax = "+"
            else:
                minmax = f"{{{self.min},}}"
        # elif self.max == 1 and self.min == 0:
        #     minmax = '?'
        return f"{value}{minmax}"

    def __repr__(self):
        return f"<{self.__class__.__name__!s} {self!s} {self.value!r}>"

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

    def generate(self):
        if self.min == self.max:
            make = self.min
        else:
            minimum = self.min
            maximum = min(10, self.max)
            make = random.randint(minimum, maximum)
        count = 0
        parts = []
        while count < make:
            value = "".join(v.generate() for v in self.value)
            parts.append(value)
            count += 1
        return "".join(parts)

    def generate_nonmatch(self):
        if self.min == self.max:
            make = self.min
        else:
            minimum = self.min
            maximum = min(10, self.max)
            make = random.randint(minimum, maximum)
        #
        # if self.max not in (int(sre_constants.MAXREPEAT), int(sre_constants.MAXREPEAT) - 1):
        #     make = random.randint(self.max, self.max + 10)
        # elif self.min > 1:
        #     make = random.randint(0, self.min - 1)
        # elif self.min == 1:
        #     make = 1
        # elif self.min == 0:
        #     make = 1
        count = 0
        parts = []
        while count < make:
            value = "".join(v.generate_nonmatch() for v in self.value)
            parts.append(value)
            count += 1
        return "".join(parts)

    # def __repr__(self):
    #     self.
    #     return f'<{self.__class__.__name__!s} {value!r}>'


class OptionalRepeat(Repeat):
    pass

class RequiredRepeat(Repeat):
    pass


class Category(Token):
    __slots__ = ("value",)

    def __new__(cls, value, position):
        instance = super().__new__(cls, position)
        instance.value = value
        return instance

    def __str__(self):
        return self.value


class DigitCategory(Category):
    def __new__(cls, position):
        return super().__new__(cls, '\d', position)

    def __str__(self):
        return f'[0-9], from {self.value!s}'

    def generate(self):
        # TODO: handle \d unicode points
        return str(random.randint(0, 9))

    def generate_nonmatch(self):
        return str(random.choice(string.ascii_letters))


class WordCategory(Category):
    def __new__(cls, position):
        return super().__new__(cls, '\w', position)

    def __str__(self):
        return f'[a-zA-Z], from {self.value!s}'

    def generate(self):
        # TODO: handle \w unicode points
        return str(random.choice(string.ascii_letters))

    def generate_nonmatch(self):
        return str(random.choice(string.digits))



class In(Token):
    __slots__ = ("value",)

    value: List[Token]

    def __new__(cls, value: List[Token], position):
        instance = super().__new__(cls, position)
        instance.value = value
        return instance

    def __str__(self):
        all_literals = {type(sub) for sub in self.value} == {Literal}
        value = "".join(str(sub) for sub in self.value)
        if all_literals:
            return f'[{value!s}]'
        return value

    def __repr__(self):
        return f"<{self.__class__.__name__!s} {self.value!r}>"

    def describe(self):
        parts = []
        for part in self.value:
            parts.append(part.describe())
        return " or ".join(parts)

    def generate(self):
        value = random.choice(self.value)
        return value.generate()
        # values = [v.generate() for v in self.value]
        # return "".join(values)

    def generate_nonmatch(self):
        reject = {None}
        current = None
        # Handle the case where a|b might otherwise yield 'b' as a valid
        # non-match for 'a'
        # if all(isinstance(x, Literal) for x in self.value):
        #     reject.update({x.generate() for x in self.value})
        while current in reject:
            value = random.choice(self.value)
            current = value.generate_nonmatch()
        return current


class LiteralIn(In):
    value: List[Literal]

    def generate_nonmatch(self):
        sentinels = {x.value for x in self.value}
        current = self.value[0].value
        choices = string.digits + string.ascii_letters + string.punctuation
        while current in sentinels:
            current = random.choice(choices)
        return current


class NegatedIn(In):
    def __str__(self):
        value = "".join(str(sub) for sub in self.value)
        return f"[^{value!s}]"


class NegatedLiteralIn(NegatedIn):
    pass


class SubPattern(Token):
    __slots__ = ("name", "number", "value")

    def __new__(cls, name, num, value, position):
        instance = super().__new__(cls, position)
        instance.name = name
        instance.number = num
        instance.value = value
        return instance

    def __str__(self):
        # Hacky, value can be a single thing like Repeat, or a series of things
        # like Literal, Repeat
        if isinstance(self.value, list):
            value = "".join(str(node) for node in self.value)
        else:
            value = str(self.value)
        if self.name:
            return f"(?P<{self.name!s}>{value!s})"
        return f"({value!s})"

    def __repr__(self):
        if self.name:
            group = f'named: {self.name!r} '
        elif self.number:
            group = f'number: {self.number!r} '
        else:
            group = ""
        return f"<{self.__class__.__name__!s} {group!s}{self.value!r}>"

    def describe(self):
        if self.name:
            return f"group (named {self.name!r}) capturing: {self.value!s}"
        else:
            return f"group (number {self.number!s}) capturing {self.value!s}"

    def generate(self):
        bits = [node.generate() for node in self.value]
        return "".join(bits)

    def generate_nonmatch(self):
        bits = [node.generate_nonmatch() for node in self.value]
        return "".join(bits)


class OrBranch(Token):
    value: List[Token]

    def __new__(cls, value: List[Token], position):
        instance = super().__new__(cls, position)
        instance.value = value
        return instance

    def __str__(self):
        branches = ["".join(str(node) for node in branch) for branch in self.value]
        # for branch in self.value:
        #     branches.append("".join(str(node) for node in branch))
        return "|".join(str(node) for node in branches)

    def __repr__(self):
        return f"<{self.__class__.__name__!s} {self.value!r}>"

    def generate(self):
        branch = random.choice(self.value)
        bits = tuple(chain.from_iterable(bit.generate() for bit in branch))
        return "".join(bits)

    def generate_nonmatch(self):
        branch = random.choice(self.value)
        bits = tuple(chain.from_iterable(bit.generate_nonmatch() for bit in branch))
        return "".join(bits)


class Empty(Token):
    __slots__ = ("value", "sre_type", "start_position")

    value: str

    def __new__(cls, position: int):
        instance = super().__new__(cls, position)
        instance.value = ""
        instance.sre_type = None
        return instance

    def __str__(self):
        return self.value


# Literal = namedtuple("Literal", ("value", "position"))
# NegatedLiteral = namedtuple("NegatedLiteral", ("value", "position"))
# LiteralGroup = namedtuple("LiteralGroup", ("value", "start", "end"))
# Negated = namedtuple("Negated", ("value"))
# Range = namedtuple("Range", ("start", "end", "position"))
# NegatedRange = namedtuple("Range", ("start", "end", "position"))
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
        names = set(self.named_groups_by_name.keys())
        numbers = set(self.named_groups_by_number.keys())
        return "".join(
            (
                f"<{self.__class__.__name__}",
                f" group names: {names!r}" if names else "",
                f" group numbers: {numbers!r}" if numbers else "",
                f" pattern: {self.pattern.data!r}>",
            )
        )

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

        self._parse_noncapturing_groups()

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

    def _parse_noncapturing_groups(self):
        # https://newbedev.com/is-it-possible-to-match-nested-brackets-with-a-regex-without-using-recursion-or-balancing-groups
        # (?=\(\?:)(?=((?:(?=.*?\((?!.*?\2)(.*\)(?!.*\3).*))(?=.*?\)(?!.*?\3)(.*)).)+?.*?(?=\2)[^(]*(?=\3$))) ... :(
        # results = tuple(self.matches('^(?:(?<p>test(?:(?P<test>\d))))$'))

        toxt = r"^(?:(?P<p>test(?:(?P<test>\d))))$"
        source = sre_parse.Tokenizer(toxt)

        starts = reversed([m.start() for m in re.finditer(re.escape("(?:"), toxt)])

        # source = sre_parse.Tokenizer(self.pattern.state.str)
        sourceget = source.get
        someshit = []
        while True:
            this = sourceget()
            # if sourcematch('(') and sourcematch('?') and sourcematch(':'):
            #     pass
            if this is None:
                break  # end of pattern
            if this == "(":
                this = sourceget()
                if this == "?":
                    this = sourceget()
                    if this == ":":
                        stack = 0
                        start = source.pos - 3
                        while True:
                            this = sourceget()
                            # print(this)
                            if this == "(":
                                stack += 1
                            if this == ")":
                                stack -= 1
                            if stack == 0:
                                end = source.pos + 1
                                someshit.append((start, end))
                                break

            # if this in "|)":
            #     break  # end of subpattern
            # sourceget()
            # state = sre_parse.State()
            # if this == "(":
            #     start = source.tell() - 1
            #     if sourcematch("?"):
            #         char = sourceget()
            #         if char == ":":
            #             # while True:
            #             #     if source.next is None:
            #             #         raise source.error("missing ), unterminated comment",
            #             #                            source.tell() - start)
            #             #     if sourceget() == ")":
            #             #         someshit.append(toxt[start:source.pos])
            #             p = sre_parse._parse(source, state, 0, nested + 1, False)
            #             contained = source.getuntil(')', name='non-capturing group')
            #             someshit.append(contained)
        return

    def _parse_noncapturing_groups1(self):
        toxt = r"^(?:(?P<p>test(?:(?P<test>\d))))$"
        source = sre_parse.Tokenizer(toxt)

        # source = sre_parse.Tokenizer(self.pattern.state.str)
        sourceget = source.get
        sourcematch = source.match

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
        elif op is sre_constants.CATEGORY_DIGIT:
            return self._category(
                op, av, handled_nodes, handled_positions, handled_reprs
            )
        elif op is sre_constants.BRANCH:
            return self._branch(op, av, handled_nodes, handled_positions, handled_reprs)
        else:
            raise ValueError(f"unexpected {op}: {av}")

    def _branch(self, op, av, *args, **kwargs):
        # dunno what the None represents
        if av[0] is None:
            av = av[1:]
        # # dunno, seems to be further wrapped
        av = av[0]
        branches = [
            self._continue_parsing(branch, *args, **kwargs) for branch in av
        ]

        # I dunno
        # for i, a in enumerate(av):
        #     if isinstance(a, sre_parse.SubPattern) and a.data == []:
        #         branches.append(Empty(self.current_position.end))
        #     for subop, subav in a:
        #         if subav == []:
        #             pass
        #         branches.append(self._continue_parsing(a, *args, **kwargs))
        return OrBranch(branches, self.current_position.end)

    def _category(self, op, av, *args, **kwargs):
        if av is sre_constants.CATEGORY_DIGIT:
            return DigitCategory(self.current_position.end)
        elif av is sre_constants.CATEGORY_WORD:
            return WordCategory(self.current_position.end)
        raise ValueError(f"unexpected {op}: {av}")

    def _any(self, op, av, *args, **kwargs):
        return Anything(self.current_position.end)
        return (None, None, None)

    def _range(
        self, op, av, handled_nodes: List, handled_positions: List, handled_reprs: List
    ):
        first, last = av
        cls = Range
        # 0-9
        if first >= 48 and last <= 57:
            cls = NumericRange
            # Starts at 1+ or ends at < 9
            if first > 48 or last < 57:
                cls = NumericSubRange
        # A-Z or a-z
        elif (first >= 65 and last <= 90) or (first >= 97 and last <= 122):
            cls = AsciiAlphabetRange
            # A-F or a-f
            if (first >= 65 and last <= 70) or (first >= 97 and last <=102):
                cls = HexAlphabetRange
        span = cls(first, last, self.current_position.end)
        return span

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

        if all(isinstance(x, Literal) for x in some_stuff):
            cls = LiteralIn
            if negated:
                cls = NegatedLiteralIn

        return cls(some_stuff, self.current_position.end)

    def _subpattern(
        self, op, av, handled_nodes: List, handled_positions: List, handled_reprs: List
    ):
        group_number, x, y, subpattern = av
        group_name = self.named_groups_by_number.get(group_number, "")
        some_stuff = self._continue_parsing(
            subpattern, handled_nodes, handled_positions, handled_reprs
        )
        # if len(some_stuff) == 1:
        #     some_stuff = some_stuff.pop()
        return SubPattern(
            group_name, group_number, some_stuff, self.current_position.end
        )

    def _min_repeat(
        self, op, av, handled_nodes: List, handled_positions: List, handled_reprs: List
    ):
        min_num, max_num, subpattern = av
        some_stuff = self._continue_parsing(
            subpattern, handled_nodes, handled_positions, handled_reprs
        )
        cls = Repeat
        if min_num == 0 and max_num == 1:
            cls = OptionalRepeat
        elif min_num > 0:
            cls = RequiredRepeat
        # if len(some_stuff) == 1:
        #     some_stuff = some_stuff.pop(0)
        cls(int(min_num), int(max_num), some_stuff, 1)
        return (None, None, None)

    def _max_repeat(
        self, op, av, handled_nodes: List, handled_positions: List, handled_reprs: List
    ):
        min_num, max_num, subpattern = av
        some_stuff = self._continue_parsing(
            subpattern, handled_nodes, handled_positions, handled_reprs
        )
        cls = Repeat
        if min_num == 0 and max_num == 1:
            cls = OptionalRepeat
        elif min_num > 0:
            cls = RequiredRepeat
        # if len(some_stuff) == 1:
        #     some_stuff = some_stuff.pop(0)
        return cls(int(min_num), int(max_num), some_stuff, 1)

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
        return NegatedLiteral(av, self.current_position.end)
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

    def _literal(self, op, av: int, *args, **kwargs):
        cls = Literal
        if av >= 48 and av <= 57:
            cls = NumericLiteral
        elif (av >= 65 and av <= 90) or (av >= 97 and av <= 122):
            cls = AsciiAlphabetLiteral
        return cls(av, position=self.current_position.end)
        bit_length = len(bit)
        advance_from = self.start_position
        advance_by = bit_length
        advance_to = advance_from + advance_by
        return (
            Literal(bit, advance_from),
            Position(advance_from, advance_by, advance_to),
            bit,
        )


def parse(pattern):
    parser = Reparser(pattern)
    results = parser.parse()
    return results, parser


# parse(pat)


if __name__ == "__main__":
    import unittest
    import sys

    class CountingTestResult(unittest.TextTestResult):
        def addSubTest(self, test, subtest, outcome):
            # handle failures calling base class
            super().addSubTest(test, subtest, outcome)
            # add to total number of tests run
            self.testsRun += 1

    class CustomAsserts:
        def setUp(self) -> None:
            """Always seed with the same number"""
            super().setUp()

        def assertRawMatches(self, raw, expected):
            random.seed(42)
            try:
                re.compile(raw)
            except re.error:
                self.fail(f"Invalid regex: {raw!r}")
            nodes, parser = parse(raw)
            output = "".join(str(n) for n in nodes)
            self.assertEqual(output, expected)

        def assertGenerates(self, raw, expected_match, expected_nonmatch='', seed=42):

            try:
                regex = re.compile(raw, re.MULTILINE | re.VERBOSE)
            except re.error:
                self.fail(f"Invalid regex: {raw!r}")
            nodes, parser = parse(raw)

            random.seed(seed)
            match_output = "".join(chain.from_iterable([n.generate() for n in nodes]))
            re_match = regex.fullmatch(match_output)
            random.seed(seed)
            nonmatch_output = "".join(chain.from_iterable([n.generate_nonmatch() for n in nodes]))
            re_nonmatch = regex.fullmatch(nonmatch_output)

            if expected_match == expected_nonmatch:
                self.fail("`expected_match` and `expected_nonmatch` should not be the same")

            if match_output == nonmatch_output:
                self.fail("`match_output` and `nonmatch_output` should not be the same")

            self.assertIsNotNone(re_match, msg="`match_output` doesn't match for `regex`")
            self.assertSequenceEqual(match_output, expected_match, msg="`match_output` is not the same as `expected_match`")

            self.assertIsNone(re_nonmatch, msg="`nonmatch_output` unexpectedly matches for `regex`")
            self.assertSequenceEqual(nonmatch_output, expected_nonmatch, msg="`nonmatch_output` is not the same as `expected_nonmatch`")

    # class Te1sts(CustomAsserts, unittest.TestCase):
    #     InOut = namedtuple("InOut", ("raw", "expected"))
    #     parameters = (
    #         # empty/blank
    #         InOut(r"^$", "^$"),
    #         # literal runs
    #         InOut(r"^testing$", "^testing$"),
    #         # multiple spaces (specifically only space!)
    #         InOut(r" +", " +"),
    #         # simple range; vowels
    #         InOut(r"[aeiou]", "[aeiou]"),
    #         # collapsing ranges
    #         InOut(r"[abcdefgh]", "[aeiou]"),
    #         # non-ascii alphanumeric
    #         InOut(r"^[^a-zA-Z0-9]$", "^[^a-zA-Z0-9]$"),

    #
    #     def test_repeats(self):
    #         raw, expected = "^a{3}$", "^a{3}$"
    #         self.assertRawMatches(raw, expected)
    #         raw, expected = "^a{3,}$", "^a{3,}$"
    #         self.assertRawMatches(raw, expected)
    #         raw, expected = "^a{3,10}$", "^a{3,10}$"
    #         self.assertRawMatches(raw, expected)
    #         raw, expected = "^a{0,10}$", "^a{0,10}$"
    #         self.assertRawMatches(raw, expected)
    #         maximum = int(sre_constants.MAXREPEAT) - 1
    #         raw, expected = f"^a{{0,{maximum}}}$", f"^a*$"
    #         self.assertRawMatches(raw, expected)
    #         raw, expected = f"^a{{1,{maximum}}}$", f"^a+$"
    #         self.assertRawMatches(raw, expected)
    #         raw, expected = "^[^@]$", "^[^@]$"
    #         self.assertRawMatches(raw, expected)
    #
    #     def test_optional(self):
    #         raw, expected = (
    #             r"(?P<product>\w+)/(?P<project_id>\w+|)",
    #             r"(?P<product>\w+)/(?P<project_id>\w+|)",
    #         )
    #         self.assertRawMatches(raw, expected)
    #         raw, expected = (
    #             r"(?P<product>\w+)/(?:(?P<project_id>\w+))?",
    #             "(?P<product>\w+)/(?P<project_id>\w+){0,1}",
    #         )
    #         self.assertRawMatches(raw, expected)


    class OrTests(CustomAsserts, unittest.TestCase):

        def test_simple(self):
            """This actually compiles into an IN query"""
            raw = r"a|b"
            expected = "a"
            expected_fail = 'p'
            self.assertGenerates(raw, expected, expected_fail)

        def test_simple2(self):
            """This actually compiles into an IN query"""
            raw = r"[1-9]|1"
            expected = "1"
            nonmatch = 'b'
            self.assertGenerates(raw, expected, nonmatch)

        def test_two_branches(self):
            raw = r"0[1-9]|1[0-2]"
            expected = "05"
            nonmatch = ':p'
            self.assertGenerates(raw, expected, nonmatch)

        def test_three_branches(self):
            raw = r'0[1-9]|[12][0-9]|3[01]'
            expected = '30'
            nonmatch = '1̘'
            self.assertGenerates(raw, expected, nonmatch)

        def test_multiple_parts_in_separate_groups(self):
            raw = r'-[A-Za-z]{2}|\d{3}'
            expected = '-hE'  # first half
            nonmatch = ':319013' # bad first literal, number too long
            self.assertGenerates(raw, expected, nonmatch)
            raw = r'[A-Za-z]{1}|\d{2}'
            expected = 'h'  # first half
            nonmatch = '3'  # 3 is too short
            self.assertGenerates(raw, expected, nonmatch)

        def test_digit_ranges(self):
            raw = r'\d{2}|\d{8}'
            expected = '43'  # second half
            expected_fail = 'Vp'
            self.assertGenerates(raw, expected, expected_fail)
            raw = r'\d{8}|\d{2}'
            expected = '43190138'  # first half
            expected_fail = 'VpiRLcfo'
            self.assertGenerates(raw, expected, expected_fail)



    class ComplexTests(CustomAsserts, unittest.TestCase):
        def test_ipv4ish(self):
            """something that looks ipv4 ish"""
            raw = r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
            expected = "251.255.254.203"
            nonmatch = '04Y¯ɽ\x80[¯×X¶ïâÒwGý1\x81{΄9îÚǷ89³+åúý\x884ÂIƜ3Ó»ʫ83\x96ƥ8Ï÷̳30\x86ĳ39Vˀ6DCĝ995®½͌9ƫ9§ÎÎä<\x8fÂ0\xa0¡'
            self.assertGenerates(raw, expected, nonmatch)

        def test_ipv4ish2(self):
            """something that looks ipv4 ish"""
            # https://rgxdb.com/r/4ODCEYFE
            raw = r"^\d{1,3}(?:\.\d{1,3}){3}$"
            expected = "251.255.254.203"
            nonmatch = '04Y¯ɽ\x80[¯×X¶ïâÒwGý1\x81{΄9îÚǷ89³+åúý\x884ÂIƜ3Ó»ʫ83\x96ƥ8Ï÷̳30\x86ĳ39Vˀ6DCĝ995®½͌9ƫ9§ÎÎä<\x8fÂ0\xa0¡'
            self.assertGenerates(raw, expected, nonmatch)

        def test_ipv6ish(self):
            raw = r'(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))'
            expected = '::251.255.254.1'
            nonmatch = '::256.255.255.256'
            self.assertGenerates(raw, expected, nonmatch)

        def test_positive_integers(self):
            raw = r"^[1-9]+[0-9]*$"
            expected = "5489"
            nonmatch = "b^"
            self.assertGenerates(raw, expected, nonmatch)

        def test_positive_or_negative_decimals_with_comma(self):
            raw = r"-?\d+(,\d*)"
            expected = "3,190"
            nonmatch = '\x93:̘ĺěą¯̓\x89˕̗LǑb'
            self.assertGenerates(raw, expected, nonmatch)

        def test_social_security_style(self):
            raw = '[0-9]{3}-[0-9]{2}-[0-9]{4}'
            expected = "431-01-8883"
            nonmatch = '&^CÍ·(W¡¯×X¶ïΛÒwĽã:\x81{ºî'
            self.assertGenerates(raw, expected, nonmatch)

        def test_language_ish(self):
            """Something like en-US or en-Latn-us"""
            raw = r'[A-Za-z]{2,4}(-[A-Za-z]{4})?(-[A-Za-z]{2}|\d{3})?$'
            expected = 'AhEV-GQ'
            nonmatch = '8991311455ƌ1826853ͪ6193ˀ74399521129b9881542088jWIMFhZppfeiJİ3458135990'
            self.assertGenerates(raw, expected, nonmatch)

        def test_pathish(self):
            raw = r'(?P<first>\w+)/(?P<second>\w+|)'
            expected = 'Vp/RLc'
            nonmatch = '43¯901388390'
            self.assertGenerates(raw, expected, nonmatch)

        def test_uuidish(self):
            raw = r"[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}"
            expected = '1fBb18eA-ab9e-f8BE-AfCB-caAC4fEdeF53'
            nonmatch = 'hxumVfBh-tSbm-PIuX-gNxt-mKIYiDhlsNWk'
            self.assertGenerates(raw, expected, nonmatch)

        def test_uuidish2(self):
            # https://rgxdb.com/r/4AK4LRIU
            raw = r"^[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}$"
            expected = 'a431fad0-38ab-feBE-A6C3-A65EAdaAceE1'
            nonmatch = 'nVvimULi-foYJ-TUuX-gSHp-WmywYiPnlyZW'
            self.assertGenerates(raw, expected, nonmatch)

        def test_hex_encoding(self):
            raw = r'[\x20-\x7E]'
            expected = '#'
            nonmatch = 'Ý'
            self.assertGenerates(raw, expected, nonmatch)

        def test_unicode_encoding(self):
            raw = r"[\u03A0-\u03FF]"
            expected = 'Σ'
            nonmatch = '&'
            self.assertGenerates(raw, expected, nonmatch)

        def test_dateish(self):
            raw = r"^\d{4}/(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])$"
            expected = '0328/01/04'
            nonmatch = 'bpiR/00/00'
            self.assertGenerates(raw, expected, nonmatch)

        def test_timeish(self):
            """12 hour format"""
            # https://digitalfortress.tech/js/top-15-commonly-used-regex/
            raw = r'^(0?[1-9]|1[0-2]):[0-5][0-9]$'
            expected = '4:18'
            nonmatch = '0:6R'
            self.assertGenerates(raw, expected, nonmatch)

        def test_timeish2(self):
            """12 hour format with AM/PM"""
            # https://digitalfortress.tech/js/top-15-commonly-used-regex/
            raw = r'^((1[0-2]|0?[1-9]):([0-5][0-9]) ?([AaPp][Mm]))$'
            expected = '12:12Am'
            nonmatch = '17:6i]('
            self.assertGenerates(raw, expected, nonmatch)

        def test_timeish3(self):
            """24 hour format, HH:MM"""
            # https://digitalfortress.tech/js/top-15-commonly-used-regex/
            raw = r'^(0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]$'
            expected = '20:12'
            nonmatch = '23:6i'
            self.assertGenerates(raw, expected, nonmatch)

        def test_timeish4(self):
            """24 hour format, HH:MM:SS"""
            # https://digitalfortress.tech/js/top-15-commonly-used-regex/
            raw = r'^(?:[01]\d|2[0123]):(?:[012345]\d):(?:[012345]\d)$'
            expected = '03:11:59'
            nonmatch = '3p:sV:dL'
            self.assertGenerates(raw, expected, nonmatch)

        def test_emailish(self):
            raw = r'^[^@]+@[^@]+\.[^@]+$'
            expected = '3z@shd].b.S43brt#'
            nonmatch = '@@@@.@@@@@'
            self.assertGenerates(raw, expected, nonmatch)

        def test_emailish2(self):
            # https://digitalfortress.tech/js/top-15-commonly-used-regex/
            raw = r'^([a-z0-9_\.\+-]+)@([\da-z\.-]+)\.([a-z\.]{2,6})$'
            expected = 'x_@e89-.ch'
            nonmatch = 'Ro@RL-.FO'
            self.assertGenerates(raw, expected, nonmatch)

        def test_emailish3(self):
            # https://digitalfortress.tech/js/top-15-commonly-used-regex/
            raw = r'^([a-zA-Z0-9._%-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6})$'
            expected = 'x3@Xv-s.agqrw'
            nonmatch = 'Rh@F-..FO'
            self.assertGenerates(raw, expected, nonmatch)

        def test_base64ish(self):
            raw = r'^[^-A-Za-z0-9+/=]|=[^=]|={3,}$'
            expected = '7'
            nonmatch = 'D'
            self.assertGenerates(raw, expected, nonmatch, seed=100)

        def test_base64ish2(self):
            raw = r'^(?:[A-Za-z\d+/]{4})*(?:[A-Za-z\d+/]{3}=|[A-Za-z\d+/]{2}==)?$'
            expected = 'A3eV/S+AGq/Rw/+o/0w+4g1ML90+/MR99wBy1d+75v1='
            nonmatch = '[pI_/+[gB/[]/Zd+rNgrpMZw/rl/KEhfG+DkwRd/'
            self.assertGenerates(raw, expected, nonmatch)

        def test_base64ish3(self):
            # https://rgxdb.com/r/1NUN74O6
            raw = r'^(?:[a-zA-Z0-9+\/]{4})*(?:|(?:[a-zA-Z0-9+\/]{3}=)|(?:[a-zA-Z0-9+\/]{2}==)|(?:[a-zA-Z0-9+\/]{1}===))$'
            expected = 'a3Ev/s+agQ/rW/+O/0W+4G1ml90+/mr99WbY1D+7L3=='
            nonmatch = 'BpbF/+BN[/tv/ZK+rggYWMZ^/YS/K^Of`+DkwkK/w+O='
            self.assertGenerates(raw, expected, nonmatch)

        def test_creditcardish(self):
            # https://gist.github.com/nerdsrescueme/1237767
            raw = r'^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|6011[0-9]{12}|622((12[6-9]|1[3-9][0-9])|([2-8][0-9][0-9])|(9(([0-1][0-9])|(2[0-5]))))[0-9]{10}|64[4-9][0-9]{13}|65[0-9]{14}|3(?:0[0-5]|[68][0-9])[0-9]{11}|3[47][0-9]{13})$'
            expected = '5331901388390643'
            nonmatch = '56oVRLcfoJToLZWS'
            self.assertGenerates(raw, expected, nonmatch)

        def test_amexish(self):
            # https://gist.github.com/nerdsrescueme/1237767
            raw = r'^3[47][0-9]{13}$'
            expected ='344319013883906'
            nonmatch = '3>bpiRLcfoJToLZ'
            self.assertGenerates(raw, expected, nonmatch)

        def test_uk_postcodeish(self):
            # https://gist.github.com/nerdsrescueme/1237767
            raw = r'^([A-Z]{1,2}[0-9][A-Z0-9]? [0-9][ABD-HJLNP-UW-Z]{2})$'
            expected = 'X3 1BP'
            nonmatch = 'ko RBC'
            self.assertGenerates(raw, expected, nonmatch)

        def test_urlish(self):
            # https://gist.github.com/nerdsrescueme/1237767
            raw = r'^(http|https|ftp):\/\/([[a-zA-Z0-9]\-\.])+(\.)([[a-zA-Z0-9]]){2,4}([[a-zA-Z0-9]\/+=%&_\.~?\-]*)$'
            # ... lol?
            expected = 'ftp://[-.]H-.].x][][//////////=%&_.~-'
            nonmatch = 'ftp://[-.]i-.].G][]c/=%&_.-]]]'
            self.assertGenerates(raw, expected, nonmatch)

        def test_urlish2(self):
            # https://digitalfortress.tech/js/top-15-commonly-used-regex/
            raw = r'^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#()?&//=]*)$'
            expected = ''
            nonmatch = ''
            self.assertGenerates(raw, expected, nonmatch)

        def test_password_complexity(self):
            # https://digitalfortress.tech/js/top-15-commonly-used-regex/
            raw = r'(?=(.*[0-9]))((?=.*[A-Za-z0-9])(?=.*[A-Z])(?=.*[a-z]))^.{8,}$'
            expected = ''
            nonmatch = ''
            self.assertGenerates(raw, expected, nonmatch)

        def test_htmlish(self):
            # https://digitalfortress.tech/js/top-15-commonly-used-regex/
            raw = r'<\/?[\w\s]*>|<.+[\W]>'
            expected = ''
            nonmatch = ''
            self.assertGenerates(raw, expected, nonmatch)

        def test_html_entity_refish(self):
            # https://rgxdb.com/r/5ZQ9UNOA
            raw = r'&(?:\#(?:(?P<dec>[0-9]+)|[Xx](?P<hex>[0-9A-Fa-f]+))|(?P<named>[A-Za-z0-9]+));'
            expected = '&#31901;'
            nonmatch = '&#oVRLc;'
            self.assertGenerates(raw, expected, nonmatch)

        def test_urlencoded_string(self):
            # https://rgxdb.com/r/48L3HPJP
            raw = r'^(?:[^%]|%[0-9A-Fa-f]{2})+$'
            expected = 'zs'
            nonmatch = '%%oV'
            self.assertGenerates(raw, expected, nonmatch)



        def test_slugish(self):
            # https://digitalfortress.tech/js/top-15-commonly-used-regex/
            raw = r'^[a-z0-9]+(?:-[a-z0-9]+)*$'
            expected = 'x3-vs0-g-rw39'
            nonmatch = 'RO-\\B-OM'
            self.assertGenerates(raw, expected, nonmatch)

        @unittest.skip("Doesn't even compile in Python; 'look-behind requires fixed-width pattern'")
        def test_semverish(self):
            # https://rgxdb.com/r/40OZ1HN5
            # jesus christ
            raw = r'(?<=^[Vv]|^)(?:(?P<major>(?:0|[1-9](?:(?:0|[1-9])+)*))[.](?P<minor>(?:0|[1-9](?:(?:0|[1-9])+)*))[.](?P<patch>(?:0|[1-9](?:(?:0|[1-9])+)*))(?:-(?P<prerelease>(?:(?:(?:[A-Za-z]|-)(?:(?:(?:0|[1-9])|(?:[A-Za-z]|-))+)?|(?:(?:(?:0|[1-9])|(?:[A-Za-z]|-))+)(?:[A-Za-z]|-)(?:(?:(?:0|[1-9])|(?:[A-Za-z]|-))+)?)|(?:0|[1-9](?:(?:0|[1-9])+)*))(?:[.](?:(?:(?:[A-Za-z]|-)(?:(?:(?:0|[1-9])|(?:[A-Za-z]|-))+)?|(?:(?:(?:0|[1-9])|(?:[A-Za-z]|-))+)(?:[A-Za-z]|-)(?:(?:(?:0|[1-9])|(?:[A-Za-z]|-))+)?)|(?:0|[1-9](?:(?:0|[1-9])+)*)))*))?(?:[+](?P<build>(?:(?:(?:[A-Za-z]|-)(?:(?:(?:0|[1-9])|(?:[A-Za-z]|-))+)?|(?:(?:(?:0|[1-9])|(?:[A-Za-z]|-))+)(?:[A-Za-z]|-)(?:(?:(?:0|[1-9])|(?:[A-Za-z]|-))+)?)|(?:(?:0|[1-9])+))(?:[.](?:(?:(?:[A-Za-z]|-)(?:(?:(?:0|[1-9])|(?:[A-Za-z]|-))+)?|(?:(?:(?:0|[1-9])|(?:[A-Za-z]|-))+)(?:[A-Za-z]|-)(?:(?:(?:0|[1-9])|(?:[A-Za-z]|-))+)?)|(?:(?:0|[1-9])+)))*))?)$'
            self.assertGenerates(raw, '', '')

        def test_macaddressish(self):
            # https://rgxdb.com/r/4SBEN3OY
            raw = r'(?:[0-9A-Fa-f]{2}(?:([:-])|)[0-9A-Fa-f]{2})(?:(?(1)\1|\.)(?:[0-9A-Fa-f]{2}([:-]?)[0-9A-Fa-f]{2})){2}'
            expected = ''
            nonmatch = ''
            self.assertGenerates(raw, expected, nonmatch)

        def test_macaddressish2(self):
            # https://rgxdb.com/r/4QCDJ0QF
            raw = r'^(?:[0-9A-Fa-f]{2}([:-]?)[0-9A-Fa-f]{2})(?:(?:\1|\.)(?:[0-9A-Fa-f]{2}([:-]?)[0-9A-Fa-f]{2})){2}$'
            expected = ''
            nonmatch = ''
            self.assertGenerates(raw, expected, nonmatch)

        def test_single_quote_with_backslash_escape(self):
            # https://rgxdb.com/r/3URVPBPQ
            raw = r"'((?:\\.|[^\\'])*)'"
            expected = ''
            nonmatch = ''
            self.assertGenerates(raw, expected, nonmatch)

        @unittest.skip("bleh, getting 'EOF while scanning triple-quoted string literal' so ignore for now")
        def test_single_quote_with_quote_escape(self):
            # https://rgxdb.com/r/5E0R5TW5
            raw = r"'((?:''|[^'])*)'"
            expected = ""
            nonmatch = ''
            self.assertGenerates(raw, expected, nonmatch)

        @unittest.skip("unknown extension ?> at position 21")
        def test_single_or_double_quote_with_backslash_escape(self):
            # https://rgxdb.com/r/1VQRNCKQ
            raw = r"""(['"])(?:.*?)(?<!\\)(?>\\\\)*?\1"""
            expected = ''
            nonmatch = ''
            self.assertGenerates(raw, expected, nonmatch)

        def test_valid_utf8(self):
            # https://rgxdb.com/r/583NSZB2
            raw = r"""\A(?:
  [\x00-\x7F]                        # ASCII
| [\xC2-\xDF][\x80-\xBF]             # non-overlong 2-byte
|  \xE0[\xA0-\xBF][\x80-\xBF]        # excluding overlongs
| [\xE1-\xEC\xEE\xEF][\x80-\xBF]{2}  # straight 3-byte
|  \xED[\x80-\x9F][\x80-\xBF]        # excluding surrogates
|  \xF0[\x90-\xBF][\x80-\xBF]{2}     # planes 1-3
| [\xF1-\xF3][\x80-\xBF]{3}          # planes 4-15
|  \xF4[\x80-\x8F][\x80-\xBF]{2}     # plane 16
)*\z"""
            expected = ''
            nonmatch = ''
            self.assertGenerates(raw, expected, nonmatch)



    class TokenTests(CustomAsserts, unittest.TestCase):
        def test_range_is_subset_of_numeric(self):
            """ Numeric subranges return values from the rest of the numeric range.

            Given a 0-9 range, if it's actually a subset (e.g. 3-7) nonmatches
            should include values from 0-3 and 8-9, because those are a more
            specific thing which should not match, than just "another character"
            """
            range = NumericSubRange.from_chars('3', '7')
            match_output = range.generate()
            self.assertEqual(match_output, '4')
            nonmatches = (
                '2',
                '1',
                '1',
                '8',
                '1',
            )
            for i, nonmatch_expected in enumerate(nonmatches):
                nonmatch_output = range.generate_nonmatch()
                self.assertEqual(nonmatch_output, nonmatch_expected)

    # class GenerationTe1sts(unittest.TestCase):
    #     def assertGenerates(self, raw, expected):
    #         try:
    #             re.compile(raw)
    #         except re.error:
    #             self.fail(f"Invalid regex: {raw!r}")
    #         nodes, parser = parse(raw)
    #         output = "".join(chain.from_iterable([n.generate() for n in nodes]))
    #         self.assertSequenceEqual(output, expected)
    #
    #     def test_iso2_codes(self):
    #         """ISO-2 country code(ish); GB, US etc"""
    #         raw = r"[A-Z][A-Z]"
    #         expected = "AH"
    #         self.assertGenerates(raw, expected)
    #
    #     def test_repeats(self):
    #         raw = "^a{3}$"
    #         expected = "aaa"
    #         self.assertGenerates(raw, expected)
    #         raw = "^a{3,}$"
    #         expected = "aaaaaaaaa"
    #         self.assertGenerates(raw, expected)
    #         raw = "^a{3,10}$"
    #         expected = "aaaaaaaaaa"
    #         self.assertGenerates(raw, expected)
    #         raw = "^a{0,10}$"
    #         expected = "a"
    #         self.assertGenerates(raw, expected)
    #         maximum = int(sre_constants.MAXREPEAT) - 1
    #         raw = f"^a{{0,{maximum}}}$"
    #         expected = "a"
    #         self.assertGenerates(raw, expected)
    #         raw = f"^a{{1,{maximum}}}$"
    #         expected = "aaaaaa"
    #         self.assertGenerates(raw, expected)
    #         raw = f"^[^@]$"
    #         expected = "\U000961d8"
    #         self.assertGenerates(raw, expected)

    unittest.main(
        module=sys.modules[__name__],
        # testRunner=unittest.TextTestRunner(resultclass=CountingTestResult),
        verbosity=2,
        catchbreak=True,
        tb_locals=True,
        failfast=False,
        buffer=False,
    )

