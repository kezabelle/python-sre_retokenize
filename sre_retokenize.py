import sre_parse
import sre_constants
import sre_compile
from collections import namedtuple
from typing import List

pat = r'^o[^a-z][a-zA-Z]n[^ABC]p*p.*t[^a]i.{1,3}o[0-9]{0, 10}n.+?al/(?P<arg1>\\d+)/✂️/(?:(?P<arg2>[a-b])/)?/([0-9]+?)/(?P<arg3>\\d+)/(?:(?P<arg4>[c-d])/)?$'
print(pat)


tree = sre_parse.parse(pat)
# tree.dump()

groupdict = {}
for k, v in tree.state.groupdict.items():
    groupdict[v] = k



class Token:
    __slots__ = ()

    def __str__(self):
        pass

    def __repr__(self):
        value = str(self)
        return f'<{self.__class__.__name__!s} {value!r}>'

    def pretty(self):
        pass

    def simplify(self):
        pass


class Literal(Token):
    __slots__ = ('value', )

    def __new__(cls, value, position):
        instance = super().__new__(cls)
        instance.value = value
        return instance

    def __str__(self):
        return self.value

    pretty = __str__
    simplify = __str__

class NegatedLiteral(Literal):
    __slots__ = ()
    def __new__(cls, value, position):
        instance = super().__new__(cls, value, position)
        return instance

    def __str__(self):
        return f'^{self.value}'


class Range(Token):
    __slots__ = ('start', 'end')
    def __new__(cls, start, end, position):
        instance = super().__new__(cls)
        instance.start = chr(start)
        instance.end = chr(end)
        return instance

    def __str__(self):
        return f'{self.start}-{self.end}'

    pretty = __str__
    simplify = __str__


class In(Token):
    __slots__ = ('value',)
    def __new__(cls, value, position):
        instance = super().__new__(cls)
        instance.value = value
        return instance

    def __str__(self):
        return f'[{self.value!s}]'

# Literal = namedtuple("Literal", ("value", "position"))
# NegatedLiteral = namedtuple("NegatedLiteral", ("value", "position"))
LiteralGroup = namedtuple("LiteralGroup", ("value", "start", "end"))
Negated = namedtuple("Negated", ("value"))
# Range = namedtuple("Range", ("start", "end", "position"))
NegatedRange = namedtuple("Range", ("start", "end", "position"))
Beginning = namedtuple("Beginning", ('value', 'position'))
End = namedtuple("End", ('value', 'position'))
Repeat = namedtuple("Repeat", ("start", "end", "value", "position"))
# In = namedtuple("In", ("value", "position"))
# MaxRepeat = namedtuple("MaxRepeat", ("start", "end", "value", "position"))

Position = namedtuple('Position', ('start', 'by', 'end'))


class Reparser:
    def __init__(self, pattern: sre_parse.SubPattern):
        self.pattern = pattern
        self.positions = [0]

    @property
    def start_position(self):
        return self.positions[-1]

    def parse(self):
        from pprint import pprint
        handled_nodes = []
        handled_positions = []
        handled_reprs = []
        pprint(self.pattern)
        self._continue_parsing(self.pattern, handled_nodes, handled_positions, handled_reprs)
        pprint(handled_nodes)
        return

    def _continue_parsing(self, nodes, handled_nodes: List, handled_positions: List, handled_reprs: List):
        for node in nodes:
            op, av = node
            # print(node)
            if op is sre_constants.AT:
                parsed_node, positions, repr = self._at(op, av)
                handled_nodes.append(parsed_node)
                handled_positions.append(positions)
                handled_reprs.append(repr)
                self.positions.append(positions.end)
            elif op is sre_constants.NOT_LITERAL:
                parsed_node, positions, repr = self._not_literal(op, av)
                handled_nodes.append(parsed_node)
                handled_positions.append(positions)
                handled_reprs.append(repr)
                self.positions.append(positions.end)
            elif op is sre_constants.LITERAL:
                parsed_node, positions, repr = self._literal(op, av)
                handled_nodes.append(parsed_node)
                handled_positions.append(positions)
                handled_reprs.append(repr)
                self.positions.append(positions.end)
            elif op is sre_constants.MIN_REPEAT:
                parsed_node, positions, repr = self._min_repeat(op, av, handled_nodes, handled_positions, handled_reprs)
                handled_nodes.append(parsed_node)
                handled_positions.append(positions)
                handled_reprs.append(repr)
                # self.positions.append(positions.end)
            elif op is sre_constants.MAX_REPEAT:
                parsed_node, positions, repr = self._max_repeat(op, av, handled_nodes, handled_positions, handled_reprs)
                handled_nodes.append(parsed_node)
                handled_positions.append(positions)
                handled_reprs.append(repr)
                # self.positions.append(positions.end)
            elif op is sre_constants.SUBPATTERN:
                parsed_node, positions, repr = self._subpattern(op, av, handled_nodes, handled_positions, handled_reprs)
                handled_nodes.append(parsed_node)
                handled_positions.append(positions)
                handled_reprs.append(repr)
            elif op is sre_constants.ANY:
                self._any(op, av)
            elif op is sre_constants.IN:
                self._in(op, av, handled_nodes, handled_positions, handled_reprs)
            elif op is sre_constants.RANGE:
                self._range(op, av, handled_nodes, handled_positions, handled_reprs)
            else:
                print(op)


        # handle promotion of literals into literalgroups

        # from pprint import pprint
        # pprint(handled_nodes)
        # pprint(handled_positions)
        # pprint(handled_reprs)
        # pprint(self.positions)

    def _any(self, op, av):
        # print(av)
        return (None, None, None)

    def _range(self, op, av, handled_nodes: List, handled_positions: List, handled_reprs: List):
        first, last = av
        span = Range(first, last, None)
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

    def _in(self, op, av, handled_nodes: List, handled_positions: List, handled_reprs: List):
        negated = av[0][0] is sre_constants.NEGATE
        if negated:
            av = av[1:]
        some_stuff = self._continue_parsing(av, handled_nodes, handled_positions, handled_reprs)
        node = In(some_stuff, None)
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

    def _subpattern(self, op, av, handled_nodes: List, handled_positions: List, handled_reprs: List):
        group_number, x, y, subpattern = av
        some_stuff = self._continue_parsing(subpattern, handled_nodes, handled_positions, handled_reprs)
        return (None, None, None)

    def _min_repeat(self, op, av, handled_nodes: List, handled_positions: List, handled_reprs: List):
        min_num, max_num, subpattern = av
        some_stuff = self._continue_parsing(subpattern, handled_nodes, handled_positions, handled_reprs)
        Repeat(int(min_num), int(max_num), some_stuff, 1)
        return (None, None, None)

    def _max_repeat(self, op, av, handled_nodes: List, handled_positions: List, handled_reprs: List):
        min_num, max_num, subpattern = av
        some_stuff = self._continue_parsing(subpattern, handled_nodes, handled_positions, handled_reprs)
        Repeat(int(min_num), int(max_num), some_stuff, 1)
        return (None, None, None)

    def _at(self, op, av):
        if av is sre_constants.AT_BEGINNING:
            return self._at_beginning(op, av)
        elif av is sre_constants.AT_END:
            return self._at_end(op, av)

    def _at_beginning(self, op, av):
        advance_from = self.start_position
        advance_by = 1
        advance_to = advance_from + advance_by
        bit = '^'
        return (
            Beginning('^', advance_from),
            Position(advance_from, advance_by, advance_to),
            bit,
        )

    def _at_end(self, op, av):
        advance_from = self.start_position
        advance_by = 1
        advance_to = advance_from + advance_by
        bit = '$'
        return (
            End('$', advance_from),
            Position(advance_from, advance_by, advance_to),
            bit,
        )
    def _not_literal(self, op, av):
        bit = chr(av)
        bit_length = len(bit)
        # [^] + character length
        advance_from = self.start_position
        advance_by = 3 + bit_length
        advance_to = advance_from + advance_by
        return (
            NegatedLiteral(bit, advance_from),
            Position(advance_from, advance_by, advance_to),
            f'[^{bit}]',
        )

    def _literal(self, op, av):
        bit = chr(av)
        bit_length = len(bit)
        advance_from = self.start_position
        advance_by = bit_length
        advance_to = advance_from + advance_by
        return (
            Literal(bit, advance_from),
            Position(advance_from, advance_by, advance_to),
            bit,
        )


def _dump(parsed_tree: sre_parse.SubPattern, groups: dict):
    dumper = Reparser()
    positions = [0]
    chars = []
    treebits = []
    segments = []
    for node in parsed_tree:
        op, av = node
        if op is sre_constants.LITERAL:
            byte = chr(av)
            position = positions[-1] + len(byte)
            chars.append(byte)
            positions.append(position)
            segments.append(Literal(byte, position))
        elif op is sre_constants.NOT_LITERAL:
            dumper.not_literal(op, av)
            chars.append('^')
            position = positions[-1] + 1
            positions.append(position)

            byte = chr(av)
            position = positions[-1] + len(byte)
            chars.append(byte)
            positions.append(position)
            segments.append(Negated(Literal(byte, position)))
        elif op is sre_constants.IN:
            negated = av[0][0] is sre_constants.NEGATE
            chars.append('[')
            position = positions[-1] + 1
            positions.append(position)

            if negated:
                av.pop(0)
                chars.append('^')
                position = positions[-1] + 1
                positions.append(position)
            for i, j in av:
                print(i, j)
                if i is sre_constants.LITERAL:
                    byte = chr(j)
                    position = positions[-1] + len(byte)
                    chars.append(byte)
                    positions.append(position)
                    segments.append(Negated(Literal(byte, position)))
                elif i is sre_constants.RANGE:
                    first = chr(j[0])
                    last = chr(j[1])

                    start_position = positions[-1] + len(first)
                    chars.append(first)
                    positions.append(start_position)

                    position = positions[-1] + len(first)
                    chars.append('-')
                    positions.append(position)

                    end_position = positions[-1] + len(last)
                    chars.append(last)
                    positions.append(end_position)

                    if negated:
                        segments.append(Negated(Range(first, last, (start_position, end_position))))
                    else:
                        segments.append(Range(first, last, (start_position, end_position)))

                else:
                    import pdb; pdb.set_trace()
            chars.append(']')
            position = positions[-1] + 1
            positions.append(position)
        # elif op in (sre_constants.MAX_REPEAT, sre_constants.MIN_REPEAT):
        #     i, j, subtree = av
        #     # print(subtree)
        #     s, suboffsets = _dump(subtree, groupdict)
        # elif op == sre_constants.SUBPATTERN:
        #     # print(op, av)
        #     # # groupnum, subtree = av
        #     groupnum, add_flags, del_flags, subtree = av
        #     # print(subtree)
        #     s, suboffsets = _dump(subtree, groupdict)
            # if groupnum in groupdict:
            #     name = groupdict[groupnum]
    # print(positions)
    # print(chars)
    print(segments)
    print("".join(chars))
    return treebits, chars

# _dump(tree, groupdict)

dumpee = Reparser(tree)
dumpee.parse()

