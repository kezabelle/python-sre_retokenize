sre_retokenize
==============

:author: Keryn Knight
:version: 0.1.0

A scruffy sketch of something I wanted to exist but couldn't find.

Brief
-----

Even if you've worked with regular expressions (regex/regexp) for years, complex ones are quasi-inscrutible at a glance, and it can be hard to visually parse what it **should** match.

It'd be cool if you could see examples that match, and examples which don't match, right?

That's what this library attempts to do. It doesn't satisfy all of the possible regex operations yet (e.g: lookaheads, lookbehinds...) and the API is currently less than friendly, but hopefully I can fix those.

Example
-------
Given an input regex like::

    [0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}$

we can generate matches like::

    a431fad0-38ab-feBE-A6C3-A65EAdaAceE1

and non-matches like::

    nVvimULi-foYJ-TUuX-gSHp-WmywYiPnlyZW

currently both *matches* and *non-matches* are a little naive in their output, but that's fixable too.

Example API usage
-----------------

I did say this was a gross API at the moment. A better one will come::

    import re
    from itertools import chain
    from sre_retokenize import parse

    raw = r"^\d{4}/(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])$"
    regex = re.compile(raw)
    nodes, parser = parse(raw)

    match_output = "".join(
        chain.from_iterable([n.generate() for n in nodes])
    )
    nonmatch_output = "".join(
        chain.from_iterable([n.generate_nonmatch() for n in nodes])
    )

    assert regex.fullmatch(match_output) is not None

Alternatives
------------

Naturally, I've spotted some un-investigated alternatives since I started hacking at this:

- `xeger`_ is a "library to generate random strings from regular expressions."
- `sre_yield`_ tries to "efficiently generate all values that can match a given regular expression" and walks "the tree as constructed by ``sre_parse`` (same thing used internally by the ``re`` module)" ... which is what I ended up doing.

Research
--------

When trying to figure out how to build this, and the 2 levels of intermediate representation, here's some of the random semi-related URLs I found myself on, which are perhaps interesting:

- a git copy of `Backrefs re parser`_ which for some reason I cannot find the actual oiginal version of, right now??
- this web archive URL for the file `jpetkau1.py`_ by Jeff Petkau, which is I think probably part of the spark that allowed me to figure out how to leverage ``sre_parse`` to make this work.
- `sre_dump`_ which was also helpful in establishing the ``sre_parse`` path as the way to go.
- some examples using  `re.Scanner`_ which otherwise appeared to be undocumented in the python library
- additional notes about `using a scanner`_ + ``sre_parse`` to do some legwork, by Armin Ronacher

.. _xeger: https://github.com/crdoconnor/xeger
.. _Backrefs re parser: https://github.com/ontheroadjp/dotfiles/blob/57549edcabd9cd3a5e5f9715657d37e482fe83ea/mac_osx/SublimeText3/Packages/backrefs/st3/backrefs/_bre_parse.py
.. _sre_yield: https://github.com/google/sre_yield
.. _jpetkau1.py: http://web.archive.org/web/20071024164712/http://www.uselesspython.com/jpetkau1.py
.. _sre_dump: http://www.dalkescientific.com/Python/sre_dump.html
.. _re.Scanner: https://www.programcreek.com/python/example/53972/re.Scanner
.. _using a scanner: https://lucumr.pocoo.org/2015/11/18/pythons-hidden-re-gems/
