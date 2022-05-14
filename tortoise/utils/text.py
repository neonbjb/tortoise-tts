import re


def split_and_recombine_text(text, desired_length=200, max_length=300):
    """Split text it into chunks of a desired length trying to keep sentences intact."""
    # normalize text, remove redundant whitespace and convert non-ascii quotes to ascii
    text = re.sub(r'\n\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[“”]', '"', text)

    rv = []
    in_quote = False
    current = ""
    split_pos = []
    pos = -1

    def seek(delta):
        nonlocal pos, in_quote, text
        is_neg = delta < 0
        for _ in range(abs(delta)):
            if is_neg:
                pos -= 1
            else:
                pos += 1
            if text[pos] == '"':
                in_quote = not in_quote
        return text[pos], text[pos+1] if pos < len(text)-1 else ""

    def commit():
        nonlocal rv, current, split_pos
        rv.append(current)
        current = ""
        split_pos = []

    while pos < len(text) - 1:
        c, next_c = seek(1)
        current += c
        # do we need to force a split?
        if len(current) >= max_length:
            if len(split_pos) > 0 and len(current) > (desired_length / 2):
                # we have at least one sentence and we are over half the desired length, seek back to the last split
                d = pos - split_pos[-1]
                seek(-d)
                current = current[:-d]
            else:
                # no full sentences, seek back until we are not in the middle of a word and split there
                while c not in '!?.\n ' and pos > 0 and len(current) > desired_length:
                    c, _ = seek(-1)
                    current = current[:-1]
            commit()
        # check for sentence boundaries
        elif not in_quote and (c in '!?\n' or (c == '.' and next_c in '\n ')):
            split_pos.append(pos)
            if len(current) >= desired_length:
                commit()
    rv.append(current)

    # clean up
    rv = [s.strip() for s in rv]
    rv = [s for s in rv if len(s) > 0]

    return rv


if __name__ == '__main__':
    import unittest

    class Test(unittest.TestCase):
        def test_split_and_recombine_text(self):
            text = """
            This is a sample sentence.
            This is another sample sentence.
            This is a longer sample sentence that should force a split inthemiddlebutinotinthislongword.
            "Don't split my quote... please"
            """
            self.assertEqual(split_and_recombine_text(text, desired_length=20, max_length=40),
                             ['This is a sample sentence.',
                              'This is another sample sentence.',
                              'This is a longer sample sentence that',
                              'should force a split',
                              'inthemiddlebutinotinthislongword.',
                              '"Don\'t split my quote... please"'])

    unittest.main()
