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
    end_pos = len(text) - 1

    def seek(delta):
        nonlocal pos, in_quote, current
        is_neg = delta < 0
        for _ in range(abs(delta)):
            if is_neg:
                pos -= 1
                current = current[:-1]
            else:
                pos += 1
                current += text[pos]
            if text[pos] == '"':
                in_quote = not in_quote
        return text[pos]

    def peek(delta):
        p = pos + delta
        return text[p] if p < end_pos and p >= 0 else ""

    def commit():
        nonlocal rv, current, split_pos
        rv.append(current)
        current = ""
        split_pos = []

    while pos < end_pos:
        c = seek(1)
        # do we need to force a split?
        if len(current) >= max_length:
            if len(split_pos) > 0 and len(current) > (desired_length / 2):
                # we have at least one sentence and we are over half the desired length, seek back to the last split
                d = pos - split_pos[-1]
                seek(-d)
            else:
                # no full sentences, seek back until we are not in the middle of a word and split there
                while c not in '!?.\n ' and pos > 0 and len(current) > desired_length:
                    c = seek(-1)
            commit()
        # check for sentence boundaries
        elif not in_quote and (c in '!?\n' or (c == '.' and peek(1) in '\n ')):
            # seek forward if we have consecutive boundary markers but still within the max length
            while pos < len(text) - 1 and len(current) < max_length and peek(1) in '!?.':
                c = seek(1)
            split_pos.append(pos)
            if len(current) >= desired_length:
                commit()
        # treat end of quote as a boundary if its followed by a space or newline
        elif in_quote and peek(1) == '"' and peek(2) in '\n ':
            seek(2)
            split_pos.append(pos)
    rv.append(current)

    # clean up, remove lines with only whitespace or punctuation
    rv = [s.strip() for s in rv]
    rv = [s for s in rv if len(s) > 0 and not re.match(r'^[\s\.,;:!?]*$', s)]

    return rv


if __name__ == '__main__':
    import os
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

        def test_split_and_recombine_text_2(self):
            text = """
            When you are really angry sometimes you use consecutive exclamation marks!!!!!! Is this a good thing to do?!?!?!
            I don't know but we should handle this situation..........................
            """
            self.assertEqual(split_and_recombine_text(text, desired_length=30, max_length=50),
                             ['When you are really angry sometimes you use',
                              'consecutive exclamation marks!!!!!!',
                              'Is this a good thing to do?!?!?!',
                              'I don\'t know but we should handle this situation.'])

        def test_split_and_recombine_text_3(self):
            text_src = os.path.join(os.path.dirname(__file__), '../data/riding_hood.txt')
            with open(text_src, 'r') as f:
                text = f.read()
            self.assertEqual(
                split_and_recombine_text(text),
                [
                    'Once upon a time there lived in a certain village a little country girl, the prettiest creature who was ever seen. Her mother was excessively fond of her; and her grandmother doted on her still more. This good woman had a little red riding hood made for her.',
                    'It suited the girl so extremely well that everybody called her Little Red Riding Hood. One day her mother, having made some cakes, said to her, "Go, my dear, and see how your grandmother is doing, for I hear she has been very ill. Take her a cake, and this little pot of butter."',
                    'Little Red Riding Hood set out immediately to go to her grandmother, who lived in another village. As she was going through the wood, she met with a wolf, who had a very great mind to eat her up, but he dared not, because of some woodcutters working nearby in the forest.',
                    'He asked her where she was going. The poor child, who did not know that it was dangerous to stay and talk to a wolf, said to him, "I am going to see my grandmother and carry her a cake and a little pot of butter from my mother." "Does she live far off?" said the wolf "Oh I say,"',
                    'answered Little Red Riding Hood; "it is beyond that mill you see there, at the first house in the village." "Well," said the wolf, "and I\'ll go and see her too. I\'ll go this way and go you that, and we shall see who will be there first."',
                    'The wolf ran as fast as he could, taking the shortest path, and the little girl took a roundabout way, entertaining herself by gathering nuts, running after butterflies, and gathering bouquets of little flowers.',
                    'It was not long before the wolf arrived at the old woman\'s house. He knocked at the door: tap, tap. "Who\'s there?" "Your grandchild, Little Red Riding Hood," replied the wolf, counterfeiting her voice; "who has brought you a cake and a little pot of butter sent you by mother."',
                    'The good grandmother, who was in bed, because she was somewhat ill, cried out, "Pull the bobbin, and the latch will go up."',
                    'The wolf pulled the bobbin, and the door opened, and then he immediately fell upon the good woman and ate her up in a moment, for it been more than three days since he had eaten.',
                    'He then shut the door and got into the grandmother\'s bed, expecting Little Red Riding Hood, who came some time afterwards and knocked at the door: tap, tap. "Who\'s there?"',
                    'Little Red Riding Hood, hearing the big voice of the wolf, was at first afraid; but believing her grandmother had a cold and was hoarse, answered, "It is your grandchild Little Red Riding Hood, who has brought you a cake and a little pot of butter mother sends you."',
                    'The wolf cried out to her, softening his voice as much as he could, "Pull the bobbin, and the latch will go up." Little Red Riding Hood pulled the bobbin, and the door opened.',
                    'The wolf, seeing her come in, said to her, hiding himself under the bedclothes, "Put the cake and the little pot of butter upon the stool, and come get into bed with me." Little Red Riding Hood took off her clothes and got into bed.',
                    'She was greatly amazed to see how her grandmother looked in her nightclothes, and said to her, "Grandmother, what big arms you have!" "All the better to hug you with, my dear." "Grandmother, what big legs you have!" "All the better to run with, my child." "Grandmother, what big ears you have!"',
                    '"All the better to hear with, my child." "Grandmother, what big eyes you have!" "All the better to see with, my child." "Grandmother, what big teeth you have got!" "All the better to eat you up with." And, saying these words, this wicked wolf fell upon Little Red Riding Hood, and ate her all up.',
                ]
            )

    unittest.main()
