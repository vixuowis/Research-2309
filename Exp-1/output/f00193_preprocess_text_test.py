from f00193_preprocess_text import *
import unittest


class TestPreprocessText(unittest.TestCase):
    def test_preprocess_text(self):
        text = "I love sci-fi and am willing to put up with a lot."
        expected_output = "love sci fi willing put lot"
        self.assertEqual(preprocess_text(text), expected_output)

        text = "Sci-fi movies/TV are usually underfunded, under-appreciated and misunderstood."
        expected_output = "sci fi movies tv usually underfunded under appreciated misunderstood"
        self.assertEqual(preprocess_text(text), expected_output)

        text = "I tried to like this, I really did, but it is to good TV sci-fi as Babylon 5 is to Star Trek (the original)."
        expected_output = "tried like really good tv sci fi babylon star trek original"
        self.assertEqual(preprocess_text(text), expected_output)

        text = "Silly prosthetics, cheap cardboard sets, stilted dialogues, CG that doesn't match the background, and painfully one-dimensional characters cannot be overcome with a 'sci-fi' setting."
        expected_output = "silly prosthetics cheap cardboard sets stilted dialogues cg match background painfully one dimensional characters cannot overcome sci fi setting"
        self.assertEqual(preprocess_text(text), expected_output)

        text = "While US viewers might like emotion and character development, sci-fi is a genre that does not take itself seriously (cf. Star Trek)."
        expected_output = "us viewers might like emotion character development sci fi genre take seriously cf star trek"
        self.assertEqual(preprocess_text(text), expected_output)


if __name__ == '__main__':
    unittest.main()
