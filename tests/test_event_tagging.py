# tests for events_synergy.event_tagging module

import pytest

from events_synergy.event_tagging.event_tagger import (
    event_tagger
)


class TestEventTagging:
    def test_event_tagger(self):
        # Define the input
        sentences = [
            "I like this sentence and hate this sentence and I like this thing",
            "The earthquake took 10 lives ."
        ]
        # Call the function to test
        result = event_tagger(sentences)
        # Define the expected output
        expected_output = [
            [
                ('like', (2, 6)),
                ('hate', (25, 29)),
                ('like', (50, 54))
            ],
            [
                ('earthquake', (4, 14)),
                ('took', (15, 19))
            ]
        ]
        # Compare the result with the expected output
        assert result == expected_output


# Run the test
if __name__ == "__main__":
    pytest.main(["-s", "test_event_tagging.py"])
