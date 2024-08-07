import regex
import re

from difflib import SequenceMatcher


def find_word_offsets(sentence, words):
    # Initialize a list to store the results
    offsets = []

    # Initialize the start search position
    search_pos = 0

    # Loop through each word to find its position in the sentence
    for word in words:
        # Find the position of the word in the sentence starting from search_pos
        word_pos = sentence.find(word, search_pos)

        # If the word is found, append its start and end positions to the offsets list
        if word_pos != -1:
            offsets.append((word_pos, word_pos + len(word)))

            # Update the search position to just after the current word's position
            search_pos = word_pos + len(word)
        else:
            # If word is not found, append None or an indicator
            offsets.append(None)

    return offsets


def find_best_match(sentence, phrase, claimed_ranges):
    # Adjust the regex to consider word boundaries. The \b ensures that we match whole words
    # but only where it makes sense based on the phrase itself.
    pattern = (
        r"\b%s\b" % regex.escape(phrase)
        if phrase[0].isalnum() and phrase[-1].isalnum()
        else r"%s" % regex.escape(phrase)
    )
    matches = regex.finditer(f"({pattern}){{e<=3}}", sentence, overlapped=True)
    best_match = None
    highest_ratio = 0.0

    for match in matches:
        start, end = match.span()
        # Exclude matches that overlap with claimed ranges
        if not any(
            start < cr_end and end > cr_start for cr_start, cr_end in claimed_ranges
        ):
            match_ratio = SequenceMatcher(None, match.group(), phrase).ratio()
            if match_ratio > highest_ratio:
                highest_ratio = match_ratio
                best_match = match

    return best_match


def find_phrase_offsets_fuzzy(sentence, phrases):
    results = []
    claimed_ranges = []
    for phrase in phrases:
        match = find_best_match(sentence, phrase, claimed_ranges)
        if match:
            start, end = match.span()
            # Claim this range
            claimed_ranges.append((start, end))
            results.append((match.group(), start, end))
    return results


def parse_kv(kv: str) -> dict:
    """
    Parses a string of key-value pairs separated by commas and ensures proper format.
    Raises ValueError if the format is incorrect.
    """
    kv_dict = {}
    pairs = kv.split(",")
    for item in pairs:
        if "=" not in item:
            raise ValueError("Each key-value pair must contain an '='.")
        key, value = item.split("=", 1)  # Split only on the first '=', allowing for '=' in values
        if not key.strip() or not value.strip():
            raise ValueError("Both key and value must be non-empty and not just whitespace.")
        kv_dict[key.strip()] = value.strip()
    return kv_dict


def get_prf(gold_tags: list, predicted_tags: list):
    # convert to sets
    gold_tags_set = set(gold_tags)
    predicted_tags_set = set(predicted_tags)
    # calculate true positives
    true_positives = len(gold_tags_set.intersection(predicted_tags_set))
    # calculate precision
    precision = true_positives / len(predicted_tags_set) if len(predicted_tags_set) > 0 else 0
    # calculate recall
    recall = true_positives / len(gold_tags_set) if len(gold_tags_set) > 0 else 0
    # calculate f1
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}
