"""
Contains two constant dictionaries for translating label from int to string and from string to int.
"""

dictionary = {chr(i): i - ord('a') for i in range(ord('a'), ord('z') + 1)}
dictionary['space'] = 26
dictionary['del'] = 27

reverse_dictionary = {v: k for k, v in dictionary.items()}