
Open In Colab

import re

# Match a pattern at the beginning of the string
pattern1 = re.compile(r'abc')
result1 = pattern1.match('abcdef')
if result1:
    print("Match found:", result1.group())

"""
Metacharacters: Special characters with a specific meaning in regular expressions.
Some common metacharacters include:
. (dot): Matches any character except a newline.
*: Matches zero or more occurrences of the preceding character or group.
+: Matches one or more occurrences of the preceding character or group.
?: Matches zero or one occurrence of the preceding character or group.
^: Anchors the match at the beginning of the string.
$: Anchors the match at the end of the string.

Character Classes: A set of characters enclosed in square brackets [].
It matches any single character from the set. For example, [aeiou] matches any vowel.
"""

# . matches any character
pattern2 = re.compile(r'.')
result2 = pattern2.search('Hello')
if result2:
    print("Character found:", result2.group())

# [aeiou] matches any vowel
pattern3 = re.compile(r'[aeiou]')
result3 = pattern3.search('Hello')
if result3:
    print("Vowel found:", result3.group())

"""
Quantifiers: Specify the number of occurrences of a character or group.
{n}: Matches exactly n occurrences.
{n,}: Matches n or more occurrences.
{n,m}: Matches between n and m occurrences.
"""

# Matches SSN format like 123-45-6789
pattern4 = re.compile(r'\d{3}-\d{2}-\d{4}')
result4 = pattern4.match('123-45-6789')
if result4:
    print("Social Security Number:", result4.group())

# Matches a literal dot using \.
pattern5 = re.compile(r'\.')
result5 = pattern5.search('www.example.com')
if result5:
    print("Dot found:", result5.group())

# Extracts day/month/year using groups
pattern6 = re.compile(r'(\d{2})/(\d{2})/(\d{4})')
result6 = pattern6.match('01/09/2024')
if result6:
    print("Day:", result6.group(1))
    print("Month:", result6.group(2))
    print("Year:", result6.group(3))

"""
Alternation: The pipe | symbol is used for alternation,
allowing you to match either the pattern on the left or the one on the right.
Example: cat|dog - Matches either "cat" or "dog."
"""

# Match either 'cat' or 'dog'
pattern7 = re.compile(r'cat|dog')
result7 = pattern7.search('I have a cat and a dog')
if result7:
    print("Animal found:", result7.group())

     
Match found: abc
Character found: H
Vowel found: e
Social Security Number: 123-45-6789
Dot found: .
Day: 01
Month: 09
Year: 2024
Animal found: cat
