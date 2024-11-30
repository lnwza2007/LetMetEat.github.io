# Define the base
base = 32

# Helper function to convert a string to its base-32 equivalent
def calculate_base32_value(word, base):
    # Create a mapping for base 32 (A=10, ..., V=31 and 0-9)
    base32_map = {chr(i + 55): i for i in range(10, 32)}  # A=10 to V=31
    base32_map.update({str(i): i for i in range(10)})     # 0-9 = 0-9

    value = 0
    for i, char in enumerate(reversed(word)):
        value += base32_map[char] * (base ** i)  # Calculate based on position
    return value

# Define the words
words = {
    "DIGITAL": "DIGITAL",
    "ANALOG": "ANALOG",
    "ELECTRICAL": "ELECTRICAL",
    "ENGINEEER": "ENGINEEER",
    "KASETSART": "KASETSART",
    "K7S5IUKG3": "K7S5IUKG3"
}

# Calculate values for each word
values = {key: calculate_base32_value(word, base) for key, word in words.items()}

# Calculate the result using the given equation
result = (
    # (values["DIGITAL"] - values["ANALOG"]) 
    (values["ELECTRICAL"] - values["ENGINEEER"]) 
    # (values["KASETSART"] - values["K7S5IUKG3"])
)

# Display the values and the result
print("Values (Base 32):")
for key, value in values.items():
    print(f"{key}: {value}")
    
print("\nResult:")
print(result)
