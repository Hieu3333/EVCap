from collections import Counter
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to extract objects and actions from a single caption
def extract_objects_actions(caption):
    doc = nlp(caption)
    objects = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN")]
    actions = [token.text for token in doc if token.pos_ == "VERB"]
    return objects, actions

# Sample retrieval process with 10 captions (assuming these captions are retrieved)
retrieved_captions = [
    "A group of women are trying to push the table to the corner of the room.",
    "The cat is sleeping on the mat.",
    "A dog runs across the park chasing a ball.",
    "A man is sitting on a chair reading a book.",
    "The table is covered with a red cloth.",
    "A cat is sitting on a windowsill.",
    "The dog jumps over the fence.",
    "A woman is painting a canvas in a studio.",
    "Children are playing with a ball in the garden.",
    "A person is eating a sandwich in the park."
]

# Initialize counters for objects and actions
object_counter = Counter()
action_counter = Counter()

# Process each caption to extract objects and actions
for caption in retrieved_captions:
    objects, actions = extract_objects_actions(caption)
    object_counter.update(objects)  # Count the frequency of objects
    action_counter.update(actions)   # Count the frequency of actions

# Function to get top-n terms based on frequency
def get_top_n_terms(counter, n):
    return [term for term, _ in counter.most_common(n)]

# Set the value of n (number of top terms to select)
n = 5

# Get the top-n frequent objects and actions
top_n_objects = get_top_n_terms(object_counter, n)
top_n_actions = get_top_n_terms(action_counter, n)

# Print results
print(f"Top-{n} Objects:", top_n_objects)
print(f"Top-{n} Actions:", top_n_actions)

