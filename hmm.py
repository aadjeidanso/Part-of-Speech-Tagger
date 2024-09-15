import numpy as np
import nltk
from collections import defaultdict, Counter
from nltk.corpus import brown
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity


# Smoothing constant for unseen words and transitions
SMOOTHING_FACTOR = 1e-12

# Load the Brown corpus with Universal tagset
nltk.download('brown')
nltk.download('universal_tagset')
tagged_sentences = brown.tagged_sents(tagset='universal')

# Load GloVe embeddings
def load_glove_embeddings(filepath):
    embeddings_index = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coeffs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coeffs
    print(f"Loaded {len(embeddings_index)} word vectors from GloVe.")
    return embeddings_index

glove_filepath = 'C:\\Users\\adjei\\Downloads\\glove.6B\\glove.6B.100d.txt'  # Adjust the path
embeddings_index = load_glove_embeddings(glove_filepath)

# Initialize counters
trigram_transition_counts = defaultdict(Counter)
emission_counts = defaultdict(Counter)
tag_counts = Counter()

# Calculate emission and trigram transition counts
for sentence in tagged_sentences:
    previous_tags = ("<START>", "<START>")
    for word, tag in sentence:
        tag_counts[tag] += 1
        emission_counts[tag][word.lower()] += 1
        trigram_transition_counts[previous_tags][tag] += 1
        previous_tags = (previous_tags[1], tag)
    trigram_transition_counts[previous_tags]["<END>"] += 1

# Convert counts to probabilities
trigram_transition_probability = {tags: {next_tag: count / sum(next_tags.values())
                                         for next_tag, count in next_tags.items()}
                                  for tags, next_tags in trigram_transition_counts.items()}

def get_word_embedding(word, embeddings_index, embedding_dim=100):
    return embeddings_index.get(word, np.zeros(embedding_dim))

# Emission probability with GloVe embeddings
def calculate_emission_probability(word, tag, embeddings_index, embedding_dim=100):
    word_embedding = get_word_embedding(word, embeddings_index, embedding_dim)
    tag_embedding = get_word_embedding(tag.lower(), embeddings_index, embedding_dim)
    
    if np.count_nonzero(word_embedding) == 0 or np.count_nonzero(tag_embedding) == 0:
        return SMOOTHING_FACTOR  # Return a small probability if either embedding is missing
    
    # Cosine similarity for emission probability
    cosine_sim = sklearn_cosine_similarity(word_embedding.reshape(1, -1), tag_embedding.reshape(1, -1))
    return cosine_sim[0][0] + SMOOTHING_FACTOR

# Build emission probabilities using GloVe embeddings
emission_probability = {tag: {word: calculate_emission_probability(word, tag, embeddings_index)
                              for word in emission_counts[tag]}
                        for tag in tag_counts}

# Starting probabilities (frequency-based)
start_probability = {tag: count / sum(tag_counts.values()) for tag, count in tag_counts.items()}

# Set of all possible states (tags)
states = list(tag_counts.keys())

def viterbi_trigram(observations, states, start_p, trigram_trans_p, emit_p):
    V = [{}]
    path = {}

    # Initialize the base cases (t == 0)
    for state in states:
        V[0][state] = start_p.get(state, SMOOTHING_FACTOR) * emit_p.get(state, {}).get(observations[0], SMOOTHING_FACTOR)
        path[state] = [state]

    # Run Viterbi for t > 0
    for t in range(1, len(observations)):
        V.append({})
        new_path = {}

        for current_state in states:
            max_prob, best_prev_state = max(
                (
                    V[t-1][prev_state] *
                    trigram_trans_p.get(
                        (path[prev_state][-2] if len(path[prev_state]) > 1 else path[prev_state][-1],
                         path[prev_state][-1]),
                        {}
                    ).get(current_state, SMOOTHING_FACTOR) *
                    emit_p.get(current_state, {}).get(observations[t], SMOOTHING_FACTOR),
                    prev_state
                )
                for prev_state in states
            )

            V[t][current_state] = max_prob
            new_path[current_state] = path[best_prev_state] + [current_state]

        path = new_path

    # Find the most probable final state
    max_prob, final_state = max((V[-1][state], state) for state in states)

    return path[final_state]

def post_process_tags(sentence, pos_tags):
    words = sentence.split()
    for i, (word, tag) in enumerate(zip(words, pos_tags)):
        if word.lower() in ['a', 'an', 'the'] and tag != 'DET':
            pos_tags[i] = 'DET'
        elif word.lower() in ['and', 'but', 'or', 'yet'] and tag != 'CONJ':
            pos_tags[i] = 'CONJ'
        elif word in ['.', ',', '!', '?'] and tag != 'PUNCT':
            pos_tags[i] = 'PUNCT'
        elif tag == "X":  # Handle unexpected "X" tags
            if word.lower() in embeddings_index:  # Word exists in GloVe
                pos_tags[i] = 'VERB' if i != 0 else 'NOUN'  # Assume verbs are more likely, except at the start
            else:
                pos_tags[i] = 'NOUN'  # Default to NOUN for unknown words
        elif word.endswith('ing') and tag != 'VERB':
            pos_tags[i] = 'VERB'
        elif word.endswith('ed') and tag != 'VERB':
            pos_tags[i] = 'VERB'
        elif word.endswith('y') and tag == 'VERB':
            pos_tags[i] = 'ADJ'
    return pos_tags

def predict_pos(sentence):
    words = sentence.lower().split()
    pos_tags = viterbi_trigram(words, states, start_probability, trigram_transition_probability, emission_probability)
    pos_tags = post_process_tags(sentence, pos_tags)
    return pos_tags



