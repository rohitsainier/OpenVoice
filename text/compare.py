import librosa
import numpy as np


def compare_audio(file1, file2):
    # Load audio files
    y1, sr1 = librosa.load(file1)
    y2, sr2 = librosa.load(file2)

    # Extract MFCC features
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2)

    # Pad or truncate MFCC feature vectors to ensure compatibility
    min_length = min(mfcc1.shape[1], mfcc2.shape[1])
    mfcc1 = mfcc1[:, :min_length]
    mfcc2 = mfcc2[:, :min_length]

    # Compute similarity (cosine similarity)
    similarity = np.dot(mfcc1.flatten(), mfcc2.flatten()) / \
        (np.linalg.norm(mfcc1) * np.linalg.norm(mfcc2))

    # Print similarity
    print("Similarity:", similarity)

    # Define a threshold (you may need to adjust this)
    threshold = 0.85

    # Compare similarity with threshold and return boolean with result
    if similarity > threshold:
        return (True, "Voices are similar with a similarity of " + str(similarity))
    else:
        return (False, "Voices are not similar because the similarity is " + str(similarity))
