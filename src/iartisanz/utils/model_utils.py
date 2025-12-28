import hashlib
import io

import xxhash


def calculate_file_hash(filepath):
    partial_hash = calculate_partial_file_hash(filepath)
    xxhash_hash = calculate_file_hash_xxhash(filepath)
    return f"{partial_hash}-{xxhash_hash}"


def calculate_file_hash_xxhash(filepath):
    hasher = xxhash.xxh64()
    with open(filepath, "rb") as file:
        while True:
            chunk = file.read(io.DEFAULT_BUFFER_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def calculate_partial_file_hash(filepath, chunk_size=io.DEFAULT_BUFFER_SIZE, offset=0):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as file:
        file.seek(offset)
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
            break  # Only read one chunk
    return hasher.hexdigest()
