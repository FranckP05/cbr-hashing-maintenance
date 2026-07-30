import time


def time_index_build(index_factory, vectors):
    index = index_factory()
    start = time.perf_counter()
    index.build(vectors)
    elapsed = time.perf_counter() - start
    return index, elapsed


def time_query(index, query_vector, k):
    start = time.perf_counter()
    result_indices, result_distances = index.query(query_vector, k)
    elapsed = time.perf_counter() - start
    return result_indices, result_distances, elapsed
