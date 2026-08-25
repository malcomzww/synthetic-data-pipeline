"""The held-out eval set that contamination is measured against.

Hand-authored for this repository and released under its MIT licence. That is
a deliberate choice over using a slice of a real benchmark:

- **Licensing.** MMLU, GSM8K and friends carry their own terms, and vendoring
  a slice of one into a repo to use as a contamination target is exactly the
  redistribution those terms govern. The brief's constraint is public data
  only; text written here is unambiguously clean.
- **Contamination of the contamination test.** A real benchmark's items are
  already in every large pretraining corpus. Using one here would mean the
  "eval set" is a document the rest of the world has also memorised, which
  muddies what the detector is being scored on.

The cost is stated plainly in the results file: 24 hand-written items do not
have the n-gram profile of a real benchmark, so the false-positive rate
measured against this set does not transfer to one. What *does* transfer is
the detector's recall on verbatim spans, which is a property of n-gram
matching rather than of the particular text.

Items are deliberately long -- 25 to 40 words each -- because an eval item
shorter than the 13-token threshold cannot be planted as detectable
contamination at all, and a fixture that cannot exercise the detector is not a
fixture.
"""

from __future__ import annotations

EVAL_SET: tuple[str, ...] = (
    "The write ahead log guarantees durability by recording every mutation to stable "
    "storage before the change is applied to the primary data pages of the database engine.",
    "A bloom filter is a probabilistic data structure that answers set membership queries "
    "with no false negatives but a tunable rate of false positives determined by the number "
    "of independent hash functions it applies.",
    "Consistent hashing maps both keys and nodes onto the same circular keyspace so that "
    "adding or removing a single node relocates only a small fraction of the total keys "
    "rather than reshuffling the entire distribution.",
    "An inverted index stores for every distinct term a posting list enumerating the "
    "documents in which that term occurs together with the positions at which each "
    "occurrence appears within the document.",
    "A token bucket rate limiter accumulates tokens at a fixed refill rate up to a maximum "
    "burst capacity and admits an incoming request only when a token is available to be "
    "consumed on its behalf.",
    "A circuit breaker trips open after a configured number of consecutive failures and "
    "then rejects calls immediately for a cooldown period, allowing a single probe request "
    "through before deciding whether to close again.",
    "A merkle tree summarises a large collection of records as a single root hash such that "
    "any modification to any leaf changes the root, which allows two replicas to locate "
    "their differences in logarithmic time.",
    "A columnar store keeps the values of one attribute contiguous on disk so that an "
    "analytic scan reads only the columns a query references and compresses each column "
    "with a codec chosen for its data type.",
    "A lock free queue coordinates producers and consumers using atomic compare and swap "
    "operations rather than mutual exclusion, which removes the risk of a thread holding a "
    "lock while it is descheduled by the operating system.",
    "The least recently used eviction policy discards the entry whose most recent access is "
    "furthest in the past, which approximates optimal replacement whenever the access "
    "pattern exhibits temporal locality of reference.",
    "A vector clock assigns each process a counter and attaches the full vector to every "
    "message, so that two events can be compared to determine whether one causally precedes "
    "the other or whether they are concurrent.",
    "Leader election protocols such as Raft require a candidate to collect votes from a "
    "strict majority of the cluster before it may serve writes, which guarantees that two "
    "leaders cannot be elected in the same term.",
    "A b-tree index keeps its nodes wide and shallow so that a lookup touches few disk "
    "pages, and it rebalances on insertion by splitting a full node and promoting the median "
    "key into the parent node above it.",
    "Copy on write defers the cost of duplicating a buffer until the moment a writer "
    "actually modifies it, so that readers which never write share a single physical copy "
    "and pay no allocation cost at all.",
    "A connection pool amortises the expense of establishing a transport session by keeping "
    "idle connections open and handing them to callers on demand, subject to a ceiling that "
    "protects the server from overload.",
    "Speculative execution improves throughput by beginning work before it is known to be "
    "necessary and discarding the result if the prediction turns out to have been wrong, "
    "trading wasted cycles for reduced latency.",
    "A priority queue orders its elements by an explicit key rather than by insertion time, "
    "and a binary heap implements it with logarithmic insertion and extraction while using "
    "no storage beyond the elements themselves.",
    "Backpressure propagates the slowness of a downstream consumer to the upstream producer "
    "so that a queue between them cannot grow without bound and exhaust the memory of the "
    "process that hosts it.",
    "A bounded channel blocks or rejects a send when its buffer is full, which converts an "
    "invisible memory leak in an unbounded queue into an explicit and observable signal that "
    "the system is beyond its capacity.",
    "Idempotent request handling allows a client to retry safely after an ambiguous timeout, "
    "because the server recognises the repeated request identifier and returns the original "
    "response instead of performing the work twice.",
    "Quorum reads and writes overlap by construction whenever the read quorum plus the write "
    "quorum exceeds the total number of replicas, which is the condition that makes a "
    "strongly consistent read possible.",
    "A rate of exponential backoff with jitter spreads client retries across time so that a "
    "fleet recovering from a common outage does not synchronise into a thundering herd that "
    "immediately overwhelms the recovered service.",
    "Compaction in a log structured merge tree rewrites overlapping sorted runs into a "
    "single larger run, which reclaims the space held by superseded versions and bounds the "
    "number of files a read must consult.",
    "A geospatial index such as an R-tree groups nearby objects into minimum bounding "
    "rectangles so that a range query can discard an entire subtree whenever its bounding "
    "rectangle does not intersect the query window.",
)
