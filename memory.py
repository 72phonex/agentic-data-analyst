import os, json, time

MEMORY_FILE = "memory.json"

def load_memories(filepath: str = MEMORY_FILE) -> list:
    """Load business rules and definitions from a small JSON file."""
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_memory(text: str, filepath: str = MEMORY_FILE) -> dict:
    """Save a new rule/definition with a unique timestamp ID."""
    clean_text = text.strip()
    memories = load_memories(filepath)
    new_item = {"id": int(time.time() * 1000), "text": clean_text}
    memories.append(new_item)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(memories, f, indent=2)
    return new_item

def delete_memory(memory_id: int, filepath: str = MEMORY_FILE) -> bool:
    """Remove a rule by ID and update memory.json."""
    memories = load_memories(filepath)
    filtered = [m for m in memories if m.get("id") != memory_id]
    if len(filtered) != len(memories):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(filtered, f, indent=2)
        return True
    return False

def find_relevant_memories(question: str, filepath: str = MEMORY_FILE) -> list:
    """Return memory strings that share at least one keyword (length > 3) with the question."""
    memories = load_memories(filepath)
    if not memories:
        return []
    # Extract keywords (>3 chars) from question, ignoring common punctuation
    words = {w.strip(".,?!:;'\"").lower() for w in question.split() if len(w.strip(".,?!:;'\"")) > 3}
    stopwords = {"what", "when", "where", "which", "whose", "show", "tell", "from", "with", "about", "dataset"}
    keywords = words - stopwords
    if not keywords:
        return []

    relevant = []
    for item in memories:
        text = item.get("text", "")
        text_lower = text.lower()
        if any(kw in text_lower for kw in keywords):
            relevant.append(text)
    return relevant
