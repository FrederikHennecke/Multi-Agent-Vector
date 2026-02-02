import json
from collections import defaultdict
import re
import matplotlib.pyplot as plt


def read_jsonl(filepath):
    """Reads a JSONL or incorrectly formatted summary file and returns a list of dictionaries."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        buffer = {}
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try parsing as JSON
            try:
                obj = json.loads(line)
                data.append(obj)
                continue
            except json.JSONDecodeError:
                pass

            # Fallback: parse custom format
            if line.startswith("ID:"):
                if buffer:
                    data.append(buffer)
                    buffer = {}
                buffer["id"] = int(line.replace("ID:", "").strip())
            elif line.startswith("Condition:"):
                buffer["condition"] = line.replace("Condition:", "").strip()
            elif line.startswith("Question:"):
                buffer["question"] = line.replace("Question:", "").strip()
            elif line.startswith("Final Answer:"):
                buffer["final"] = line.replace("Final Answer:", "").strip()
            elif line.startswith("Correct:"):
                value = line.replace("Correct:", "").strip()
                buffer["correct"] = value.lower() == "true"
            elif line.startswith("Verifier:"):
                buffer["ver_out"] = buffer.get("ver_out", "") + " " + line.replace("Verifier:", "").strip()
            elif line.startswith("-"):
                continue
        if buffer:
            data.append(buffer)
    return data


def read_gold(filepath):
    """Reads the gold answers from a JSONL training file and returns a dict mapping question -> gold."""
    gold = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                question = obj.get("input")
                label = obj.get("label", "")
                match = re.search(r"ANSWER\s*:\s*([A-J])\b", label, flags=re.IGNORECASE)
                if match:
                    gold_answer = match.group(1).strip()
                    gold[question] = gold_answer
    return gold


def merge_with_gold(entries, gold_dict):
    for entry in entries:
        q = entry.get("question")
        if q in gold_dict:
            entry["gold"] = gold_dict[q]
        else:
            entry["gold"] = None
    return entries


def summarize_entries(entries):
    for entry in entries:
        print("-" * 50)
        print(f"ID: {entry.get('id')}")
        print(f"Condition: {entry.get('condition')}")
        print(f"Question: {entry.get('question')}")
        print(f"Final Answer: {entry.get('final')}")
        print(f"Gold Answer: {entry.get('gold')}")
        print(f"Correct: {entry.get('correct')}")


def count_correct_by_condition(entries):
    counts = defaultdict(int)
    for entry in entries:
        if entry.get("correct"):
            condition = entry.get("condition", "unknown")
            counts[condition] += 1
    print("\nCorrect Answers per Condition:")
    for condition, count in counts.items():
        print(f"{condition}: {count}")
    return counts


def plot_correct_by_condition(counts):
    plt.figure(figsize=(8, 5))
    plt.bar(counts.keys(), counts.values())
    plt.xlabel("Condition")
    plt.ylabel("Number of Correct Answers")
    plt.title("Correct Answers per Condition")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("correct.png")
    plt.show()


def find_conflicting_final_answers(entries):
    grouped = defaultdict(dict)
    conflicts = []
    for entry in entries:
        qid = entry.get("id")
        condition = entry.get("condition", "unknown")
        final_answer = str(entry.get("final"))
        if qid not in grouped:
            grouped[qid] = {condition: final_answer}
        else:
            grouped[qid][condition] = final_answer

    for qid, answers in grouped.items():
        if len(set(answers.values())) > 1:
            conflicts.append((qid, answers))

    print("\nIDs with conflicting final answers across conditions:")
    if conflicts:
        for qid, answers in conflicts:
            print(f"ID {qid}: {answers}")
    else:
        print("None")


def plot_conversation_lengths(entries):
    lengths = defaultdict(int)
    for entry in entries:
        condition = entry.get("condition", "unknown")
        # approximate conversation length: length of final answer string
        lengths[condition] += len(str(entry.get("final", "")))

    plt.figure(figsize=(8, 5))
    plt.bar(lengths.keys(), lengths.values())
    plt.xlabel("Condition")
    plt.ylabel("Total Length of Final Answers (chars)")
    plt.title("Conversation Length per Condition (approx)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("conversation_lengths.png")
    plt.show()

def plot_avg_incorrect_by_condition(entries):
    incorrect_counts = {'prompt': 72, 'vector': 83, 'lora': 500 }
    total_counts = {'prompt': 500, 'vector': 500, 'lora': 500 }

    avg_incorrect = {cond: incorrect_counts[cond] / total_counts[cond] for cond in total_counts}

    plt.figure(figsize=(8, 5))
    plt.bar(avg_incorrect.keys(), avg_incorrect.values())
    plt.xlabel("Condition")
    plt.ylabel("Average 'INCORRECT' Flags by Verifier")
    plt.title("Average Verifier INCORRECT Judgments per Condition")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("incorrect.png")
    plt.show()

def main():
    filepath = "results/comparative_results.jsonl"  # change to your filename
    gold_file = "experiments_8B/mmlu_pro:law/data/generator.jsonl"

    entries = read_jsonl(filepath)
    gold_dict = read_gold(gold_file)
    entries = merge_with_gold(entries, gold_dict)

    print(f"Loaded {len(entries)} entries.\n")
    summarize_entries(entries)
    count_correct_by_condition(entries)

    counts = count_correct_by_condition(entries)
    #plot_correct_by_condition(counts)
    #find_conflicting_final_answers(entries)
    #plot_conversation_lengths(entries)
    #plot_avg_incorrect_by_condition(entries)


if __name__ == "__main__":
    main()

"""
interesting:
time LORA: 1779.06s
time VECTOR: 1068.61s
time PROMPT: 470.09s
"""
