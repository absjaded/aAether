"""
Data Gen: Synthetic eDSL Corpus for Aether-beta v2.0 Syntax Prior.
Generates valid expressions based on Grammar v0 rules.
"""
import json
import random
from pathlib import Path

# Grammar Elements
MODALITIES = ["spatial", "kinematic", "relational", "temporal"]
FACTS = ["proximity_breach", "id_confirmed", "path_clear", "anomaly_detected"]
VALENCES = ["hesitate", "abort"]

def generate_expression():
    mode = random.choice(["hold", "nominal", "escalation"])
    
    if mode == "hold":
        return "(hold)"
    
    modality = random.choice(MODALITIES)
    if mode == "nominal":
        return f"(nominal {modality})"
    
    # Escalation
    valence = random.choice(VALENCES)
    fact = random.choice(FACTS)
    return f"({valence} {modality} ({fact}))"

def main(count=10000, output_path="src/aether_lab/data/synthetic_edsl.jsonl"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {count} synthetic eDSL expressions...")
    
    with open(output_path, "w") as f:
        for i in range(count):
            expr = generate_expression()
            # Wrap in a format compatible with load_data
            record = {
                "scene_id": f"syn_{i:05d}",
                "edsl": expr,
                "metadata": {
                    "source": "synthetic_grammar_v0",
                    "domain_id": random.randint(0, 1)
                }
            }
            f.write(json.dumps(record) + "\n")
            
    print(f"Done -> {output_path}")

if __name__ == "__main__":
    main()
