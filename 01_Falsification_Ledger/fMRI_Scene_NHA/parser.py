import torch
from models.grammar import UNIT_TO_IDX

class LeanParser:
    """
    Parses raw structured eDSL syntax strings into semantic indices.
    Example: "(detect anomaly)" -> Subject: 'anomaly', Predicate: 'detect'
    """
    def __init__(self):
        self.vocab_map = UNIT_TO_IDX

    def parse(self, edsl_strings: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parses a list of eDSL strings.
        Returns (sub_indices, pred_indices) as LongTensors.
        """
        sub_indices = []
        pred_indices = []
        
        for s in edsl_strings:
            # Clean string
            s_clean = s.strip().replace("(", "").replace(")", "").strip()
            parts = s_clean.split()
            
            if len(parts) >= 2:
                predicate = parts[0]
                subject = parts[1]
            elif len(parts) == 1:
                predicate = parts[0]
                subject = parts[0] # Fallback
            else:
                predicate = "hold"
                subject = "normal"
                
            p_idx = self.vocab_map.get(predicate, self.vocab_map.get("hold", 0))
            s_idx = self.vocab_map.get(subject, self.vocab_map.get("normal", 0))
            
            pred_indices.append(p_idx)
            sub_indices.append(s_idx)
            
        return torch.tensor(sub_indices, dtype=torch.long), torch.tensor(pred_indices, dtype=torch.long)
