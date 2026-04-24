from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from peft import PeftModel
import re
import json
import numpy as np
from typing import Dict, List


_THEOREM_RE = re.compile(r"[a-z][a-z0-9_]*")  


class Evaluator:
    def __init__(self, model, tokenizer, num_evals=3,lora_weights=None):
        self.model = model
        self.tokenizer = tokenizer
        self.num_evals = num_evals
        if lora_weights:
            pass
    
    def evaluate(self, theorem_name,goal,cond_text ,new_cond) -> float:
        scores = []
        for i in range(self.num_evals):
            score = self._single_evaluation(theorem_name,goal,cond_text ,new_cond, eval_id=i)
            scores.append(score)
        avg_score = sum(scores) / len(scores)
        avg_score = avg_score / 10.0
        
        return avg_score
    
    
    
    def _single_evaluation(self, theorem_name,goal,cond_text ,new_cond, eval_id: int) -> float:

        prompt = self._build_evaluation_prompt(theorem_name,goal,cond_text ,new_cond, eval_id)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        )
        
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
           
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,   
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
             
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
       
        score = parse_score(response)
        
        return score
    
    def _build_evaluation_prompt(self, theorem_name,goal,cond_text ,new_cond, eval_id) -> str:
        
        prompt = f"""You are a strict value evaluator for geometric proof search.

        ## TASK
        Evaluate how useful it is to apply the given theorem at the current state.
        Assume invalid/illegal actions have already been filtered out by the symbolic solver.

        ## EVALUATION CRITERIA
        Score based on these factors (in order of importance):

        1. **Goal Proximity**: Does applying this theorem bring us closer to proving the goal?
        - Directly establishes goal conclusion → high score
        - Creates intermediate conditions needed for goal → medium-high score
        - No obvious connection to goal → low score

        2. **Information Gain**: Are the new facts non-trivial and useful?
        - Introduces new geometric relationships (e.g., parallel, congruent, cyclic, collinear) → positive
        - Merely restates known facts or produces trivial tautologies → negative

        3. **Enabling Power**: Does this step unlock important follow-up theorems?
        - Enables angle chasing, similar triangles, power of a point, etc. → positive
        - Dead-end with no natural continuation → negative

        4. **Redundancy Penalty**: Avoid rewarding steps that:
        - Derive already-known facts
        - Apply symmetric/equivalent theorems redundantly

        ---

        ## INPUT

        **[Goal]**
        {goal}

        **[Current Known Facts]**
        {cond_text}

        **[Action: Theorem Applied]**
        {theorem_name}

        **[New Facts Derived]**
        {new_cond}

        ---

        ## SCORING GUIDE
        | Score | Meaning |
        |-------|---------|
        | 0-2   | No new facts, redundant, or irrelevant to goal |
        | 3-4   | Minor new facts, but weak/indirect relevance |
        | 5-6   | Moderate progress; new facts may help indirectly |
        | 7-8   | Strong progress; enables key structures likely needed |
        | 9-10  | Directly proves goal or makes it reachable in 1-2 steps |

        ---

        ## OUTPUT FORMAT
        Return ONLY a valid JSON object with an integer score:
        {{"score": <integer 0-10>}}
        """


        if eval_id > 0: 
            prompt += f"\n (evaluation {eval_id + 1}th) "   
        
        return prompt
    
    def _parse_conditions_from_state(self, state: str) -> str:
      
        if "Conditions:" in state:
            conditions = state.split("Conditions:")[1].strip()
          
            if conditions.startswith("[") and conditions.endswith("]"):
                cond_list = eval(conditions) 
                formatted = "\n".join(f"  {i+1}. {cond}" for i, cond in enumerate(cond_list[:10]))
                if len(cond_list) > 10:
                    formatted += f"\n  ...  {len(cond_list)-10} conditions"
                return formatted
        return state
    
    def _parse_and_normalize_score(self, response: str) -> float:
        
        numbers = re.findall(r'\d+\.?\d*', response)
        
        if numbers:
            score = float(numbers[0])  
            if 0 <= score <= 10:
                return score 
           
            elif 0 <= score <= 100:
                return score / 10.0
          
            elif 0 <= score <= 1:
                return score
            else:
            
                return 0
        return 0




def parse_score(response: str) -> int:
   
    json_block = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_block:
        try:
            data = json.loads(json_block.group(1))
            return int(data["score"])
        except:
            pass

    json_match = re.search(r'\{\s*"score"\s*:\s*(\d+)\s*\}', response)
    if json_match:
        return int(json_match.group(1))
    
    score_match = re.search(r'["\']?score["\']?\s*:\s*(\d+)', response, re.IGNORECASE)
    if score_match:
        return int(score_match.group(1))
    
    num_match = re.search(r'\b(\d|10)\b', response)
    if num_match:
        return int(num_match.group(1))
    
    return -1




def build_messages(sample: Dict, reflection_context: dict = None) -> List[Dict]:

    inst = sample.get("instruction", "").strip()
    inp = sample.get("input", "").strip()
    sys_msg = "You are a helpful math assistant. Reply with only the final answer."
    user = (f"{inst}\n\n{inp}\n\n"
            "Return only the next theorem name as a single snake_case token, "
            "without any extra text.")
    if reflection_context:
        avoid = reflection_context.get("avoid", [])
        prefer = reflection_context.get("prefer", [])
        if avoid or prefer:
            user += "\n\nReflection from previous attempts:"
            if avoid:
                user += f"\nAvoid: [{', '.join(avoid)}]"
            if prefer:
                user += f"\nPrefer: [{', '.join(prefer)}]"
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user},
    ]


def norm_label(x: str) -> str:
    return x.strip().strip('"').strip("'").replace(" ", "").replace("\n", "").lower()



class Expander:
    def __init__(self, model, tokenizer, lora_weights=None):
        self.model = model
        self.tokenizer = tokenizer
        self.reflection_context = None
        self._avoid_set = set()

    def update_reflection_context(self, reflection: dict):
        new_avoid = reflection.get("avoid", [])
        self._avoid_set.update(new_avoid)
        self.reflection_context = {
            "avoid": list(self._avoid_set),
            "prefer": reflection.get("prefer", []),
        }
        

    def generate(self, state: str, top_k: int = 5) -> list:

        eos_ids = []
        try:
            eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if isinstance(eot_id, int) and eot_id > 0:
                eos_ids.append(eot_id)
        except Exception:
            pass
        if self.tokenizer.eos_token_id is not None:
            if isinstance(self.tokenizer.eos_token_id, list):
                eos_ids.extend([i for i in self.tokenizer.eos_token_id if i is not None])
            else:
                eos_ids.append(self.tokenizer.eos_token_id)
        eos_ids = list(dict.fromkeys(eos_ids)) if eos_ids else None
        
        
        messages = build_messages(state, self.reflection_context)
        
        actions = []
        action_confidences = []
        
        chat_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(chat_text, return_tensors="pt", add_special_tokens=False).to(self.model.device)

        input_len = inputs["input_ids"].shape[1]  
        
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.7,
                top_p=1.0,
                do_sample=(0.7 > 0.0),
                repetition_penalty=1.1,
                eos_token_id=eos_ids,
                pad_token_id=self.tokenizer.pad_token_id,
                num_return_sequences=5, 
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True,  
                output_logits=True,
       
            )
        
        scores = out.scores
        gen_ids = out.sequences
        
       
        if hasattr(out, 'scores') and out.scores:
               
                confidences = self._calculate_confidence_from_logits(
                out.sequences, 
                out.logits,    
                input_len     
            )
        else:
           
            confidences = [1.0 / top_k] * top_k
        
        
       
        candidates_raw = []
        candidates_norm = []
        for i in range(gen_ids.shape[0]):
            new_tokens = gen_ids[i, input_len:]
            raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            candidates_raw.append(raw)
            actions.append(norm_label(raw.split()[0] if raw else ""))
            action_confidences.append(confidences[i])
           
            candidates_norm.append(norm_label(raw.split()[0] if raw else ""))

        best_conf = {}  
        for a, c in zip(actions, action_confidences):
            if not a:
                continue
            if (a not in best_conf) or (c > best_conf[a]):
                best_conf[a] = c

       
        items = sorted(best_conf.items(), key=lambda kv: kv[1], reverse=True)

        top_k = 5
        items = items[:top_k]

        actions = [a for a, _ in items]
        action_confidences = [c for _, c in items]
        
       
        total_confidence = sum(action_confidences)
        if total_confidence > 0:
            action_confidences = [c / total_confidence for c in action_confidences]
        else:
            action_confidences = [1.0 / len(action_confidences)] * len(action_confidences)

        
        return list(zip(actions, action_confidences))



    def _calculate_sequence_confidence(self, sequences, scores, prompt):
       
        import torch.nn.functional as F
        
        confidences = []
        num_sequences = sequences.shape[0]
        
      
        input_length = self.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        
        for seq_idx in range(num_sequences):
            log_probs = []
            for token_idx, score in enumerate(scores):
             
                token_logits = score[seq_idx]

                token_probs = F.softmax(token_logits, dim=-1)

                actual_token_id = sequences[seq_idx, input_length + token_idx]

                token_log_prob = torch.log(token_probs[actual_token_id])
                log_probs.append(token_log_prob.item())
            
            avg_log_prob = sum(log_probs) / len(log_probs) if log_probs else -float('inf')
            confidence = np.exp(avg_log_prob)
            confidences.append(confidence)
        
        return confidences

    def _calculate_confidence_from_logits(self, sequences, logits_list, input_length):
        
        import torch.nn.functional as F
        
        num_sequences = sequences.shape[0]  # 5
        confidences = []
        
        for seq_idx in range(num_sequences):
            log_probs = []
            
            for step_idx, step_logits in enumerate(logits_list):
                seq_logits = step_logits[seq_idx]  
                
                probs = F.softmax(seq_logits, dim=-1)
                token_position = input_length + step_idx
                
                if token_position < sequences.shape[1]:  
                    actual_token_id = sequences[seq_idx, token_position]

                    token_log_prob = torch.log(probs[actual_token_id] + 1e-10)  
                    log_probs.append(token_log_prob.item())
            
            if log_probs:
                avg_log_prob = sum(log_probs) / len(log_probs)
                confidence = np.exp(avg_log_prob)
            else:
                confidence = 0.0
            
            confidences.append(confidence)
        
        return confidences


    
    
    def insert_first_param(self, text, insert_value="1"):
        
        pattern = r'(\w+\()([^)]+)\)'
        def replacement(match):
            func_name = match.group(1)  
            params = match.group(2)    
            return f"{func_name}{insert_value},{params})"
        
        return re.sub(pattern, replacement, text)
    
    def _clean_action(self, text: str) -> str:

        pattern = r'([A-Za-z]+\([A-Za-z0-9, ()+-/*]+\))'  
        matches = re.findall(pattern, text)
        if matches:
            original = matches[0]
            result = self.insert_first_param(original, "1")
            return result

        text = ""
        if len(text) > 100: 
            return ""
        
        return text

    

    def _get_fallback_actions(self, state: str, num_actions: int) -> list:
        
        common_theorems = [
            "Perpendicular(A,B)",
            "Parallel(A,B)",
            "Equal(Angle(A),Angle(B))",
            "Collinear(A,B,C)",
            "Congruent(Triangle(A),Triangle(B))"
        ]
        return common_theorems[:num_actions]





class Reflector:
    """Analyze failed theorem applications and suggest avoid/prefer guidance."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def reflect(self, conditions: str, goal: str, failed_theorems: list) -> dict:
        prompt = self._build_prompt(conditions, goal, failed_theorems)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=1024, truncation=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )
        return self._parse_response(response, failed_theorems)

    def _build_prompt(self, conditions: str, goal: str, failed_theorems: list) -> str:
        failed_lines = "\n".join(
            f"  • {thm} → Error: {reason}" for thm, reason in failed_theorems
        )
        prompt = (
            "[Task] Analyze failed theorem applications rejected by the symbolic verifier. "
            "Identify error patterns and suggest theorem categories for future attempts.\n"
            f"[Current Conditions] {conditions}\n"
            f"[Goal] {goal}\n"
            f"[Failed Theorems]\n{failed_lines}\n\n"
            '[Output] Return ONLY a JSON object with two fields:\n'
            '"avoid"   List of specific theorems that should not be retried.\n'
            '"prefer"  List of 2-3 theorem category keywords most likely to help'
        )
        return prompt

    def _parse_response(self, response: str, failed_theorems: list) -> dict:
        result = {"avoid": [], "prefer": []}

        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed.get("avoid"), list):
                    result["avoid"] = [str(t) for t in parsed["avoid"]]
                if isinstance(parsed.get("prefer"), list):
                    result["prefer"] = [str(t) for t in parsed["prefer"]]
                return result
            except (json.JSONDecodeError, KeyError):
                pass

        result["avoid"] = [thm for thm, _ in failed_theorems]
        return result
