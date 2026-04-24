from formalgeo.problem import Problem
from formalgeo.data import DatasetLoader
from formalgeo.tools import load_json, save_json # get_used
from formalgeo.tools import draw_solution_hypertree, draw_theorem_dag
from formalgeo.tools import get_solution_hypertree, get_theorem_dag, show_solution
from formalgeo.solver import Interactor
from formalgeo.parse import parse_theorem_seqs, inverse_parse_solution, parse_one_theorem
from formalgeo.parse import parse_predicate_gdl, parse_theorem_gdl, parse_problem_cdl
from formalgeo.parse import inverse_parse_one, inverse_parse_one_theorem
from formalgeo.tools import get_used_pid_and_theorem

import sympy as sp



DROP_PREDICATES_DEFAULT = {"Point", "Line", "Angle"}
COMPRESS_COLLINEAR_DEFAULT = True

def _rotations(seq):
    n = len(seq)
    return [tuple(seq[i:] + seq[:i]) for i in range(n)]

def _canon_cycle(seq):
    if len(seq) <= 1:
        return tuple(seq)
    return min(_rotations(list(seq)))

def _canon_cycle_with_reverse(seq):
    if len(seq) <= 1:
        return tuple(seq)
    seq = tuple(seq)
    cand = _rotations(list(seq)) + _rotations(list(reversed(seq)))
    return min(cand)

def _first_item_str(obj):
    if isinstance(obj, tuple) and obj and isinstance(obj[0], tuple):
        return "".join(obj[0])
    if isinstance(obj, tuple):
        return "".join(obj)
    return str(obj)

def _sym_to_readable(problem, sym: sp.Symbol) -> str:
    cond = problem.condition
    if sym in getattr(cond, "attr_of_sym", {}):
        attr_name, items = cond.attr_of_sym[sym]
        if attr_name == "Free":
            return _first_item_str(items)
        return f"{attr_name}({_first_item_str(items)})"
    return str(sym)

def _expr_to_readable(problem, expr: sp.Expr) -> str:
    s = str(expr).replace(" ", "")
    syms = sorted(list(expr.free_symbols), key=lambda x: len(str(x)), reverse=True)
    for sym in syms:
        s = s.replace(str(sym), _sym_to_readable(problem, sym))
    return s

def _equation_to_readable(problem, eq_expr: sp.Expr) -> str:
    expr = sp.simplify(eq_expr)
    if len(expr.free_symbols) == 0:
        return f"Equation({_expr_to_readable(problem, expr)}=0)"

    if len(expr.free_symbols) <= 2:
        best = None
        for sym in list(expr.free_symbols):
            try:
                sols = sp.solve(expr, sym, dict=True)
            except Exception:
                continue
            if not sols:
                continue
            rhs = sols[0].get(sym)
            if rhs is None:
                continue
            score = (len(rhs.free_symbols), len(str(rhs)))
            if best is None or score < best[0]:
                best = (score, sym, rhs)

        if best is not None:
            _, sym, rhs = best
            lhs_txt = _sym_to_readable(problem, sym)
            rhs_txt = _expr_to_readable(problem, rhs)
            return f"Equal({lhs_txt},{rhs_txt})"

    return f"Equation({_expr_to_readable(problem, expr)}=0)"

def condition_to_str(problem, predicate, item):
    if predicate == "Equation":
        return _equation_to_readable(problem, item)
    return inverse_parse_one(predicate, item, problem)

def _compress_collinear_items(col_items):
    pair_to_id = {}
    parent = []

    def pid(pair):
        if pair not in pair_to_id:
            pair_to_id[pair] = len(parent)
            parent.append(len(parent))
        return pair_to_id[pair]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for pts in col_items:
        if isinstance(pts, list):
            pts = tuple(pts)
        if not isinstance(pts, tuple) or len(pts) < 3:
            continue
        uniq = list(dict.fromkeys(pts))
        if len(uniq) < 3:
            continue
        pairs = []
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a, b = uniq[i], uniq[j]
                pairs.append(pid(tuple(sorted((a, b)))))
        if not pairs:
            continue
        base = pairs[0]
        for other in pairs[1:]:
            union(base, other)

    comp_points = {}
    for (a, b), idx in pair_to_id.items():
        r = find(idx)
        s = comp_points.setdefault(r, set())
        s.add(a)
        s.add(b)

    compressed = []
    for pts in comp_points.values():
        if len(pts) >= 3:
            compressed.append(tuple(sorted(pts)))
    compressed.sort()
    return compressed

def extract_unique_conditions(
    problem,
    canonicalizer: PredicateCanonicalizer,
    include_solved_values: bool = False,
    drop_predicates=None,
    compress_collinear: bool = COMPRESS_COLLINEAR_DEFAULT,
    allow_predicates=None,
):
    drop_predicates = DROP_PREDICATES_DEFAULT if drop_predicates is None else set(drop_predicates)
    seen = set()
    cond_list = []
    collinear_items = []

    for predicate, item, *_ in problem.condition.items:
        if predicate in drop_predicates:
            continue
        if allow_predicates is not None and predicate not in allow_predicates:
            continue

        if predicate == "Collinear":
            collinear_items.append(item)
            continue

        key, canon_item = canonicalizer.canonicalize(predicate, item)
        if key in seen:
            continue
        seen.add(key)
        cond_list.append(condition_to_str(problem, predicate, canon_item))

    if collinear_items:
        if compress_collinear:
            for pts in _compress_collinear_items(collinear_items):
                cond_list.append(condition_to_str(problem, "Collinear", pts))

    
    return cond_list

def goal_to_str(problem, goal):
    if isinstance(goal, str):
        return goal
    if isinstance(goal, (list, tuple)) and len(goal) >= 2:
        predicate, item = goal[0], goal[1]
        return condition_to_str(problem, predicate, item)
    return str(goal)

def _pred_name_from_sig(sig: str) -> str:
    return sig.split("(", 1)[0] if "(" in sig else sig

def build_delta_predicate_set(predicate_gdl: dict) -> set:
    allow = set()
    for sec in ("Entity", "Relation", "Attribution"):
        for sig in predicate_gdl.get(sec, {}).keys():
            allow.add(_pred_name_from_sig(sig))
    allow.update({"Equal", "Equation"})
    return allow


class FormalGeoEnv:
    """Interface to FormalGeo symbolic reasoning system."""
    
    def __init__(self, dataset_name="formalgeo7k_v2", datasets_path="/root/autodl-fs/GeoLMA/data/"):
        
        self.dl = DatasetLoader(dataset_name=dataset_name, datasets_path=datasets_path)
        self.solver = Interactor(self.dl.predicate_GDL, self.dl.theorem_GDL)
        
       
        self.current_problem = None
        self.theorem_history = []  
        self.step_size = 0                 
        self.max_steps = 50 
        self.current_conditions = set()  
        self.goal_conditions = set()    
        
        self.predicate_GDL = self.dl.predicate_GDL  
        self.theorem_GDL = self.dl.theorem_GDL     
        
        self.parsed_predicate_GDL = parse_predicate_gdl(self.predicate_GDL)
        self.parsed_theorem_GDL = parse_theorem_gdl(self.theorem_GDL, self.parsed_predicate_GDL)
        
       
        self.canonicalizer = PredicateCanonicalizer(self.predicate_GDL)
        self.allow_predicates = build_delta_predicate_set(self.predicate_GDL)

    
    def init(self, problem_id: int):
        self.solver = Interactor(self.dl.predicate_GDL, self.dl.theorem_GDL)
        base_problem = self.dl.get_problem(pid=int(problem_id))
        self.solver.load_problem(base_problem)

        p = Problem()
        p.load_problem_by_copy(self.solver.problem)
        return p , base_problem


    def build_input_block(self, conditions, goal, used_theorems):
        
        cond_text = ", ".join(conditions)
        goal_text = goal if goal else "None"

        if used_theorems:
            thm_text = "[" + ", ".join(used_theorems) + "]"
        else:
            thm_text = "None"

        input_block = (
            f"Condition: [{cond_text}]\n"
            f"Goal: {goal_text}\n"
            f"Used Theorems: {thm_text}"
        )
        return input_block, cond_text
    
    
    def _used_theorems_from_timing(self, p):
        used = []
        for step in sorted(p.timing.keys()):
            th = p.timing[step][0]

            if not isinstance(th, tuple):
                continue

            if th[0] in ["extended", "solve_eq", "prerequisite"]:
                continue

            th_str = inverse_parse_one_theorem(th, p.parsed_theorem_GDL)
            if th_str not in used:
                used.append(th_str)

        return used


    def render(self, problem_state, init_state, max_conditions=100, max_per_predicate=15):
       
        self.solver.problem = problem_state
        p = self.solver.problem

        INSTRUCTION_TEXT = (
            "Based on the known conditions of the problem and the theorems used, "
            "predict the next theorem to be used."
        )

        base_conditions, goal = init_state

        current = extract_unique_conditions(
            p,
            self.canonicalizer,
            include_solved_values=False,
            drop_predicates=DROP_PREDICATES_DEFAULT,
            compress_collinear=COMPRESS_COLLINEAR_DEFAULT,
            allow_predicates=self.allow_predicates,
        )

        seen = set()
        cond_list = []
        for x in base_conditions + current:
            if x in seen:
                continue
            seen.add(x)
            cond_list.append(x)

        if max_conditions is not None and len(cond_list) > max_conditions:
            cond_list = cond_list[:max_conditions]

        used_theorems = self._used_theorems_from_timing(p)

        goal_text = goal_to_str(p, goal)
        input_text,cond_text = self.build_input_block(cond_list, goal_text, used_theorems)

        sample = {"instruction": INSTRUCTION_TEXT, "input": input_text}
        return used_theorems, sample, seen, goal_text

    def step_from_state(self, problem_state: Problem, action: str):

        p = Problem()
        p.load_problem_by_copy(problem_state)
        self.solver.problem = p
        t_name = action
        success = False
        
        if not t_name:
            return problem_state, False, {"error": "empty action", "action": action, "success": False}

        try:
            success = self.solver.apply_theorem(t_name=t_name, t_branch=None, t_para=None)
        except Exception as e:
            print(e)
            return problem_state, False, {
                "error": str(e),
                "invalid_theorem": t_name,
                "success": success,
                "action": action
            }

        self.solver.problem.check_goal()
        done = self.solver.problem.goal.solved

        new_state = Problem()
        new_state.load_problem_by_copy(self.solver.problem)

        return new_state, done, {"success": success, "action": t_name}

    
    
    def _get_current_state(self) -> str:
        try:

            conditions = []

            if 'construction_cdl' in self.current_problem:
                conditions.extend(self.current_problem['construction_cdl'])
            if 'text_cdl' in self.current_problem:
                conditions.extend(self.current_problem['text_cdl'])
            if 'goal_cdl' in self.current_problem:
                goals = self.current_problem['goal_cdl']


            state = {
                'conditions': conditions[:20],
                'goals': goals,
                'steps': self.step_size,
                'theorem_history': self.theorem_history[-5:]
            }
            return f"Conditions: {conditions}; Goals: {goals}; Steps: {self.step_size}"

        except:
            return f"State at step {self.step_size}"


    
    def step(self, action: str) -> tuple:

        self.step_size += 1
        if self.step_size > self.max_steps:
            return self._get_current_state(), False
        
        if action in self.theorem_history:
            return self._get_current_state(), False
        theorem_name, theorem_branch, theorem_params = parse_one_theorem(action)
        
        if not theorem_name:
            return self._get_current_state(), False

        try:
            success = self.solver.apply_theorem(theorem_name, theorem_branch, theorem_params)
            
            if success:
                self.theorem_history.append(action)
                
            self.solver.problem.check_goal()
            done = self.solver.problem.goal.solved
            
            return self._get_current_state(), done
            
        except Exception as e:
            print(f"***: {e}")
            return self._get_current_state(), False
        
    
    def _parse_action(self, action: str) -> tuple:
        if not action:
            return None, "1"
        
        
        if '_' in action:
            parts = action.rsplit('_', 1)
            if parts[-1].isdigit():
                return parts[0], parts[1]
            else:
                return action, "1"
        
        return action, "1"
    
    def copy_state(self) -> 'Problem':
        problem_copy = Problem()
        problem_copy.load_problem_by_copy(self.solver.problem)
        return problem_copy
    
    def restore_state(self, problem_state: 'Problem'):
        self.solver.problem = problem_state
