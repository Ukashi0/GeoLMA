import math
from collections import defaultdict


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state  
        self.parent = parent 
        self.action = action 
        self.children = []   
        self.N = defaultdict(int) 
        self.W = defaultdict(float)   


class MCTS:
    def __init__(self, env, expander, evaluator, reflector, cpuct=1.0):
        self.env = env  
        self.expander = expander  
        self.evaluator = evaluator  
        self.reflector = reflector 
        self.cpuct = cpuct  
        self.action_priors = {}  
    
    def join_conditions(self, problem_json):
       
        conditions = []
        for key in ["construction_cdl", "text_cdl", "image_cdl"]:
            if key in problem_json and isinstance(problem_json[key], list):
                conditions.extend(problem_json[key])
       
        conditions = list(dict.fromkeys(conditions))
        return conditions
    
    def process_problem(self, problem_json):
        
        conditions = self.join_conditions(problem_json)
        goal = problem_json.get("goal_cdl", "None")
        
        return (conditions, goal)

    def search(self, problem_id, num_iters=1000):

        root_state, base_problem = self.env.init(problem_id)
        init_state = self.process_problem(base_problem)
        self.root = MCTSNode(root_state)
        self._last_failed = []
        self._last_cond_text = ""
        self._last_goal_text = ""
        last_done, last_used = False, []
        for _ in range(num_iters):
            node = self._select(self.root)
            if node is None:
                continue
            last_done, last_used  = self._expand_and_eval(node, init_state)
            if last_done:
                break
            self._reflect()
        return last_done, self._best_path(last_used)

    def _select(self, node):
        while node.children:
            best_score, best_child = max(
                ((self._uct(node, a), c) for c,a in node.children),
                key=lambda x: x[0]
            )
            node = best_child
        return node

    def _uct(self, node, action):
        N_tot = sum(node.N.values())
        Q = node.W[action] / (1 + node.N[action])
        P = self.action_priors.get(action, 1.0 / len(node.children) if node.children else 1.0)
        return Q + self.cpuct * P * math.sqrt(N_tot) / (1 + node.N[action])

    
    def _expand_and_eval(self, node, init_state):
        used_theorems, state_text, cond_text, goal_text = self.env.render(node.state, init_state)
        action_prior_pairs = self.expander.generate(state_text)

        self._last_failed = []
        self._last_cond_text = ", ".join(cond_text)
        self._last_goal_text = goal_text

        for action, prior in action_prior_pairs:
            self.action_priors[action] = prior

            new_state, done, info = self.env.step_from_state(node.state, action)
            if not info['success']:
                error_reason = info.get('error', 'theorem application failed')
                self._last_failed.append((action, error_reason))
                continue

            child = MCTSNode(new_state, parent=node, action=action)

            node.children.append((child, action))
            if done:
                v = 1.0
                self._backprop(child, v)
                used_theorems_solved, _ ,_ ,_   = self.env.render(new_state, init_state)
                return True, used_theorems_solved
            else:
                _, _, cond_text_new, _ = self.env.render(new_state, init_state)
                new_cond = cond_text_new - cond_text
                new_cond_str = ", ".join(new_cond)
                cond_text_str = ", ".join(cond_text)
                v = self.evaluator.evaluate(action, goal_text, cond_text_str, new_cond_str)
                self._backprop(child, v)


        return False, used_theorems

    def _backprop(self, node, v):
        while node.parent:
            a = node.action
            node.parent.N[a] += 1
            node.parent.W[a] += v
            node = node.parent 

    def _path(self, node):
        seq = []
        while node.parent:
            seq.append(node.action)
            node = node.parent
        return list(reversed(seq))
    
    def _reflect(self):
        if not self._last_failed:
            return

        reflection = self.reflector.reflect(
            conditions=self._last_cond_text,
            goal=self._last_goal_text,
            failed_theorems=self._last_failed,
        )
        self.expander.update_reflection_context(reflection)

    def _best_path(self, used_theorems):
        if not self.root.children:
            return used_theorems
        return self._path(max(self.root.children, key=lambda ca: self.root.N[ca[1]])[0])