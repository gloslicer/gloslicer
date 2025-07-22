import ast
import argparse
import os
import json
import uuid
from typing import Dict, List, Set, Any, Tuple
import re

def strip_python_comments(source: str) -> str:
    source = re.sub(r'(?m)#.*$', '', source)
    source = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', source, flags=re.DOTALL)
    source = '\n'.join([line for line in source.split('\n') if line.strip() != ''])
    return source

def build_strip_lineno_mapping(raw_lines, strip_lines):
    mapping = {}
    j = 0
    for i, line in enumerate(raw_lines):
        l = line.strip()
        if l == '' or l.startswith('#') or l.startswith('"""') or l.startswith("'''"):
            mapping[i] = None
            continue
        while j < len(strip_lines) and strip_lines[j].strip() != l:
            j += 1
        if j < len(strip_lines) and strip_lines[j].strip() == l:
            mapping[i] = j
            j += 1
        else:
            mapping[i] = None
    return mapping

def map_label_lines(labels, lineno_map):
    new_labels = []
    for l in labels:
        orig_rel = l['rel_line']
        strip_rel = lineno_map.get(orig_rel, None) if orig_rel is not None else None
        if strip_rel is None:
            continue
        l2 = dict(l)
        l2['rel_line'] = strip_rel
        new_labels.append(l2)
    return new_labels

class ProgramGraphSlicer:
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.python_files = self._collect_python_files(project_root)

        self.ast_trees: Dict[str, ast.AST] = {}
        self.parent_maps: Dict[str, Dict[ast.AST, ast.AST]] = {}
        self.source_by_file: Dict[str, str] = {}

        for path in self.python_files:
            with open(path, 'r', encoding='utf-8') as f:
                src = f.read()
            tree = ast.parse(src)
            self.ast_trees[path] = tree
            self.source_by_file[path] = src
            self.parent_maps[path] = self._build_parent_map(tree)

        self.cfg_edges: Dict[str, List[Dict[str, Any]]] = {p: [] for p in self.python_files}
        self.cg_edges:  Dict[str, List[Dict[str, Any]]] = {p: [] for p in self.python_files}

        self.func_index: Dict[str, Tuple[str, ast.FunctionDef]] = {}
        self._index_functions()
        self._build_cfgs()
        self._build_call_graphs()

    def _collect_python_files(self, root: str) -> List[str]:
        files = []
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.endswith('.py'):
                    files.append(os.path.join(dp, fn))
        return files

    def _build_parent_map(self, tree: ast.AST) -> Dict[ast.AST, ast.AST]:
        parent = {}
        for p in ast.walk(tree):
            for c in ast.iter_child_nodes(p):
                parent[c] = p
        return parent

    def _find_enclosing(self, node: ast.AST, parents: Dict[ast.AST, ast.AST], typ: Any):
        while node and not isinstance(node, typ):
            node = parents.get(node)
        return node
    
    def _find_def_use_names(self, func_node, var):
        def_nodes = set()
        use_nodes = set()
        for n in ast.walk(func_node):
            if isinstance(n, ast.Name) and n.id == var:
                if isinstance(n.ctx, ast.Store):
                    def_nodes.add(n)
                elif isinstance(n.ctx, ast.Load):
                    use_nodes.add(n)
        for arg in getattr(func_node.args, 'args', []):
            if arg.arg == var:
                def_nodes.add(arg)
        return def_nodes, use_nodes

    def _find_call_sites(self, func_node, func_path):
        call_sites = []
        func_key = self._node_key(func_node)
        
        for file_path, cg_edges in self.cg_edges.items():
            for edge in cg_edges:
                if edge.get('dst') == func_key and edge.get('dst_file') == func_path:
                    call_line = edge.get('src_line')
                    if call_line:
                        for node in ast.walk(self.ast_trees[file_path]):
                            if (isinstance(node, ast.Call) and 
                                getattr(node, 'lineno', None) == call_line):
                                call_sites.append((file_path, node, call_line))
        return call_sites

    def _find_assignments_receiving_call(self, call_node, call_file_path):
        assignments = []
        call_line = getattr(call_node, 'lineno', None)
        if not call_line:
            return assignments
            
        for node in ast.walk(self.ast_trees[call_file_path]):
            if isinstance(node, ast.Assign):
                if (hasattr(node.value, 'lineno') and 
                    node.value.lineno == call_line and
                    isinstance(node.value, ast.Call)):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            assignments.append(target)
        return assignments

    def _build_dfg(self, func_node, var, def_nodes, use_nodes, func_path):
        dfg = {d: set() for d in def_nodes}
        all_var_nodes = [n for n in ast.walk(func_node) if isinstance(n, ast.Name) and n.id == var]
        all_var_nodes.sort(key=lambda n: getattr(n, 'lineno', -1))
        for i, node in enumerate(all_var_nodes):
            if isinstance(node.ctx, ast.Store):
                for j in range(i + 1, len(all_var_nodes)):
                    next_node = all_var_nodes[j]
                    if isinstance(next_node.ctx, ast.Store):
                        break
                    elif isinstance(next_node.ctx, ast.Load):
                        dfg[node].add(next_node)

        param_names = [arg.arg for arg in getattr(func_node.args, 'args', [])]
        if var in param_names:
            param_index = param_names.index(var)
            param_arg = func_node.args.args[param_index]
            
            call_sites = self._find_call_sites(func_node, func_path)
            for call_file_path, call_node, call_line in call_sites:
                if param_index < len(call_node.args):
                    actual_arg = call_node.args[param_index]
                    if isinstance(actual_arg, ast.Name):
                        for def_node in ast.walk(self.ast_trees[call_file_path]):
                            if (isinstance(def_node, ast.Name) and 
                                def_node.id == actual_arg.id and 
                                isinstance(def_node.ctx, ast.Store) and
                                getattr(def_node, 'lineno', float('inf')) < call_line):
                                for use_node in use_nodes:
                                    if use_node in all_var_nodes:
                                        dfg.setdefault(def_node, set()).add(use_node)
        
        for return_node in ast.walk(func_node):
            if isinstance(return_node, ast.Return) and return_node.value:
                for use_node in ast.walk(return_node.value):
                    if (isinstance(use_node, ast.Name) and use_node.id == var and
                        isinstance(use_node.ctx, ast.Load)):
                        
                        call_sites = self._find_call_sites(func_node, func_path)
                        for call_file_path, call_node, call_line in call_sites:
                            receiving_vars = self._find_assignments_receiving_call(call_node, call_file_path)
                            for recv_var in receiving_vars:
                                dfg.setdefault(use_node, set()).add(recv_var)
        
        return dfg

    def _get_function_statements(self, func_node):
        statements = []
        for node in ast.walk(func_node):
            if isinstance(node, ast.stmt) and node != func_node:
                statements.append(node)
        return statements

    def _find_var_def_statements(self, func_node, var):
        def_stmts = []
        for stmt in self._get_function_statements(func_node):
            if self._statement_defines_var(stmt, var):
                def_stmts.append(stmt)
        return def_stmts

    def _find_var_use_statements(self, func_node, var):
        use_stmts = []
        for stmt in self._get_function_statements(func_node):
            if self._statement_uses_var(stmt, var):
                use_stmts.append(stmt)
        return use_stmts

    def _statement_defines_var(self, stmt, var):
        for node in ast.walk(stmt):
            if (isinstance(node, ast.Name) and 
                node.id == var and 
                isinstance(node.ctx, ast.Store)):
                return True
        return False

    def _statement_uses_var(self, stmt, var):
        for node in ast.walk(stmt):
            if (isinstance(node, ast.Name) and 
                node.id == var and 
                isinstance(node.ctx, ast.Load)):
                return True
        return False

    def _get_statement_defined_vars(self, stmt):
        defined_vars = set()
        for node in ast.walk(stmt):
            if (isinstance(node, ast.Name) and 
                isinstance(node.ctx, ast.Store)):
                defined_vars.add(node.id)
        return defined_vars

    def _get_statement_used_vars(self, stmt):
        used_vars = set()
        for node in ast.walk(stmt):
            if (isinstance(node, ast.Name) and 
                isinstance(node.ctx, ast.Load)):
                used_vars.add(node.id)
        return used_vars

    def get_forward_nodes(self, func_node, var, path):
        all_statements = self._get_function_statements(func_node)
        seed_def_stmts = self._find_var_def_statements(func_node, var)
        
        forward_slice = set()
        
        for stmt in all_statements:
            if self._statement_uses_var(stmt, var):
                forward_slice.add(stmt)
        
        changed = True
        while changed:
            changed = False
            new_stmts = set()            
            for influenced_stmt in forward_slice:
                defined_vars = self._get_statement_defined_vars(influenced_stmt)
                for stmt in all_statements:
                    if stmt not in forward_slice:
                        for def_var in defined_vars:
                            if self._statement_uses_var(stmt, def_var):
                                new_stmts.add(stmt)
            
            if new_stmts:
                forward_slice.update(new_stmts)
                changed = True
        
        return forward_slice

    def get_backward_nodes(self, func_node, var, path):
        all_statements = self._get_function_statements(func_node)
        seed_use_stmts = self._find_var_use_statements(func_node, var)
        
        backward_slice = set()

        for stmt in all_statements:
            if self._statement_defines_var(stmt, var):
                backward_slice.add(stmt)
        
        changed = True
        while changed:
            changed = False
            new_stmts = set()
            
            for dependent_stmt in backward_slice:
                used_vars = self._get_statement_used_vars(dependent_stmt)
                
                for stmt in all_statements:
                    if stmt not in backward_slice:
                        for used_var in used_vars:
                            if self._statement_defines_var(stmt, used_var):
                                new_stmts.add(stmt)
            
            if new_stmts:
                backward_slice.update(new_stmts)
                changed = True
        
        return backward_slice

    def _index_functions(self):
        for path, tree in self.ast_trees.items():
            pm = self.parent_maps[path]
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    cls = self._find_enclosing(node, pm, ast.ClassDef)
                    qn = f"{cls.name}.{node.name}" if cls else node.name
                    self.func_index[qn + f'#{path}'] = (path, node)

    def _build_cfgs(self):
        for path, tree in self.ast_trees.items():
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For)):
                    stmts = node.body + getattr(node, 'orelse', [])
                    for stmt in stmts:
                        if hasattr(node, 'lineno') and hasattr(stmt, 'lineno'):
                            self.cfg_edges[path].append({
                                'src': self._node_key(node),
                                'dst': self._node_key(stmt),
                                'src_line': node.lineno,
                                'dst_line': stmt.lineno,
                                'type': 'CFG'
                            })

    def _build_call_graphs(self):
        for path, tree in self.ast_trees.items():
            pm = self.parent_maps[path]
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    caller = self._find_enclosing(node, pm, ast.FunctionDef)
                    name = None
                    if isinstance(node.func, ast.Name):
                        name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        name = node.func.attr
                    if name and caller:
                        for qn, (callee_path, callee_node) in self.func_index.items():
                            if callee_node.name == name:
                                self.cg_edges[path].append({
                                    'src': self._node_key(caller),
                                    'dst': self._node_key(callee_node),
                                    'src_file': path,
                                    'dst_file': callee_path,
                                    'src_line': getattr(node, 'lineno', None),
                                    'dst_line': getattr(callee_node, 'lineno', None),
                                    'type': 'CG'
                                })

    def _find_defs_uses(self, path: str, var: str) -> Tuple[Set[ast.AST], Set[ast.AST]]:
        defs, uses = set(), set()
        tree, pm = self.ast_trees[path], self.parent_maps[path]
        # parameters
        for p, fn in [v for v in self.func_index.values() if v[0] == path]:
            for arg in fn.args.args:
                if arg.arg == var:
                    defs.add(fn)
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and n.id == var:
                stmt = self._find_enclosing(n, pm, ast.stmt)
                if stmt:
                    if isinstance(n.ctx, ast.Store):
                        defs.add(stmt)
                    else:
                        uses.add(stmt)
            if isinstance(n, ast.Return) and n.value:
                for sub in ast.walk(n.value):
                    if isinstance(sub, ast.Name) and sub.id == var:
                        uses.add(n)
        return defs, uses

    def _build_edges(self, path: str, var: str):
        defs, uses = self._find_defs_uses(path, var)
        bwd, fwd = [], []
        for u in uses:
            for d in defs:
                if hasattr(u, 'lineno') and hasattr(d, 'lineno') and d.lineno <= u.lineno:
                    bwd.append({'src': self._node_key(u), 'dst': self._node_key(d),
                                'src_line': u.lineno, 'dst_line': d.lineno, 'type': 'DFG'})
        for d in defs:
            for u in uses:
                if hasattr(d, 'lineno') and hasattr(u, 'lineno') and u.lineno >= d.lineno:
                    fwd.append({'src': self._node_key(d), 'dst': self._node_key(u),
                                'src_line': d.lineno, 'dst_line': u.lineno, 'type': 'DFG'})
        for c in [n for n in ast.walk(self.ast_trees[path]) if isinstance(n, ast.Call)]:
            if isinstance(c.func, ast.Name):
                fname = c.func.id
                for idx, arg in enumerate(c.args):
                    if isinstance(arg, ast.Name):
                        for qn, (fp, fn) in self.func_index.items():
                            if fn.name == fname and idx < len(fn.args.args):
                                param = fn.args.args[idx]
                                if hasattr(arg, 'lineno') and hasattr(param, 'lineno'):
                                    bwd.append({'src': self._node_key(arg), 'dst': self._node_key(param),
                                                'src_line': arg.lineno, 'dst_line': param.lineno, 'type': 'DFG'})
                                    fwd.append({'src': self._node_key(param), 'dst': self._node_key(arg),
                                                'src_line': param.lineno, 'dst_line': arg.lineno, 'type': 'DFG'})
        pm = self.parent_maps[path]
        for node in defs | uses:
            for e in self.cfg_edges[path]:
                if e['dst'] == self._node_key(node):
                    bwd.append(e)
                if e['src'] == self._node_key(node):
                    fwd.append(e)
            func = self._find_enclosing(node, pm, ast.FunctionDef)
            if func:
                for edges in self.cg_edges.values():
                    for e in edges:
                        if e['dst'] == self._node_key(func):
                            bwd.append(e)
                        if e['src'] == self._node_key(func):
                            fwd.append(e)
        return bwd, fwd

    def _node_key(self, node: ast.AST) -> str:
        return str(id(node))

    def _expand_slice_with_cfg_cg(self, path: str, nodes: Set[ast.AST], is_backward: bool) -> Set[ast.AST]:
        pm = self.parent_maps[path]
        result = set(nodes)
        ids = {self._node_key(n) for n in nodes}
        for e in self.cfg_edges[path]:
            if is_backward and e['dst'] in ids:
                for n in ast.walk(self.ast_trees[path]):
                    if self._node_key(n) == e['src']:
                        result.add(n)
            if not is_backward and e['src'] in ids:
                for n in ast.walk(self.ast_trees[path]):
                    if self._node_key(n) == e['dst']:
                        result.add(n)
        funcs = {self._find_enclosing(n, pm, ast.FunctionDef) for n in nodes}
        func_ids = {self._node_key(f) for f in funcs if f}
        for pth, edges in self.cg_edges.items():
            for e in edges:
                if is_backward and e['dst'] in func_ids:
                    for n in ast.walk(self.ast_trees.get(pth, self.ast_trees[path])):
                        if self._node_key(n) == e['src']:
                            result.add(n)
                if not is_backward and e['src'] in func_ids:
                    for n in ast.walk(self.ast_trees.get(pth, self.ast_trees[path])):
                        if self._node_key(n) == e['dst']:
                            result.add(n)
        return result

    def _is_intra(self, node, func_node, file_path):
        node_func = self._find_enclosing(node, self.parent_maps[file_path], ast.FunctionDef)
        return (node_func is func_node)

    def _label_nodes(self, nodes: Set[ast.AST], func_node: ast.FunctionDef, func_path: str) -> List[Dict]:
        labels = []
        for node in nodes:
            node_file_path = func_path
            node_func = None
            
            found_in_current = False
            for n in ast.walk(self.ast_trees[func_path]):
                if n is node:
                    found_in_current = True
                    break
            
            if found_in_current:
                node_func = self._find_enclosing(node, self.parent_maps[func_path], ast.FunctionDef)
            else:
                for file_path, tree in self.ast_trees.items():
                    for n in ast.walk(tree):
                        if n is node:
                            node_file_path = file_path
                            node_func = self._find_enclosing(node, self.parent_maps[file_path], ast.FunctionDef)
                            break
                    if node_file_path != func_path:
                        break
            
            is_intra = (node_func is func_node and node_file_path == func_path)
            abs_line = getattr(node, 'lineno', None)
            rel_line = abs_line - func_node.lineno if is_intra and abs_line is not None else None
            
            labels.append({
                "abs_line": abs_line,
                "rel_line": rel_line,
                "file_path": os.path.relpath(node_file_path, self.project_root),
                "func_name": node_func.name if node_func else None,
                "intra": is_intra
            })
        return labels

    def _convert_statements_to_labels(self, statements: Set[ast.AST], func_node: ast.FunctionDef, func_path: str) -> List[Dict]:
        labels = []
        for stmt in statements:
            abs_line = getattr(stmt, 'lineno', None)
            if abs_line is None:
                continue
                
            is_intra = True
            rel_line = abs_line - func_node.lineno if abs_line is not None else None
            
            labels.append({
                "abs_line": abs_line,
                "rel_line": rel_line,
                "file_path": os.path.relpath(func_path, self.project_root),
                "func_name": func_node.name,
                "intra": is_intra
            })
        return labels

    def _get_related_func_codes(self, func_node, func_path, input_mode):
        related_codes = []
        used = set()
        this_func_key = self._node_key(func_node)
        used.add(this_func_key)
        if input_mode in ("func_plus_called", "func_plus_all"):
            for cg in self.cg_edges.get(func_path, []):
                if cg["src"] == this_func_key:
                    for qn2, (p2, fn2) in self.func_index.items():
                        if self._node_key(fn2) == cg["dst"] and self._node_key(fn2) not in used:
                            code = ast.get_source_segment(self.source_by_file[p2], fn2)
                            if code:
                                related_codes.append(code)
                                used.add(self._node_key(fn2))
        if input_mode in ("func_plus_caller", "func_plus_all"):
            for cg in self.cg_edges.get(func_path, []):
                if cg["dst"] == this_func_key:
                    for qn2, (p2, fn2) in self.func_index.items():
                        if self._node_key(fn2) == cg["src"] and self._node_key(fn2) not in used:
                            code = ast.get_source_segment(self.source_by_file[p2], fn2)
                            if code:
                                related_codes.append(code)
                                used.add(self._node_key(fn2))
        return related_codes

    def perform_slicing_flat(self, input_mode="func_only") -> List[Dict[str, Any]]:
        project_code = "\n".join(
            self.source_by_file[p] for p in sorted(self.source_by_file.keys())
        )
        project_code = strip_python_comments(project_code)
        flat: List[Dict[str, Any]] = []
        for qn, (path, func_node) in self.func_index.items():
            raw_func_code = ast.get_source_segment(self.source_by_file[path], func_node)
            if raw_func_code is None:
                continue
            raw_func_lines = raw_func_code.splitlines()
            func_code = strip_python_comments(raw_func_code)
            strip_func_lines = func_code.splitlines()
            lineno_map = build_strip_lineno_mapping(raw_func_lines, strip_func_lines)
            input_code = func_code

            if input_mode != "func_only":
                related_codes = self._get_related_func_codes(func_node, path, input_mode)
                if related_codes:
                    input_code = func_code + "\n\n# ==== CONTEXT BEGIN ====\n" + "\n\n".join(
                        strip_python_comments(code) for code in related_codes
                    )

            var_nodes = [
                n for n in ast.walk(func_node)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
            ]
            for n in var_nodes:
                var = n.id
                ln  = n.lineno
                col = getattr(n, 'col_offset', None)
                rel_line = ln - func_node.lineno
                strip_var_line = lineno_map.get(rel_line, None)
                if strip_var_line is None:
                    continue

                f_nodes = self.get_forward_nodes(func_node, var, path)
                b_nodes = self.get_backward_nodes(func_node, var, path)
                f_nodes = self._expand_slice_with_cfg_cg_recursive(path, f_nodes, is_backward=False)
                b_nodes = self._expand_slice_with_cfg_cg_recursive(path, b_nodes, is_backward=True)
                backward_labels = self._convert_statements_to_labels(b_nodes, func_node, path)
                forward_labels  = self._convert_statements_to_labels(f_nodes, func_node, path)
                backward_labels = map_label_lines(backward_labels, lineno_map)
                forward_labels  = map_label_lines(forward_labels, lineno_map)

                flat.append({
                    "eid":            str(uuid.uuid4()),
                    "input_func_code": input_code,
                    "file_path":      os.path.relpath(path, self.project_root),
                    "project_code":   project_code,
                    "variable":       var,
                    "variable_loc":   [strip_var_line, col],
                    "line_number":    strip_var_line,
                    "backward_labels": backward_labels,
                    "forward_labels":  forward_labels,
                    "raw_func_code":   raw_func_code,
                })

        return flat
    
    def _expand_slice_with_cfg_cg(self, path: str, nodes: Set[ast.AST], is_backward: bool) -> Set[ast.AST]:
        pm = self.parent_maps[path]
        result = set(nodes)
        queue = list(nodes)
        seen = {self._node_key(n) for n in nodes}
        while queue:
            curr_nodes = set(queue)
            queue = []
            ids = {self._node_key(n) for n in curr_nodes}
            for e in self.cfg_edges[path]:
                if is_backward and e['dst'] in ids:
                    for n in ast.walk(self.ast_trees[path]):
                        if self._node_key(n) == e['src'] and self._node_key(n) not in seen:
                            result.add(n)
                            queue.append(n)
                            seen.add(self._node_key(n))
                if not is_backward and e['src'] in ids:
                    for n in ast.walk(self.ast_trees[path]):
                        if self._node_key(n) == e['dst'] and self._node_key(n) not in seen:
                            result.add(n)
                            queue.append(n)
                            seen.add(self._node_key(n))
            funcs = {self._find_enclosing(n, pm, ast.FunctionDef) for n in curr_nodes}
            func_ids = {self._node_key(f) for f in funcs if f}
            for pth, edges in self.cg_edges.items():
                for e in edges:
                    if is_backward and e['dst'] in func_ids:
                        for n in ast.walk(self.ast_trees.get(pth, self.ast_trees[path])):
                            if self._node_key(n) == e['src'] and self._node_key(n) not in seen:
                                result.add(n)
                                queue.append(n)
                                seen.add(self._node_key(n))
                    if not is_backward and e['src'] in func_ids:
                        for n in ast.walk(self.ast_trees.get(pth, self.ast_trees[path])):
                            if self._node_key(n) == e['dst'] and self._node_key(n) not in seen:
                                result.add(n)
                                queue.append(n)
                                seen.add(self._node_key(n))
        return result
    
    def _expand_slice_with_cfg_cg_recursive(self, path, nodeset, is_backward):
        result = set(nodeset)
        queue = list(nodeset)
        seen = set(self._node_key(n) for n in nodeset)
        while queue:
            current_nodes = set(queue)
            queue = []
            new_nodes = self._expand_slice_with_cfg_cg(path, current_nodes, is_backward) - result
            for n in new_nodes:
                if self._node_key(n) not in seen:
                    result.add(n)
                    queue.append(n)
                    seen.add(self._node_key(n))
        return result

    def export_json(self, output: str, input_mode="func_only", split="train"):
        data = self.perform_slicing_flat(input_mode=input_mode)
        out_data = []
        for item in data:
            if split == "test":
                forward_slices = sorted({l['abs_line'] for l in item['forward_labels'] if l['abs_line'] is not None})
                backward_slices = sorted({l['abs_line'] for l in item['backward_labels'] if l['abs_line'] is not None})
                variable_loc = [
                    item['backward_labels'][0]['abs_line'] if item['backward_labels'] else None,
                    item['variable_loc'][1]
                ]
                line_number = variable_loc[0]
                out_data.append({
                    "eid": item["eid"],
                    "variable": item["variable"],
                    "code": item["project_code"],
                    "variable_loc": variable_loc,
                    "line_number": line_number,
                    "forward_slice": forward_slices,
                    "backward_slice": backward_slices
                })
            else:
                forward_slices = sorted({l['rel_line'] for l in item['forward_labels'] if l['rel_line'] is not None})
                backward_slices = sorted({l['rel_line'] for l in item['backward_labels'] if l['rel_line'] is not None})
                variable_loc = [
                    item['backward_labels'][0]['rel_line'] if item['backward_labels'] else None,
                    item['variable_loc'][1]
                ]
                line_number = variable_loc[0]
                out_data.append({
                    "eid": item["eid"],
                    "variable": item["variable"],
                    "code": item["input_func_code"],
                    "variable_loc": variable_loc,
                    "line_number": line_number,
                    "forward_slice": forward_slices,
                    "backward_slice": backward_slices
                })

        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(out_data)} slices to {output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir',    dest='root', required=True, help="Python source code root directory")
    parser.add_argument('-o', '--output', dest='out',  required=True, help="Output JSON file path")
    parser.add_argument('--input_mode', default='func_only', choices=['func_only', 'func_plus_called', 'func_plus_caller', 'func_plus_all'], help="input mode for slicing")
    parser.add_argument('--split', required=True, choices=['train', 'val', 'test'], help="dataset split type (train/val/test)")
    args = parser.parse_args()
    slicer = ProgramGraphSlicer(args.root)
    slicer.export_json(args.out, input_mode=args.input_mode, split=args.split)