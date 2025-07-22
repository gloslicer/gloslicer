# llm_verification.py

import ast
import argparse
import os
import json
from typing import Dict, List, Set, Any, Tuple, Optional

class ProgramGraphSlicer:
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.python_files = self._collect_python_files(project_root)

        self.ast_trees: Dict[str, ast.AST] = {}
        self.parent_maps: Dict[str, Dict[ast.AST, ast.AST]] = {}
        self.source_by_file: Dict[str, str] = {}
        self.dfg_edges: Dict[str, List[Dict[str, Any]]] = {}
        self.cfg_edges: Dict[str, List[Dict[str, Any]]] = {}
        self.cg_edges: Dict[str, List[Dict[str, Any]]] = {}

        for path in self.python_files:
            src = open(path, 'r', encoding='utf-8').read()
            tree = ast.parse(src)
            self.ast_trees[path] = tree
            self.source_by_file[path] = src
            self.parent_maps[path] = self._build_parent_map(tree)
            self.dfg_edges[path] = []
            self.cfg_edges[path] = []
            self.cg_edges[path] = []

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

    def _index_functions(self):
        for path, tree in self.ast_trees.items():
            pm = self.parent_maps[path]
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    cls = self._find_enclosing(node, pm, ast.ClassDef)
                    if cls:
                        qn = f"{cls.name}.{node.name}"
                    else:
                        qn = node.name
                    self.func_index[qn] = (path, node)

    def _build_cfgs(self):
        for path, tree in self.ast_trees.items():
            self._build_cfg_for_file(path, tree)

    def _build_cfg_for_file(self, path: str, tree: ast.AST):
        pm = self.parent_maps[path]
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                controlled_stmts = []
                if isinstance(node, ast.If):
                    controlled_stmts.extend(node.body)
                    controlled_stmts.extend(node.orelse)
                elif isinstance(node, (ast.While, ast.For)):
                    controlled_stmts.extend(node.body)
                    if hasattr(node, 'orelse'):
                        controlled_stmts.extend(node.orelse)
                for stmt in controlled_stmts:
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
            self._build_call_graph_for_file(path, tree)

    def _build_call_graph_for_file(self, path: str, tree: ast.AST):
        pm = self.parent_maps[path]
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                caller_func = self._find_enclosing(node, pm, ast.FunctionDef)
                callee_name = None
                if isinstance(node.func, ast.Name):
                    callee_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    callee_name = node.func.attr
                if callee_name and caller_func:
                    for qn, (callee_path, callee_func) in self.func_index.items():
                        if callee_func.name == callee_name:
                            self.cg_edges[path].append({
                                'src': self._node_key(caller_func),
                                'dst': self._node_key(callee_func),
                                'src_line': node.lineno if hasattr(node, 'lineno') else None,
                                'dst_line': callee_func.lineno if hasattr(callee_func, 'lineno') else None,
                                'src_file': path,
                                'dst_file': callee_path,
                                'type': 'CG'
                            })

    def _find_defs_uses(self, path: str, var: str) -> Tuple[Set[ast.AST], Set[ast.AST]]:
        defs, uses = set(), set()
        tree, pm = self.ast_trees[path], self.parent_maps[path]
        for fp, fn in [v for v in self.func_index.values() if v[0] == path]:
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

    def _build_edges(self, path: str, var: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        defs, uses = self._find_defs_uses(path, var)
        bwd_edges: List[Dict[str, Any]] = []
        fwd_edges: List[Dict[str, Any]] = []
        for u in uses:
            for d in defs:
                if hasattr(u, 'lineno') and hasattr(d, 'lineno') and d.lineno <= u.lineno:
                    bwd_edges.append({
                        'src': self._node_key(u), 'dst': self._node_key(d),
                        'src_line': u.lineno, 'dst_line': d.lineno,
                        'type': 'DFG'
                    })
        for d in defs:
            for u in uses:
                if hasattr(d, 'lineno') and hasattr(u, 'lineno') and u.lineno >= d.lineno:
                    fwd_edges.append({
                        'src': self._node_key(d), 'dst': self._node_key(u),
                        'src_line': d.lineno, 'dst_line': u.lineno,
                        'type': 'DFG'
                    })
        for call in [n for n in ast.walk(self.ast_trees[path]) if isinstance(n, ast.Call)]:
            if isinstance(call.func, ast.Name):
                func_name = call.func.id
                for idx, arg in enumerate(call.args):
                    if isinstance(arg, ast.Name):
                        actual = arg
                        for qn, (fp, fn) in self.func_index.items():
                            if fn.name == func_name:
                                if idx < len(fn.args.args):
                                    param = fn.args.args[idx]
                                    if hasattr(actual, 'lineno') and hasattr(param, 'lineno'):
                                        bwd_edges.append({
                                            'src': self._node_key(actual),
                                            'dst': self._node_key(param),
                                            'src_line': actual.lineno,
                                            'dst_line': param.lineno,
                                            'type': 'DFG'
                                        })
                                        fwd_edges.append({
                                            'src': self._node_key(param),
                                            'dst': self._node_key(actual),
                                            'src_line': param.lineno,
                                            'dst_line': actual.lineno,
                                            'type': 'DFG'
                                        })
        pm = self.parent_maps[path]
        for node in uses | defs:
            for edge in self.cfg_edges[path]:
                if edge['dst'] == self._node_key(node):
                    bwd_edges.append(edge)
        for node in defs:
            for edge in self.cfg_edges[path]:
                if edge['src'] == self._node_key(node):
                    fwd_edges.append(edge)
        for node in uses | defs:
            func = self._find_enclosing(node, pm, ast.FunctionDef)
            if func:
                for path_key, edges in self.cg_edges.items():
                    for edge in edges:
                        if edge['dst'] == self._node_key(func):
                            bwd_edges.append(edge)
        for node in defs:
            func = self._find_enclosing(node, pm, ast.FunctionDef)
            if func:
                for path_key, edges in self.cg_edges.items():
                    for edge in edges:
                        if edge['src'] == self._node_key(func):
                            fwd_edges.append(edge)
        return bwd_edges, fwd_edges

    def _node_key(self, node: ast.AST) -> str:
        return str(id(node))

    def _expand_slice_with_cfg_cg(self, path: str, nodeset: Set[ast.AST], is_backward: bool) -> Set[ast.AST]:
        pm = self.parent_maps[path]
        result = nodeset.copy()
        node_ids = {self._node_key(n) for n in nodeset}
        for edge in self.cfg_edges[path]:
            if is_backward and edge['dst'] in node_ids:
                for n in ast.walk(self.ast_trees[path]):
                    if self._node_key(n) == edge['src']:
                        result.add(n)
                        break
            elif not is_backward and edge['src'] in node_ids:
                for n in ast.walk(self.ast_trees[path]):
                    if self._node_key(n) == edge['dst']:
                        result.add(n)
                        break
        funcs_in_slice = {self._find_enclosing(n, pm, ast.FunctionDef) for n in nodeset if self._find_enclosing(n, pm, ast.FunctionDef)}
        funcs_in_slice = {f for f in funcs_in_slice if f is not None}
        func_ids = {self._node_key(f) for f in funcs_in_slice}
        for path_key, edges in self.cg_edges.items():
            for edge in edges:
                if is_backward and edge['dst'] in func_ids:
                    src_file = edge.get('src_file', path_key)
                    if src_file in self.ast_trees:
                        for n in ast.walk(self.ast_trees[src_file]):
                            if self._node_key(n) == edge['src']:
                                result.add(n)
                                break
                elif not is_backward and edge['src'] in func_ids:
                    dst_file = edge.get('dst_file', path_key)
                    if dst_file in self.ast_trees:
                        for n in ast.walk(self.ast_trees[dst_file]):
                            if self._node_key(n) == edge['dst']:
                                result.add(n)
                                break
        return result

    def export_variable_level_json(self, output: str):
        results = []
        for path in self.python_files:
            full_code = self.source_by_file[path]
            for qn, (func_path, func_node) in self.func_index.items():
                if func_path != path:
                    continue
                for node in ast.walk(func_node):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        var = node.id
                        ln = node.lineno
                        col = node.col_offset
                        bwd_edges, fwd_edges = self._build_edges(path, var)
                        defs, uses = self._find_defs_uses(path, var)
                        b_nodes = {u for u in uses if hasattr(u, 'lineno') and u.lineno >= ln} | defs
                        f_nodes = defs | {u for u in uses if hasattr(u, 'lineno') and u.lineno >= ln}
                        b_nodes = self._expand_slice_with_cfg_cg(path, b_nodes, True)
                        f_nodes = self._expand_slice_with_cfg_cg(path, f_nodes, False)
                        forward_lines = sorted({n.lineno for n in f_nodes if hasattr(n, 'lineno')})
                        backward_lines = sorted({n.lineno for n in b_nodes if hasattr(n, 'lineno')})
                        results.append({
                            "file_name": os.path.relpath(path, self.project_root),
                            "variable": var,
                            "line": ln,
                            "col_offset": col,
                            "full_code": full_code,
                            "forward_slice": forward_lines,
                            "backward_slice": backward_lines
                        })
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(results)} variable slicing entries to {output}")

    def export_json(self, output: str):
        data = self.perform_slicing()
        if not os.path.exists(output):
            os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(data)} function slices to {output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', dest='root', required=True, help="Python source code root directory")
    parser.add_argument('-o', '--output', dest='out', required=True, help="Output JSON file path")
    parser.add_argument('--mode', default='var', choices=['func', 'var'])
    args = parser.parse_args()
    slicer = ProgramGraphSlicer(args.root)
    if args.mode == 'var':
        slicer.export_variable_level_json(args.out)
    else:
        slicer.export_json(args.out)
