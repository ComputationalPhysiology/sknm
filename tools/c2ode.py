"""Transcribe gotran-generated C++ headers back into gotranx .ode files.

Strategy: C expression -> valid Python expression text -> ast.parse -> AST
transform (IfExp->Conditional, Compare->Lt/Le/.., BoolOp->And/Or, pow->**)
-> ast.unparse.  Python's own parser handles precedence, so we never have to
reason about it by hand.

Usage:

    python3 tools/c2ode.py base_model_IM        # -> base_model_IM.ode
    python3 tools/c2ode.py PBM                  # -> PBM.ode
    python3 tools/c2ode.py --all

The headers are all gotran output, so one parser covers them: the only per-model
information is the citation written into the .ode as a comment.  Check the result
with `tools/validate_<model>.py`, which compares `rhs` against the compiled C
elementwise -- reading the transcription is not a check.
"""

import argparse
import ast
import re
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "references" / "SKNM_code"

#: Per-model provenance, written into the .ode header. Keyed by the header's stem, which is
#: also the .ode's stem and the generated module's name.
MODELS = {
    "base_model_IM": [
        "Immature wild-type hiPSC-CM base model, Jaeger, Wall & Tveito,",
        "PLoS Comput Biol 17(2):e1008089 (2021).  Parameterisation as used by",
        "Jaeger & Tveito, Sci Rep 13:16434 (2023).",
    ],
    "PBM": [
        "Phantom bursting model of the pancreatic beta cell, Bertram & Sherman,",
        "Bull Math Biol 66:1313-1344 (2004).  Parameterisation as used by",
        "Jaeger & Tveito, Sci Rep 13:16434 (2023).",
    ],
}

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("model", nargs="?", choices=sorted(MODELS), help="which header to transcribe")
parser.add_argument("--all", action="store_true", help="transcribe every known model")
args = parser.parse_args()
if not args.all and args.model is None:
    parser.error("give a model name or --all")
targets = sorted(MODELS) if args.all else [args.model]


# ---------------------------------------------------------------- init values
def init_values(src, fn):
    """{name: literal} in declaration (index) order from an init_*_values fn."""
    body = src.split("void " + fn + "(")[1].split("\n}")[0]
    out = OrderedDict()
    for m in re.finditer(r"\[(\d+)\]\s*=\s*([^;]+);\s*//\s*(\w+);", body):
        out[m.group(3)] = (int(m.group(1)), m.group(2).strip())
    return out


# ------------------------------------------------- C expression -> Python text
def strip_ns(s):
    return re.sub(r"\bstd::", "", s)


def match_paren(s, i):
    d = 0
    for k in range(i, len(s)):
        if s[k] in "([":
            d += 1
        elif s[k] in ")]":
            d -= 1
            if d == 0:
                return k
    raise ValueError("unbalanced: " + s[i : i + 60])


def split_top(s, sep):
    """Split on `sep` occurring at paren depth 0."""
    parts, d, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            d += 1
        elif ch in ")]":
            d -= 1
        if ch == sep and d == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def find_ternary(s):
    """Locate `?` and its matching `:` at depth 0, honouring nested ?: pairs."""
    d = 0
    for i, ch in enumerate(s):
        if ch in "([":
            d += 1
        elif ch in ")]":
            d -= 1
        elif ch == "?" and d == 0:
            d2 = q = 0
            for k in range(i + 1, len(s)):
                c = s[k]
                if c in "([":
                    d2 += 1
                elif c in ")]":
                    d2 -= 1
                elif c == "?" and d2 == 0:
                    q += 1
                elif c == ":" and d2 == 0:
                    if q == 0:
                        return i, k
                    q -= 1
            raise ValueError("no matching ':' in " + s)
    return None


def deternary(s):
    """Rewrite top-level C ternaries as Python conditional expressions."""
    r = find_ternary(s)
    if r is None:
        return s
    i, k = r
    cond, a, b = s[:i], s[i + 1 : k], s[k + 1 :]
    return "((%s) if (%s) else (%s))" % (deternary(a), cond.strip(), deternary(b))


def to_python(s):
    """Recurse into parens/commas first, then handle ternaries at this level."""
    out, i = [], 0
    while i < len(s):
        if s[i] in "([":
            j = match_paren(s, i)
            inner = ",".join(to_python(p) for p in split_top(s[i + 1 : j], ","))
            out.append(s[i] + inner + s[j])
            i = j + 1
        else:
            out.append(s[i])
            i += 1
    return deternary("".join(out))


# --------------------------------------------------------------- AST rewrite
CMP = {
    ast.Lt: "Lt",
    ast.LtE: "Le",
    ast.Gt: "Gt",
    ast.GtE: "Ge",
    ast.Eq: "Eq",
    ast.NotEq: "Ne",
}


class Rewrite(ast.NodeTransformer):
    def visit_IfExp(self, node):
        self.generic_visit(node)
        return ast.Call(
            ast.Name("Conditional", ast.Load()),
            [node.test, node.body, node.orelse],
            [],
        )

    def visit_Compare(self, node):
        self.generic_visit(node)
        assert len(node.ops) == 1, ast.dump(node)
        return ast.Call(
            ast.Name(CMP[type(node.ops[0])], ast.Load()),
            [node.left, node.comparators[0]],
            [],
        )

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        name = "And" if isinstance(node.op, ast.And) else "Or"
        return ast.Call(ast.Name(name, ast.Load()), node.values, [])

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            if node.func.id == "pow":
                return ast.BinOp(node.args[0], ast.Pow(), node.args[1])
            if node.func.id == "fabs":
                node.func.id = "abs"
        return node

    def visit_Name(self, node):
        if node.id == "t":  # the independent variable
            node.id = "time"
        return node


def convert(cexpr):
    py = to_python(strip_ns(cexpr))
    py = py.replace("&&", " and ").replace("||", " or ")
    tree = ast.parse(py, mode="eval")
    tree = ast.fix_missing_locations(Rewrite().visit(tree))
    return ast.unparse(tree)


# ------------------------------------------------------------------- emit
def block(kind, comp, entries):
    lines = ['%s("%s",' % (kind, comp)]
    lines += [
        "%s = %s%s" % (n, v, "," if i < len(entries) - 1 else "")
        for i, (n, v) in enumerate(entries)
    ]
    lines.append(")")
    return "\n".join(lines)


def transcribe(model):
    """Write <model>.ode from references/SKNM_code/<model>.h, and report what it found."""
    header = REFERENCE / f"{model}.h"
    src = header.read_text()

    states_init = init_values(src, "init_state_values")
    params_init = init_values(src, "init_parameters_values")
    state_by_idx = {i: n for n, (i, _) in states_init.items()}

    rhs_body = src.split(
        "void rhs(const double* states, const double t, const double* parameters,\n"
        "  double* values)\n{"
    )[1].split("\n}")[0]
    # Drop the "Assign states"/"Assign parameters" preamble.
    rhs_body = "// Expressions for the" + rhs_body.split("// Expressions for the", 1)[1]

    components = OrderedDict()  # name -> [(lhs, rhs_ode), ...]
    used_in = OrderedDict()  # symbol -> first component that references it
    current = None

    for chunk in rhs_body.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        for m in re.finditer(r"// Expressions for the (.+?) component", chunk):
            current = m.group(1)
            components.setdefault(current, [])
        stmt = re.sub(r"//.*", "", chunk).strip()
        stmt = " ".join(stmt.split())
        if not stmt:
            continue
        lhs, _, rhs = stmt.partition("=")
        lhs = lhs.strip()
        m = re.match(r"values\[(\d+)\]$", lhs)
        if m:
            target = "d%s_dt" % state_by_idx[int(m.group(1))]
        else:
            m2 = re.match(r"const double (\w+)$", lhs)
            if not m2:
                raise SystemExit("unparsed statement: %r" % stmt)
            target = m2.group(1)
        expr = convert(rhs.strip())
        components[current].append((target, expr))
        for sym in set(re.findall(r"\b[A-Za-z_]\w*\b", expr)):
            used_in.setdefault(sym, current)

    state_component = {}
    for comp, assigns in components.items():
        for lhs, _ in assigns:
            if lhs.startswith("d") and lhs.endswith("_dt"):
                state_component.setdefault(lhs[1:-3], comp)

    param_component = OrderedDict()
    for p in params_init:
        param_component[p] = used_in.get(p, "Misc")
    unused = [p for p in params_init if p not in used_in]

    out = [
        "# Transcribed from references/SKNM_code/%s.h (gotran-generated C++)" % model,
        *["# " + line for line in MODELS[model]],
        "# Generated by tools/c2ode.py -- do not edit by hand.",
        "",
    ]

    order = list(components)
    for comp in order:
        ps = [(p, params_init[p][1]) for p in params_init if param_component[p] == comp]
        if ps:
            out += [block("parameters", comp, ps), ""]
    if unused:
        out += [block("parameters", "Misc", [(p, params_init[p][1]) for p in unused]), ""]

    for comp in order:
        ss = [(s, states_init[s][1]) for s in states_init if state_component.get(s) == comp]
        if ss:
            out += [block("states", comp, ss), ""]

    for comp in order:
        out.append('expressions("%s")' % comp)
        for lhs, expr in components[comp]:
            out.append("%s = %s" % (lhs, expr))
        out.append("")

    destination = ROOT / f"{model}.ode"
    destination.write_text("\n".join(out) + "\n")

    n_expr = sum(len(v) for v in components.values())
    print(model)
    print("  components    :", len(components))
    print("  states        :", len(states_init), "grouped:", len(state_component))
    print("  parameters    :", len(params_init), "unused in rhs:", unused)
    print("  assignments   :", n_expr)
    print("  wrote        ", destination.relative_to(ROOT))


for name in targets:
    transcribe(name)
