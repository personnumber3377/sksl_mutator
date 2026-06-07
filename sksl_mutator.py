# sksl_mutator.py
# Drop-in Python custom mutator for Skia SkSL / RuntimeEffect / PipelineStage fuzzing.
#
# Supports both call styles:
#   custom_mutator(buf: bytearray, add_buf, max_size: int, callback=None)
# and the common libFuzzer Python bridge style:
#   custom_mutator(buf: bytearray, max_size: int, seed: int, callback)
#
# The mutator is intentionally "mostly syntax aware":
#   - tries to preserve blocks/statements/functions
#   - tracks approximate global/local variable declarations
#   - mutates tokens, expressions, statements, structs, uniforms, loops, arrays, matrices
#   - 5% dumb token havoc mode
#
# It does not depend on your existing GLSL parser. That makes it good for first-pass fuzzing.

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional


# ----------------------------
# SkSL-ish vocabulary
# ----------------------------

SCALAR_TYPES = [
    "bool", "int", "uint", "short", "ushort", "float", "half",
]

VECTOR_TYPES = [
    "bool2", "bool3", "bool4",
    "int2", "int3", "int4",
    "uint2", "uint3", "uint4",
    "short2", "short3", "short4",
    "ushort2", "ushort3", "ushort4",
    "float2", "float3", "float4",
    "half2", "half3", "half4",
]

MATRIX_TYPES = [
    "float2x2", "float3x3", "float4x4",
    "half2x2", "half3x3", "half4x4",
    # SkSL often supports rectangular matrices too, depending on settings.
    "float2x3", "float3x2", "float2x4", "float4x2", "float3x4", "float4x3",
    "half2x3", "half3x2", "half2x4", "half4x2", "half3x4", "half4x3",
]

OPAQUE_TYPES = [
    "shader", "colorFilter", "blender",
    "sampler2D", "sampler", "texture2D",
]

TYPES = SCALAR_TYPES + VECTOR_TYPES + MATRIX_TYPES + OPAQUE_TYPES

QUALIFIERS = [
    "uniform", "const", "in", "out", "inout", "layout", "noinline",
    "flat", "readonly", "writeonly", "coherent",
]

BUILTINS = [
    "sin", "cos", "tan", "asin", "acos", "atan",
    "pow", "exp", "log", "sqrt", "inversesqrt",
    "abs", "sign", "floor", "ceil", "fract", "mod",
    "min", "max", "clamp", "mix", "step", "smoothstep",
    "length", "distance", "dot", "cross", "normalize",
    "any", "all", "not",
    "sample",
    "toLinearSrgb", "fromLinearSrgb",
]

SPECIAL_NAMES = [
    "sk_FragColor", "sk_FragCoord", "sk_Clockwise",
    "coords", "color", "src", "dst",
]

LITERALS = [
    "0", "1", "-1", "2", "3", "4", "8", "16", "32", "64", "255",
    "0.0", "1.0", "-1.0", "0.5", "2.0", "3.14159",
    "true", "false",
]

BINOPS = ["+", "-", "*", "/", "%", "&&", "||", "==", "!=", "<", ">", "<=", ">=", "&", "|", "^"]
ASSIGNOPS = ["=", "+=", "-=", "*=", "/="]
UNARYOPS = ["+", "-", "!"]
SWIZZLES = [
    "x", "y", "z", "w", "xy", "yx", "xyz", "zyx", "xyzw", "wzyx",
    "r", "g", "b", "a", "rg", "gr", "rgb", "bgr", "rgba", "argb",
]

IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
TYPE_RE = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in sorted(TYPES, key=len, reverse=True)) + r")\b")
DECL_RE = re.compile(
    r"\b(?P<qual>(?:uniform|const|in|out|inout|flat|readonly|writeonly|coherent|noinline)\s+)*"
    r"(?P<type>" + "|".join(re.escape(t) for t in sorted(TYPES, key=len, reverse=True)) + r")\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<array>(?:\[[^\]]*\])*)"
)
FUNC_RE = re.compile(
    r"\b(?P<ret>" + "|".join(re.escape(t) for t in sorted(TYPES + ["void"], key=len, reverse=True)) + r")\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\((?P<params>[^)]*)\)\s*\{"
)
STRUCT_RE = re.compile(r"\bstruct\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)?\s*\{(?P<body>.*?)\}\s*(?P<vars>[^;]*)\;", re.S)


# ----------------------------
# Helpers
# ----------------------------

@dataclass
class VarInfo:
    name: str
    typ: str
    scope_depth: int = 0
    is_global: bool = False
    is_array: bool = False


@dataclass
class Context:
    globals: dict[str, VarInfo] = field(default_factory=dict)
    locals: dict[str, VarInfo] = field(default_factory=dict)
    funcs: dict[str, tuple[str, list[str]]] = field(default_factory=dict)
    structs: dict[str, list[tuple[str, str]]] = field(default_factory=dict)

    def all_vars(self) -> dict[str, VarInfo]:
        out = {}
        out.update(self.globals)
        out.update(self.locals)
        return out

    def vars_by_type(self, typ: str) -> list[str]:
        exact = [v.name for v in self.all_vars().values() if v.typ == typ]
        if exact:
            return exact

        # Allow scalar/vector family substitutions.
        family = type_family(typ)
        fuzzy = [v.name for v in self.all_vars().values() if type_family(v.typ) == family]
        return fuzzy


def _seed_from_buf(buf: bytes, seed: Optional[int] = None) -> int:
    if seed is not None:
        return int(seed) & 0xFFFFFFFF
    h = hashlib.sha256(buf[:4096]).digest()
    return int.from_bytes(h[:8], "little")


def decode(buf: bytearray | bytes) -> str:
    # Prefer UTF-8; SkSL is source text. latin1 fallback preserves bytes.
    b = bytes(buf)
    try:
        return b.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return b.decode("latin1", errors="ignore")


def encode(s: str, max_size: int) -> bytearray:
    b = s.encode("utf-8", errors="ignore")
    if len(b) > max_size:
        b = b[:max_size]
        # Avoid ending in the middle of a UTF-8 sequence.
        b = b.decode("utf-8", errors="ignore").encode("utf-8")
    return bytearray(b)


def clamp_source(s: str, max_size: int) -> str:
    if len(s.encode("utf-8", errors="ignore")) <= max_size:
        return s
    return s.encode("utf-8", errors="ignore")[:max_size].decode("utf-8", errors="ignore")


def coin(rng: random.Random, p: float) -> bool:
    return rng.random() < p


def choice(rng: random.Random, xs):
    return xs[rng.randrange(len(xs))] if xs else None


def type_family(t: str) -> str:
    if t.startswith("half"):
        return "half"
    if t.startswith("float"):
        return "float"
    if t.startswith("int"):
        return "int"
    if t.startswith("uint"):
        return "uint"
    if t.startswith("bool"):
        return "bool"
    return t


def vector_width(t: str) -> int:
    m = re.search(r"([234])$", t)
    if m:
        return int(m.group(1))
    return 1


def matrix_shape(t: str) -> tuple[int, int] | None:
    m = re.search(r"([234])x([234])$", t)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def scalar_for(t: str) -> str:
    if t.startswith("half"):
        return "half"
    if t.startswith("float"):
        return "float"
    if t.startswith("int"):
        return "int"
    if t.startswith("uint"):
        return "uint"
    if t.startswith("bool"):
        return "bool"
    return "float"


def fresh_name(rng: random.Random, prefix="v") -> str:
    return f"{prefix}_{rng.randrange(1_000_000):x}"


def split_statements(block_text: str) -> list[str]:
    # Best-effort statement splitter that respects braces and parentheses.
    out, start, depth_paren, depth_brace = [], 0, 0, 0
    for i, ch in enumerate(block_text):
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(0, depth_paren - 1)
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace = max(0, depth_brace - 1)
        elif ch == ";" and depth_paren == 0 and depth_brace == 0:
            out.append(block_text[start:i + 1])
            start = i + 1
    tail = block_text[start:].strip()
    if tail:
        out.append(tail)
    return out


def find_top_level_blocks(src: str) -> list[tuple[int, int]]:
    # Returns ranges of function bodies / block bodies: indices including braces.
    ranges = []
    stack = []
    in_line = in_block = in_str = False
    quote = ""
    i = 0
    while i < len(src):
        ch = src[i]
        nxt = src[i + 1] if i + 1 < len(src) else ""

        if in_line:
            if ch == "\n":
                in_line = False
            i += 1
            continue
        if in_block:
            if ch == "*" and nxt == "/":
                in_block = False
                i += 2
                continue
            i += 1
            continue
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                in_str = False
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block = True
            i += 2
            continue
        if ch in ("'", '"'):
            in_str = True
            quote = ch
            i += 1
            continue

        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            ranges.append((start, i + 1))
        i += 1
    return ranges


def extract_functions(src: str) -> list[tuple[int, int, str, str, str]]:
    funcs = []
    for m in FUNC_RE.finditer(src):
        body_start = m.end() - 1
        end = matching_brace(src, body_start)
        if end is not None:
            funcs.append((m.start(), end + 1, m.group("ret"), m.group("name"), m.group("params")))
    return funcs


def matching_brace(src: str, open_idx: int) -> Optional[int]:
    depth = 0
    for i in range(open_idx, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def analyze(src: str) -> Context:
    ctx = Context()

    # Struct fields.
    for sm in STRUCT_RE.finditer(src):
        name = sm.group("name") or ""
        if not name:
            continue
        fields = []
        for dm in DECL_RE.finditer(sm.group("body")):
            fields.append((dm.group("type"), dm.group("name")))
        ctx.structs[name] = fields

    # Function signatures.
    for fm in FUNC_RE.finditer(src):
        params = []
        for dm in DECL_RE.finditer(fm.group("params")):
            params.append(dm.group("type"))
        ctx.funcs[fm.group("name")] = (fm.group("ret"), params)

    # Approx global/local: declarations before first function body are globals; rest locals.
    funcs = extract_functions(src)
    first_func = min([f[0] for f in funcs], default=len(src))
    for dm in DECL_RE.finditer(src):
        name = dm.group("name")
        typ = dm.group("type")
        if name in BUILTINS or name in TYPES:
            continue
        is_global = dm.start() < first_func or "uniform" in (dm.group("qual") or "")
        vi = VarInfo(name=name, typ=typ, is_global=is_global, is_array=bool(dm.group("array")))
        if is_global:
            ctx.globals[name] = vi
        else:
            ctx.locals[name] = vi

    return ctx


# ----------------------------
# Generation
# ----------------------------

def gen_literal_for_type(typ: str, rng: random.Random) -> str:
    fam = type_family(typ)
    if fam == "bool":
        return choice(rng, ["true", "false", "bool(0)", "bool(1)"])
    if fam in ("int", "short"):
        return str(choice(rng, [-16, -2, -1, 0, 1, 2, 4, 8, 16, 255]))
    if fam in ("uint", "ushort"):
        return str(choice(rng, [0, 1, 2, 4, 8, 16, 255])) + "u"
    return choice(rng, ["0.0", "1.0", "-1.0", "0.5", "2.0", "3.14159"])


def gen_expr(typ: str, ctx: Context, rng: random.Random, depth: int = 0) -> str:
    if depth > 3:
        vars_ = ctx.vars_by_type(typ)
        if vars_ and coin(rng, 0.7):
            return choice(rng, vars_)
        return gen_constructor(typ, ctx, rng, depth + 1)

    vars_ = ctx.vars_by_type(typ)
    options = []

    if vars_:
        options.append(lambda: choice(rng, vars_))

    options += [
        lambda: gen_constructor(typ, ctx, rng, depth + 1),
        lambda: gen_literal_for_type(typ, rng),
        lambda: f"({gen_expr(typ, ctx, rng, depth + 1)} {choice(rng, BINOPS[:4])} {gen_expr(typ, ctx, rng, depth + 1)})",
        lambda: f"({gen_expr('bool', ctx, rng, depth + 1)} ? {gen_expr(typ, ctx, rng, depth + 1)} : {gen_expr(typ, ctx, rng, depth + 1)})",
    ]

    fam = type_family(typ)
    if fam in ("float", "half"):
        options.append(lambda: f"{choice(rng, ['sin','cos','sqrt','abs','floor','fract'])}({gen_expr(typ, ctx, rng, depth + 1)})")
    if fam == "bool":
        options.append(lambda: f"({gen_expr('float', ctx, rng, depth + 1)} {choice(rng, ['<','>','==','!='])} {gen_expr('float', ctx, rng, depth + 1)})")

    # Function calls with matching return.
    funcs = [name for name, (ret, params) in ctx.funcs.items() if ret == typ and name != "main"]
    if funcs:
        def call_user():
            name = choice(rng, funcs)
            _, params = ctx.funcs[name]
            args = [gen_expr(p, ctx, rng, depth + 1) for p in params[:6]]
            return f"{name}({', '.join(args)})"
        options.append(call_user)

    return choice(rng, options)()


def gen_constructor(typ: str, ctx: Context, rng: random.Random, depth: int = 0) -> str:
    if typ in SCALAR_TYPES:
        return gen_literal_for_type(typ, rng)

    if typ in VECTOR_TYPES:
        n = vector_width(typ)
        base = scalar_for(typ)
        if coin(rng, 0.5):
            return f"{typ}({gen_literal_for_type(base, rng)})"
        args = [gen_expr(base, ctx, rng, depth + 1) for _ in range(n)]
        return f"{typ}({', '.join(args)})"

    shape = matrix_shape(typ)
    if shape:
        cols, rows = shape
        base = scalar_for(typ)
        if coin(rng, 0.35):
            return f"{typ}({gen_literal_for_type(base, rng)})"
        args = [gen_expr(base, ctx, rng, depth + 1) for _ in range(cols * rows)]
        return f"{typ}({', '.join(args)})"

    if typ in ctx.structs:
        args = [gen_expr(ft, ctx, rng, depth + 1) for ft, _ in ctx.structs[typ]]
        return f"{typ}({', '.join(args)})"

    return gen_literal_for_type("float", rng)


def gen_decl(ctx: Context, rng: random.Random) -> str:
    typ = choice(rng, SCALAR_TYPES + VECTOR_TYPES + MATRIX_TYPES)
    name = fresh_name(rng)
    init = gen_expr(typ, ctx, rng)
    ctx.locals[name] = VarInfo(name, typ)
    return f"{typ} {name} = {init};"


def gen_stmt(ctx: Context, rng: random.Random) -> str:
    vars_ = list(ctx.all_vars().values())
    assignable = [v for v in vars_ if v.typ not in OPAQUE_TYPES]
    kinds = ["decl", "expr", "if", "for", "return"]
    k = choice(rng, kinds)

    if k == "decl" or not assignable:
        return gen_decl(ctx, rng)

    if k == "expr":
        v = choice(rng, assignable)
        return f"{v.name} {choice(rng, ASSIGNOPS)} {gen_expr(v.typ, ctx, rng)};"

    if k == "if":
        return (
            f"if ({gen_expr('bool', ctx, rng)}) {{\n"
            f"    {gen_stmt(ctx, rng)}\n"
            f"}} else {{\n"
            f"    {gen_stmt(ctx, rng)}\n"
            f"}}"
        )

    if k == "for":
        idx = fresh_name(rng, "i")
        return (
            f"for (int {idx}=0; {idx}<3; ++{idx}) {{\n"
            f"    {gen_stmt(ctx, rng)}\n"
            f"}}"
        )

    # Prefer legal-ish return type for common RuntimeEffect main.
    if "main" in ctx.funcs:
        ret, _ = ctx.funcs["main"]
        if ret != "void":
            return f"return {gen_expr(ret, ctx, rng)};"
    return f"{gen_expr('float', ctx, rng)};"


def gen_struct(ctx: Context, rng: random.Random) -> str:
    name = fresh_name(rng, "S")
    fields = []
    for _ in range(rng.randint(1, 5)):
        ft = choice(rng, SCALAR_TYPES + VECTOR_TYPES + MATRIX_TYPES)
        fn = fresh_name(rng, "f")
        fields.append((ft, fn))
    ctx.structs[name] = fields
    body = "\n".join(f"    {ft} {fn};" for ft, fn in fields)
    return f"struct {name} {{\n{body}\n}};\n"


def ensure_runtime_main(src: str, rng: random.Random) -> str:
    # Many GLSL seeds have void main(), but RuntimeEffect usually likes half4 main(float2 coords).
    # Don't always rewrite; diversity is good.
    if "main" not in src:
        return src + "\nhalf4 main(float2 coords) { return half4(1); }\n"
    if coin(rng, 0.15):
        src = re.sub(r"\bvoid\s+main\s*\(\s*(?:void)?\s*\)", "half4 main(float2 coords)", src, count=1)
        # If we changed void-main to half4-main and there is no return, add one at end of first main block.
        funcs = extract_functions(src)
        for start, end, ret, name, params in funcs:
            if name == "main" and ret == "half4" and "return" not in src[start:end]:
                close = end - 1
                src = src[:close] + "\nreturn half4(1);\n" + src[close:]
                break
    return src


def glsl_to_sksl_cleanup(src: str, rng: random.Random) -> str:
    # Remove GLSL-specific version/extension lines often rejected by SkSL.
    src = re.sub(r"^\s*#version[^\n]*\n", "", src, flags=re.M)
    if coin(rng, 0.85):
        src = re.sub(r"^\s*#extension[^\n]*\n", "", src, flags=re.M)

    replacements = {
        "vec2": "float2", "vec3": "float3", "vec4": "float4",
        "ivec2": "int2", "ivec3": "int3", "ivec4": "int4",
        "uvec2": "uint2", "uvec3": "uint3", "uvec4": "uint4",
        "bvec2": "bool2", "bvec3": "bool3", "bvec4": "bool4",
        "mat2": "float2x2", "mat3": "float3x3", "mat4": "float4x4",
        "gl_FragColor": "sk_FragColor",
        "texture2D": "sample",
        "texture": "sample",
    }

    # Usually translate GLSL type spellings to SkSL spellings.
    if coin(rng, 0.80):
        for a, b in replacements.items():
            src = re.sub(rf"\b{re.escape(a)}\b", b, src)

    # Drop precision qualifiers sometimes; SkSL accepts half/float distinction instead.
    if coin(rng, 0.55):
        src = re.sub(r"\b(?:lowp|mediump|highp)\s+", "", src)

    return src


# ----------------------------
# Mutations
# ----------------------------

def mutate_identifier(src: str, ctx: Context, rng: random.Random) -> str:
    names = [n for n in ctx.all_vars().keys() if n not in SPECIAL_NAMES]
    if len(names) < 2:
        return src
    old = choice(rng, names)
    old_t = ctx.all_vars()[old].typ
    same = [n for n in names if n != old and ctx.all_vars()[n].typ == old_t]
    new = choice(rng, same or [n for n in names if n != old])
    return re.sub(rf"\b{re.escape(old)}\b", new, src, count=rng.randint(1, 3))


def mutate_type(src: str, rng: random.Random) -> str:
    matches = list(TYPE_RE.finditer(src))
    if not matches:
        return src
    m = choice(rng, matches)
    old = m.group(0)
    fam = type_family(old)
    pool = [t for t in TYPES if type_family(t) == fam] or TYPES
    new = choice(rng, pool)
    return src[:m.start()] + new + src[m.end():]


def mutate_literal(src: str, rng: random.Random) -> str:
    lit_re = re.compile(r"\b(?:\d+\.\d+|\d+u?|-?\d+|true|false)\b")
    matches = list(lit_re.finditer(src))
    if not matches:
        return src
    m = choice(rng, matches)
    new = choice(rng, LITERALS)
    return src[:m.start()] + new + src[m.end():]


def mutate_operator(src: str, rng: random.Random) -> str:
    ops = sorted(BINOPS + ASSIGNOPS + ["++", "--"], key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(o) for o in ops))
    matches = list(pattern.finditer(src))
    if not matches:
        return src
    m = choice(rng, matches)
    old = m.group(0)
    if old in ASSIGNOPS:
        new = choice(rng, ASSIGNOPS)
    elif old in ("++", "--"):
        new = choice(rng, ["++", "--"])
    else:
        new = choice(rng, BINOPS)
    return src[:m.start()] + new + src[m.end():]


def mutate_swizzle(src: str, rng: random.Random) -> str:
    matches = list(re.finditer(r"\.([xyzwrgba]{1,4})\b", src))
    if not matches:
        return src
    m = choice(rng, matches)
    new = "." + choice(rng, SWIZZLES)
    return src[:m.start()] + new + src[m.end():]


def insert_statement(src: str, ctx: Context, rng: random.Random) -> str:
    blocks = find_top_level_blocks(src)
    if not blocks:
        return src + "\n" + gen_stmt(ctx, rng) + "\n"
    start, end = choice(rng, blocks)
    insert_at = end - 1
    stmt = "\n    " + gen_stmt(ctx, rng) + "\n"
    return src[:insert_at] + stmt + src[insert_at:]


def delete_statement(src: str, rng: random.Random) -> str:
    blocks = find_top_level_blocks(src)
    rng.shuffle(blocks)
    for start, end in blocks:
        inner = src[start + 1:end - 1]
        stmts = split_statements(inner)
        if stmts:
            victim = choice(rng, stmts)
            new_inner = inner.replace(victim, "", 1)
            return src[:start + 1] + new_inner + src[end - 1:]
    return src


def duplicate_or_move_statement(src: str, rng: random.Random) -> str:
    blocks = find_top_level_blocks(src)
    if not blocks:
        return src
    start, end = choice(rng, blocks)
    inner = src[start + 1:end - 1]
    stmts = split_statements(inner)
    if not stmts:
        return src
    stmt = choice(rng, stmts)
    insert_pos = end - 1
    if coin(rng, 0.5):
        # duplicate
        return src[:insert_pos] + "\n" + stmt + "\n" + src[insert_pos:]
    # move within same block
    new_inner = inner.replace(stmt, "", 1)
    pieces = split_statements(new_inner)
    idx = rng.randrange(len(pieces) + 1) if pieces else 0
    pieces.insert(idx, stmt)
    return src[:start + 1] + "\n".join(pieces) + src[end - 1:]


def wrap_expr_in_constructor(src: str, ctx: Context, rng: random.Random) -> str:
    matches = list(IDENT_RE.finditer(src))
    if not matches:
        return src
    m = choice(rng, matches)
    ident = m.group(0)
    if ident in TYPES or ident in BUILTINS or ident in QUALIFIERS:
        return src
    typ = choice(rng, VECTOR_TYPES + MATRIX_TYPES + SCALAR_TYPES)
    new = f"{typ}({ident})"
    return src[:m.start()] + new + src[m.end():]


def add_global(src: str, ctx: Context, rng: random.Random) -> str:
    if coin(rng, 0.25):
        return gen_struct(ctx, rng) + src

    typ = choice(rng, SCALAR_TYPES + VECTOR_TYPES + MATRIX_TYPES)
    name = fresh_name(rng, "u")
    q = choice(rng, ["uniform ", "const ", ""])
    init = ""
    if q != "uniform " and coin(rng, 0.8):
        init = " = " + gen_constructor(typ, ctx, rng)
    decl = f"{q}{typ} {name}{init};\n"
    ctx.globals[name] = VarInfo(name, typ, is_global=True)
    return decl + src


def mutate_layout_or_qualifier(src: str, rng: random.Random) -> str:
    if coin(rng, 0.5):
        # Insert SkSL-ish layout binding.
        return f"layout(binding={rng.randrange(0, 8)}) " + src
    qs = ["uniform", "const", "in", "out", "inout", "noinline"]
    matches = list(re.finditer(r"\b(?:" + "|".join(qs) + r")\b", src))
    if not matches:
        return src
    m = choice(rng, matches)
    return src[:m.start()] + choice(rng, qs) + src[m.end():]


def dumb_token_havoc(src: str, rng: random.Random) -> str:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+\.\d+|\d+|==|!=|<=|>=|\+\+|--|&&|\|\||[{}()\[\];,.:+\-*/%<>=!?]", src)
    if not tokens:
        return src

    op = rng.choice(["shuffle_window", "delete", "duplicate", "replace", "insert_garbage"])

    if op == "shuffle_window" and len(tokens) > 4:
        i = rng.randrange(0, len(tokens) - 3)
        j = min(len(tokens), i + rng.randint(3, 12))
        window = tokens[i:j]
        rng.shuffle(window)
        tokens[i:j] = window
    elif op == "delete" and len(tokens) > 1:
        del tokens[rng.randrange(len(tokens))]
    elif op == "duplicate":
        i = rng.randrange(len(tokens))
        tokens.insert(i, tokens[i])
    elif op == "replace":
        i = rng.randrange(len(tokens))
        tokens[i] = choice(rng, TYPES + BUILTINS + LITERALS + ["{", "}", ";", "(", ")"])
    else:
        i = rng.randrange(len(tokens) + 1)
        tokens.insert(i, choice(rng, ["@", "#", "layout", "half4x4", "struct", "return", ";;;"]))

    return untokenize(tokens)


def untokenize(tokens: list[str]) -> str:
    out = []
    no_space_before = set(")]};,.:")
    no_space_after = set("([{.")
    prev = ""
    for t in tokens:
        if not out:
            out.append(t)
        elif t in no_space_before or prev in no_space_after:
            out.append(t)
        else:
            out.append(" " + t)
        prev = t
    return "".join(out)


def smart_mutate(src: str, rng: random.Random, max_size: int) -> str:
    src = glsl_to_sksl_cleanup(src, rng)
    src = ensure_runtime_main(src, rng)
    ctx = analyze(src)

    mutations = [
        lambda s: mutate_identifier(s, ctx, rng),
        lambda s: mutate_type(s, rng),
        lambda s: mutate_literal(s, rng),
        lambda s: mutate_operator(s, rng),
        lambda s: mutate_swizzle(s, rng),
        lambda s: insert_statement(s, ctx, rng),
        lambda s: delete_statement(s, rng),
        lambda s: duplicate_or_move_statement(s, rng),
        lambda s: wrap_expr_in_constructor(s, ctx, rng),
        lambda s: add_global(s, ctx, rng),
        lambda s: mutate_layout_or_qualifier(s, rng),
    ]

    # Apply 1-4 smart mutations.
    for _ in range(rng.randint(1, 4)):
        src = choice(rng, mutations)(src)
        src = clamp_source(src, max_size)

    return src


# ----------------------------
# Crossover
# ----------------------------

def extract_global_prefix(src: str) -> str:
    funcs = extract_functions(src)
    if not funcs:
        return src
    first = min(f[0] for f in funcs)
    return src[:first]


def extract_function_bodies(src: str) -> list[str]:
    bodies = []
    for start, end, ret, name, params in extract_functions(src):
        bodies.append(src[start:end])
    return bodies


def crossover_sources(a: str, b: str, rng: random.Random, max_size: int) -> str:
    a = glsl_to_sksl_cleanup(a, rng)
    b = glsl_to_sksl_cleanup(b, rng)

    ga, gb = extract_global_prefix(a), extract_global_prefix(b)
    fa, fb = extract_function_bodies(a), extract_function_bodies(b)

    mode = rng.choice(["globals_mix", "function_splice", "statement_splice", "block_append"])

    if mode == "globals_mix":
        funcs = fa or fb
        src = ga + "\n" + gb + "\n" + "\n".join(funcs[:3])

    elif mode == "function_splice":
        funcs = []
        funcs.extend(rng.sample(fa, min(len(fa), rng.randint(0, 2))) if fa else [])
        funcs.extend(rng.sample(fb, min(len(fb), rng.randint(1, 3))) if fb else [])
        rng.shuffle(funcs)
        src = ga + "\n" + gb + "\n" + "\n".join(funcs)

    elif mode == "statement_splice":
        src = a
        blocks_b = find_top_level_blocks(b)
        blocks_a = find_top_level_blocks(src)
        if blocks_a and blocks_b:
            bs, be = choice(rng, blocks_b)
            stmts = split_statements(b[bs + 1:be - 1])
            if stmts:
                stmt = choice(rng, stmts)
                as_, ae = choice(rng, blocks_a)
                src = src[:ae - 1] + "\n" + stmt + "\n" + src[ae - 1:]

    else:
        src = a + "\n/* crossover */\n" + gb
        if fb:
            src += "\n" + choice(rng, fb)

    src = ensure_runtime_main(src, rng)
    return clamp_source(src, max_size)


# ----------------------------
# Public libFuzzer API
# ----------------------------

def _parse_mutator_args(buf, add_buf=None, max_size=None, callback=None):
    """Accept both your requested signature and common bridge signature."""
    seed = None
    real_add_buf = None
    real_callback = callback

    # Bridge style: custom_mutator(data, MaxSize, Seed, callback)
    if isinstance(add_buf, int) and isinstance(max_size, int):
        real_max_size = add_buf
        seed = max_size
        real_add_buf = None
    else:
        real_max_size = max_size if isinstance(max_size, int) else len(buf) * 2 + 1024
        real_add_buf = add_buf

    return real_add_buf, int(real_max_size), seed, real_callback


def custom_mutator(buf: bytearray, add_buf=None, max_size: int | None = None, callback=None) -> bytearray:
    add_buf, max_size, seed, callback = _parse_mutator_args(buf, add_buf, max_size, callback)
    data = bytes(buf)
    rng = random.Random(_seed_from_buf(data, seed))

    # 5% dumb mutation exactly as requested.
    if coin(rng, 0.05):
        out = dumb_token_havoc(decode(buf), rng)
        return encode(out, max_size)

    # Sometimes let libFuzzer do byte-level mutation first, then clean it up as SkSL.
    if callback is not None and coin(rng, 0.15):
        try:
            tmp = bytearray(buf)
            callback(tmp, max_size)
            src = decode(tmp)
        except Exception:
            src = decode(buf)
    else:
        src = decode(buf)

    # If add_buf exists, occasionally use it as a crossover-ish donor.
    if add_buf and coin(rng, 0.20):
        try:
            src = crossover_sources(src, decode(add_buf), rng, max_size)
        except Exception:
            pass

    try:
        out = smart_mutate(src, rng, max_size)
    except Exception:
        # Never let mutator bugs kill fuzzing.
        out = dumb_token_havoc(src, rng)

    return encode(out, max_size)


def custom_crossover(data1: bytearray, data2: bytearray, max_size: int, seed: int) -> bytearray:
    rng = random.Random(int(seed) & 0xFFFFFFFF)
    a = decode(data1)
    b = decode(data2)

    # 5% deliberately dumb crossover.
    if coin(rng, 0.05):
        ta = re.findall(r"\w+|[^\w\s]", a)
        tb = re.findall(r"\w+|[^\w\s]", b)
        cuta = rng.randrange(len(ta) + 1) if ta else 0
        cutb = rng.randrange(len(tb) + 1) if tb else 0
        return encode(untokenize(ta[:cuta] + tb[cutb:]), max_size)

    try:
        out = crossover_sources(a, b, rng, max_size)
    except Exception:
        # Byte-ish fallback.
        ba, bb = bytes(data1), bytes(data2)
        ca = rng.randrange(len(ba) + 1) if ba else 0
        cb = rng.randrange(len(bb) + 1) if bb else 0
        out_b = ba[:ca] + bb[cb:]
        return bytearray(out_b[:max_size])

    return encode(out, max_size)


# Optional local smoke-test:
if __name__ == "__main__":
    sample = bytearray(b"""
uniform half4 colorGreen, colorRed;
half4 main(float2 coords) {
    float x = coords.x;
    return x > 0 ? colorGreen : colorRed;
}
""")
    for i in range(5):
        sample = custom_mutator(sample, None, 4096)
        print(sample.decode("utf-8", "ignore"))
        print("---")
