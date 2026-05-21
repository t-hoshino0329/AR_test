import tkinter as tk
import cmath
import math
from typing import Optional


# --- Backend: EML primitive (paper constraint: single operator) ---
# eml(x, y) = exp(x) - ln(y)
# All evaluation uses complex arithmetic and this single primitive.
def _is_inf(z: complex) -> bool:
    return math.isinf(z.real) or math.isinf(z.imag)


def _safe_log(z: complex) -> complex:
    z = complex(z)
    if z == 0:
        # ln(0) = -inf
        return complex(float("-inf"), 0.0)
    if _is_inf(z):
        return complex(float("inf"), cmath.phase(z))
    return cmath.log(z)


def _safe_exp(z: complex) -> complex:
    z = complex(z)
    if math.isinf(z.real):
        return complex(0.0, 0.0) if z.real < 0 else complex(float("inf"), 0.0)
    return cmath.exp(z)


def eml(x: complex, y: complex) -> complex:
    try:
        return _safe_exp(x) - _safe_log(y)
    except (OverflowError, ValueError):
        return complex(float("inf"), 0.0)


# --- EML Tree Node (for visualization + evaluation) ---
class EMLNode:
    def __init__(self, kind: str, value=None, text=None, left=None, right=None):
        self.kind = kind
        self.value = value
        self.text = text
        self.left = left
        self.right = right

    def pretty(self, indent: int = 0) -> str:
        pad = "  " * indent
        if self.kind == "const":
            return f"{pad}{self.text}"
        return (
            f"{pad}eml(\n"
            f"{self.left.pretty(indent + 1)},\n"
            f"{self.right.pretty(indent + 1)}\n"
            f"{pad})"
        )

    def label(self) -> str:
        return self.text if self.kind == "const" else "eml"


def _const_node(value: float, text: Optional[str] = None) -> EMLNode:
    if text is None:
        text = f"{value:.12g}"
    return EMLNode("const", value=value, text=text)


def one_node() -> EMLNode:
    return _const_node(1.0, "1")


def _eml_node(x: EMLNode, y: EMLNode) -> EMLNode:
    return EMLNode("eml", left=x, right=y)


# --- Compiler macros (pure EML expansions) ---
def eml_exp_node(x: EMLNode) -> EMLNode:
    return _eml_node(x, one_node())


def eml_ln_node(x: EMLNode) -> EMLNode:
    return _eml_node(one_node(), _eml_node(_eml_node(one_node(), x), one_node()))


def eml_zero_node() -> EMLNode:
    return eml_ln_node(one_node())


def eml_sub_node(x: EMLNode, y: EMLNode) -> EMLNode:
    return _eml_node(eml_ln_node(x), eml_exp_node(y))


def eml_neg_node(x: EMLNode) -> EMLNode:
    return eml_sub_node(eml_zero_node(), x)


def eml_add_node(x: EMLNode, y: EMLNode) -> EMLNode:
    return eml_sub_node(x, eml_neg_node(y))


def eml_mul_node(x: EMLNode, y: EMLNode) -> EMLNode:
    return eml_exp_node(eml_add_node(eml_ln_node(x), eml_ln_node(y)))


def eml_div_node(x: EMLNode, y: EMLNode) -> EMLNode:
    return eml_exp_node(eml_sub_node(eml_ln_node(x), eml_ln_node(y)))


def eml_pow_node(x: EMLNode, y: EMLNode) -> EMLNode:
    return eml_exp_node(eml_mul_node(y, eml_ln_node(x)))


def eml_two_node() -> EMLNode:
    return eml_add_node(one_node(), one_node())


def eml_four_node() -> EMLNode:
    return eml_add_node(eml_two_node(), eml_two_node())


def eml_half_node() -> EMLNode:
    return eml_div_node(one_node(), eml_two_node())


def eml_sqrt_node(x: EMLNode) -> EMLNode:
    return eml_exp_node(eml_mul_node(eml_half_node(), eml_ln_node(x)))


def eml_i_node() -> EMLNode:
    return eml_sqrt_node(eml_neg_node(one_node()))


def eml_sin_node(x: EMLNode) -> EMLNode:
    i = eml_i_node()
    ix = eml_mul_node(i, x)
    num = eml_sub_node(eml_exp_node(ix), eml_exp_node(eml_neg_node(ix)))
    den = eml_mul_node(eml_two_node(), i)
    return eml_div_node(num, den)


def eml_cos_node(x: EMLNode) -> EMLNode:
    i = eml_i_node()
    ix = eml_mul_node(i, x)
    num = eml_add_node(eml_exp_node(ix), eml_exp_node(eml_neg_node(ix)))
    return eml_div_node(num, eml_two_node())


def eml_tan_node(x: EMLNode) -> EMLNode:
    return eml_div_node(eml_sin_node(x), eml_cos_node(x))


def eml_sinh_node(x: EMLNode) -> EMLNode:
    num = eml_sub_node(eml_exp_node(x), eml_exp_node(eml_neg_node(x)))
    return eml_div_node(num, eml_two_node())


def eml_cosh_node(x: EMLNode) -> EMLNode:
    num = eml_add_node(eml_exp_node(x), eml_exp_node(eml_neg_node(x)))
    return eml_div_node(num, eml_two_node())


def eml_tanh_node(x: EMLNode) -> EMLNode:
    return eml_div_node(eml_sinh_node(x), eml_cosh_node(x))


def eml_asin_node(x: EMLNode) -> EMLNode:
    i = eml_i_node()
    x2 = eml_mul_node(x, x)
    inside = eml_sub_node(one_node(), x2)
    root = eml_sqrt_node(inside)
    term = eml_add_node(eml_mul_node(i, x), root)
    return eml_mul_node(eml_neg_node(i), eml_ln_node(term))


def eml_acos_node(x: EMLNode) -> EMLNode:
    i = eml_i_node()
    x2 = eml_mul_node(x, x)
    inside = eml_sub_node(one_node(), x2)
    root = eml_sqrt_node(inside)
    term = eml_add_node(x, eml_mul_node(i, root))
    return eml_mul_node(eml_neg_node(i), eml_ln_node(term))


def eml_atan_node(x: EMLNode) -> EMLNode:
    i = eml_i_node()
    ix = eml_mul_node(i, x)
    num = eml_sub_node(
        eml_ln_node(eml_sub_node(one_node(), ix)),
        eml_ln_node(eml_add_node(one_node(), ix)),
    )
    return eml_mul_node(eml_mul_node(i, eml_half_node()), num)


def eml_pi_node() -> EMLNode:
    return eml_mul_node(eml_four_node(), eml_atan_node(one_node()))


# --- Expression AST ---
class NumberNode:
    def __init__(self, value: float, text: str):
        self.value = value
        self.text = text


class ConstantNode:
    def __init__(self, name: str):
        self.name = name


class UnaryNode:
    def __init__(self, op: str, operand):
        self.op = op
        self.operand = operand


class BinaryNode:
    def __init__(self, op: str, left, right):
        self.op = op
        self.left = left
        self.right = right


class FuncNode:
    def __init__(self, name: str, arg):
        self.name = name
        self.arg = arg


FUNCTIONS = {
    "exp",
    "ln",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "sqrt",
}
CONSTANTS = {"e", "pi", "i"}


def tokenize(expr: str):
    tokens = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit() or ch == ".":
            start = i
            has_dot = ch == "."
            i += 1
            while i < len(expr) and (expr[i].isdigit() or (expr[i] == "." and not has_dot)):
                if expr[i] == ".":
                    has_dot = True
                i += 1
            if i < len(expr) and expr[i] in "eE":
                i += 1
                if i < len(expr) and expr[i] in "+-":
                    i += 1
                if i >= len(expr) or not expr[i].isdigit():
                    raise ValueError("Invalid exponent")
                while i < len(expr) and expr[i].isdigit():
                    i += 1
            num_text = expr[start:i]
            tokens.append(("NUMBER", num_text))
            continue
        if ch.isalpha():
            start = i
            i += 1
            while i < len(expr) and expr[i].isalpha():
                i += 1
            ident = expr[start:i]
            tokens.append(("IDENT", ident))
            continue
        if ch in "+-*/^()":
            tokens.append((ch, ch))
            i += 1
            continue
        raise ValueError(f"Invalid character: {ch}")
    return tokens


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _accept(self, kind):
        tok = self._peek()
        if tok and tok[0] == kind:
            self.pos += 1
            return tok
        return None

    def _expect(self, kind):
        tok = self._accept(kind)
        if not tok:
            raise ValueError(f"Expected {kind}")
        return tok

    def parse(self):
        node = self._parse_add_sub()
        if self._peek() is not None:
            raise ValueError("Unexpected token")
        return node

    def _parse_add_sub(self):
        node = self._parse_mul_div()
        while True:
            if self._accept("+"):
                node = BinaryNode("+", node, self._parse_mul_div())
            elif self._accept("-"):
                node = BinaryNode("-", node, self._parse_mul_div())
            else:
                break
        return node

    def _parse_mul_div(self):
        node = self._parse_pow()
        while True:
            if self._accept("*"):
                node = BinaryNode("*", node, self._parse_pow())
            elif self._accept("/"):
                node = BinaryNode("/", node, self._parse_pow())
            else:
                break
        return node

    def _parse_pow(self):
        node = self._parse_unary()
        if self._accept("^"):
            node = BinaryNode("^", node, self._parse_pow())
        return node

    def _parse_unary(self):
        if self._accept("+"):
            return UnaryNode("+", self._parse_unary())
        if self._accept("-"):
            return UnaryNode("-", self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self):
        tok = self._peek()
        if not tok:
            raise ValueError("Unexpected end")
        if tok[0] == "NUMBER":
            self.pos += 1
            return NumberNode(float(tok[1]), tok[1])
        if tok[0] == "IDENT":
            self.pos += 1
            ident = tok[1]
            if self._accept("("):
                arg = self._parse_add_sub()
                self._expect(")")
                if ident not in FUNCTIONS:
                    raise ValueError(f"Unknown function: {ident}")
                return FuncNode(ident, arg)
            if ident in CONSTANTS:
                return ConstantNode(ident)
            raise ValueError(f"Unknown identifier: {ident}")
        if self._accept("("):
            node = self._parse_add_sub()
            self._expect(")")
            return node
        raise ValueError("Unexpected token")


def compile_to_eml(node) -> EMLNode:
    if isinstance(node, NumberNode):
        return _const_node(node.value, node.text)
    if isinstance(node, ConstantNode):
        if node.name == "e":
            return _eml_node(one_node(), one_node())
        if node.name == "pi":
            return eml_pi_node()
        if node.name == "i":
            return eml_i_node()
    if isinstance(node, UnaryNode):
        compiled = compile_to_eml(node.operand)
        if node.op == "+":
            return compiled
        if node.op == "-":
            return eml_neg_node(compiled)
    if isinstance(node, BinaryNode):
        left = compile_to_eml(node.left)
        right = compile_to_eml(node.right)
        if node.op == "+":
            return eml_add_node(left, right)
        if node.op == "-":
            return eml_sub_node(left, right)
        if node.op == "*":
            return eml_mul_node(left, right)
        if node.op == "/":
            return eml_div_node(left, right)
        if node.op == "^":
            return eml_pow_node(left, right)
    if isinstance(node, FuncNode):
        arg = compile_to_eml(node.arg)
        if node.name == "exp":
            return eml_exp_node(arg)
        if node.name == "ln":
            return eml_ln_node(arg)
        if node.name == "sin":
            return eml_sin_node(arg)
        if node.name == "cos":
            return eml_cos_node(arg)
        if node.name == "tan":
            return eml_tan_node(arg)
        if node.name == "asin":
            return eml_asin_node(arg)
        if node.name == "acos":
            return eml_acos_node(arg)
        if node.name == "atan":
            return eml_atan_node(arg)
        if node.name == "sinh":
            return eml_sinh_node(arg)
        if node.name == "cosh":
            return eml_cosh_node(arg)
        if node.name == "tanh":
            return eml_tanh_node(arg)
        if node.name == "sqrt":
            return eml_sqrt_node(arg)
    raise ValueError("Unsupported expression")


def eval_eml(node: EMLNode) -> complex:
    if node.kind == "const":
        return complex(float(node.value), 0.0)
    return eml(eval_eml(node.left), eval_eml(node.right))


def format_complex(z: complex) -> str:
    if math.isinf(z.real) or math.isinf(z.imag):
        return str(z)
    if math.isnan(z.real) or math.isnan(z.imag):
        return "nan"
    if abs(z.imag) < 1e-12:
        return f"{z.real:.12g}"
    return f"{z.real:.6g}{z.imag:+.6g}j"


# --- Frontend (GUI) ---
class EMLCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title("Pure EML Calculator")
        self.root.geometry("980x640")
        self.display_var = tk.StringVar(value="0")
        self.expr_var = tk.StringVar(value="Expr:")
        self.current_input = ""
        self._create_widgets()

    def _create_widgets(self):
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief="raised")
        main_pane.pack(fill="both", expand=True)

        left_frame = tk.Frame(main_pane, bg="#f2f2f2")
        right_frame = tk.Frame(main_pane, bg="#f7f2ea")
        main_pane.add(left_frame, minsize=360)
        main_pane.add(right_frame, minsize=480)

        entry = tk.Entry(
            left_frame,
            textvariable=self.display_var,
            font=("Courier", 24),
            justify="right",
            bd=10,
            relief="flat",
            bg="#f4f4f4",
            fg="#000000",
        )
        entry.pack(fill="x", padx=12, pady=(16, 8))

        btn_frame = tk.Frame(left_frame, bg="#f2f2f2")
        btn_frame.pack(expand=True, fill="both", padx=8, pady=8)

        expr_label = tk.Label(
            right_frame,
            textvariable=self.expr_var,
            font=("Courier", 12, "bold"),
            bg="#f7f2ea",
            fg="#3b3b3b",
            anchor="w",
        )
        expr_label.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))

        self.tree_canvas = tk.Canvas(
            right_frame,
            bg="#fffaf5",
            highlightthickness=1,
            highlightbackground="#e5d8c8",
        )
        v_scroll = tk.Scrollbar(right_frame, orient="vertical", command=self.tree_canvas.yview)
        h_scroll = tk.Scrollbar(right_frame, orient="horizontal", command=self.tree_canvas.xview)
        self.tree_canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        self.tree_canvas.grid(row=1, column=0, sticky="nsew", padx=12)
        v_scroll.grid(row=1, column=1, sticky="ns", pady=4)
        h_scroll.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))
        right_frame.rowconfigure(1, weight=1)
        right_frame.columnconfigure(0, weight=1)

        self._set_tree_display("", None, "")

        buttons = [
            ("sin", "cos", "tan", "ln", "exp"),
            ("asin", "acos", "atan", "sqrt", "^"),
            ("7", "8", "9", "/", "pi"),
            ("4", "5", "6", "*", "e"),
            ("1", "2", "3", "-", "i"),
            ("0", ".", "(", ")", "+"),
            ("C", "DEL", "=", "", ""),
        ]

        for r, row in enumerate(buttons):
            btn_frame.rowconfigure(r, weight=1)
            for c, label in enumerate(row):
                btn_frame.columnconfigure(c, weight=1)
                if not label:
                    continue
                color = "#eeeeee"
                if label in ("/", "*", "-", "+", "=", "^"):
                    color = "#ff9500"
                if label in ("C", "DEL"):
                    color = "#ff3b30"
                btn = tk.Button(
                    btn_frame,
                    text=label,
                    font=("Arial", 14, "bold"),
                    bg=color,
                    command=lambda x=label: self._click(x),
                )
                btn.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)

    def _set_tree_display(self, expr: str, tree: Optional[EMLNode], error: str):
        if expr:
            self.expr_var.set(f"Expr: {expr}")
        else:
            self.expr_var.set("Expr:")
        self._draw_tree(tree, error)

    def _draw_tree(self, tree: Optional[EMLNode], error: str):
        self.tree_canvas.delete("all")
        if error:
            self.tree_canvas.create_text(
                20,
                20,
                anchor="nw",
                text=error,
                font=("Courier", 12, "bold"),
                fill="#b03a2e",
            )
            self.tree_canvas.config(scrollregion=(0, 0, 400, 200))
            return
        if tree is None:
            self.tree_canvas.config(scrollregion=(0, 0, 400, 200))
            return

        positions = {}
        max_depth = [0]
        next_leaf = [0]

        def assign(node: EMLNode, depth: int) -> float:
            max_depth[0] = max(max_depth[0], depth)
            if node.kind == "const":
                x = float(next_leaf[0])
                next_leaf[0] += 1
            else:
                x_left = assign(node.left, depth + 1)
                x_right = assign(node.right, depth + 1)
                x = (x_left + x_right) / 2.0
            positions[node] = (x, depth)
            return x

        assign(tree, 0)

        base_x_spacing = 120
        base_y_spacing = 90
        base_margin = 40
        base_radius = 22
        base_font = 10

        min_x = min(x for x, _ in positions.values())
        max_x = max(x for x, _ in positions.values())
        base_width = base_margin * 2 + (max_x - min_x + 1) * base_x_spacing
        base_height = base_margin * 2 + (max_depth[0] + 1) * base_y_spacing

        self.tree_canvas.update_idletasks()
        view_w = max(self.tree_canvas.winfo_width(), 1)
        view_h = max(self.tree_canvas.winfo_height(), 1)
        scale = min(view_w / base_width, view_h / base_height, 1.0)

        x_spacing = base_x_spacing * scale
        y_spacing = base_y_spacing * scale
        margin = base_margin * scale
        radius = max(6.0, base_radius * scale)
        font_size = max(7, int(round(base_font * scale)))

        width = base_width * scale
        height = base_height * scale

        def to_px(x: float, depth: float):
            return (margin + (x - min_x) * x_spacing, margin + depth * y_spacing)

        for node, (x, depth) in positions.items():
            if node.kind != "const":
                x1, y1 = to_px(x, depth)
                x2, y2 = to_px(*positions[node.left])
                x3, y3 = to_px(*positions[node.right])
                self.tree_canvas.create_line(
                    x1,
                    y1 + radius,
                    x2,
                    y2 - radius,
                    fill="#c0b6a7",
                    width=2,
                )
                self.tree_canvas.create_line(
                    x1,
                    y1 + radius,
                    x3,
                    y3 - radius,
                    fill="#c0b6a7",
                    width=2,
                )

        for node, (x, depth) in positions.items():
            x, y = to_px(x, depth)
            if node.kind == "const":
                fill = "#ffe7c7"
                outline = "#e2a86f"
            else:
                fill = "#d7ecff"
                outline = "#6aa7e5"
            self.tree_canvas.create_oval(
                x - radius - 2,
                y - radius - 2,
                x + radius + 2,
                y + radius + 2,
                fill="#000000",
                outline="",
            )
            self.tree_canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill=fill,
                outline=outline,
                width=2,
            )
            self.tree_canvas.create_text(
                x,
                y,
                text=node.label(),
                font=("Helvetica", font_size, "bold"),
                fill="#1f1f1f",
            )

        self.tree_canvas.config(scrollregion=(0, 0, width, height))

    def _click(self, label):
        if label == "C":
            self.current_input = ""
            self.display_var.set("0")
            self._set_tree_display("", None, "")
            return
        if label == "DEL":
            self.current_input = self.current_input[:-1]
            self.display_var.set(self.current_input or "0")
            return
        if label == "=":
            self._calculate()
            return

        if label in FUNCTIONS:
            self.current_input += f"{label}("
        else:
            if self.display_var.get() == "0" and label.isdigit():
                self.current_input = label
            else:
                self.current_input += label
        self.display_var.set(self.current_input)

    def _calculate(self):
        expr = self.current_input
        try:
            tokens = tokenize(expr)
            ast = Parser(tokens).parse()
            eml_tree = compile_to_eml(ast)
            result = eval_eml(eml_tree)
            formatted = format_complex(result)
            self.display_var.set(formatted)
            self.current_input = formatted
            self._set_tree_display(expr, eml_tree, "")
        except Exception as exc:
            self.display_var.set("Error")
            self.current_input = ""
            self._set_tree_display(expr, None, f"Error: {exc}")


if __name__ == "__main__":
    root = tk.Tk()
    app = EMLCalculator(root)
    root.mainloop()
