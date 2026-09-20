"""Private reuse of the frozen atomic engine, changing only budget and schedule.

The original file stays byte-for-byte unchanged. AST edits are restricted to
fit/validate_completed: epoch constants 30/31 become 100/101, and the cosine
scheduler becomes an identity LambdaLR solely to preserve resumable state.
"""
import ast
import hashlib
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/real_video/train.py"
SOURCE_SHA256 = "90f26847fa0cb7915f40880187358e4cb0fbf3d65670292e5ba910503318bafe"


class Recipe(ast.NodeTransformer):
    def __init__(self):
        self.epochs = self.scheduler = 0

    def visit_Constant(self, node):
        if type(node.value) is int and node.value in (30, 31):
            self.epochs += 1
            return ast.copy_location(ast.Constant(node.value + 70), node)
        if isinstance(node.value, str):
            node.value = node.value.replace("30 epochs", "100 epochs").replace("30-epoch", "100-epoch")
        return node

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "CosineAnnealingLR":
            if ast.unparse(node) != 'torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=config[\'min_lr\'])':
                raise ValueError("Unexpected original scheduler expression")
            self.scheduler += 1
            return ast.copy_location(ast.parse("torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)", mode="eval").body, node)
        return self.generic_visit(node)


def load():
    source = SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Frozen atomic engine source differs")
    tree = ast.parse(source, filename=str(SOURCE))
    edits = Recipe()
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.FunctionDef) and node.name in ("fit", "validate_completed"):
            tree.body[index] = edits.visit(node)
    if edits.epochs != 10 or edits.scheduler != 1:
        raise ValueError(f"Unexpected bounded engine transformation: {edits.epochs}, {edits.scheduler}")
    ast.fix_missing_locations(tree)
    module = types.ModuleType("external_dinowm_raw_v2_atomic_engine")
    module.__file__ = str(SOURCE)
    exec(compile(tree, str(SOURCE), "exec"), module.__dict__)
    return module
