"""工具包：calculator / python_sandbox / file_search / wiki。"""
from . import calculator, file_search, python_sandbox, wiki

TOOL_MODULES = {
    "calculator": calculator,
    "python_sandbox": python_sandbox,
    "file_search": file_search,
    "wiki": wiki,
}


def all_schemas():
    return [m.TOOL_SCHEMA for m in TOOL_MODULES.values()]


def dispatch(name: str, args: dict) -> str:
    if name not in TOOL_MODULES:
        raise KeyError(f"未知工具: {name}；可选: {list(TOOL_MODULES)}")
    return TOOL_MODULES[name].run(args)
