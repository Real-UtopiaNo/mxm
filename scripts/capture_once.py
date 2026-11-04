"""示例任务：只截图，不产生点击或按键。"""

from pet_alliance import ScriptContext


def run(ctx: ScriptContext) -> None:
    ctx.screenshot("capture_once")
