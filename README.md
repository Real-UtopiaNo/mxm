# 雷电模拟器游戏自动化


通过雷电自带的 ADB 获取 Android 原始画面、模拟点击/滑动/按键，后续每个任务作为独立 Python 脚本增量添加。Daily_Hero 使用真实游戏截图的视觉特征进行判断，支持 16:9 横屏，推荐 **1920×1080**。

## 新电脑从零运行

准备 Windows 10/11、Git、**64 位 Python 3.11 或以上**（安装时勾选加入 PATH），以及雷电模拟器。先在雷电安装游戏、登录账号，在模拟器设置中开启 **ADB 本地连接/本地调试**，设置为 1920×1080 横屏，并启动游戏。不同版本的设置名称可能略有不同。

在 PowerShell 中运行：

```powershell
git clone https://github.com/Real-UtopiaNo/mxm.git
cd mxm
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

初始化会创建本目录的 `.venv`、安装视觉依赖，再检查 ADB、在线设备、所有视觉模板和实际截图；不会向游戏发送点击。默认自动查找 PATH、环境变量、运行中的雷电/ADB、注册表和常见安装位置，并选择唯一在线的设备，无需照搬原电脑的安装目录或设备编号。

将游戏停在**天星镇主页面**，先预演再运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_daily_hero.ps1 -Preview
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_daily_hero.ps1
```

按 **Ctrl+C** 停止。无需激活虚拟环境，也无需管理员权限。更新代码后重新执行 `setup.ps1` 即可安装新增依赖。Python 不在 PATH 时可传入 `setup.ps1 -PythonPath 'D:\Python\python.exe'`；暂时不连接模拟器可用 `setup.ps1 -SkipDeviceCheck`，稍后执行下文的 `doctor`。

仓库包含代码、视觉模板和测试样本；Python、雷电、游戏本体和游戏账号需要在新电脑单独准备。脚本从已有游戏画面开始，不自动安装游戏或登录。

## 使用

以下命令均在 clone 的仓库目录内执行。截图和输入基础功能只使用 Python 标准库；`Daily_Hero` 和环境检查需要 `setup.ps1` 安装的视觉依赖：

```powershell
.\.venv\Scripts\python.exe -m pet_alliance doctor
.\.venv\Scripts\python.exe -m pet_alliance devices
.\.venv\Scripts\python.exe -m pet_alliance capture
.\.venv\Scripts\python.exe -m pet_alliance capture --output artifacts/screenshots/current.png
```

截图默认保存在 `artifacts/screenshots/`，每次生成带时间戳的新 PNG。获取的是游戏内容，不包含模拟器标题栏和工具栏；窗口缩放、移动不会改变截图坐标。

公共默认配置在 [automation.toml](automation.toml)。如果自动发现失败，复制 [automation.local.example.toml](automation.local.example.toml) 为 `automation.local.toml`，填写雷电安装目录中 `adb.exe` 的完整路径。该文件已被 Git 忽略，只覆盖填写的配置项；例如：

```toml
[device]
adb_path = 'D:\leidian\LDPlayer9\adb.exe'
serial = ''
```

`serial` 留空自动选择唯一在线设备。多个实例同时在线时，先执行 `devices` 查看编号，再填写 `serial`，避免操作其他实例。也可单次覆盖：

```powershell
.\.venv\Scripts\python.exe -m pet_alliance --serial emulator-5554 capture
.\.venv\Scripts\python.exe -m pet_alliance --adb 'D:\其他目录\adb.exe' devices
.\run_daily_hero.ps1 -AdbPath 'D:\其他目录\adb.exe' -Serial emulator-5554 -Preview
```

配置优先级是命令行参数 → `automation.local.toml` → `automation.toml`。显式使用 `--config` 时只读取指定的文件；相对路径以该配置文件所在目录为基准。也支持 `ADB_PATH` 指向 ADB，或 `LDPLAYER_HOME` 指向雷电安装目录。

如果设备列表为空，请在雷电设置中开启本地 ADB 调试，并保持模拟器启动；如果设备显示 `offline`，等待启动完成或重启模拟器后重试。若雷电使用 TCP 地址连接，可先运行 `& 'D:\leidian\LDPlayer9\adb.exe' connect 127.0.0.1:5555`，其中端口应使用当前实例实际的 ADB 端口，再执行 `devices`。排查后用 `doctor` 再次确认截图可用。

## 输入操作

坐标以原始截图左上角为 `(0, 0)`，向右增加 x，向下增加 y；1920×1080 截图的范围为 `0 ≤ x < 1920`、`0 ≤ y < 1080`。修改模拟器分辨率或切换横竖屏后，需要重新抓图确认手动坐标。Daily_Hero 会自动映射同为 16:9 的实际分辨率。

```powershell
# 先预演：只打印动作，不发送点击
python -m pet_alliance --dry-run tap 960 540

# 以下命令会实际操作游戏
python -m pet_alliance tap 960 540
python -m pet_alliance swipe 1400 540 500 540 --duration 400
python -m pet_alliance key back
```

## 增量添加任务

在 `scripts/` 新建文件，例如 `daily_task.py`，定义 `run(ctx)`。文件名使用英文字母、数字、下划线，并以字母开头。示例：

```python
from pet_alliance import ScriptContext


def run(ctx: ScriptContext) -> None:
    ctx.screenshot("before")
    ctx.tap(960, 540)
    ctx.wait(1)
    ctx.screenshot("after")
```

```powershell
python -m pet_alliance scripts
python -m pet_alliance run capture_once
python -m pet_alliance --dry-run run daily_task
python -m pet_alliance run daily_task
```

`ctx.capture()` 返回含 `data`（PNG 字节）、`width`、`height` 的对象，可作为后续按钮图像匹配、文字识别和状态判断的输入。`ctx.screenshot()` 保存并返回绝对路径。其他接口包括 `ctx.tap()`、`ctx.swipe()`、`ctx.key()`、`ctx.wait()` 和 `ctx.log`。新视觉任务可以复用 `pet_alliance.vision` 的模板匹配和坐标转换。

`--dry-run` 拦截通过 `ctx.tap/swipe/key` 发出的输入；截图、等待和脚本中的普通 Python 代码仍会执行。任务脚本应通过这些接口操作游戏；直接使用底层 `ctx.device` 会绕过预演。

运行在前台，按 **Ctrl+C** 停止。

## Daily_Hero：每日英雄级副本

已使用真实游戏截图建立 OpenCV 彩色模板库。通过“页面特征 → 区域名称和推荐等级 → 右上角感叹号 → 副本感叹号 → 英雄模式 → 挑战 → 跳过 → 下一关”的视觉判断执行。点击位置由匹配结果计算；模板统一在 1920×1080 下识别，并映射回设备实际分辨率，支持同比例的 16:9 横屏。

完成 `setup.ps1` 后，把游戏放在**天星镇主页面**，运行：

```powershell
.\.venv\Scripts\python.exe -m pet_alliance run Daily_Hero
```

也可以使用启动脚本：

```powershell
.\run_daily_hero.ps1
```

新环境初始化（网络可访问 Python 包源）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

只检查当前截图、显示下一步计划，不点击游戏：

```powershell
.\.venv\Scripts\python.exe -m pet_alliance --dry-run run Daily_Hero
# 或
.\run_daily_hero.ps1 -Preview
```

脚本也支持从区域地图、大地图或已打开的副本面板开始。预演只判断当前页面，不会假装走完尚未打开的页面。

- 大地图选择同时匹配区域名称、推荐等级和感叹号的区域；已处理区域在本次运行内不重复进入。当前区域的红色菱形可能遮住感叹号，其他区域处理后会补查一次该区域的副本标记。
- 区域入口使用匹配结果及该区域的 `click_offset` 计算点击位置；奇幻梦境、异界幻世点击图标，避免落在下方不可点击的推荐等级上。不同区域的偏移可在 `assets/daily_hero/profile.json` 分别调整。
- 区域内只进入带感叹号的副本；感叹号使用内部图像特征匹配，覆盖浅色地图和最右侧入口。聊天预览遮挡弗瑞斯时，结合稳定地标和标记残留颜色补查；完成和放弃的副本在本次运行内不再重复处理。
- 切换后必须同时识别英雄模式龙形标志和“挑战”按钮，才开始战斗。
- 所有导航、挑战、跳过、下一关、退出和领奖均采用“点一次 → 截图确认 → 未生效再补点”。任意两次点击默认至少间隔 **1 秒**；点击后观察 **2 秒**，只有原页面的按钮仍能重新识别时才补点，最多补点 **3 次**。加载中或原按钮已消失时只等待，不沿用旧坐标点击。
- 战斗中看到“跳过”立即点击。结算界面优先于背景战斗界面识别；获胜后继续“下一关”，按钮变灰时退出当前副本，若出现通关奖励则识别并领取。
- 失败后点击“退出战斗”。如果回到英雄面板，直接继续挑战，保留已经打过的关卡；如果返回地图，则重新进入同一副本。**不会把“回放”当成重试。**
- 默认首次挑战之外最多额外重试 **10 次**，即同一副本累计第 **11 次失败**时放弃；计数不因中间打赢某一关而重置，换副本后重新计数。下次运行会重新考虑这些副本。
- 未知弹窗、加载卡住、资源不足等导致页面无法识别时停止并保存现场。不会盲点弹窗；Ctrl+C 随时停止。

参数在 `automation.toml` 的 `[daily_hero]` 下：`min_click_interval` 是两次点击的最小间隔（秒），`click_retry_interval` 是点击后确认未生效的观察时间（秒），`max_click_retries` 是单次点击的补点上限。`max_retries` 单独控制战斗失败重试次数；`transition_timeout` 控制页面等待超时；`max_runtime` 控制总运行时长。`max_dungeons = 0` 表示遍历全部未完成副本，设为 `1` 可只处理一个副本。

每次运行生成 `artifacts/runs/Daily_Hero_时间戳/`，包含动作匹配分数、点击位置的 `events.jsonl`、完成/跳过/失败统计的 `summary.json`，以及失败和停止时的截图。

游戏逻辑位于 `pet_alliance/daily_hero.py`，通用视觉模块位于 `pet_alliance/vision.py`，模板及区域关联配置位于 `assets/daily_hero/`。其他任务可复用视觉模块并拥有各自的模板目录。游戏 UI 改版时更新模板及测试截图。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试使用模拟 ADB 和真实游戏截图回放，不向游戏发送输入。包括模式区分、感叹号关联、分辨率映射、灰色按钮、结算残留、失败重试及两种退出路径，以及漏点、延迟响应、点击节奏、加载中不补点和补点前重新定位。
