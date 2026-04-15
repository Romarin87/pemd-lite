from pathlib import Path
import logging
import os

from pemd_lite import load
from pemd_lite.charges import LigParGenBackend
from pemd_lite.forcefield import ChargeTransferPolicy, ForcefieldGenerator
from pemd_lite.logging_utils import configure_workflow_logging
from pemd_lite.md import GromacsRunner
from pemd_lite.pack import PackBuilder
from pemd_lite.polymer import PolymerBuildOptions, PolymerBuilder
from pemd_lite.relax import BoxEstimator, RelaxOptions, RelaxRunner


# ============================================================
# PEMD-Lite 单体系流程模板
# ------------------------------------------------------------
# 说明：
# 1. md.json 继续尽量保持老 PEMD 风格，只放体系参数。
# 2. 运行顺序、步数、是否执行某一步，都在这个 md.py 里控制。
# 3. 你可以直接注释掉某一步，或者修改执行顺序。
# ============================================================

work_dir = Path(__file__).resolve().parent
logger = configure_workflow_logging(work_dir / "workflow.log")
project = load(work_dir / "md.json")
logger.info("Workflow start: root=%s json=%s", work_dir, work_dir / "md.json")


def require_stage_output(name, value, hint):
    if value is None:
        raise RuntimeError(f"{name} is required here. {hint}")


# ============================================================
# 一、全局可改参数
# ------------------------------------------------------------
# 这些参数不写在 md.json，而是直接写在流程脚本里。
# 这样可以把体系参数和执行流程分开维护。
# ============================================================

# GROMACS 可执行程序：
# 推荐显式指定外部 gmx_mpi，例如：
# export PEMD_GMX_EXEC=/root/shared-nvme/soft/gromacs-2026.1/bin/gmx_mpi
# GROMACS 线程数：
# 这里会写入 PEMD_GMX_NTOMP，控制底层 mdrun 的 -ntomp。
GMX_THREADS = project.run.gmx_threads   # 默认 8
os.environ["PEMD_GMX_NTOMP"] = str(GMX_THREADS)

# 是否执行各阶段
RUN_BUILD_POLYMER = True          # 建短链和长链
RUN_POLYMER_FORCEFIELD = True     # 短链电荷 + 长链力场/电荷
RUN_SMALL_MOLECULE_FF = True      # Li / 阴离子等小分子力场
RUN_RELAX_CHAIN = True            # 单链驰豫
RUN_PACK_CELL = True              # 建盒子
RUN_BOX_MD = False                 # 盒子 MD（EM/NVT/NPT/production）


# ============================================================
# 二、建短链和长链
# ============================================================

polymer = None
if RUN_BUILD_POLYMER:
    # 建链参数：
    # 1. BUILD_RESUME = True 时，已有链文件会直接复用
    # 2. FORCE_REBUILD_* = True 时，只强制重建对应那一根链
    # 3. OPTIMIZE_EVERY_N_STEPS 控制长链增长时每几步做一次 MMFF
    #    标准脚本默认每一步都做一次 MMFF。
    BUILD_RESUME = True
    FORCE_REBUILD_SHORT_CHAIN = False
    FORCE_REBUILD_LONG_CHAIN = False
    OPTIMIZE_EVERY_N_STEPS = 1

    polymer = PolymerBuilder(
        project,
        options=PolymerBuildOptions(
            optimize_every_n_steps=OPTIMIZE_EVERY_N_STEPS,
            resume=BUILD_RESUME,
            force_rebuild_short_chain=FORCE_REBUILD_SHORT_CHAIN,
            force_rebuild_long_chain=FORCE_REBUILD_LONG_CHAIN,
        ),
    ).build_required()
    logger.info("Build section completed.")


# ============================================================
# 三、聚合物电荷和力场
# ------------------------------------------------------------
# 包含：
# 1. 短链 LigParGen 电荷
# 2. 长链 OPLS-AA 力场
# 3. 长链电荷迁移
# 4. 小分子力场（Li、阴离子等）
# ============================================================

forcefield = None
if RUN_POLYMER_FORCEFIELD:
    # 电荷后端：
    # 当前默认使用 LigParGen。
    CHARGE_BACKEND = LigParGenBackend()
    # 电荷迁移最短短链长度：
    # 默认 3，若你想强制要求 N4 才允许迁移，可以改成 4。
    MIN_SHORT_LENGTH = 3
    # 端部电荷迁移时，每一端取几个重复单元参与映射。
    # 当前默认 1。
    END_REPEATING_UNITS = 1

    require_stage_output(
        "polymer",
        polymer,
        "Set RUN_BUILD_POLYMER = True. If you want to reuse existing chains, keep BUILD_RESUME = True in the build section.",
    )

    ff = ForcefieldGenerator(
        project,
        charge_backend=CHARGE_BACKEND,
        charge_policy=ChargeTransferPolicy(
            min_short_length=MIN_SHORT_LENGTH,
            end_repeating_units=END_REPEATING_UNITS,
        ),
    )
    forcefield = ff.generate_polymer(
        short_mol=polymer.short_mol,
        long_mol=polymer.long_mol,
        short_pdb=polymer.short_pdb,
        long_pdb=polymer.long_pdb,
    )

    # 注意：
    # 小分子力场生成依赖当前这个 ff 对象，所以它虽然是单独开关，
    # 但仍然要求 RUN_POLYMER_FORCEFIELD = True。
    if RUN_SMALL_MOLECULE_FF:
        ff.generate_small_molecule_forcefields()
    logger.info("Forcefield section completed.")


# ============================================================
# 四、单链驰豫
# ------------------------------------------------------------
# 说明：
# 这里是“单链 relax”，不是盒子里的正式 MD。
# 通常用于后面估算盒子大小。
# ============================================================

relax = None
if RUN_RELAX_CHAIN:
    # 单链驰豫参数：
    # 这里控制的是单链 relax，不是盒子里的正式 MD。
    # RELAX_TEMPERATURE：单链 relax 温度，单位 K。
    # 温度高一些通常更容易把链拉开，但也更容易在早期引入不稳定。
    RELAX_TEMPERATURE = 500   # 默认 1000 K
    # RELAX_DT_PS：积分步长，单位 ps。
    # 越小越稳，但同样步数下对应的物理时间越短。
    RELAX_DT_PS = 0.0005
    # 盒子写法支持三种：
    # 1. 浮点数：固定盒子边长（单位 nm），例如 6.0
    # 2. "editconf"：等价于 gmx editconf -c -d 1.2
    # 3. "editconf -d x.x"：手动指定边界距离，例如 "editconf -d 1.5"
    RELAX_BOX = "editconf -d 1.2"

    # RELAX_RUN_*：是否允许执行对应阶段。
    # 注意：某个阶段即使这里设成 True，也仍然需要出现在 RELAX_STAGE_ORDER 里才会实际执行。
    RELAX_RUN_EM = True
    RELAX_RUN_NPT = True
    RELAX_RUN_NVT = False
    # RELAX_STAGE_ORDER：实际执行顺序。
    # 常见写法有 ["em", "npt"] 或 ["em", "nvt", "npt"]。
    RELAX_STAGE_ORDER = ["em", "npt"]

    # EM 阶段参数
    # RELAX_EM_OUTPUT：EM 输出文件前缀；None 表示自动命名为 {polymer_name}_relax_em。
    RELAX_EM_OUTPUT = None

    # NPT 阶段参数
    # RELAX_NPT_PRESSURE：NPT 目标压力，单位 bar。
    RELAX_NPT_PRESSURE = 1.0                           # 默认 1.0 bar
    # RELAX_NPT_STEPS：NPT 总步数。当前 dt=0.0005 ps 时，200000 steps 对应 100 ps。
    RELAX_NPT_STEPS = 200000                           # 默认 200000 steps = 100 ps
    # RELAX_NPT_TAU_T_PS：NPT 温度耦合时间常数，单位 ps。
    # 越小温度耦合越强，越大则更平缓。
    RELAX_NPT_TAU_T_PS = 1.0
    # RELAX_NPT_TAU_P_PS：NPT 压力耦合时间常数，单位 ps。
    # 越大表示压耦合更慢，通常也更稳。
    RELAX_NPT_TAU_P_PS = 10.0
    # RELAX_NPT_OUTPUT：NPT 输出文件前缀；None 表示自动命名为 {polymer_name}_relax_npt。
    RELAX_NPT_OUTPUT = None

    # NVT 阶段参数
    # RELAX_NVT_STEPS：NVT 总步数。只有在 RELAX_RUN_NVT=True 且 stage_order 包含 nvt 时才会生效。
    RELAX_NVT_STEPS = 200000                           # 默认 200000 steps = 100 ps
    # RELAX_NVT_TAU_T_PS：NVT 温度耦合时间常数，单位 ps。
    RELAX_NVT_TAU_T_PS = 1.0
    # RELAX_NVT_GEN_VEL：是否在 NVT 开始前重新生成速度。
    # 常用于把 NVT 当作真正预热阶段；若前一步已经是 MD，通常保持 False。
    RELAX_NVT_GEN_VEL = False
    # RELAX_NVT_OUTPUT：NVT 输出文件前缀；None 表示自动命名为 {polymer_name}_relax_nvt。
    RELAX_NVT_OUTPUT = None

    require_stage_output(
        "forcefield",
        forcefield,
        "Set RUN_POLYMER_FORCEFIELD = True. Relax uses the parameterized polymer GRO from the forcefield section.",
    )
    relax = RelaxRunner(project).run(
        forcefield.polymer_gro,
        options=RelaxOptions(
            temperature=RELAX_TEMPERATURE,
            pressure=RELAX_NPT_PRESSURE,
            box_mode=RELAX_BOX,
            dt_ps=RELAX_DT_PS,
            tau_t_ps=RELAX_NPT_TAU_T_PS,
            tau_p_ps=RELAX_NPT_TAU_P_PS,
            run_em=RELAX_RUN_EM,
            run_npt=RELAX_RUN_NPT,
            run_nvt=RELAX_RUN_NVT,
            stage_order=RELAX_STAGE_ORDER,
            npt_steps=RELAX_NPT_STEPS,
            nvt_steps=RELAX_NVT_STEPS,
            nvt_tau_t_ps=RELAX_NVT_TAU_T_PS,
            nvt_gen_vel=RELAX_NVT_GEN_VEL,
            em_output=RELAX_EM_OUTPUT,
            npt_output=RELAX_NPT_OUTPUT,
            nvt_output=RELAX_NVT_OUTPUT,
        ),
    )
    logger.info("Relax section completed.")


# ============================================================
# 五、建盒子
# ------------------------------------------------------------
# 先根据驰豫后的长链估算 add_length，再调用 packmol。
# 如果你已经有自己的盒子大小策略，可以直接替换这一段。
# ============================================================

box = None
pack = None
if RUN_PACK_CELL:
    # 盒子估算参数：
    # 默认值从 md.json 的 run 区读取；这里显式写出来方便改单体系脚本。
    # PACK_DENSITY：packmol 建盒子时使用的目标密度，单位 g/cm^3。
    PACK_DENSITY = project.run.density            # 默认 0.3
    # BOX_SCALE：按链尺寸估算 add_length 时的放大倍数。
    # 若想让盒子整体更宽松，可以增大这个值。
    BOX_SCALE = project.run.add_length_scale      # 默认 1.0
    # BOX_MIN_ADD_A：在链尺寸基础上额外增加的最小盒长，单位埃。
    # 即使链本身很短，也至少会额外加这么多空间。
    BOX_MIN_ADD_A = project.run.add_length_min_a  # 默认 100.0 埃

    if relax is not None:
        box_source_pdb = relax.relaxed_pdb
    else:
        require_stage_output(
            "polymer",
            polymer,
            "Set RUN_BUILD_POLYMER = True. Pack needs either a relaxed chain or the built long-chain PDB.",
        )
        box_source_pdb = polymer.long_pdb

    box = BoxEstimator().estimate_from_pdb(
        box_source_pdb,
        scale=BOX_SCALE,
        min_add_a=BOX_MIN_ADD_A,
    )
    original_density = project.run.density
    try:
        project.run.density = PACK_DENSITY
        pack = PackBuilder(project).run(
            add_length_a=box.add_length_a,
            polymer_pdb=box_source_pdb,
        )
    finally:
        project.run.density = original_density
    logger.info("Pack section completed.")


# ============================================================
# 六、盒子 MD
# ------------------------------------------------------------
# 推荐思路：
# - 顺序控制放在这里
# - 参数值也放在这里
# - 每一步默认自动接上一步输出
#
# 例如默认：
# poly_x_pack_cell.pdb -> poly_x_boxmd_conf.gro -> poly_x_boxmd_em.gro -> poly_x_boxmd_nvt.gro -> poly_x_boxmd_npt.gro
#
# 如果你想改顺序，例如先 NPT 再 EM，也可以直接改：
# gmx.build_flow(...).pdb_to_gro().npt(...).em(...).run()
# ============================================================

md = None
if RUN_BOX_MD:
    # 盒子 MD 参数：
    # 顺序、是否执行、步数，都在这里改。
    # 输出文件名前缀默认自动命名为：
    # {polymer_name}_boxmd_em / {polymer_name}_boxmd_nvt / {polymer_name}_boxmd_npt / {polymer_name}_boxmd_production
    # RUN_*：是否执行对应盒子 MD 阶段。
    RUN_EM = True
    RUN_NVT = True
    RUN_NPT = True
    RUN_PRODUCTION = False

    # *_STEPS：各阶段步数。
    # 物理时间 = 步数 x 该 mdp 中的 dt；默认 box MD 仍沿用底层 md.py 的 dt 配置。
    NVT_STEPS = 200000              # 默认 200000 steps = 200 ps
    NPT_STEPS = 200000              # 默认 200000 steps = 200 ps
    PRODUCTION_STEPS = 5000000      # 默认 5000000 steps = 5000 ps = 5 ns

    # *_TEMPERATURE / NPT_PRESSURE：盒子 MD 的目标热力学条件。
    # 默认都从 project.run 读取，便于和项目级参数保持一致。
    NVT_TEMPERATURE = project.run.production_temperature   # 默认 298 K
    NPT_TEMPERATURE = project.run.production_temperature   # 默认 298 K
    NPT_PRESSURE = project.run.production_pressure         # 默认 1.0 bar
    PRODUCTION_TEMPERATURE = project.run.production_temperature

    require_stage_output(
        "pack",
        pack,
        "Set RUN_PACK_CELL = True. Box MD needs the pack_cell.pdb generated in the packing section.",
    )

    gmx = GromacsRunner(project)
    flow = gmx.build_flow(pack.pack_pdb.name).pdb_to_gro()

    if RUN_EM:
        flow = flow.em()

    if RUN_NVT:
        flow = flow.nvt(
            steps=NVT_STEPS,
            temperature=NVT_TEMPERATURE,
        )

    if RUN_NPT:
        flow = flow.npt(
            steps=NPT_STEPS,
            temperature=NPT_TEMPERATURE,
            pressure=NPT_PRESSURE,
        )

    if RUN_PRODUCTION:
        flow = flow.production(
            steps=PRODUCTION_STEPS,
            temperature=PRODUCTION_TEMPERATURE,
        )

    md = flow.run()
    logger.info("Box MD section completed.")


# ============================================================
# 七、结果输出
# ============================================================

print("PEMD-Lite workflow completed.")

if polymer is not None:
    print("short_chain_pdb =", polymer.short_pdb)
    print("long_chain_pdb  =", polymer.long_pdb)

if forcefield is not None:
    print("polymer_bonded_itp    =", forcefield.polymer_bonded_itp)
    print("polymer_nonbonded_itp =", forcefield.polymer_nonbonded_itp)

if relax is not None:
    print("relaxed_pdb =", relax.relaxed_pdb)

if box is not None:
    print("box_add_length_a =", box.add_length_a)

if pack is not None:
    print("pack_cell_pdb =", pack.pack_pdb)

if md is not None:
    print("md_completed_stages =", md.completed_stages)


# ============================================================
# 八、可选写法示例
# ------------------------------------------------------------
# 1. 只建模到力场：
# RUN_RELAX_CHAIN = False
# RUN_PACK_CELL = False
# RUN_BOX_MD = False
#
# 2. 只做到单链驰豫：
# RUN_PACK_CELL = False
# RUN_BOX_MD = False
#
# 3. 盒子只跑 EM + NPT：
# RUN_EM = True
# RUN_NVT = False
# RUN_NPT = True
#
# 4. 自定义流程顺序：
# gmx = GromacsRunner(project)
# md = (
#     gmx.build_flow(pack.pack_pdb.name)
#     .pdb_to_gro()
#     .npt(output="npt1", steps=50000)
#     .em(output="em2")
#     .run()
# )
# ============================================================
