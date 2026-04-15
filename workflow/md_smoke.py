from pathlib import Path

from pemd_lite import load
from pemd_lite.charges import LigParGenBackend
from pemd_lite.forcefield import ChargeTransferPolicy, ForcefieldGenerator
from pemd_lite.logging_utils import configure_workflow_logging
from pemd_lite.pack import PackBuilder
from pemd_lite.polymer import PolymerBuildOptions, PolymerBuilder
from pemd_lite.relax import BoxEstimator, RelaxOptions, RelaxRunner


# ============================================================
# PEMD-Lite 单体系 smoke 流程模板
# ------------------------------------------------------------
# 说明：
# 1. 这份脚本用于快速 smoke test，目标是尽快验证到 relax/pack。
# 2. 体系参数仍然从 md.json 读取，流程参数在这里单独控制。
# 3. 这份脚本默认缩短 relax 步数，并把 relax 温度固定为 500 K。
# ============================================================

work_dir = Path(__file__).resolve().parent
logger = configure_workflow_logging(work_dir / "workflow_smoke.log")
project = load(work_dir / "md.json")
project.polymer.length_long = 20
logger.info("Smoke workflow start: root=%s json=%s", work_dir, work_dir / "md.json")


def require_stage_output(name, value, hint):
    if value is None:
        raise RuntimeError(f"{name} is required here. {hint}")


# ============================================================
# 一、全局可改参数
# ------------------------------------------------------------
# smoke 脚本默认只跑到 pack，不跑 box MD。
# ============================================================

RUN_BUILD_POLYMER = True
RUN_POLYMER_FORCEFIELD = True
RUN_SMALL_MOLECULE_FF = True
RUN_RELAX_CHAIN = True
RUN_PACK_CELL = True


# ============================================================
# 二、建短链和长链
# ============================================================

polymer = None
if RUN_BUILD_POLYMER:
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
# ============================================================

forcefield = None
if RUN_POLYMER_FORCEFIELD:
    CHARGE_BACKEND = LigParGenBackend()
    MIN_SHORT_LENGTH = 3
    END_REPEATING_UNITS = 1

    require_stage_output(
        "polymer",
        polymer,
        "Set RUN_BUILD_POLYMER = True. Smoke flow still needs the built short and long chains.",
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

    if RUN_SMALL_MOLECULE_FF:
        ff.generate_small_molecule_forcefields()
    logger.info("Forcefield section completed.")


# ============================================================
# 四、单链驰豫
# ------------------------------------------------------------
# smoke 版 relax 默认更短，只跑 EM + NPT。
# ============================================================

relax = None
if RUN_RELAX_CHAIN:
    RELAX_TEMPERATURE = 500.0
    RELAX_DT_PS = 0.0005
    RELAX_BOX = "editconf -d 1.2"

    RELAX_RUN_EM = True
    RELAX_RUN_NPT = True
    RELAX_RUN_NVT = False
    RELAX_STAGE_ORDER = ["em", "npt"]

    RELAX_EM_OUTPUT = "smoke_relax_em"

    RELAX_NPT_PRESSURE = 1.0
    RELAX_NPT_STEPS = 20000
    RELAX_NPT_TAU_T_PS = 1.0
    RELAX_NPT_TAU_P_PS = 10.0
    RELAX_NPT_OUTPUT = "smoke_relax_npt"

    RELAX_NVT_STEPS = 10000
    RELAX_NVT_TAU_T_PS = 1.0
    RELAX_NVT_GEN_VEL = False
    RELAX_NVT_OUTPUT = "smoke_relax_nvt"

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
# ============================================================

box = None
pack = None
if RUN_PACK_CELL:
    PACK_DENSITY = 0.3
    BOX_SCALE = project.run.add_length_scale
    BOX_MIN_ADD_A = project.run.add_length_min_a

    require_stage_output(
        "relax",
        relax,
        "Set RUN_RELAX_CHAIN = True. Smoke flow packs from the relaxed long-chain structure.",
    )

    box = BoxEstimator().estimate_from_pdb(
        relax.relaxed_pdb,
        scale=BOX_SCALE,
        min_add_a=BOX_MIN_ADD_A,
    )
    original_density = project.run.density
    try:
        project.run.density = PACK_DENSITY
        pack = PackBuilder(project).run(
            add_length_a=box.add_length_a,
            polymer_pdb=relax.relaxed_pdb,
        )
    finally:
        project.run.density = original_density
    logger.info("Pack section completed.")


# ============================================================
# 六、结果输出
# ============================================================

print("PEMD-Lite smoke workflow completed.")

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
