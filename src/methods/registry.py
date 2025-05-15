from .rebmbo import REBMBOClassic, REBMBOSparse, REBMBODeep
from .baselines import TuRBO, BALLETICI, EARLBO, ClassicBO, TwoStepEI, KnowledgeGradient

METHOD_REGISTRY = {
    'rebmbo_classic': REBMBOClassic,
    'rebmbo_sparse': REBMBOSparse,
    'rebmbo_deep': REBMBODeep,
    'turbo': TuRBO,
    'ballet_ici': BALLETICI,
    'earl_bo': EARLBO,
    'classic_bo': ClassicBO,
    'two_step_ei': TwoStepEI,
    'kg': KnowledgeGradient,
}
