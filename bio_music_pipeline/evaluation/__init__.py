"""Evaluation module initialization."""

from .validator import (
    StatisticalValidator,
    InformationTransferClassifier,
    HumanEvaluationSurvey,
    run_comprehensive_evaluation
)
from .ablation import (
    ablate_bio_vector,
    create_ablated_dataset,
    run_ablation_study,
    compute_sequence_statistics,
    compare_statistics,
    ABLATION_GROUPS,
    BIO_VECTOR_COMPONENTS
)
from .visualizations import (
    visualize_bio_vectors,
    plot_correlation_heatmap,
    render_piano_roll,
    visualize_attention,
    create_all_visualizations
)
from .musical_quality import (
    MusicalQualityMetrics,
    compare_musical_quality
)
from .diversity import (
    DiversityAnalyzer
)
from .perplexity_metrics import (
    ShannonEntropyMetrics,
    compute_model_perplexity,
    compute_complexity_for_maestro
)
from .idyom_integration import (
    IDyOMWrapper,
    compare_idyom_vs_shannon,
    plot_idyom_vs_shannon
)

__all__ = [
    'StatisticalValidator',
    'InformationTransferClassifier',
    'HumanEvaluationSurvey',
    'run_comprehensive_evaluation',
    'ablate_bio_vector',
    'create_ablated_dataset',
    'run_ablation_study',
    'compute_sequence_statistics',
    'compare_statistics',
    'ABLATION_GROUPS',
    'BIO_VECTOR_COMPONENTS',
    'visualize_bio_vectors',
    'plot_correlation_heatmap',
    'render_piano_roll',
    'visualize_attention',
    'create_all_visualizations',
    'MusicalQualityMetrics',
    'compare_musical_quality',
    'DiversityAnalyzer',
    'ShannonEntropyMetrics',
    'compute_model_perplexity',
    'compute_complexity_for_maestro',
    'IDyOMWrapper',
    'compare_idyom_vs_shannon',
    'plot_idyom_vs_shannon'
]
