"""
Stage 5: Evaluation and validation module.

Provides comprehensive evaluation including:
- Disentanglement metrics
- Correlation analysis between bio-features and musical parameters
- Permutation tests for significance
- Classifier-based information transfer verification
- Human evaluation survey generation
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
import json
from pathlib import Path
from .musical_quality import MusicalQualityMetrics


class StatisticalValidator:
    """Statistical validation of conditioning effectiveness."""
    
    def __init__(self, significance_level: float = 0.05, n_permutations: int = 1000):
        self.significance_level = significance_level
        self.n_permutations = n_permutations

    @staticmethod
    def _bootstrap_mean_ci(values: np.ndarray,
                           alpha: float = 0.05,
                           n_bootstrap: int = 2000,
                           seed: int = 42) -> Tuple[float, float]:
        """Bootstrap confidence interval for the mean."""
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return 0.0, 0.0
        if values.size == 1:
            v = float(values[0])
            return v, v

        rng = np.random.RandomState(seed)
        boot_means = np.empty(n_bootstrap, dtype=np.float64)
        n = values.size
        for i in range(n_bootstrap):
            sample = values[rng.choice(n, size=n, replace=True)]
            boot_means[i] = np.mean(sample)

        lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
        upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
        return lower, upper
    
    def compute_disentanglement_gap(self, 
                                   conditioned_scores: np.ndarray,
                                   unconditioned_scores: np.ndarray) -> Dict:
        """
        Compute disentanglement gap between conditioned and unconditioned generation.
        
        Args:
            conditioned_scores: Performance scores from conditioned model
            unconditioned_scores: Performance scores from unconditioned baseline
            
        Returns:
            Dictionary with gap statistics and significance test results
        """
        # Compute mean gap
        mean_gap = np.mean(conditioned_scores) - np.mean(unconditioned_scores)
        
        # Two-sample Welch t-test (more robust for unequal variances)
        t_stat, p_value = stats.ttest_ind(conditioned_scores, unconditioned_scores, equal_var=False)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(conditioned_scores)**2 + np.std(unconditioned_scores)**2) / 2)
        cohens_d = mean_gap / pooled_std if pooled_std > 0 else 0
        
        rng = np.random.RandomState(42)
        n_cond = len(conditioned_scores)
        n_uncond = len(unconditioned_scores)
        boot_gaps = np.empty(2000, dtype=np.float64)
        for i in range(2000):
            cond_sample = conditioned_scores[rng.choice(n_cond, size=n_cond, replace=True)]
            uncond_sample = unconditioned_scores[rng.choice(n_uncond, size=n_uncond, replace=True)]
            boot_gaps[i] = np.mean(cond_sample) - np.mean(uncond_sample)
        ci_low = float(np.percentile(boot_gaps, 2.5))
        ci_high = float(np.percentile(boot_gaps, 97.5))

        return {
            'mean_gap': float(mean_gap),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < self.significance_level),
            'cohens_d': float(cohens_d),
            'mean_gap_ci_95': [float(ci_low), float(ci_high)],
            'conditioned_mean': float(np.mean(conditioned_scores)),
            'unconditioned_mean': float(np.mean(unconditioned_scores)),
            'conditioned_std': float(np.std(conditioned_scores)),
            'unconditioned_std': float(np.std(unconditioned_scores))
        }
    
    def permutation_test(self, 
                        original_stat: float,
                        data1: np.ndarray,
                        data2: np.ndarray) -> Dict:
        """
        Perform permutation test for significance.
        
        Args:
            original_stat: Original test statistic
            data1: First dataset
            data2: Second dataset
            
        Returns:
            Permutation test results
        """
        combined = np.concatenate([data1, data2])
        n1 = len(data1)
        
        permuted_stats = []
        for _ in range(self.n_permutations):
            np.random.shuffle(combined)
            perm_stat = np.mean(combined[:n1]) - np.mean(combined[n1:])
            permuted_stats.append(perm_stat)
        
        # Two-tailed p-value
        extreme_count = sum(abs(s) >= abs(original_stat) for s in permuted_stats)
        p_value = extreme_count / self.n_permutations
        
        return {
            'original_statistic': float(original_stat),
            'permutation_p_value': float(p_value),
            'significant': bool(p_value < self.significance_level),
            'n_permutations': self.n_permutations
        }
    
    def test_correlation_significance(self, 
                                     x: np.ndarray, 
                                     y: np.ndarray,
                                     method: str = 'pearson') -> Dict:
        """
        Test correlation significance with noise robustness check.
        
        Args:
            x: First variable
            y: Second variable
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Correlation analysis results
        """
        # Compute correlation
        if method == 'pearson':
            corr, p_value = stats.pearsonr(x, y)
        elif method == 'spearman':
            corr, p_value = stats.spearmanr(x, y)
        elif method == 'kendall':
            corr, p_value = stats.kendalltau(x, y)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        # Confidence interval via bootstrap
        n_bootstrap = 1000
        bootstrap_corrs = []
        n = len(x)
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, n, replace=True)
            if method == 'pearson':
                boot_corr, _ = stats.pearsonr(x[indices], y[indices])
            elif method == 'spearman':
                boot_corr, _ = stats.spearmanr(x[indices], y[indices])
            else:
                boot_corr, _ = stats.kendalltau(x[indices], y[indices])
            bootstrap_corrs.append(boot_corr)
        
        ci_lower = np.percentile(bootstrap_corrs, 2.5)
        ci_upper = np.percentile(bootstrap_corrs, 97.5)
        
        # Noise robustness test
        noisy_corrs = []
        for noise_level in [0.01, 0.05, 0.1]:
            noisy_x = x + np.random.normal(0, noise_level * np.std(x), len(x))
            if method == 'pearson':
                noisy_corr, _ = stats.pearsonr(noisy_x, y)
            else:
                noisy_corr, _ = stats.spearmanr(noisy_x, y)
            noisy_corrs.append({
                'noise_level': noise_level,
                'correlation': float(noisy_corr),
                'robust': abs(noisy_corr - corr) < 0.1
            })
        
        return {
            'correlation': float(corr),
            'p_value': float(p_value),
            'significant': bool(p_value < self.significance_level),
            'ci_95_lower': float(ci_lower),
            'ci_95_upper': float(ci_upper),
            'method': method,
            'noise_robustness': noisy_corrs
        }
    
    def shuffle_test(self, 
                    bio_vectors: np.ndarray,
                    musical_features: np.ndarray,
                    metric_fn,
                    n_shuffles: int = 100) -> Dict:
        """
        Test if relationship survives feature shuffling.
        
        Args:
            bio_vectors: Bio-feature vectors
            musical_features: Extracted musical features
            metric_fn: Function to compute metric between bio and music features
            n_shuffles: Number of shuffle iterations
            
        Returns:
            Shuffle test results
        """
        # Original metric
        original_metric = metric_fn(bio_vectors, musical_features)
        
        # Shuffled metrics
        shuffled_metrics = []
        for _ in range(n_shuffles):
            shuffled_bio = np.random.permutation(bio_vectors)
            shuffled_metric = metric_fn(shuffled_bio, musical_features)
            shuffled_metrics.append(shuffled_metric)
        
        # Compare
        shuffled_mean = np.mean(shuffled_metrics)
        shuffled_std = np.std(shuffled_metrics)
        
        z_score = (original_metric - shuffled_mean) / shuffled_std if shuffled_std > 0 else 0
        p_value = sum(m >= original_metric for m in shuffled_metrics) / n_shuffles
        
        return {
            'original_metric': float(original_metric),
            'shuffled_mean': float(shuffled_mean),
            'shuffled_std': float(shuffled_std),
            'z_score': float(z_score),
            'p_value': float(p_value),
            'significant': bool(p_value < self.significance_level),
            'metric_retained': bool(original_metric > shuffled_mean + 2 * shuffled_std)
        }


class InformationTransferClassifier:
    """Classifier to verify information transfer from bio-vectors to music."""
    
    def __init__(self, n_clusters: int = 5, vocab: Optional[Dict[str, int]] = None,
                 significance_level: float = 0.05):
        self.n_clusters = n_clusters
        self.significance_level = significance_level
        self.cluster_labels = None
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.vocab = vocab or {}
        self.idx_to_token = {v: k for k, v in self.vocab.items()} if self.vocab else {}
    
    def cluster_bio_vectors(self, bio_vectors: np.ndarray) -> np.ndarray:
        """Cluster bio-vectors for classification task."""
        import warnings
        from sklearn.cluster import KMeans
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler

        clean = np.nan_to_num(bio_vectors.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        clean = np.clip(clean, -1e4, 1e4)

        # Robust per-dimension normalization to reduce outlier-driven overflow in KMeans.
        scale = np.percentile(np.abs(clean), 95, axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        normalized = np.clip(clean / scale, -20.0, 20.0)

        scaled = StandardScaler().fit_transform(normalized)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        scaled = np.clip(scaled, -20.0, 20.0)

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            labels = kmeans.fit_predict(scaled)

        if any(issubclass(w.category, RuntimeWarning) for w in caught):
            agglomerative = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage='ward'
            )
            labels = agglomerative.fit_predict(scaled)

        self.cluster_labels = labels
        return self.cluster_labels
    
    def extract_musical_features(self, generated_sequences: torch.Tensor) -> np.ndarray:
        """
        Extract statistical features from generated sequences.
        
        Features include:
        - Note density
        - Pitch statistics
        - Rhythm patterns
        - Velocity distribution
        """
        sequences = generated_sequences.cpu().numpy()
        features = []

        for seq in sequences:
            note_pitches = []
            velocities = []
            shift_values = []
            content_tokens = []
            eos_pos = len(seq)

            for i, tok_id in enumerate(seq):
                tok = self.idx_to_token.get(int(tok_id), "")
                if tok == "EOS" and eos_pos == len(seq):
                    eos_pos = i
                if tok.startswith("NOTE_ON_"):
                    note_pitches.append(int(tok.split("_")[2]))
                    content_tokens.append(tok_id)
                elif tok.startswith("VEL_"):
                    velocities.append(int(tok.split("_")[1]))
                    content_tokens.append(tok_id)
                elif tok.startswith("SHIFT_"):
                    shift_values.append(int(tok.split("_")[1]))
                    content_tokens.append(tok_id)
                elif tok and tok not in {"BOS", "EOS", "PAD"}:
                    content_tokens.append(tok_id)

            effective_len = max(eos_pos, 1)
            feat = np.array([
                len(note_pitches) / effective_len,
                np.mean(note_pitches) if note_pitches else 0.0,
                np.std(note_pitches) if len(note_pitches) > 1 else 0.0,
                np.mean(velocities) if velocities else 0.0,
                np.std(shift_values) if len(shift_values) > 1 else 0.0,
                len(set(content_tokens)) / max(len(self.vocab), 1),
            ])
            features.append(feat)

        feature_array = np.array(features, dtype=np.float64)
        return np.nan_to_num(feature_array, nan=0.0, posinf=1e6, neginf=-1e6)
    
    def train_and_evaluate(self, 
                          musical_features: np.ndarray,
                          bio_cluster_labels: np.ndarray,
                          n_folds: int = 5) -> Dict:
        """
        Train classifier to predict bio-clusters from musical features.
        
        Args:
            musical_features: Extracted musical features
            bio_cluster_labels: Cluster labels from bio-vectors
            n_folds: Cross-validation folds
            
        Returns:
            Classification results
        """
        labels = np.asarray(bio_cluster_labels, dtype=int)
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        effective_n_classes = len(unique_labels)
        class_count_map = {int(lbl): int(cnt) for lbl, cnt in zip(unique_labels, label_counts)}
        class_counts = np.bincount(labels) if labels.size > 0 else np.array([0])
        min_count = int(class_counts[class_counts > 0].min()) if np.any(class_counts > 0) else 1
        if min_count < 2:
            n_splits = 2
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            n_splits = max(2, min(n_folds, min_count))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Cross-validated metrics
        cv_scores = cross_val_score(
            self.classifier, musical_features, labels, cv=cv, scoring='accuracy'
        )
        cv_balanced_scores = cross_val_score(
            self.classifier, musical_features, labels, cv=cv, scoring='balanced_accuracy'
        )
        
        # Train final model
        self.classifier.fit(musical_features, labels)
        predictions = self.classifier.predict(musical_features)
        train_accuracy = accuracy_score(labels, predictions)
        train_balanced_accuracy = balanced_accuracy_score(labels, predictions)

        cv_ci_low, cv_ci_high = StatisticalValidator._bootstrap_mean_ci(
            cv_scores, alpha=0.05, n_bootstrap=2000, seed=42
        )
        cv_bal_ci_low, cv_bal_ci_high = StatisticalValidator._bootstrap_mean_ci(
            cv_balanced_scores, alpha=0.05, n_bootstrap=2000, seed=42
        )
        chance_level = float(1.0 / max(effective_n_classes, 1))
        
        return {
            'cv_mean_accuracy': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'cv_mean_accuracy_ci_95': [float(cv_ci_low), float(cv_ci_high)],
            'cv_mean_balanced_accuracy': float(cv_balanced_scores.mean()),
            'cv_std_balanced_accuracy': float(cv_balanced_scores.std()),
            'cv_mean_balanced_accuracy_ci_95': [float(cv_bal_ci_low), float(cv_bal_ci_high)],
            'train_accuracy': float(train_accuracy),
            'train_balanced_accuracy': float(train_balanced_accuracy),
            'n_classes_requested': self.n_clusters,
            'n_classes_effective': int(effective_n_classes),
            'class_counts': class_count_map,
            'chance_level': chance_level,
            'above_chance': bool(cv_scores.mean() > chance_level),
            'above_chance_balanced': bool(cv_balanced_scores.mean() > chance_level),
            'classification_report': classification_report(
                labels, predictions, output_dict=True
            ),
            'confusion_matrix': confusion_matrix(labels, predictions).tolist()
        }
    
    def permutation_importance_test(self,
                                   musical_features: np.ndarray,
                                   bio_cluster_labels: np.ndarray,
                                   n_permutations: int = 100) -> Dict:
        """
        Permutation test for classifier accuracy significance.
        
        Args:
            musical_features: Musical features
            bio_cluster_labels: True labels
            n_permutations: Number of permutations
            
        Returns:
            Permutation test results
        """
        labels = np.asarray(bio_cluster_labels, dtype=int)
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        effective_n_classes = len(unique_labels)
        class_count_map = {int(lbl): int(cnt) for lbl, cnt in zip(unique_labels, label_counts)}

        class_counts = np.bincount(labels) if labels.size > 0 else np.array([0])
        min_count = int(class_counts[class_counts > 0].min()) if np.any(class_counts > 0) else 1
        if min_count < 2:
            n_splits = 2
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            n_splits = max(2, min(5, min_count))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        original_scores = cross_val_score(
            self.classifier, musical_features, labels, cv=cv, scoring='accuracy'
        )
        original_balanced_scores = cross_val_score(
            self.classifier, musical_features, labels, cv=cv, scoring='balanced_accuracy'
        )
        original_acc = float(np.mean(original_scores))
        original_balanced_acc = float(np.mean(original_balanced_scores))

        perm_accuracies = []
        perm_balanced_accuracies = []
        for _ in range(n_permutations):
            shuffled_labels = np.random.permutation(labels)
            perm_scores = cross_val_score(
                self.classifier, musical_features, shuffled_labels, cv=cv, scoring='accuracy'
            )
            perm_balanced_scores = cross_val_score(
                self.classifier, musical_features, shuffled_labels, cv=cv, scoring='balanced_accuracy'
            )
            perm_accuracies.append(float(np.mean(perm_scores)))
            perm_balanced_accuracies.append(float(np.mean(perm_balanced_scores)))

        p_value = (sum(acc >= original_acc for acc in perm_accuracies) + 1) / (n_permutations + 1)
        p_value_balanced = (
            sum(acc >= original_balanced_acc for acc in perm_balanced_accuracies) + 1
        ) / (n_permutations + 1)
        perm_mean = float(np.mean(perm_accuracies))
        perm_std = float(np.std(perm_accuracies))
        perm_bal_mean = float(np.mean(perm_balanced_accuracies))
        perm_bal_std = float(np.std(perm_balanced_accuracies))
        z_score = float((original_acc - perm_mean) / perm_std) if perm_std > 0 else 0.0
        z_score_balanced = (
            float((original_balanced_acc - perm_bal_mean) / perm_bal_std) if perm_bal_std > 0 else 0.0
        )
        chance_level = float(1.0 / max(effective_n_classes, 1))
        
        return {
            'original_accuracy': original_acc,
            'original_balanced_accuracy': original_balanced_acc,
            'permuted_mean_accuracy': perm_mean,
            'permuted_std_accuracy': perm_std,
            'permuted_mean_balanced_accuracy': perm_bal_mean,
            'permuted_std_balanced_accuracy': perm_bal_std,
            'z_score_accuracy': z_score,
            'z_score_balanced_accuracy': z_score_balanced,
            'permutation_p_value': float(p_value),
            'permutation_p_value_balanced_accuracy': float(p_value_balanced),
            'significant': bool(p_value < self.significance_level),
            'significant_balanced_accuracy': bool(p_value_balanced < self.significance_level),
            'chance_level': chance_level,
            'n_classes_requested': self.n_clusters,
            'n_classes_effective': int(effective_n_classes),
            'class_counts': class_count_map,
            'n_permutations': int(n_permutations)
        }


class HumanEvaluationSurvey:
    """Generate and analyze human evaluation surveys."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_survey_html(self,
                            midi_files: Dict[str, List[str]],
                            n_trials: int = 20,
                            seed: int = 42) -> str:
        """
        Generate HTML survey for blind human evaluation.

        Args:
            midi_files: Dictionary mapping condition to list of file paths
            n_trials: Number of evaluation trials
            seed: Random seed

        Returns:
            Path to generated HTML file
        """
        np.random.seed(seed)

        # Prepare trial data with randomization
        conditions = list(midi_files.keys())
        trials = []

        import random
        for i in range(n_trials):
            # Randomly select condition and file
            condition = random.choice(conditions)
            file_idx = random.randint(0, len(midi_files[condition]) - 1)
            file_path = Path(midi_files[condition][file_idx])
            session_id = file_path.stem

            trials.append({
                'trial_id': i + 1,
                'condition': condition,
                'file': str(file_path),
                'session_id': session_id,
                'random_order': [int(x) for x in np.random.permutation(len(conditions))]
            })

        # Add attention checks (every 5th trial)
        attention_check_indices = [4, 9, 14, 19]
        for idx in attention_check_indices:
            if idx < len(trials):
                trials[idx]['is_attention_check'] = True

        # Generate HTML
        html_content = self._render_survey_template(trials, conditions)

        html_path = self.output_dir / 'human_evaluation_survey.html'
        with open(html_path, 'w') as f:
            f.write(html_content)

        return str(html_path)

    def _render_survey_template(self, trials: List[Dict], conditions: List[str]) -> str:
        """Render survey HTML template."""
        trials_json = json.dumps(trials)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bio-Music Human Evaluation Survey</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .trial {{ background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 8px; }}
        .rating {{ margin: 10px 0; }}
        .rating label {{ display: inline-block; margin: 0 10px; }}
        button {{ background: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; }}
        button:hover {{ background: #45a049; }}
        .attention-check {{ background: #fff3cd; border: 2px solid #ffc107; }}
        #results {{ display: none; }}
    </style>
</head>
<body>
    <h1>Bio-Music Generation Evaluation</h1>
    <p>Please listen to each audio sample and rate it on the following dimensions:</p>
    <ul>
        <li>Musicality (1-5): How musical/coherent does it sound?</li>
        <li>Structure (1-5): How well-structured is the piece?</li>
        <li>Variety (1-5): How varied/interesting is the content?</li>
    </ul>

    <div id="survey"></div>
    <button onclick="submitSurvey()">Submit Evaluation</button>

    <div id="results">
        <h2>Thank you for your participation!</h2>
        <p>Your responses have been recorded.</p>
    </div>

    <script>
        const trials = {trials_json};
        const conditions = {json.dumps(conditions)};

        function renderSurvey() {{
            const container = document.getElementById('survey');
            let html = '';

            trials.forEach((trial, idx) => {{
                const isAttentionCheck = trial.is_attention_check || false;
                const checkClass = isAttentionCheck ? 'attention-check' : '';

                html += `<div class="trial ${{checkClass}}" id="trial-${{idx}}">`;
                html += `<h3>Trial ${{trial.trial_id}}</h3>`;
                html += `<p>Condition: ${{trial.condition}}</p>`;
                html += `<audio controls><source src="/api/download/${{trial.session_id}}/midi" type="audio/midi"></audio>`;

                if (isAttentionCheck) {{
                    html += `<p><strong>Attention Check:</strong> Please select "3" for all ratings in this trial.</p>`;
                }}

                html += `<div class="rating">`;
                html += `<label>Musicality: `;
                html += `<select id="musicality-${{idx}}">`;
                for (let i = 1; i <= 5; i++) html += `<option value="${{i}}">${{i}}</option>`;
                html += `</select></label>`;

                html += `<label>Structure: `;
                html += `<select id="structure-${{idx}}">`;
                for (let i = 1; i <= 5; i++) html += `<option value="${{i}}">${{i}}</option>`;
                html += `</select></label>`;

                html += `<label>Variety: `;
                html += `<select id="variety-${{idx}}">`;
                for (let i = 1; i <= 5; i++) html += `<option value="${{i}}">${{i}}</option>`;
                html += `</select></label>`;
                html += `</div>`;
                html += `</div>`;
            }});

            container.innerHTML = html;
        }}

        async function submitSurvey() {{
            const responses = trials.map((trial, idx) => ({{
                trial_id: trial.trial_id,
                session_id: trial.session_id,
                condition: trial.condition,
                musicality: parseInt(document.getElementById(`musicality-${{idx}}`).value),
                structure: parseInt(document.getElementById(`structure-${{idx}}`).value),
                variety: parseInt(document.getElementById(`variety-${{idx}}`).value),
                is_attention_check: trial.is_attention_check || false
            }}));

            // Send to server
            try {{
                const response = await fetch('/api/survey/submit', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify(responses)
                }});

                if (response.ok) {{
                    // Save locally as backup
                    localStorage.setItem('surveyResponses', JSON.stringify(responses));

                    // Show results
                    document.getElementById('survey').style.display = 'none';
                    document.getElementById('results').style.display = 'block';
                }} else {{
                    alert('Failed to submit survey. Please try again.');
                }}
            }} catch (error) {{
                console.error('Submission error:', error);
                alert('Failed to submit survey. Please check your connection.');
            }}
        }}

        renderSurvey();
    </script>
</body>
</html>"""
        return html
    
    def compute_reliability(self, responses: List[Dict]) -> Dict:
        """
        Compute inter-rater reliability metrics.
        
        Args:
            responses: List of response dictionaries
            
        Returns:
            Reliability metrics
        """
        # Extract ratings
        musicality = [r['musicality'] for r in responses if not r.get('is_attention_check')]
        structure = [r['structure'] for r in responses if not r.get('is_attention_check')]
        variety = [r['variety'] for r in responses if not r.get('is_attention_check')]
        
        # Cronbach's alpha approximation
        def cronbach_alpha(items):
            n_items = len(items)
            if n_items < 2:
                return 0.0
            total_var = np.var(items)
            item_vars = [np.var(item) for item in items]
            alpha = (n_items / (n_items - 1)) * (1 - sum(item_vars) / total_var)
            return max(0, min(1, alpha))
        
        # Attention check performance
        attention_responses = [r for r in responses if r.get('is_attention_check')]
        attention_correct = sum(
            1 for r in attention_responses 
            if r['musicality'] == 3 and r['structure'] == 3 and r['variety'] == 3
        )
        attention_rate = attention_correct / len(attention_responses) if attention_responses else 1.0
        
        return {
            'cronbach_alpha_musicality': float(cronbach_alpha(musicality)),
            'cronbach_alpha_structure': float(cronbach_alpha(structure)),
            'cronbach_alpha_variety': float(cronbach_alpha(variety)),
            'attention_check_pass_rate': float(attention_rate),
            'n_valid_responses': len(responses) - len(attention_responses),
            'n_total_responses': len(responses)
        }


def run_comprehensive_evaluation(config: Dict,
                                 test_data: List,
                                 generated_samples: Dict,
                                 bio_vectors: np.ndarray,
                                 output_dir: str,
                                 vocab: Optional[Dict[str, int]] = None,
                                 generated_metadata: Optional[Dict] = None) -> Dict:
    """
    Run comprehensive evaluation suite.
    
    Args:
        config: Configuration dictionary
        test_data: Test dataset samples
        generated_samples: Dictionary of generated samples by condition
        bio_vectors: Bio-feature vectors
        output_dir: Output directory for results
        
    Returns:
        Complete evaluation results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    validator = StatisticalValidator(
        significance_level=config.get('significance_level', 0.05),
        n_permutations=config.get('n_permutations', 1000)
    )
    
    results = {}
    
    # 1. Musical quality + disentanglement analysis
    quality_summaries = {}
    per_condition_scores = {}
    for condition, sequences in generated_samples.items():
        scores, summary = compute_musical_quality_scores(np.asarray(sequences), vocab=vocab)
        per_condition_scores[condition] = scores
        quality_summaries[condition] = summary
    results['musical_quality'] = quality_summaries

    if 'conditioned' in generated_samples and 'unconditional' in generated_samples:
        cond_quality = per_condition_scores.get('conditioned', np.array([]))
        uncond_quality = per_condition_scores.get('unconditional', np.array([]))
        results['disentanglement'] = validator.compute_disentanglement_gap(cond_quality, uncond_quality)
        results['disentanglement']['metric'] = 'composite_musical_quality'
    
    # 2. Information transfer
    n_clusters = int(config.get('n_bio_clusters', 3))
    it_classifier = InformationTransferClassifier(
        n_clusters=n_clusters,
        vocab=vocab,
        significance_level=float(config.get('significance_level', 0.05))
    )
    cluster_labels = it_classifier.cluster_bio_vectors(bio_vectors)

    if 'conditioned' in generated_samples:
        n_gen = len(generated_samples['conditioned'])
        if generated_metadata and 'conditioned_bio_indices' in generated_metadata:
            raw_indices = np.asarray(generated_metadata['conditioned_bio_indices'], dtype=int)
            valid = (raw_indices >= 0) & (raw_indices < len(cluster_labels))
            indices = raw_indices[valid]
            if len(indices) >= n_gen:
                indices = indices[:n_gen]
            elif len(indices) > 0:
                pad = np.random.choice(indices, size=n_gen - len(indices), replace=True)
                indices = np.concatenate([indices, pad])
            else:
                indices = np.arange(n_gen)
        else:
            indices = np.arange(n_gen)

        subset_labels = cluster_labels[indices]

        mock_generated = torch.stack([torch.tensor(s) for s in generated_samples['conditioned']])

        musical_features = it_classifier.extract_musical_features(mock_generated)
        unique_labels, counts = np.unique(subset_labels, return_counts=True)
        results['information_transfer_labels'] = {
            'unique_labels': [int(x) for x in unique_labels.tolist()],
            'counts': [int(x) for x in counts.tolist()],
            'n_samples': int(len(subset_labels))
        }
        if len(unique_labels) < 2 or int(np.min(counts)) < 2:
            effective_classes = int(len(unique_labels))
            chance_level = float(1.0 / max(effective_classes, 1))
            results['information_transfer'] = {
                'error': 'Insufficient label support for reliable classification (need >=2 samples per class).',
                'cv_mean_accuracy': 0.0,
                'above_chance': False,
                'chance_level': chance_level,
                'n_classes_requested': int(n_clusters),
                'n_classes_effective': effective_classes
            }
            results['permutation_test'] = {
                'error': 'Skipped due to insufficient per-class support.',
                'significant': False,
                'permutation_p_value': 1.0,
                'chance_level': chance_level,
                'n_classes_requested': int(n_clusters),
                'n_classes_effective': effective_classes
            }
        else:
            results['information_transfer'] = it_classifier.train_and_evaluate(musical_features, subset_labels)
            results['permutation_test'] = it_classifier.permutation_importance_test(
                musical_features, subset_labels, n_permutations=int(config.get('n_permutations', 1000))
            )
    
    # 3. Save results
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def compute_sequence_quality(sequence: np.ndarray, vocab: Optional[Dict[str, int]] = None) -> float:
    """
    Compute a structural quality score for a token sequence.

    Higher is better. The score rewards:
    - non-trivial note activity
    - token diversity (without extreme randomness)
    - explicit sequence termination (EOS before max length)
    """
    seq = np.asarray(sequence)
    if len(seq) == 0:
        return 0.0

    eos_id = vocab.get("EOS") if vocab else None
    pad_id = vocab.get("PAD") if vocab else None
    idx_to_token = {v: k for k, v in vocab.items()} if vocab else {}

    if eos_id is not None and np.any(seq == eos_id):
        eos_pos = int(np.where(seq == eos_id)[0][0])
    else:
        eos_pos = len(seq)

    effective = seq[:max(eos_pos, 1)]
    if pad_id is not None:
        effective = effective[effective != pad_id]
    if len(effective) == 0:
        return 0.0

    note_on = 0
    note_off = 0
    shift = 0
    for tok_id in effective:
        tok = idx_to_token.get(int(tok_id), "")
        if tok.startswith("NOTE_ON_"):
            note_on += 1
        elif tok.startswith("NOTE_OFF_"):
            note_off += 1
        elif tok.startswith("SHIFT_"):
            shift += 1

    note_density = note_on / len(effective)
    balance = 1.0 - min(abs(note_on - note_off) / max(note_on + note_off, 1), 1.0)
    diversity = len(set(effective.tolist())) / max(len(effective), 1)
    eos_bonus = 1.0 if eos_pos < len(seq) - 1 else 0.0

    score = (
        0.35 * np.clip(note_density * 5.0, 0.0, 1.0) +
        0.25 * diversity +
        0.25 * balance +
        0.15 * eos_bonus
    )
    return float(np.clip(score, 0.0, 1.0))


def compute_musical_quality_scores(sequences: np.ndarray,
                                   vocab: Optional[Dict[str, int]] = None) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute per-sequence composite musical quality scores and summary metrics.

    Composite score combines:
    - tonal stability
    - rhythmic regularity
    - melodic coherence
    - harmonic richness
    - structural token quality (fallback-safe)
    """
    if len(sequences) == 0:
        return np.array([]), {'composite_mean': 0.0, 'composite_std': 0.0}

    metrics_calc = MusicalQualityMetrics(vocab=vocab or {})
    all_metrics = metrics_calc.compute_all_metrics(sequences)
    core_metrics = ['tonal_stability', 'rhythmic_regularity', 'melodic_coherence', 'harmonic_richness']

    scores = []
    for i, seq in enumerate(sequences):
        core_values = []
        for metric_name in core_metrics:
            values = all_metrics.get(metric_name, [])
            if i < len(values):
                core_values.append(float(values[i]))
        structural = compute_sequence_quality(seq, vocab=vocab)
        if core_values:
            composite = 0.8 * float(np.mean(core_values)) + 0.2 * structural
        else:
            composite = structural
        scores.append(float(np.clip(composite, 0.0, 1.0)))

    summary = {
        'composite_mean': float(np.mean(scores)),
        'composite_std': float(np.std(scores)),
        'structural_mean': float(np.mean([compute_sequence_quality(s, vocab=vocab) for s in sequences])),
    }
    for metric_name in core_metrics + ['self_similarity', 'repetition_rate']:
        vals = all_metrics.get(metric_name, [])
        summary[f'{metric_name}_mean'] = float(np.mean(vals)) if vals else 0.0

    return np.array(scores, dtype=np.float64), summary
